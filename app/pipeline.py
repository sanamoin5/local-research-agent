import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from . import config as cfg
from .repository import Repository
from .schemas import Event, Settings, TaskPlan
from .reporting import (
    StructuredReport,
    _analyze_and_judge,
    _clean_report_markdown,
    _direct_synthesis,
    _extract_facts_from_source,
    _source_ids,
    _synthesize_final,
    build_limitations_seed,
    build_preview,
    build_report_metadata,
    coordinate_synthesis,
    detect_conflicts,
    generate_structured_report,
)
from .services import (
    async_playwright,
    autonomous_step,
    build_knowledge_summary,
    content_hash,
    extract_content,
    fetch_browser,
    fetch_http,
    normalize_url,
    should_retry_with_browser,
    web_search,
)

TRACE_LOG_DIR = Path(os.getenv("LRA_TRACE_LOG_DIR", "trace_logs"))
TRACE_LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RunConfig:
    ollama_base_url: str


_GLOBAL_REPO: Repository | None = None
_GLOBAL_BUS: "EventBus | None" = None


def set_runtime(repo: Repository, bus: "EventBus") -> None:
    global _GLOBAL_REPO, _GLOBAL_BUS
    _GLOBAL_REPO = repo
    _GLOBAL_BUS = bus


class EventBus:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Event]] = {}

    def queue_for(self, task_id: str) -> asyncio.Queue[Event]:
        if task_id not in self._queues:
            self._queues[task_id] = asyncio.Queue()
        return self._queues[task_id]

    async def emit(self, task_id: str, event_type: str, payload: dict[str, Any]) -> None:
        await self.queue_for(task_id).put(Event(type=event_type, payload=payload))

    async def close(self, task_id: str) -> None:
        await self.queue_for(task_id).put(Event(type="__close__", payload={}))


async def run_direct_pipeline(task_id: str, config: RunConfig, repo: Repository | None = None, bus: EventBus | None = None) -> dict[str, Any]:
    repo = repo or _GLOBAL_REPO
    bus = bus or _GLOBAL_BUS
    if repo is None or bus is None:
        raise RuntimeError("run_direct_pipeline requires runtime repo+bus")

    task = repo.get_task(task_id)
    if not task:
        return {"status": "failed", "error": "task_not_found"}

    settings = Settings(**json.loads(task["settings_snapshot_json"]))
    task_text = task["input_text"]
    repo.mark_started(task_id)

    try:
        return await asyncio.wait_for(
            _run_pipeline_inner(task_id, config, repo, bus, settings, task_text),
            timeout=settings.max_total_runtime_sec,
        )
    except asyncio.TimeoutError:
        await _save_partial_report(task_id, repo, bus, settings, task_text, config)
        return {"status": "completed", "error": "timeout_partial"}


async def _save_partial_report(task_id: str, repo: Repository, bus: EventBus, settings: Settings, task_text: str, config: RunConfig) -> None:
    """On timeout, synthesize whatever sources we have into a partial report."""
    sources = repo.list_sources(task_id)
    usable = [s for s in sources if s.get("quality") in ("good", "medium") and s.get("extraction_status") == "ok"]
    await bus.emit(task_id, "trace", {"trace_type": "warning", "label": f"Time limit reached — saving partial report ({len(usable)} sources)", "detail": ""})
    log_path = TRACE_LOG_DIR / f"{task_id}.txt"
    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n[TIMEOUT] Time limit {settings.max_total_runtime_sec}s reached — partial report with {len(usable)} sources\n")
    except Exception:
        pass
    if usable:
        from .reporting import build_limitations_seed, build_preview, build_report_metadata, detect_conflicts, generate_structured_report
        skipped = len(sources) - len(usable)
        conflicts = detect_conflicts(usable)
        limitations = build_limitations_seed(usable_count=len(usable), skipped_count=skipped, fallback_used=False, conflict_count=len(conflicts), domain_count=1)
        limitations.append(f"Research was cut short by the {settings.max_total_runtime_sec}s time limit.")
        try:
            structured, synthesis_mode, clean_markdown = await asyncio.wait_for(
                generate_structured_report(task_text=task_text, sources=usable, conflicts=conflicts, limitations_seed=limitations, ollama_base_url=config.ollama_base_url, model_name=settings.model_name),
                timeout=600,
            )
        except Exception:
            from .reporting import StructuredReport
            structured = StructuredReport(summary=f"Partial research on \"{task_text[:80]}\" — time limit reached with {len(usable)} sources.", findings=[{"id": "finding_1", "text": "Research was incomplete due to time limit.", "source_ids": ["src_0"], "confidence": "low"}], limitations=limitations, sources=[{"id": f"src_{i}", "title": s.get("title", ""), "url": s.get("url", ""), "quality": s.get("quality", "medium"), "note": "partial"} for i, s in enumerate(usable, 1)])
            synthesis_mode = "timeout_fallback"
            clean_markdown = f"# {task_text[:80]}\n\nResearch was cut short by the time limit. {len(usable)} sources were collected but synthesis failed."
        preview = build_preview(clean_markdown)
        metadata = build_report_metadata(usable_count=len(usable), skipped_count=skipped, conflict_count=len(conflicts), execution_mode="autonomous_partial", fallback_used=False, started_at=repo.get_task(task_id).get("started_at"))
        metadata["synthesis_mode"] = synthesis_mode
        repo.store_report(task_id, markdown=clean_markdown, structured_output=structured.model_dump(), output_preview=preview, usable_count=len(usable), skipped_count=skipped, conflict_count=len(conflicts), report_metadata=metadata)
        repo.update_task_status(task_id, "completed")
        await bus.emit(task_id, "output_ready", {"task_id": task_id})
    else:
        repo.update_task_status(task_id, "failed", error_message="Time limit reached with no usable sources", terminal_reason="timeout")
        await bus.emit(task_id, "error", {"message": f"Research timed out after {settings.max_total_runtime_sec}s with no usable sources"})
    await bus.close(task_id)


async def _run_pipeline_inner(task_id: str, config: RunConfig, repo: Repository, bus: EventBus, settings: Settings, task_text: str) -> dict[str, Any]:

    # ── Load goal + criteria from plan (set during planning phase) ──
    plan_raw = repo.get_task(task_id).get("plan_json", "{}")
    try:
        plan = TaskPlan(**json.loads(plan_raw))
        goal = plan.goal
        criteria = plan.success_criteria or [f"Find comprehensive information about: {task_text[:100]}"]
    except Exception:
        goal = task_text[:300]
        criteria = [f"Find comprehensive information about: {task_text[:100]}"]

    usable_sources: list[dict[str, Any]] = []
    skipped_sources = 0
    all_urls_seen: set[str] = set()
    search_history: list[str] = []
    step_idx = 1
    last_observation = "This is the very first iteration. No research done yet — start by searching."
    max_iterations = settings.max_iterations

    log_path = TRACE_LOG_DIR / f"{task_id}.txt"
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"=== Task: {task_id} ===\n")
    log_file.write(f"=== Query: {task_text} ===\n")
    log_file.write(f"=== Started: {datetime.now(timezone.utc).isoformat()} ===\n\n")

    async def _trace(tt: str, label: str, detail: str = "") -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        detail_trimmed = detail[:cfg.TRACE_DETAIL_MAX_LENGTH]
        await bus.emit(task_id, "trace", {"trace_type": tt, "label": label, "detail": detail_trimmed})
        repo.add_trace(task_id, tt, label, detail_trimmed)
        log_file.write(f"[{ts}] [{tt:14s}] {label}\n")
        if detail_trimmed:
            for line in detail_trimmed.splitlines():
                log_file.write(f"    {line}\n")
        log_file.flush()

    async def _search_and_fetch(queries: list[str], iteration: int) -> int:
        """Run a search+fetch cycle. Returns count of new usable sources found."""
        nonlocal skipped_sources, step_idx
        new_usable = 0

        # ── SEARCH ──
        step_start = datetime.now(timezone.utc)
        new_merged: dict[str, dict[str, Any]] = {}
        for qi, query in enumerate(queries):
            if qi > 0:
                await asyncio.sleep(settings.inter_query_delay)
            await _trace("search", f"Searching: {query}", f"Provider: duckduckgo | Max: {settings.max_pages_per_query}")
            try:
                rows = await web_search(query, settings.max_pages_per_query)
            except Exception as exc:
                await _trace("warning", f"Search failed: {query}", f"{exc.__class__.__name__}: {str(exc)[:200]}")
                continue
            await _trace("search_result", f"Found {len(rows)} results", "\n".join(f"  {r.get('title', '?')[:60]}" for r in rows[:5]))
            for row in rows:
                if not row.get("url"):
                    continue
                # Skip junk domains (search engines, redirect pages)
                _netloc = row["url"].lower().split("//")[-1].split("/")[0].lstrip("www.")
                if _netloc in cfg.JUNK_DOMAINS:
                    continue
                norm = normalize_url(row["url"])
                if norm not in all_urls_seen:
                    row["normalized_url"] = norm
                    new_merged.setdefault(norm, row)
        all_urls_seen.update(new_merged.keys())
        candidates = list(new_merged.values())[:settings.max_pages_per_query]
        await _trace("info", f"{len(candidates)} new candidates ({len(all_urls_seen)} total seen)")
        repo.add_step(task_id, step_idx, "search", "completed", step_start, f"Iteration {iteration}: {len(candidates)} new URLs", {"candidate_count": len(candidates)})
        step_idx += 1

        # ── FETCH + EXTRACT ──
        for ci, candidate in enumerate(candidates, 1):
            normalized = candidate["normalized_url"]
            cache = repo.get_cache(normalized) if settings.cache_enabled else None
            if cache and repo.cache_reusable(cache):
                source = {
                    **candidate, "normalized_url": normalized,
                    "title": cache.get("title") or candidate.get("title"),
                    "content_text": cache.get("extracted_text") or "",
                    "quality": cache.get("quality") or "medium",
                    "fetch_status": "ok", "extraction_status": "ok",
                    "extraction_method": cache.get("extraction_method") or "http_trafilatura",
                    "cache_hit": True,
                }
                repo.upsert_source(task_id, source)
                usable_sources.append(source)
                new_usable += 1
                await _trace("extract", f"[{ci}/{len(candidates)}] Cache hit: {source['title'][:60]}", f"Quality: {source['quality']}")
                await bus.emit(task_id, "source_found", {"url": candidate["url"], "message": "Cached"})
                continue

            await _trace("fetch", f"[{ci}/{len(candidates)}] Fetching {candidate['url'][:80]}", "")
            http_result = await fetch_http(candidate["url"], settings.fetch_timeout_sec, settings.max_page_size_bytes)
            extraction = extract_content(http_result.html or "", task_text, candidate.get("query", "")) if http_result.ok and http_result.html else None
            chosen_fetch, chosen_extraction, method = http_result, extraction, "http_trafilatura"

            retry, reason = should_retry_with_browser(http_result, extraction)
            if retry:
                await _trace("warning", f"HTTP weak ({reason}), trying browser", candidate['url'][:80])
                browser_result = await fetch_browser(candidate["url"], settings.fetch_timeout_sec + 2)
                browser_extraction = extract_content(browser_result.html or "", task_text, candidate.get("query", ""), fallback_title=browser_result.title) if browser_result.ok and browser_result.html else None
                if browser_result.ok and browser_extraction and browser_extraction["quality"] in {"medium", "good"}:
                    chosen_fetch, chosen_extraction, method = browser_result, browser_extraction, "browser_trafilatura"

            if not chosen_fetch.ok or not chosen_extraction or chosen_extraction["quality"] == "poor":
                skipped_sources += 1
                repo.upsert_source(task_id, {
                    **candidate, "normalized_url": normalized,
                    "fetch_status": chosen_fetch.fetch_status, "http_status": chosen_fetch.http_status,
                    "content_type": chosen_fetch.content_type, "content_length_bytes": chosen_fetch.content_length_bytes,
                    "extraction_status": "skipped", "quality": "poor", "extraction_method": method,
                    "error_reason": chosen_fetch.error_reason or "low_quality",
                })
                await _trace("skip", f"Skipped: {candidate['url'][:60]}", chosen_fetch.error_reason or "low_quality")
                await bus.emit(task_id, "source_skipped", {"url": candidate["url"], "reason": chosen_fetch.error_reason or "low_quality"})
                continue

            source = {
                **candidate, "normalized_url": normalized,
                "title": chosen_extraction["title"], "content_text": chosen_extraction["text"][:cfg.MAX_CONTENT_PER_SOURCE_STORED],
                "quality": chosen_extraction["quality"], "fetch_status": "ok", "extraction_status": "ok",
                "extraction_method": method, "content_type": chosen_fetch.content_type,
                "content_length_bytes": chosen_fetch.content_length_bytes, "cache_hit": False,
            }
            repo.upsert_source(task_id, source)
            usable_sources.append(source)
            new_usable += 1
            await _trace("extract", f"Usable: {source['title'][:60]}", f"Quality: {source['quality']} | Method: {method}")
            await bus.emit(task_id, "source_found", {"url": candidate["url"], "message": f"Usable ({method})"})

            if settings.cache_enabled:
                repo.upsert_cache(normalized, candidate["url"], {
                    "fetch_method": chosen_fetch.fetch_method, "content_type": chosen_fetch.content_type,
                    "http_status": chosen_fetch.http_status, "html_text": chosen_fetch.html,
                    "extracted_text": source["content_text"], "title": source["title"],
                    "quality": source["quality"], "extraction_method": source["extraction_method"],
                    "fetch_status": "ok", "extraction_status": "ok", "error_reason": None,
                    "content_hash": content_hash(source["content_text"]),
                })

        return new_usable

    # ════════════════════════════════════════════════════════════
    #  AUTONOMOUS LOOP:  THINK → ACT → OBSERVE → repeat
    #  One unified model call per iteration handles:
    #  THOUGHTS → REASONING → PLAN → CRITICISM → ACTION
    # ════════════════════════════════════════════════════════════

    await _trace("info", f"Goal: {goal}", f"Success criteria:\n" + "\n".join(f"  {i}. {c}" for i, c in enumerate(criteria, 1)))
    await bus.emit(task_id, "step_started", {"step": "autonomous_loop", "message": "Starting autonomous research loop"})

    consecutive_empty = 0  # counts iterations that found zero new sources

    for iteration in range(1, max_iterations + 1):
        await _trace("info", f"══ Iteration {iteration}/{max_iterations} ══", f"Sources so far: {len(usable_sources)} | Skipped: {skipped_sources}")

        # ── UNIFIED THINK: analysis + action decision ──
        knowledge = build_knowledge_summary(usable_sources)
        await _trace("model_call", f"[THINK] Autonomous reasoning (iter {iteration})",
                      f"Feeding observation from last round into the thinker...")

        await _trace("info", f"[OBSERVATION → THINKER] What happened last round",
                      last_observation[:800])

        step = await autonomous_step(
            goal, criteria, knowledge, search_history, last_observation,
            iteration, config.ollama_base_url, settings.model_name,
            temperature=settings.reasoning_temperature,
        )

        await _trace("model_result", f"[THOUGHTS] Agent's analysis (iter {iteration})",
                      step["thoughts"][:600])
        await _trace("model_result", f"[REASONING] Why this next action",
                      step["reasoning"][:400])
        await _trace("model_result", f"[CRITICISM] Self-critique",
                      step["criticism"][:400])
        await _trace("model_result", f"[PLAN] Next steps",
                      step["plan"][:400])
        await _trace("model_result", f"[DECISION] Action: {step['action']}",
                      "Queries:\n" + "\n".join(f"  {i}. {q}" for i, q in enumerate(step["queries"], 1))
                      if step["queries"] else "Ready to synthesize")

        good_sources = [s for s in usable_sources if s.get("quality") == "good"]
        min_to_complete = cfg.MIN_SOURCES_TO_COMPLETE if not good_sources else cfg.MIN_SOURCES_TO_COMPLETE_WITH_GOOD

        # COMPLETE if: agent decided so with enough sources, OR stuck with enough sources
        if step["action"] == "COMPLETE" and len(usable_sources) >= min_to_complete:
            await _trace("info", f"Agent decided COMPLETE at iteration {iteration} ({len(usable_sources)} sources, {len(good_sources)} good) — proceeding to synthesis")
            break

        if consecutive_empty >= cfg.MAX_CONSECUTIVE_EMPTY_ROUNDS and len(usable_sources) >= min_to_complete:
            await _trace("info", f"[STUCK] No new sources for {consecutive_empty} consecutive rounds with {len(usable_sources)} sources — moving to synthesis")
            break

        # ── ACT: search + fetch ──
        queries = step["queries"]

        # Guard: strip near-duplicates (e.g. "topic guide 3" when "topic guide 2" already searched)
        def _query_stem(q: str) -> str:
            return re.sub(r"\s+\d+$", "", q.lower().strip())

        history_stems = {_query_stem(q) for q in search_history}
        history_lower = {q.lower() for q in search_history}
        novel = [q for q in queries if q.lower() not in history_lower and _query_stem(q) not in history_stems]

        if not novel and queries:
            await _trace("warning", f"[GUARD] All {len(queries)} queries are duplicates or near-duplicates — generating fresh criteria-based angles")
            from .services import _criteria_fallback_queries
            queries = _criteria_fallback_queries(criteria, search_history, iteration, task_text)
            queries = [q for q in queries if q.lower() not in history_lower and _query_stem(q) not in history_stems]

        if not queries:
            await _trace("warning", f"[GUARD] Could not generate any novel queries at iter {iteration} — skipping search")
            consecutive_empty += 1
            await asyncio.sleep(settings.inter_iteration_cooldown)
            continue

        await _trace("info", f"[ACT] Executing {len(queries)} searches")
        new_count = await _search_and_fetch(queries, iteration)
        search_history.extend(queries)

        # ── OBSERVE: build observation of what was found for next iteration ──
        good_count = sum(1 for s in usable_sources if s.get("quality") == "good")
        med_count = sum(1 for s in usable_sources if s.get("quality") == "medium")
        quality_summary = f"Quality breakdown: {good_count} good, {med_count} medium"

        if new_count > 0:
            consecutive_empty = 0
            new_snippets = [
                f"- ({s.get('quality', '?')}) {s.get('title', '?')[:50]}: {(s.get('content_text') or '')[:cfg.OBSERVATION_SNIPPET_LENGTH].strip()}"
                for s in usable_sources[-new_count:]
            ]
            last_observation = (
                f"Found {new_count} new usable sources this round (total: {len(usable_sources)}). {quality_summary}.\n"
                f"New sources:\n" + "\n".join(new_snippets[:6])
            )
        elif len(usable_sources) > 0:
            consecutive_empty += 1
            last_observation = (
                f"No NEW sources found this round (searches returned duplicates or low-quality pages). "
                f"Total usable: {len(usable_sources)}, skipped: {skipped_sources}. {quality_summary}. "
                f"Try COMPLETELY DIFFERENT angles — focus on criteria not yet covered."
            )
        else:
            consecutive_empty += 1
            last_observation = (
                f"Still no usable sources after {iteration} iterations. "
                f"All {skipped_sources} pages were low quality or failed to fetch. "
                f"Try much broader or different search queries."
            )

        await _trace("info", f"[OBSERVE] Round {iteration} results",
                      last_observation[:600])

        await asyncio.sleep(settings.inter_iteration_cooldown)

    # ════════════════════════════════════════════════════════════
    #  SYNTHESIS
    # ════════════════════════════════════════════════════════════

    step_start = datetime.now(timezone.utc)
    await _trace("analysis", "Analyzing source conflicts", f"Comparing {len(usable_sources)} sources")
    await bus.emit(task_id, "conflict_analysis_started", {"message": "Analyzing source conflicts"})
    conflicts = detect_conflicts(usable_sources)
    await _trace("analysis", f"Conflict analysis: {len(conflicts)} found", "")
    repo.add_step(task_id, step_idx, "analyze_conflicts", "completed", step_start, f"Detected {len(conflicts)} conflicts", {"conflict_count": len(conflicts)})
    step_idx += 1
    await bus.emit(task_id, "conflict_analysis_completed", {"conflict_count": len(conflicts)})

    domain_count = len({(s.get("url", "").split("/")[2] if "/" in s.get("url", "") else s.get("url", "")) for s in usable_sources if s.get("url")})
    limitations = build_limitations_seed(
        usable_count=len(usable_sources), skipped_count=skipped_sources,
        fallback_used=False, conflict_count=len(conflicts), domain_count=domain_count,
    )

    step_start = datetime.now(timezone.utc)

    quality_order = {"good": 0, "medium": 1, "poor": 2}
    ranked = sorted(usable_sources, key=lambda s: quality_order.get(s.get("quality", "medium"), 1))
    top_sources = _source_ids(ranked[:settings.top_sources_cap])

    # ── COORDINATOR: decide synthesis strategy ──
    await _trace("model_call", "[COORDINATOR] Deciding synthesis strategy",
                  f"Examining {len(top_sources)} sources to determine: DIRECT (1 call) or MULTI_AGENT (themed analysts + synthesizer)")

    plan = await coordinate_synthesis(
        task_text, top_sources, config.ollama_base_url, settings.model_name,
    )
    strategy = plan["strategy"]
    groups = plan.get("groups", [])

    strategy_detail = f"Strategy: {strategy}\nReason: {plan['reason']}"
    if groups:
        strategy_detail += "\nThematic groups:\n" + "\n".join(
            f"  Group {i+1} [{g['theme']}]: sources {g['indices']}" for i, g in enumerate(groups)
        )
    await _trace("model_result", f"[COORDINATOR] Strategy: {strategy}",
                  strategy_detail)

    await bus.emit(task_id, "final_synthesis_started", {"message": f"Synthesis strategy: {strategy}"})

    detailed_findings_md = ""

    if strategy == "DIRECT":
        # ── DIRECT: multi-section report from sources ──
        n_criteria_d = len(criteria) if criteria else 0
        await _trace("model_call", f"[WRITER] Direct multi-section report ({n_criteria_d} sections)",
                      f"Writing report from {len(top_sources)} sources (focused topic)")
        raw_report = await _direct_synthesis(
            task_text, top_sources, config.ollama_base_url, settings.model_name,
            temperature=settings.synthesis_temperature,
            max_tokens=settings.synthesis_max_tokens,
            criteria=criteria,
        )
        clean_markdown = _clean_report_markdown(raw_report)
        detailed_findings_md = clean_markdown
        synthesis_mode = "direct"
        batches = [top_sources]

        await _trace("model_result", "[WRITER] Report complete",
                      clean_markdown[:600])

    else:
        # ── MULTI_AGENT: one fact-extractor per source + one synthesizer ──
        batches = []
        if groups:
            for g in groups:
                batch = [top_sources[i] for i in g["indices"] if i < len(top_sources)]
                if batch:
                    batches.append(batch)
        else:
            for bi in range(0, len(top_sources), cfg.ANALYST_BATCH_SIZE):
                batches.append(top_sources[bi:bi + cfg.ANALYST_BATCH_SIZE])

        await _trace("info", f"[MULTI_AGENT] {len(top_sources)} fact-extractor agents (1 per source) + 1 synthesizer")

        per_source_extracts: list[str] = []
        for si, source in enumerate(top_sources, 1):
            title = source.get("title", "?")[:60]
            await _trace("model_call", f"[EXTRACTOR {si}/{len(top_sources)}] {title}",
                          f"Extracting criterion-specific facts from source")
            try:
                extract = await _extract_facts_from_source(
                    task_text, source,
                    config.ollama_base_url, settings.model_name,
                    temperature=settings.synthesis_temperature,
                    max_tokens=min(settings.synthesis_max_tokens, 2048),
                    criteria=criteria,
                )
                if isinstance(extract, str) and len(extract.strip()) > 20:
                    per_source_extracts.append(extract.strip())
                    await _trace("model_result", f"[EXTRACTOR {si}] {title} — facts extracted",
                                  extract[:400])
                else:
                    await _trace("warning", f"[EXTRACTOR {si}] {title} — no useful facts")
            except Exception as exc:
                await _trace("warning", f"[EXTRACTOR {si}] {title} — failed: {exc.__class__.__name__}")

        if not per_source_extracts:
            facts = []
            for s in top_sources[:8]:
                text = (s.get("content_text") or "")
                sentences = [sent.strip() for sent in re.split(r"[.!?]\s+", text[:1500]) if len(sent.strip()) > 30]
                if sentences:
                    facts.append("- " + "\n- ".join(sentences[:3]))
            per_source_extracts = facts if facts else ["No usable information could be extracted."]

        # Detailed findings: raw concatenation of per-source extracts
        findings_parts = []
        for si, extract in enumerate(per_source_extracts):
            src_title = top_sources[si]["title"][:60] if si < len(top_sources) else f"Source {si+1}"
            findings_parts.append(f"## {src_title}\n\n{extract.strip()}")
        detailed_findings_md = "\n\n---\n\n".join(findings_parts)

        # ── ANALYSIS: deep comparison, judgment, recommendations ──
        await _trace("model_call", f"[ANALYST] Deep analysis of {len(per_source_extracts)} source extracts — comparing, judging, recommending")
        analysis = await _analyze_and_judge(
            task_text, per_source_extracts, config.ollama_base_url, settings.model_name,
            temperature=settings.synthesis_temperature,
            max_tokens=min(settings.synthesis_max_tokens, 4096),
            criteria=criteria,
        )
        if analysis.strip():
            await _trace("model_result", "[ANALYST] Expert analysis complete", analysis[:600])
        else:
            await _trace("warning", "[ANALYST] Analysis returned empty — synthesizer will work from extracts only")

        # ── SYNTHESIS: per-section writers + bookends → long comprehensive report ──
        n_criteria = len(criteria) if criteria else 0
        await _trace("model_call",
                      f"[SYNTHESIZER] Writing {n_criteria} sections + bookends from {len(per_source_extracts)} extracts + expert analysis",
                      "Each section gets its own dedicated writer agent for maximum depth")

        async def _synth_progress(label: str, detail: str) -> None:
            await _trace("model_result", label, detail)

        raw_report = await _synthesize_final(
            task_text, per_source_extracts, config.ollama_base_url, settings.model_name,
            temperature=settings.synthesis_temperature,
            max_tokens=settings.synthesis_max_tokens,
            criteria=criteria,
            analysis=analysis,
            progress_callback=_synth_progress,
        )
        clean_markdown = _clean_report_markdown(raw_report)
        synthesis_mode = "multi_agent"

        await _trace("model_result", f"[SYNTHESIZER] Report complete ({len(clean_markdown)} chars)",
                      clean_markdown[:600])

    if len(clean_markdown) < 50:
        clean_markdown = f"# {task_text}\n\nInsufficient data for a complete report."

    preview = build_preview(clean_markdown)

    findings = []
    for fi in range(min(len(batches), 6)):
        batch = batches[fi]
        src_ids = [s["source_id"] for s in batch[:2]]
        text = (batch[0].get("content_text") or "")
        import re as _re
        sentences = [s.strip() for s in _re.split(r"[.!?]\s+", text[:500]) if len(s.strip()) > 20]
        if sentences:
            findings.append({"id": f"finding_{fi+1}", "text": sentences[0][:300], "source_ids": src_ids, "confidence": "medium"})
    if not findings:
        findings = [{"id": "finding_1", "text": "Research completed.", "source_ids": ["src_1"], "confidence": "medium"}]

    source_rows = [
        {"id": s["source_id"], "title": s.get("title") or "Untitled", "url": s.get("url", ""), "quality": s.get("quality", "medium"), "note": "analyzed"}
        for s in top_sources
    ]
    structured = StructuredReport(
        summary=clean_markdown[:500].split("\n\n")[0].lstrip("# ").strip(),
        findings=findings,
        conflicts=conflicts,
        limitations=limitations,
        sources=source_rows,
    )

    metadata = build_report_metadata(
        usable_count=len(usable_sources), skipped_count=skipped_sources,
        conflict_count=len(structured.conflicts), execution_mode="autonomous",
        fallback_used=False, started_at=repo.get_task(task_id).get("started_at"),
    )
    metadata["synthesis_mode"] = synthesis_mode
    metadata["iterations_used"] = iteration
    metadata["detailed_findings"] = detailed_findings_md

    await _trace("model_result", f"Report finalized ({synthesis_mode})",
                  f"Preview:\n{clean_markdown[:600]}")
    repo.add_step(task_id, step_idx, "final_synthesis", "completed", step_start, f"Synthesized report ({len(structured.findings)} findings)", {"synthesis_mode": synthesis_mode})
    repo.store_report(
        task_id, markdown=clean_markdown, structured_output=structured.model_dump(),
        output_preview=preview, usable_count=len(usable_sources), skipped_count=skipped_sources,
        conflict_count=len(structured.conflicts), report_metadata=metadata,
    )
    repo.update_task_status(task_id, "completed")
    await bus.emit(task_id, "final_synthesis_completed", {"synthesis_mode": synthesis_mode})
    await bus.emit(task_id, "output_ready", {"task_id": task_id})
    await bus.close(task_id)

    log_file.write(f"\n=== Completed: {datetime.now(timezone.utc).isoformat()} ===\n")
    log_file.write(f"=== Iterations: {iteration} | Sources: {len(usable_sources)} | Skipped: {skipped_sources} ===\n")
    log_file.close()

    return {"status": "completed", "usable_sources": len(usable_sources), "iterations": iteration}


async def health_report(ollama_base_url: str, repo: Repository) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ollama_base_url}/api/tags")
            resp.raise_for_status()
            checks["ollama"] = {"ok": True, "models": len(resp.json().get("models", []))}
    except Exception as exc:
        checks["ollama"] = {"ok": False, "error": exc.__class__.__name__}

    try:
        repo.list_tasks()
        checks["sqlite"] = {"ok": True}
    except Exception as exc:
        checks["sqlite"] = {"ok": False, "error": str(exc)}

    from .services import DDGS
    checks["search"] = {"ok": DDGS is not None, "provider": "duckduckgo" if DDGS is not None else "none"}
    checks["playwright"] = {"ok": async_playwright is not None}
    checks["overall"] = all(v.get("ok") for v in checks.values())
    return checks
