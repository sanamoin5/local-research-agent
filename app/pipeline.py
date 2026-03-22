import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from .repository import Repository
from .schemas import Event, Settings
from .reporting import build_limitations_seed, build_preview, build_report_metadata, detect_conflicts, generate_structured_report, render_markdown_report
from .services import (
    async_playwright,
    content_hash,
    extract_content,
    fetch_browser,
    fetch_http,
    generate_queries,
    normalize_url,
    ollama_generate,
    should_retry_with_browser,
    tavily_search,
)


@dataclass
class RunConfig:
    ollama_base_url: str
    tavily_api_key: str


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


async def summarize(task_text: str, sources: list[dict[str, Any]], ollama_base_url: str, model: str) -> str:
    if not sources:
        return (
            "## Summary\nInsufficient extractable sources were found to build a reliable synthesis.\n\n"
            "## Key Findings\n- Multiple candidate URLs were fetched, but usable extraction quality was too low.\n\n"
            "## Sources\n1. No usable sources were available for this run.\n\n"
            "## Limitations\n- This output is constrained by zero usable extracted pages."
        )

    blocks = [
        f"[{i}] {s['title']}\nURL: {s['url']}\nMethod: {s['extraction_method']}\nContent:\n{s['content_text'][:3500]}"
        for i, s in enumerate(sources, start=1)
    ]
    prompt = (
        "You are writing a source-backed research deliverable. Use ONLY provided source text.\n"
        "Return markdown with sections exactly: Summary, Key Findings, Sources, Limitations.\n"
        f"Task: {task_text}\n\nSources:\n" + "\n\n".join(blocks)
    )
    return await ollama_generate(prompt, ollama_base_url, model)


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

    usable_sources: list[dict[str, Any]] = []
    skipped_sources = 0
    step_idx = 1

    step_start = datetime.now(timezone.utc)
    await bus.emit(task_id, "step_started", {"step": "generate_queries", "message": "Generating search queries"})
    queries = await generate_queries(task_text, config.ollama_base_url, settings.model_name)
    repo.add_step(task_id, step_idx, "generate_queries", "completed", step_start, f"Generated {len(queries)} search queries", {"queries": queries})
    step_idx += 1

    step_start = datetime.now(timezone.utc)
    await bus.emit(task_id, "step_started", {"step": "search", "message": "Searching provider for candidate URLs"})
    merged: dict[str, dict[str, Any]] = {}
    for query in queries:
        try:
            rows = await tavily_search(config.tavily_api_key, query, settings.max_pages_per_query)
        except Exception as exc:
            await bus.emit(task_id, "warning", {"message": f"Search failure for '{query}': {exc.__class__.__name__}"})
            continue
        for row in rows:
            if not row.get("url"):
                continue
            row["normalized_url"] = normalize_url(row["url"])
            merged.setdefault(row["normalized_url"], row)
    candidates = list(merged.values())[: settings.max_pages_per_query]
    repo.add_step(task_id, step_idx, "search", "completed", step_start, f"Selected {len(candidates)} unique URLs", {"candidate_count": len(candidates)})
    step_idx += 1

    for candidate in candidates:
        normalized = candidate["normalized_url"]
        cache = repo.get_cache(normalized) if settings.cache_enabled else None
        if cache and repo.cache_reusable(cache):
            source = {
                **candidate,
                "normalized_url": normalized,
                "title": cache.get("title") or candidate.get("title"),
                "content_text": cache.get("extracted_text") or "",
                "quality": cache.get("quality") or "medium",
                "fetch_status": "ok",
                "extraction_status": "ok",
                "extraction_method": cache.get("extraction_method") or "http_trafilatura",
                "cache_hit": True,
            }
            repo.upsert_source(task_id, source)
            usable_sources.append(source)
            await bus.emit(task_id, "source_found", {"url": candidate["url"], "message": "Loaded cached extraction"})
            continue

        http_result = await fetch_http(candidate["url"], settings.fetch_timeout_sec, settings.max_page_size_bytes)
        extraction = extract_content(http_result.html or "", task_text, candidate.get("query", "")) if http_result.ok and http_result.html else None

        chosen_fetch = http_result
        chosen_extraction = extraction
        method = "http_trafilatura"

        retry, reason = should_retry_with_browser(http_result, extraction)
        if retry:
            await bus.emit(task_id, "warning", {"message": f"HTTP extraction weak ({reason}); retrying browser"})
            browser_result = await fetch_browser(candidate["url"], settings.fetch_timeout_sec + 2)
            browser_extraction = extract_content(browser_result.html or "", task_text, candidate.get("query", ""), fallback_title=browser_result.title) if browser_result.ok and browser_result.html else None
            if browser_result.ok and browser_extraction and browser_extraction["quality"] in {"medium", "good"}:
                chosen_fetch = browser_result
                chosen_extraction = browser_extraction
                method = "browser_trafilatura"

        if not chosen_fetch.ok or not chosen_extraction or chosen_extraction["quality"] == "poor":
            skipped_sources += 1
            repo.upsert_source(
                task_id,
                {
                    **candidate,
                    "normalized_url": normalized,
                    "fetch_status": chosen_fetch.fetch_status,
                    "http_status": chosen_fetch.http_status,
                    "content_type": chosen_fetch.content_type,
                    "content_length_bytes": chosen_fetch.content_length_bytes,
                    "extraction_status": "skipped",
                    "quality": "poor",
                    "extraction_method": method,
                    "error_reason": chosen_fetch.error_reason or "low_quality",
                },
            )
            await bus.emit(task_id, "source_skipped", {"url": candidate["url"], "reason": chosen_fetch.error_reason or "low_quality"})
            continue

        source = {
            **candidate,
            "normalized_url": normalized,
            "title": chosen_extraction["title"],
            "content_text": chosen_extraction["text"][:30_000],
            "quality": chosen_extraction["quality"],
            "fetch_status": "ok",
            "extraction_status": "ok",
            "extraction_method": method,
            "content_type": chosen_fetch.content_type,
            "content_length_bytes": chosen_fetch.content_length_bytes,
            "cache_hit": False,
        }
        repo.upsert_source(task_id, source)
        usable_sources.append(source)
        await bus.emit(task_id, "source_found", {"url": candidate["url"], "message": f"Usable source ({method})"})

        if settings.cache_enabled:
            repo.upsert_cache(
                normalized,
                candidate["url"],
                {
                    "fetch_method": chosen_fetch.fetch_method,
                    "content_type": chosen_fetch.content_type,
                    "http_status": chosen_fetch.http_status,
                    "html_text": chosen_fetch.html,
                    "extracted_text": source["content_text"],
                    "title": source["title"],
                    "quality": source["quality"],
                    "extraction_method": source["extraction_method"],
                    "fetch_status": "ok",
                    "extraction_status": "ok",
                    "error_reason": None,
                    "content_hash": content_hash(source["content_text"]),
                },
            )

    step_start = datetime.now(timezone.utc)
    await bus.emit(task_id, "conflict_analysis_started", {"message": "Analyzing source conflicts"})
    conflicts = detect_conflicts(usable_sources)
    repo.add_step(task_id, step_idx, "analyze_conflicts", "completed", step_start, f"Detected {len(conflicts)} conflicts", {"conflict_count": len(conflicts)})
    step_idx += 1
    await bus.emit(task_id, "conflict_analysis_completed", {"conflict_count": len(conflicts)})

    domain_count = len({(s.get("url", "").split("/")[2] if "/" in s.get("url", "") else s.get("url", "")) for s in usable_sources if s.get("url")})
    limitations = build_limitations_seed(
        usable_count=len(usable_sources),
        skipped_count=skipped_sources,
        fallback_used=repo.get_task(task_id).get("execution_mode") == "direct_fallback",
        conflict_count=len(conflicts),
        domain_count=domain_count,
    )

    step_start = datetime.now(timezone.utc)
    await bus.emit(task_id, "final_synthesis_started", {"message": "Generating final structured report"})
    structured, synthesis_mode = await generate_structured_report(
        task_text=task_text,
        sources=usable_sources,
        conflicts=conflicts,
        limitations_seed=limitations,
        ollama_base_url=config.ollama_base_url,
        model_name=settings.model_name,
    )
    markdown = render_markdown_report(structured)
    preview = build_preview(markdown)
    metadata = build_report_metadata(
        usable_count=len(usable_sources),
        skipped_count=skipped_sources,
        conflict_count=len(structured.conflicts),
        execution_mode=repo.get_task(task_id).get("execution_mode") or "agent",
        fallback_used=repo.get_task(task_id).get("execution_mode") == "direct_fallback",
        started_at=repo.get_task(task_id).get("started_at"),
    )
    metadata["synthesis_mode"] = synthesis_mode

    repo.add_step(task_id, step_idx, "final_synthesis", "completed", step_start, f"Synthesized report with {len(structured.findings)} findings", {"synthesis_mode": synthesis_mode})
    repo.store_report(
        task_id,
        markdown=markdown,
        structured_output=structured.model_dump(),
        output_preview=preview,
        usable_count=len(usable_sources),
        skipped_count=skipped_sources,
        conflict_count=len(structured.conflicts),
        report_metadata=metadata,
    )
    repo.update_task_status(task_id, "completed")
    await bus.emit(task_id, "final_synthesis_completed", {"synthesis_mode": synthesis_mode})
    await bus.emit(task_id, "output_ready", {"task_id": task_id})
    await bus.close(task_id)
    return {"status": "completed", "usable_sources": len(usable_sources)}


async def health_report(ollama_base_url: str, repo: Repository, tavily_api_key: str) -> dict[str, Any]:
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

    checks["tavily"] = {"ok": bool(tavily_api_key), "configured": bool(tavily_api_key)}
    checks["playwright"] = {"ok": async_playwright is not None}
    checks["overall"] = all(v.get("ok") for v in checks.values())
    return checks
