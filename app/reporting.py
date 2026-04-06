import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from . import config as cfg
from .services import ollama_generate


class ReportFinding(BaseModel):
    id: str
    text: str
    source_ids: list[str] = Field(min_length=1)
    confidence: str


class ReportConflict(BaseModel):
    topic: str
    description: str
    source_ids: list[str]


class ReportSource(BaseModel):
    id: str
    title: str
    url: str
    quality: str
    note: str


class StructuredReport(BaseModel):
    summary: str
    findings: list[ReportFinding]
    conflicts: list[ReportConflict] = []
    limitations: list[str] = Field(min_length=1)
    sources: list[ReportSource]


def _source_ids(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, s in enumerate(sources, start=1):
        copy = dict(s)
        copy["source_id"] = f"src_{i}"
        out.append(copy)
    return out


def build_limitations_seed(*, usable_count: int, skipped_count: int, fallback_used: bool, conflict_count: int, domain_count: int) -> list[str]:
    items = []
    if usable_count < 2:
        items.append("Few usable sources were available, so coverage may be incomplete.")
    if skipped_count > usable_count:
        items.append("Many candidate sources were skipped due to fetch or extraction quality issues.")
    if domain_count <= 1:
        items.append("Most usable evidence comes from a narrow set of domains.")
    if conflict_count > 0:
        items.append("Some claims conflict across sources and remain unresolved.")
    if fallback_used:
        items.append("Execution switched to direct fallback mode for reliability.")
    if not items:
        items.append("This report is based only on currently retrievable web content and may miss newer or inaccessible sources.")
    return items


def detect_conflicts(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    pattern = re.compile(r"\b(supports?|does not support|recommended|not recommended|required|optional)\b", re.I)
    topic_map: dict[str, list[tuple[str, str]]] = {}
    for i, s in enumerate(sources, start=1):
        text = (s.get("content_text") or "")[:2000]
        hits = pattern.findall(text)
        if not hits:
            continue
        sid = s.get("source_id", f"src_{i}")
        key = (s.get("title") or s.get("url") or "topic").split(" ")[0].lower()
        topic_map.setdefault(key, []).append((sid, " ".join(sorted(set([h.lower() for h in hits])))))

    for topic, entries in topic_map.items():
        if len(entries) < 2:
            continue
        phrases = {p for _, p in entries}
        if len(phrases) > 1:
            conflicts.append(
                {
                    "topic": topic,
                    "description": "Sources describe materially different claims for this topic.",
                    "source_ids": [sid for sid, _ in entries[:4]],
                }
            )
    return conflicts


def build_preview(markdown: str) -> str:
    text = re.sub(r"[#*`_\[\]()\-]", " ", markdown)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:220]


def build_report_metadata(*, usable_count: int, skipped_count: int, conflict_count: int, execution_mode: str, fallback_used: bool, started_at: str | None) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    duration_ms = None
    if started_at:
        try:
            duration_ms = int((now - datetime.fromisoformat(started_at)).total_seconds() * 1000)
        except Exception:
            duration_ms = None
    return {
        "usable_source_count": usable_count,
        "skipped_source_count": skipped_count,
        "conflict_count": conflict_count,
        "execution_mode": execution_mode,
        "fallback_used": fallback_used,
        "generated_at": now.isoformat(),
        "duration_ms": duration_ms,
    }


# ════════════════════════════════════════════════════════════════
#  Coordinator + synthesis pipeline
# ════════════════════════════════════════════════════════════════


async def coordinate_synthesis(
    task_text: str,
    sources: list[dict[str, Any]],
    ollama_base_url: str,
    model_name: str,
    temperature: float = cfg.COORDINATOR_TEMPERATURE,
) -> dict[str, Any]:
    """
    Coordinator agent: decides the synthesis strategy.

    Looks at the task complexity, number of sources, and content diversity
    to pick the fastest effective approach:
      DIRECT      — 1 LLM call, for simple/focused topics with similar sources
      MULTI_AGENT — analysts per theme + synthesizer, for complex/diverse topics

    For MULTI_AGENT, also groups sources by thematic angle so each analyst
    focuses on a coherent cluster rather than an arbitrary batch.
    """
    source_summaries = []
    for i, s in enumerate(sources):
        title = s.get("title") or "Untitled"
        snippet = (s.get("content_text") or "")[:120].strip().replace("\n", " ")
        source_summaries.append(f"  [{i}] {title}: {snippet}")

    prompt = (
        "You are a research coordinator. Given a research task and the sources collected, "
        "decide how to synthesize a report.\n\n"
        f"TASK: {task_text}\n\n"
        f"SOURCES ({len(sources)} total):\n" + "\n".join(source_summaries) + "\n\n"
        "Decide:\n"
        "- Are the sources covering ONE focused angle, or MULTIPLE distinct angles?\n"
        "- Is this a simple factual question or a complex multi-faceted topic?\n\n"
        "Respond in EXACTLY this format:\n\n"
        "STRATEGY: DIRECT or MULTI_AGENT\n"
        "REASON: (one sentence why)\n"
    )

    if len(sources) > 6:
        prompt += (
            "GROUPS: (only if MULTI_AGENT — group source indices by theme, one group per line)\n"
            "Group 1 [theme name]: 0, 1, 3\n"
            "Group 2 [theme name]: 2, 4, 5\n"
            "Group 3 [theme name]: 6, 7, 8\n"
        )

    try:
        output = await ollama_generate(prompt, ollama_base_url, model_name, timeout=cfg.COORDINATOR_TIMEOUT, temperature=temperature)
    except Exception:
        if len(sources) <= 5:
            return {"strategy": "DIRECT", "reason": "Coordinator failed, few sources — defaulting to direct", "groups": []}
        return {"strategy": "MULTI_AGENT", "reason": "Coordinator failed, many sources — defaulting to multi-agent", "groups": []}

    strategy = "DIRECT"
    reason = ""
    groups: list[dict[str, Any]] = []

    for line in output.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("STRATEGY:"):
            val = stripped.split(":", 1)[1].strip().upper()
            strategy = "MULTI_AGENT" if "MULTI" in val else "DIRECT"
        elif upper.startswith("REASON:"):
            reason = stripped.split(":", 1)[1].strip()
        elif upper.startswith("GROUP") and "[" in stripped:
            try:
                theme_part = stripped.split("[")[1].split("]")[0].strip()
                indices_part = stripped.split("]")[1].lstrip(":").strip()
                indices = [int(x.strip()) for x in indices_part.split(",") if x.strip().isdigit()]
                valid_indices = [idx for idx in indices if 0 <= idx < len(sources)]
                if valid_indices:
                    groups.append({"theme": theme_part, "indices": valid_indices})
            except (IndexError, ValueError):
                pass

    if strategy == "MULTI_AGENT" and not groups and len(sources) > 3:
        n = len(sources)
        groups = [
            {"theme": "primary", "indices": list(range(0, min(3, n)))},
            {"theme": "secondary", "indices": list(range(3, min(6, n)))},
        ]
        if n > 6:
            groups.append({"theme": "additional", "indices": list(range(6, min(9, n)))})

    if strategy == "DIRECT" and len(sources) > cfg.DIRECT_SOURCE_THRESHOLD:
        strategy = "MULTI_AGENT"
        reason = reason or "Too many sources for direct synthesis"

    return {"strategy": strategy, "reason": reason or "Coordinator decision", "groups": groups}


async def _direct_synthesis(
    task_text: str,
    sources: list[dict[str, Any]],
    ollama_base_url: str,
    model_name: str,
    temperature: float = cfg.SYNTHESIS_TEMPERATURE,
    max_tokens: int = cfg.SYNTHESIS_MAX_TOKENS,
) -> str:
    """Single-shot synthesis for focused topics. One LLM call, no intermediaries."""
    source_blocks = []
    for s in sources:
        title = s.get("title") or "Untitled"
        text = (s.get("content_text") or "")[:cfg.SOURCE_CONTENT_LENGTH]
        source_blocks.append(f"SOURCE: {title}\n{text}")

    prompt = (
        "You are a research writer. Write a DETAILED, comprehensive research report based on the sources below.\n\n"
        f"RESEARCH QUESTION: {task_text}\n\n"
        + "\n\n---\n\n".join(source_blocks) + "\n\n"
        "Write a thorough, in-depth report in markdown format:\n"
        "- Start with a clear title using #\n"
        "- Write a substantial executive summary paragraph\n"
        "- Create as many logical sections with ## headings as the topic demands\n"
        "- Each section should have detailed paragraphs with specific facts, examples, numbers, and context\n"
        "- Include relevant background, history, and technical details\n"
        "- End with a thorough ## Conclusion summarizing key takeaways\n\n"
        "IMPORTANT: Write as much useful detail as possible. Include everything from the sources.\n"
        "Do not be brief. Cover every angle, every fact, every detail. More is better.\n\n"
        "STRICT RULES:\n"
        "- Do NOT include any URLs, links, or source references\n"
        "- Do NOT include a sources/references section\n"
        "- Write for a reader who wants to deeply UNDERSTAND the topic\n\n"
        "REPORT:\n"
    )

    try:
        return await ollama_generate(prompt, ollama_base_url, model_name, timeout=cfg.SYNTHESIS_TIMEOUT, max_tokens=max_tokens, temperature=temperature)
    except Exception:
        facts = []
        for s in sources[:6]:
            text = (s.get("content_text") or "")
            sentences = [sent.strip() for sent in re.split(r"[.!?]\s+", text[:2000]) if len(sent.strip()) > 30]
            if sentences:
                facts.extend(sentences[:3])
        return f"# {task_text}\n\n" + "\n\n".join(facts) if facts else f"# {task_text}\n\nInsufficient data for report."


async def _analyze_source_batch(
    task_text: str,
    batch: list[dict[str, Any]],
    batch_label: str,
    ollama_base_url: str,
    model_name: str,
    temperature: float = cfg.ANALYST_TEMPERATURE,
    max_tokens: int = cfg.ANALYST_MAX_TOKENS,
) -> str:
    """Analyst agent: extract key insights from a batch of 2-3 sources."""
    source_blocks = []
    for s in batch:
        title = s.get("title") or "Untitled"
        text = (s.get("content_text") or "")[:cfg.SOURCE_CONTENT_LENGTH]
        source_blocks.append(f"SOURCE: {title}\n{text}")

    prompt = (
        "You are a thorough research analyst. Read the sources below and extract ALL information relevant to the research question.\n\n"
        f"RESEARCH QUESTION: {task_text}\n\n"
        + "\n\n---\n\n".join(source_blocks) + "\n\n"
        "Write a detailed analysis covering:\n"
        "- Every important fact, data point, statistic, and claim\n"
        "- Expert opinions, notable quotes, and key arguments\n"
        "- Specific examples, numbers, dates, names, and technical details\n"
        "- Historical context or background information\n"
        "- How this information answers the research question\n\n"
        "Be THOROUGH — extract every useful detail. Do not skip or summarize away information.\n"
        "Write as many paragraphs as needed to capture all the information.\n"
        "Do NOT include URLs or source references. State facts directly.\n\n"
        "ANALYSIS:\n"
    )

    try:
        return await ollama_generate(prompt, ollama_base_url, model_name, timeout=cfg.ANALYST_TIMEOUT, max_tokens=max_tokens, temperature=temperature)
    except Exception:
        facts = []
        for s in batch:
            text = (s.get("content_text") or "")
            sentences = [sent.strip() for sent in re.split(r"[.!?]\s+", text[:2000]) if len(sent.strip()) > 30]
            if sentences:
                facts.extend(sentences[:3])
        return "\n".join(facts) if facts else "Analysis unavailable for this batch."


async def _synthesize_final(
    task_text: str,
    analyses: list[str],
    ollama_base_url: str,
    model_name: str,
    temperature: float = cfg.SYNTHESIS_TEMPERATURE,
    max_tokens: int = cfg.SYNTHESIS_MAX_TOKENS,
) -> str:
    """Synthesizer agent: combine all analyses into a clean, reader-friendly report."""
    analyses_text = "\n\n---\n\n".join(
        f"ANALYSIS {i}:\n{a.strip()}" for i, a in enumerate(analyses, 1) if a.strip()
    )

    prompt = (
        "You are a senior research writer. Below are research notes on a topic gathered from multiple sources. "
        "Your job is to combine ALL of these notes into ONE thorough, well-written research report.\n\n"
        f"RESEARCH QUESTION: {task_text}\n\n"
        f"RESEARCH NOTES:\n{analyses_text}\n\n"
        "Write a DETAILED, comprehensive research report in markdown format. Requirements:\n"
        "- Start with a clear, informative title using #\n"
        "- Write a substantial executive summary\n"
        "- Create as many logical sections with ## headings as needed to cover all aspects\n"
        "- Each section should have detailed paragraphs with specific facts, examples, and context\n"
        "- Include ALL specific facts, numbers, dates, names, and technical details from the notes\n"
        "- If the notes contain contradictory information, present both sides and note the disagreement\n"
        "- Provide background, context, comparisons, and implications where relevant\n"
        "- End with a thorough ## Conclusion with key takeaways and forward-looking insights\n\n"
        "IMPORTANT: Include EVERY piece of useful information from the notes.\n"
        "Do not summarize away ANY details. More information is always better.\n"
        "Do not repeat yourself — but do not leave anything out either.\n\n"
        "STRICT RULES:\n"
        "- Do NOT include any URLs, links, or source references\n"
        "- Do NOT include a sources/references section\n"
        "- Do NOT use source IDs like [1], [src_1], etc.\n"
        "- Do NOT mention 'notes', 'research notes', or 'analysis'\n"
        "- Write as if you are the expert who did the research yourself\n"
        "- Write for a reader who wants to deeply UNDERSTAND the topic\n\n"
        "REPORT:\n"
    )

    try:
        return await ollama_generate(prompt, ollama_base_url, model_name, timeout=cfg.SYNTHESIS_TIMEOUT, max_tokens=max_tokens, temperature=temperature)
    except Exception:
        return f"# {task_text}\n\n" + "\n\n".join(a.strip() for a in analyses if a.strip())


def _clean_report_markdown(raw: str) -> str:
    """Remove any URLs, source references, or metadata the model might have snuck in."""
    lines = []
    skip_section = False
    for line in raw.splitlines():
        lower = line.strip().lower()
        if lower.startswith("## source") or lower.startswith("## reference") or lower.startswith("## bibliography") or lower.startswith("## works cited") or lower.startswith("## citations"):
            skip_section = True
            continue
        if skip_section and line.startswith("## "):
            skip_section = False
        if skip_section:
            continue
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        cleaned = re.sub(r"https?://\S+", "", cleaned)
        cleaned = re.sub(r"\(sources?:?\s*src_\d+[^)]*\)", "", cleaned)
        cleaned = re.sub(r"\bsrc_\d+\b", "", cleaned)
        # Catch [1], [2], (Source 1), (Source: 2), [Source 1] patterns
        cleaned = re.sub(r"\[(\d{1,2})\]", "", cleaned)
        cleaned = re.sub(r"\(Source:?\s*\d{1,2}\)", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\[Source:?\s*\d{1,2}\]", "", cleaned, flags=re.I)
        # Catch "According to Source 1" / "as noted in Analysis 2"
        cleaned = re.sub(r"\b(according to|as noted in|as mentioned in|per|from)\s+(source|analysis|analyst)\s*\d*\b", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        lines.append(cleaned.rstrip())
    return "\n".join(lines).strip()


def render_markdown_report(report: StructuredReport) -> str:
    """Render the StructuredReport for internal storage (backward compat)."""
    lines = ["## Summary", "", report.summary.strip(), "", "## Key Findings", ""]
    for f in report.findings:
        lines.append(f"- {f.text}")

    if report.conflicts:
        lines.extend(["", "## Source Conflicts", ""])
        for c in report.conflicts:
            lines.append(f"- **{c.topic}**: {c.description}")

    lines.extend(["", "## Limitations", ""])
    for lim in report.limitations:
        lines.append(f"- {lim}")
    return "\n".join(lines)


async def generate_structured_report(
    *,
    task_text: str,
    sources: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    limitations_seed: list[str],
    ollama_base_url: str,
    model_name: str,
    top_sources_cap: int = cfg.TOP_SOURCES_CAP,
    analyst_batch_size: int = cfg.ANALYST_BATCH_SIZE,
    synthesis_temperature: float = cfg.SYNTHESIS_TEMPERATURE,
    synthesis_max_tokens: int = cfg.SYNTHESIS_MAX_TOKENS,
    analyst_temperature: float = cfg.ANALYST_TEMPERATURE,
    analyst_max_tokens: int = cfg.ANALYST_MAX_TOKENS,
) -> tuple[StructuredReport, str]:
    """Multi-agent synthesis: analysts extract, synthesizer combines, output is clean markdown."""
    sources_with_ids = _source_ids(sources)

    quality_order = {"good": 0, "medium": 1, "poor": 2}
    ranked = sorted(sources_with_ids, key=lambda s: quality_order.get(s.get("quality", "medium"), 1))
    top_sources = ranked[:top_sources_cap]

    batch_size = analyst_batch_size
    batches: list[list[dict[str, Any]]] = []
    for i in range(0, len(top_sources), batch_size):
        batches.append(top_sources[i:i + batch_size])

    valid_analyses: list[str] = []
    for i, batch in enumerate(batches):
        try:
            analysis = await _analyze_source_batch(task_text, batch, f"batch_{i+1}", ollama_base_url, model_name, temperature=analyst_temperature, max_tokens=analyst_max_tokens)
            if isinstance(analysis, str) and len(analysis.strip()) > 20:
                valid_analyses.append(analysis)
        except Exception:
            pass

    if not valid_analyses:
        facts = []
        for s in sources_with_ids[:8]:
            text = (s.get("content_text") or "")
            sentences = [sent.strip() for sent in re.split(r"[.!?]\s+", text[:1500]) if len(sent.strip()) > 30]
            if sentences:
                facts.append(sentences[0])
        valid_analyses = ["\n".join(facts)] if facts else ["No usable information could be extracted from the sources."]

    # ── Stage 2: Final synthesis ──
    raw_report = await _synthesize_final(task_text, valid_analyses, ollama_base_url, model_name, temperature=synthesis_temperature, max_tokens=synthesis_max_tokens)
    clean_markdown = _clean_report_markdown(raw_report)

    if len(clean_markdown) < 50:
        clean_markdown = f"# {task_text}\n\n" + "\n\n".join(valid_analyses)

    # Build StructuredReport for internal storage
    findings = []
    for i, analysis in enumerate(valid_analyses[:8], 1):
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", analysis[:500]) if len(s.strip()) > 20]
        if sentences:
            batch_idx = min(i - 1, len(batches) - 1)
            src_ids = [s["source_id"] for s in batches[batch_idx]]
            findings.append({
                "id": f"finding_{i}",
                "text": sentences[0][:300],
                "source_ids": src_ids[:2],
                "confidence": "medium",
            })

    if not findings:
        findings = [{"id": "finding_1", "text": "Research completed but findings were limited.", "source_ids": ["src_1"], "confidence": "low"}]

    source_rows = [
        {"id": s["source_id"], "title": s.get("title") or "Untitled", "url": s.get("url", ""), "quality": s.get("quality", "medium"), "note": "analyzed"}
        for s in sources_with_ids
    ]

    structured = StructuredReport(
        summary=clean_markdown[:500].split("\n\n")[0].lstrip("# ").strip(),
        findings=findings,
        conflicts=conflicts,
        limitations=limitations_seed,
        sources=source_rows,
    )

    return structured, "multi_agent", clean_markdown


async def generate_structured_report_legacy(
    *,
    task_text: str,
    sources: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    limitations_seed: list[str],
    ollama_base_url: str,
    model_name: str,
) -> tuple[StructuredReport, str]:
    """Kept for backward compat — single-shot synthesis."""
    sources_with_ids = _source_ids(sources)
    source_blocks = []
    for s in sources_with_ids:
        sid = s["source_id"]
        title = s.get("title") or "Untitled"
        snippet = (s.get("content_text") or "")[:800]
        source_blocks.append(f"[{sid}] {title}\n{snippet}")

    prompt = (
        "You are a research analyst. Write a research report using ONLY the provided sources.\n\n"
        f"TASK: {task_text}\n\n"
        "SOURCES:\n" + "\n---\n".join(source_blocks) + "\n\n"
        "Write the report with these exact sections:\n\n"
        "## Summary\n(2-3 sentences)\n\n"
        "## Key Findings\n(5-8 bullet points)\n\n"
        "## Limitations\n(what this research might be missing)\n\n"
        "Report:\n"
    )

    try:
        raw = await ollama_generate(prompt, ollama_base_url, model_name)
        findings = []
        for i, line in enumerate(raw.splitlines(), 1):
            stripped = line.strip().lstrip("-*•· ")
            if len(stripped) > 15 and i <= 20:
                src_id = sources_with_ids[min(i - 1, len(sources_with_ids) - 1)]["source_id"]
                findings.append({"id": f"finding_{i}", "text": stripped[:300], "source_ids": [src_id], "confidence": "medium"})
        if findings:
            summary = findings[0]["text"]
            source_rows = [
                {"id": s["source_id"], "title": s.get("title") or "Untitled", "url": s.get("url", ""), "quality": s.get("quality", "medium"), "note": "analyzed"}
                for s in sources_with_ids
            ]
            return StructuredReport(summary=summary, findings=findings[:8], conflicts=conflicts, limitations=limitations_seed, sources=source_rows), "model_markdown"
    except Exception:
        pass

    findings = []
    for i, s in enumerate(sources_with_ids[:8], start=1):
        text = (s.get("content_text") or "")
        sentences = [sent.strip() for sent in re.split(r"[.!?]\s+", text[:1500]) if len(sent.strip()) > 30]
        best = sentences[0] if sentences else s.get("title", "Relevant source found")
        findings.append({"id": f"finding_{i}", "text": best[:300], "source_ids": [s["source_id"]], "confidence": "medium"})
    if not findings:
        findings = [{"id": "finding_1", "text": "Insufficient usable sources.", "source_ids": ["src_1"], "confidence": "low"}]
    summary = f"Research on \"{task_text[:80]}\" found {len(sources_with_ids)} relevant sources."
    source_rows = [
        {"id": s["source_id"], "title": s.get("title") or "Untitled", "url": s.get("url", ""), "quality": s.get("quality", "medium"), "note": "analyzed"}
        for s in sources_with_ids
    ]
    return StructuredReport(summary=summary, findings=findings, conflicts=conflicts, limitations=limitations_seed, sources=source_rows), "fallback_extract"
