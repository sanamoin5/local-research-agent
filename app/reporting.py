import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from . import config as cfg
from .services import ollama_generate

try:
    import chromadb
except Exception:  # pragma: no cover
    chromadb = None

try:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction as _STEmbedFn
except Exception:  # pragma: no cover
    _STEmbedFn = None


def _best_embedding_device() -> str:
    """Return the best available device for sentence-transformer embeddings."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class _FactStore:
    """
    Transient in-memory vector store for extracted facts.

    Splits each source extract into individual bullet-point facts, embeds them
    via ChromaDB's default sentence-transformer, and allows semantic retrieval
    per criterion. Falls back to regex Q-label matching when ChromaDB is unavailable.
    """

    def __init__(self, extracts: list[str]) -> None:
        self._extracts = extracts
        self._collection: Any | None = None
        self._available = False
        if chromadb is None or not extracts:
            return
        try:
            client = chromadb.Client()
            embed_kwargs: dict[str, Any] = {"metadata": {"hnsw:space": "cosine"}}
            if _STEmbedFn is not None:
                device = _best_embedding_device()
                try:
                    embed_kwargs["embedding_function"] = _STEmbedFn(
                        model_name="all-MiniLM-L6-v2",
                        device=device,
                    )
                except Exception:
                    pass  # fall back to ChromaDB default (onnxruntime on CPU)
            col = client.create_collection(name="facts", **embed_kwargs)
            docs: list[str] = []
            ids: list[str] = []
            idx = 0
            for ei, extract in enumerate(extracts):
                for line in extract.splitlines():
                    stripped = line.strip().lstrip("-•* ")
                    if len(stripped) > 20 and not stripped.upper().startswith(("Q", "OTHER RELEVANT", "SOURCE")):
                        docs.append(stripped)
                        ids.append(f"f_{ei}_{idx}")
                        idx += 1
                    elif len(stripped) > 20:
                        docs.append(stripped)
                        ids.append(f"f_{ei}_{idx}")
                        idx += 1
            if docs:
                col.add(documents=docs, ids=ids)
                self._collection = col
                self._available = True
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._available

    def query(self, criterion: str, n_results: int = 30) -> str:
        """Retrieve the most relevant facts for a given criterion."""
        if not self._available or not self._collection:
            return ""
        try:
            results = self._collection.query(
                query_texts=[criterion],
                n_results=min(n_results, self._collection.count()),
            )
            docs = results.get("documents", [[]])[0]
            return "\n".join(f"- {d}" for d in docs if d.strip())
        except Exception:
            return ""

    def query_all(self, criterion: str, n_results: int = 30) -> tuple[str, str]:
        """Return (relevant_text, other_context) for a criterion."""
        relevant = self.query(criterion, n_results)
        return relevant, ""


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
    criteria: list[str] | None = None,
) -> str:
    """
    Synthesis for fewer sources — still uses multi-section writing when criteria exist.
    Each criterion gets its own dedicated writer call for depth.
    """
    source_blocks = []
    for s in sources:
        title = s.get("title") or "Untitled"
        text = (s.get("content_text") or "")[:cfg.SOURCE_CONTENT_LENGTH]
        source_blocks.append(f"SOURCE: {title}\n{text}")
    all_source_text = "\n\n---\n\n".join(source_blocks)

    if not criteria or len(criteria) < 2:
        # Very simple task — single call is fine
        prompt = (
            f"Write a comprehensive research report on: {task_text}\n\n"
            f"SOURCES:\n{all_source_text}\n\n"
            "Write a DETAILED markdown report. Include an executive summary, "
            "multiple detailed sections, actionable recommendations, and a conclusion.\n"
            "Be specific — name tools, strategies, numbers. Be opinionated — recommend the best options.\n\n"
            "REPORT:\n"
        )
        try:
            return await ollama_generate(prompt, ollama_base_url, model_name, timeout=cfg.SYNTHESIS_TIMEOUT, max_tokens=max_tokens, temperature=temperature)
        except Exception:
            return f"# {task_text}\n\n{all_source_text}"

    # Multi-section approach even for direct synthesis
    section_markdowns: list[str] = []
    section_summaries: list[str] = []

    for criterion in criteria:
        section_title = criterion[:120].rstrip(".,")
        section_md = await _write_section(
            task_text, section_title, all_source_text[:8000], "",
            ollama_base_url, model_name, temperature=temperature, max_tokens=min(max_tokens, cfg.SECTION_WRITER_MAX_TOKENS),
        )
        if section_md.strip():
            if not section_md.strip().startswith("##"):
                section_md = f"## {section_title}\n\n{section_md}"
            section_markdowns.append(section_md)
            first_para = section_md.split("\n\n")[1] if "\n\n" in section_md else section_md[:200]
            section_summaries.append(f"Section: {section_title}\nSummary: {first_para[:300]}")

    summaries_text = "\n\n".join(section_summaries) if section_summaries else "No sections."
    exec_summary, recommendations, conclusion = await _write_bookends(
        task_text, summaries_text, criteria, ollama_base_url, model_name,
        temperature=temperature, max_tokens=min(max_tokens, cfg.SECTION_WRITER_MAX_TOKENS),
    )

    title_kw = task_text[:80].rstrip(".,!?")
    parts = [f"# {title_kw}\n"]
    if exec_summary:
        parts.append(f"## Executive Summary\n\n{exec_summary}\n")
    for section in section_markdowns:
        parts.append(f"\n{section}\n")
    if recommendations:
        parts.append(f"\n## Actionable Recommendations\n\n{recommendations}\n")
    if conclusion:
        parts.append(f"\n## Conclusion\n\n{conclusion}\n")
    return "\n".join(parts)


async def _extract_facts_from_source(
    task_text: str,
    source: dict[str, Any],
    ollama_base_url: str,
    model_name: str,
    temperature: float = cfg.ANALYST_TEMPERATURE,
    max_tokens: int = cfg.ANALYST_MAX_TOKENS,
    criteria: list[str] | None = None,
) -> str:
    """
    Fact-extractor agent: one agent per source, outputs structured bullet-point facts.
    Asks the model to answer each criterion specifically from THIS source only.
    """
    title = source.get("title") or "Untitled"
    text = (source.get("content_text") or "")[:cfg.SOURCE_CONTENT_LENGTH]

    criteria_qs = ""
    if criteria:
        criteria_qs = "QUESTIONS TO ANSWER FROM THIS SOURCE:\n" + "\n".join(
            f"  Q{i+1}: {c}" for i, c in enumerate(criteria)
        ) + "\n\n"

    prompt = (
        f"You are a fact extractor. Read the source below and extract specific, concrete facts.\n\n"
        f"RESEARCH TOPIC: {task_text}\n\n"
        + criteria_qs
        + f"SOURCE TITLE: {title}\n"
        f"SOURCE CONTENT:\n{text}\n\n"
        "Extract facts in this EXACT format:\n\n"
        + (
            "\n".join(
                f"Q{i+1} FACTS:\n- [specific fact from source, or NONE]\n- [another fact if found]"
                for i in range(len(criteria or []))
            ) + "\n\n"
            if criteria else ""
        )
        + "OTHER RELEVANT FACTS:\n"
        "- [any other useful specific facts from this source]\n\n"
        "STRICT RULES:\n"
        "- Only include facts EXPLICITLY stated in the source content above\n"
        "- Name specific tools, platforms, companies, and techniques mentioned in the source\n"
        "- Include numbers, statistics, and concrete details when present\n"
        "- Write NONE if the source does not address a question\n"
        "- Do NOT add your own knowledge — only extract from the source\n"
        "- Do NOT include URLs\n"
    )

    try:
        return await ollama_generate(
            prompt, ollama_base_url, model_name,
            timeout=cfg.ANALYST_TIMEOUT, max_tokens=max_tokens, temperature=temperature,
        )
    except Exception:
        # Fallback: grab first meaningful sentences from the source
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", text[:2000]) if len(s.strip()) > 30]
        return "\n".join(f"- {s}" for s in sentences[:5]) if sentences else "- No extractable facts."


async def _analyze_source_batch(
    task_text: str,
    batch: list[dict[str, Any]],
    batch_label: str,
    ollama_base_url: str,
    model_name: str,
    temperature: float = cfg.ANALYST_TEMPERATURE,
    max_tokens: int = cfg.ANALYST_MAX_TOKENS,
    criteria: list[str] | None = None,
) -> str:
    """
    Analyst agent: runs _extract_facts_from_source on each source in the batch
    sequentially and concatenates results. Kept for backward compatibility with
    the coordinator's thematic grouping, but delegates to per-source extraction.
    """
    parts = []
    for s in batch:
        result = await _extract_facts_from_source(
            task_text, s, ollama_base_url, model_name,
            temperature=temperature, max_tokens=min(max_tokens, cfg.ANALYST_MAX_TOKENS),
            criteria=criteria,
        )
        parts.append(f"[Source: {(s.get('title') or 'Untitled')[:60]}]\n{result.strip()}")
    return "\n\n---\n\n".join(parts) if parts else "No facts extracted."


async def _analyze_and_judge(
    task_text: str,
    extracts: list[str],
    ollama_base_url: str,
    model_name: str,
    temperature: float = cfg.SYNTHESIS_TEMPERATURE,
    max_tokens: int = cfg.SYNTHESIS_MAX_TOKENS,
    criteria: list[str] | None = None,
) -> str:
    """
    Analysis/judgment agent: reads all per-source fact extracts, then produces
    a deep comparative analysis with the model's own reasoning, trade-offs,
    pros/cons, and recommendations. This is the "thinking" step before writing.
    """
    extracts_text = "\n\n---\n\n".join(
        f"EXTRACT {i}:\n{a.strip()}" for i, a in enumerate(extracts, 1) if a.strip()
    )

    criteria_block = ""
    if criteria:
        criteria_block = "RESEARCH CRITERIA:\n" + "\n".join(
            f"  {i+1}. {c}" for i, c in enumerate(criteria)
        ) + "\n\n"

    prompt = (
        "You are a senior research analyst. You have fact extracts from multiple sources.\n"
        "Your job is to THINK DEEPLY about what these facts mean and provide expert analysis.\n\n"
        f"RESEARCH QUESTION: {task_text}\n\n"
        + criteria_block
        + f"COLLECTED FACTS:\n{extracts_text}\n\n"
        "Now write your ANALYSIS. For each criterion:\n\n"
    )

    if criteria:
        for i, c in enumerate(criteria):
            short = c[:80].rstrip(".,")
            prompt += (
                f"CRITERION {i+1}: {short}\n"
                f"- What did the sources say about this?\n"
                f"- What are the best options and why?\n"
                f"- What are the trade-offs, pros, and cons?\n"
                f"- What would you recommend and why?\n\n"
            )

    prompt += (
        "OVERALL JUDGMENT:\n"
        "- What is the best overall strategy based on all the evidence?\n"
        "- What should the reader do first, second, third?\n"
        "- What are the biggest risks or pitfalls?\n\n"
        "Rules:\n"
        "- Use the facts from the extracts as evidence, but ADD your own reasoning and judgment\n"
        "- Compare options and explain WHY one is better than another\n"
        "- Be opinionated — don't just list options, RECOMMEND the best ones\n"
        "- Be specific: name tools, amounts, timelines, strategies\n"
        "- Think about what a smart advisor would say, not just what the websites said\n"
    )

    try:
        return await ollama_generate(
            prompt, ollama_base_url, model_name,
            timeout=cfg.ANALYSIS_TIMEOUT, max_tokens=max_tokens, temperature=temperature,
        )
    except Exception:
        return ""


async def _write_section(
    task_text: str,
    section_title: str,
    extracts_text: str,
    analysis_text: str,
    ollama_base_url: str,
    model_name: str,
    temperature: float = cfg.SYNTHESIS_TEMPERATURE,
    max_tokens: int = cfg.SECTION_WRITER_MAX_TOKENS,
) -> str:
    """
    Section writer agent: writes ONE deep report section with multiple paragraphs.
    Each section gets its own dedicated LLM call so the model can focus entirely
    on producing rich, detailed content for this one topic.
    """
    prompt = (
        f"You are writing ONE section of a research report.\n\n"
        f"OVERALL RESEARCH QUESTION: {task_text}\n\n"
        f"THIS SECTION'S TOPIC: {section_title}\n\n"
        f"RELEVANT FACTS FROM SOURCES:\n{extracts_text}\n\n"
    )
    if analysis_text.strip():
        prompt += f"EXPERT ANALYSIS ON THIS TOPIC:\n{analysis_text}\n\n"

    prompt += (
        f"Write the section titled: ## {section_title}\n\n"
        "REQUIREMENTS:\n"
        "- Write 4-8 detailed paragraphs (at least 400 words total)\n"
        "- Start with an overview paragraph that frames the topic\n"
        "- Present ALL specific facts from the sources: name tools, platforms, numbers, techniques\n"
        "- Compare different options with pros and cons\n"
        "- Add your expert judgment: recommend the best approaches and explain WHY\n"
        "- Include practical, actionable advice the reader can use immediately\n"
        "- Use sub-headings (###) to organize if the topic has distinct sub-areas\n"
        "- End with a brief recommendation for this specific topic\n\n"
        "STRICT RULES:\n"
        "- Do NOT write a brief 2-sentence paragraph — write DETAILED, LONG content\n"
        "- Do NOT use generic advice like 'do your research' — be specific\n"
        "- Do NOT include URLs, source references, or source IDs\n"
        "- Do NOT repeat the section title in the first sentence\n"
        "- Every paragraph must contain concrete, useful information\n\n"
        f"## {section_title}\n"
    )

    try:
        result = await ollama_generate(
            prompt, ollama_base_url, model_name,
            timeout=cfg.SECTION_WRITER_TIMEOUT, max_tokens=max_tokens, temperature=temperature,
        )
        return result.strip()
    except Exception:
        return ""


async def _write_bookends(
    task_text: str,
    section_summaries: str,
    criteria: list[str] | None,
    ollama_base_url: str,
    model_name: str,
    temperature: float = cfg.SYNTHESIS_TEMPERATURE,
    max_tokens: int = 4096,
) -> tuple[str, str, str]:
    """
    Bookend writer: produces the executive summary, actionable recommendations,
    and conclusion that tie the whole report together. Runs AFTER all sections
    are written so it can summarize the actual content.
    """
    prompt = (
        "You are finalizing a research report. Below are summaries of each section that was already written.\n\n"
        f"RESEARCH QUESTION: {task_text}\n\n"
        f"SECTIONS ALREADY WRITTEN:\n{section_summaries}\n\n"
        "Now write THREE parts to complete the report:\n\n"
        "EXECUTIVE SUMMARY:\n"
        "Write 3-5 detailed paragraphs summarizing the key findings across ALL sections. "
        "Highlight the most important discoveries, the best options found, and the top recommendations. "
        "This should give a busy reader the full picture without reading the rest.\n\n"
        "ACTIONABLE RECOMMENDATIONS:\n"
        "Write a numbered list of 8-15 specific, concrete action steps the reader should take. "
        "Each recommendation should include WHAT to do, HOW to do it, and specific tools or platforms. "
        "Order from most important to least important.\n\n"
        "CONCLUSION:\n"
        "Write 2-3 paragraphs that tie everything together, highlight the most critical takeaways, "
        "and provide forward-looking perspective. Be opinionated about what matters most.\n\n"
        "Rules:\n"
        "- Be SPECIFIC and CONCRETE — name tools, amounts, strategies\n"
        "- Do NOT include URLs or source references\n"
        "- Write for an intelligent reader who wants actionable depth\n"
    )

    try:
        output = await ollama_generate(
            prompt, ollama_base_url, model_name,
            timeout=cfg.SYNTHESIS_TIMEOUT, max_tokens=max_tokens, temperature=temperature,
        )
    except Exception:
        return ("Research findings are summarized in the sections below.", "", "")

    buckets: dict[str, list[str]] = {"exec": [], "recs": [], "concl": []}
    current = "exec"  # default to exec if model skips the header

    for line in output.splitlines():
        upper = line.strip().upper()
        if upper.startswith("EXECUTIVE SUMMARY"):
            current = "exec"
            continue
        elif upper.startswith("ACTIONABLE RECOMMENDATION"):
            current = "recs"
            continue
        elif upper.startswith("CONCLUSION"):
            current = "concl"
            continue
        buckets[current].append(line)

    exec_summary = "\n".join(buckets["exec"]).strip()
    recommendations = "\n".join(buckets["recs"]).strip()
    conclusion = "\n".join(buckets["concl"]).strip()

    return (exec_summary, recommendations, conclusion)


async def _synthesize_final(
    task_text: str,
    extracts: list[str],
    ollama_base_url: str,
    model_name: str,
    temperature: float = cfg.SYNTHESIS_TEMPERATURE,
    max_tokens: int = cfg.SYNTHESIS_MAX_TOKENS,
    criteria: list[str] | None = None,
    analysis: str = "",
    progress_callback: Any | None = None,
) -> str:
    """
    Multi-agent report assembly: one writer per section + bookends.

    Instead of one LLM call writing everything (producing thin content),
    each section gets its own dedicated call, then a final call ties it together.
    All calls are sequential — one at a time through the local Ollama instance.

    progress_callback: optional async callable(label: str, detail: str) for trace logging.
    """
    all_extracts = "\n\n".join(a.strip() for a in extracts if a.strip())
    per_section_tokens = min(max_tokens, cfg.SECTION_WRITER_MAX_TOKENS)

    if not criteria:
        # No criteria — single-call fallback
        prompt = (
            f"Write a comprehensive research report on: {task_text}\n\n"
            f"FACTS:\n{all_extracts}\n\n"
            "Write a DETAILED markdown report with multiple sections.\n"
            "REPORT:\n"
        )
        try:
            return await ollama_generate(
                prompt, ollama_base_url, model_name,
                timeout=cfg.SYNTHESIS_TIMEOUT, max_tokens=max_tokens, temperature=temperature,
            )
        except Exception:
            return f"# {task_text}\n\n" + all_extracts

    # ── Build semantic fact store (ChromaDB) or fall back to regex matching ──
    fact_store = _FactStore(extracts)
    retrieval_mode = "chromadb" if fact_store.available else "regex"

    section_markdowns: list[str] = []
    section_summaries: list[str] = []

    for i, criterion in enumerate(criteria):
        section_title = criterion[:120].rstrip(".,")

        if fact_store.available:
            # Semantic retrieval: query ChromaDB with the criterion text
            extracts_for_section = fact_store.query(criterion, n_results=40)
            if len(extracts_for_section) < 100:
                extracts_for_section = all_extracts[:8000]
        else:
            # Regex fallback: match by Q-label structure from extractor output
            q_label = f"Q{i+1}"
            relevant_parts: list[str] = []
            other_parts: list[str] = []
            for extract in extracts:
                lines = extract.splitlines()
                relevant_lines: list[str] = []
                other_lines: list[str] = []
                in_relevant = False
                for line in lines:
                    if line.strip().upper().startswith(f"{q_label} FACTS") or line.strip().upper().startswith(f"{q_label}:"):
                        in_relevant = True
                        continue
                    elif re.match(r"^Q\d+ (FACTS|:)", line.strip(), re.I):
                        in_relevant = False
                    elif line.strip().upper().startswith("OTHER RELEVANT"):
                        in_relevant = False

                    if in_relevant and line.strip() and line.strip() != "- NONE":
                        relevant_lines.append(line)
                    elif line.strip() and line.strip() != "- NONE":
                        other_lines.append(line)

                if relevant_lines:
                    relevant_parts.append("\n".join(relevant_lines))
                if other_lines:
                    other_parts.append("\n".join(other_lines))

            if not relevant_parts:
                extracts_for_section = all_extracts[:8000]
            else:
                extracts_for_section = "\n\n".join(relevant_parts)
                other_text = "\n".join(other_parts)[:2000]
                if other_text:
                    extracts_for_section += f"\n\nADDITIONAL CONTEXT:\n{other_text}"

        # Extract analysis relevant to this criterion
        analysis_for_section = ""
        if analysis:
            crit_marker = f"CRITERION {i+1}"
            lines = analysis.splitlines()
            capture = False
            captured: list[str] = []
            for line in lines:
                if crit_marker in line.upper():
                    capture = True
                    continue
                elif capture and re.match(r"^CRITERION \d+", line.strip(), re.I):
                    break
                elif capture and line.strip().upper().startswith("OVERALL"):
                    break
                elif capture:
                    captured.append(line)
            analysis_for_section = "\n".join(captured).strip()

        if progress_callback:
            await progress_callback(
                f"[SECTION {i+1}/{len(criteria)}] Writing: {section_title[:60]}",
                f"Facts: {len(extracts_for_section)} chars ({retrieval_mode}) | Analysis: {len(analysis_for_section)} chars"
            )

        section_md = await _write_section(
            task_text, section_title, extracts_for_section, analysis_for_section,
            ollama_base_url, model_name, temperature=temperature, max_tokens=per_section_tokens,
        )

        if section_md.strip():
            if not section_md.strip().startswith("##"):
                section_md = f"## {section_title}\n\n{section_md}"
            section_markdowns.append(section_md)
            first_para = section_md.split("\n\n")[1] if "\n\n" in section_md else section_md[:200]
            section_summaries.append(f"Section: {section_title}\nSummary: {first_para[:300]}")

            if progress_callback:
                await progress_callback(
                    f"[SECTION {i+1}/{len(criteria)}] Done ({len(section_md)} chars)",
                    section_md[:400]
                )
        else:
            if progress_callback:
                await progress_callback(
                    f"[SECTION {i+1}/{len(criteria)}] Empty — section skipped",
                    f"Topic: {section_title}"
                )

    if progress_callback:
        await progress_callback(
            "[BOOKENDS] Writing executive summary, recommendations, and conclusion",
            f"Summarizing {len(section_markdowns)} completed sections"
        )

    summaries_text = "\n\n".join(section_summaries) if section_summaries else "No sections were written."
    exec_summary, recommendations, conclusion = await _write_bookends(
        task_text, summaries_text, criteria, ollama_base_url, model_name,
        temperature=temperature, max_tokens=per_section_tokens,
    )

    # ── Assemble the full report ──
    title_kw = task_text[:80].rstrip(".,!?")
    parts = [f"# {title_kw}\n"]

    if exec_summary:
        parts.append(f"## Executive Summary\n\n{exec_summary}\n")

    for section in section_markdowns:
        parts.append(f"\n{section}\n")

    if recommendations:
        parts.append(f"\n## Actionable Recommendations\n\n{recommendations}\n")

    if conclusion:
        parts.append(f"\n## Conclusion\n\n{conclusion}\n")

    return "\n".join(parts)


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
    criteria: list[str] | None = None,
) -> tuple[StructuredReport, str]:
    """
    Per-source fact extraction → structured synthesis pipeline.

    Stage 1: One fact-extractor agent per source (parallel, focused).
             Each extracts criterion-specific bullets from a single source.
    Stage 2: One synthesizer that combines all extracts into a grounded report.
    """
    sources_with_ids = _source_ids(sources)
    quality_order = {"good": 0, "medium": 1, "poor": 2}
    ranked = sorted(sources_with_ids, key=lambda s: quality_order.get(s.get("quality", "medium"), 1))
    top_sources = ranked[:top_sources_cap]

    # ── Stage 1: per-source fact extraction (sequential to avoid overloading local Ollama) ──
    per_source_extracts: list[str] = []
    for s in top_sources:
        try:
            extract = await _extract_facts_from_source(
                task_text, s, ollama_base_url, model_name,
                temperature=analyst_temperature,
                max_tokens=min(analyst_max_tokens, 2048),
                criteria=criteria,
            )
            if isinstance(extract, str) and len(extract.strip()) > 20:
                per_source_extracts.append(extract.strip())
        except Exception:
            pass

    if not per_source_extracts:
        # Hard fallback: first sentences from each source
        for s in top_sources[:6]:
            text = (s.get("content_text") or "")
            sentences = [sent.strip() for sent in re.split(r"[.!?]\s+", text[:2000]) if len(sent.strip()) > 30]
            if sentences:
                per_source_extracts.append("- " + "\n- ".join(sentences[:4]))
        if not per_source_extracts:
            per_source_extracts = ["No usable information could be extracted from the collected sources."]

    # ── Stage 2: Deep analysis and judgment ──
    analysis = await _analyze_and_judge(
        task_text, per_source_extracts, ollama_base_url, model_name,
        temperature=synthesis_temperature, max_tokens=min(synthesis_max_tokens, 4096),
        criteria=criteria,
    )

    # ── Stage 3: Final synthesis (extracts + analysis → long report) ──
    raw_report = await _synthesize_final(
        task_text, per_source_extracts, ollama_base_url, model_name,
        temperature=synthesis_temperature, max_tokens=synthesis_max_tokens,
        criteria=criteria, analysis=analysis,
    )
    clean_markdown = _clean_report_markdown(raw_report)

    # Batches for findings/metadata (kept for StructuredReport building below)
    batches: list[list[dict[str, Any]]] = []
    for i in range(0, len(top_sources), analyst_batch_size):
        batches.append(top_sources[i:i + analyst_batch_size])

    if len(clean_markdown) < 50:
        clean_markdown = f"# {task_text}\n\n" + "\n\n".join(per_source_extracts)

    # Build StructuredReport for internal storage
    findings = []
    for i, extract in enumerate(per_source_extracts[:8], 1):
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", extract[:500]) if len(s.strip()) > 20]
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
