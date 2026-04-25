import asyncio
import hashlib
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

from . import config as cfg
from .retry_utils import run_with_retries

try:
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    from playwright.async_api import async_playwright
except Exception:  # pragma: no cover
    async_playwright = None
    PlaywrightTimeoutError = Exception

try:
    import trafilatura
except Exception:  # pragma: no cover
    trafilatura = None

try:
    from duckduckgo_search import DDGS
except Exception:  # pragma: no cover
    DDGS = None


@dataclass
class FetchResult:
    ok: bool
    html: str | None
    title: str | None
    fetch_status: str
    error_reason: str | None
    http_status: int | None = None
    content_type: str | None = None
    content_length_bytes: int | None = None
    fetch_method: str = "http"


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    query = urlencode([(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if not k.startswith("utm_")])
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", query, ""))


def is_valid_scheme(url: str) -> bool:
    return urlparse(url).scheme in {"http", "https"}


async def ollama_generate(
    prompt: str, base_url: str, model: str,
    timeout: int = cfg.OLLAMA_DEFAULT_TIMEOUT,
    max_tokens: int = cfg.DEFAULT_MAX_TOKENS,
    temperature: float | None = None,
) -> str:
    async def _call() -> str:
        options: dict[str, Any] = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "options": options},
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
    return await run_with_retries(_call, retries=cfg.OLLAMA_RETRIES)


_QUERY_META_BLACKLIST = frozenset({
    "search", "complete", "none", "action", "query", "queries",
    "null", "n/a", "ready", "synthesize", "searching", "done",
})


def _clean_query(raw: str) -> str:
    """Strip formatting artifacts that LLMs add to generated queries."""
    q = raw.strip()
    if not q:
        return ""
    # Reject single meta-words the model emits instead of real queries
    if q.lower() in _QUERY_META_BLACKLIST or q.lower().startswith("ready to synthesize"):
        return ""
    q = re.sub(r"^[\d]+[.):\-]\s*", "", q)
    q = re.sub(r"^[-*•·]\s*", "", q)
    q = re.sub(r"^(Query\s*\d*[.:]\s*)", "", q, flags=re.I)
    # Strip sentence-opener action verbs — "Develop a letter about X" → "a letter about X"
    q = re.sub(r"^(Search\s+(for\s+)?)", "", q, flags=re.I)
    q = re.sub(r"^Find\s+(information\s+)?(about\s+|on\s+)?", "", q, flags=re.I)
    q = re.sub(r"^(I\s+need\s+to\s+|I\s+should\s+|I\s+want\s+to\s+|Let\s+me\s+)", "", q, flags=re.I)
    q = re.sub(r"^(Look\s+(up\s+|into\s+)?|Research\s+|Investigate\s+|Explore\s+)", "", q, flags=re.I)
    q = re.sub(
        r"^(Develop|Write|Create|Generate|Build|Make|Provide|Describe|Explain|Analyze|Examine|Discuss|Consider)\s+",
        "", q, flags=re.I,
    )
    q = re.sub(
        r"^(Best\s+practices?\s+(for\s+)?|Templates?\s+(for\s+)?|Examples?\s+(of\s+)?|Guidelines?\s+(for\s+)?)",
        "", q, flags=re.I,
    )
    if len(q) >= 2 and q[0] == q[-1] and q[0] in "\"'""''":
        q = q[1:-1]
    q = q.strip().rstrip(".")
    # Reject queries that are still sentence-like (not Google keyword style)
    if len(q.split()) >= cfg.MAX_QUERY_WORD_COUNT:
        return ""
    return q


async def generate_queries(
    task_text: str, base_url: str, model: str,
    temperature: float = cfg.QUERY_GEN_TEMPERATURE,
) -> list[str]:
    prompt = (
        "You are a search query generator. Write 3 short web search queries to research this topic.\n\n"
        "Rules:\n"
        "- One query per line, nothing else\n"
        "- NO quotes around queries\n"
        "- NO numbering or bullet points\n"
        "- Each query 3-8 words\n"
        "- Use different angles for each\n\n"
        f"Topic: {task_text}\n\n"
        "Queries:\n"
    )
    try:
        output = await ollama_generate(prompt, base_url, model, temperature=temperature)
        queries = [_clean_query(line) for line in output.splitlines() if line.strip()]
        cleaned = [q for q in queries if 5 <= len(q) <= 200][:4]
        return cleaned or [_extract_keywords(task_text)]
    except Exception:
        return [_extract_keywords(task_text)]


async def generate_gap_queries(
    task_text: str,
    existing_titles: list[str],
    base_url: str,
    model: str,
    temperature: float = cfg.GAP_QUERY_TEMPERATURE,
) -> list[str]:
    """After initial research, generate queries that target gaps."""
    found = "\n".join(f"- {t}" for t in existing_titles[:6]) if existing_titles else "- Nothing found yet"
    prompt = (
        "You are a research assistant. I searched for information but need more.\n\n"
        f"Research topic: {task_text}\n\n"
        f"Sources already found:\n{found}\n\n"
        "What important aspects are NOT yet covered? Write 2-3 NEW search queries.\n\n"
        "Rules:\n"
        "- One query per line, nothing else\n"
        "- NO quotes, NO numbering\n"
        "- Each query 3-8 words, like a Google search\n"
        "- Focus on what is MISSING\n"
        "- Do NOT repeat or rephrase what was already found\n\n"
        "New queries:\n"
    )
    try:
        output = await ollama_generate(prompt, base_url, model, temperature=temperature)
        queries = [_clean_query(line) for line in output.splitlines() if line.strip()]
        return [q for q in queries if 5 <= len(q) <= 200][:3]
    except Exception:
        return []


async def web_search(query: str, max_results: int) -> list[dict[str, Any]]:
    """Free web search via DuckDuckGo with exponential backoff on rate limits."""
    if DDGS is None:
        raise RuntimeError("duckduckgo-search package is not installed")

    def _sync_search() -> list[dict[str, Any]]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    last_exc: Exception | None = None
    for attempt in range(cfg.SEARCH_RETRY_COUNT):
        try:
            if attempt > 0:
                backoff = min(cfg.SEARCH_BACKOFF_BASE * (2 ** (attempt - 1)), cfg.SEARCH_BACKOFF_MAX)
                await asyncio.sleep(backoff)
            rows = await asyncio.to_thread(_sync_search)
            out = []
            for idx, row in enumerate(rows, start=1):
                out.append(
                    {
                        "query": query,
                        "title": row.get("title") or "Untitled",
                        "url": row.get("href"),
                        "snippet": row.get("body") or "",
                        "rank": idx,
                        "provider": "duckduckgo",
                    }
                )
            return out
        except Exception as exc:
            last_exc = exc
    raise last_exc or RuntimeError("web_search failed after retries")


async def fetch_http(url: str, timeout_sec: int, max_bytes: int) -> FetchResult:
    if not is_valid_scheme(url):
        return FetchResult(False, None, None, "skipped", "unsupported_scheme")

    async def _call() -> FetchResult:
        async with httpx.AsyncClient(timeout=timeout_sec, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    return FetchResult(False, None, None, "failed", f"http_{resp.status_code}", http_status=resp.status_code)
                content_type = resp.headers.get("content-type", "")
                if "html" not in content_type.lower():
                    return FetchResult(False, None, None, "skipped", "non_html", content_type=content_type)
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        return FetchResult(False, None, None, "skipped", "too_large", content_type=content_type, content_length_bytes=total)
                    chunks.append(chunk)
                html = b"".join(chunks).decode("utf-8", errors="ignore")
                return FetchResult(True, html, None, "ok", None, http_status=resp.status_code, content_type=content_type, content_length_bytes=total, fetch_method="http")
    try:
        return await run_with_retries(_call, retries=cfg.FETCH_RETRIES)
    except httpx.TimeoutException:
        return FetchResult(False, None, None, "failed", "timeout")
    except Exception as exc:
        return FetchResult(False, None, None, "failed", f"fetch_error:{exc.__class__.__name__}")


async def fetch_browser(url: str, timeout_sec: int) -> FetchResult:
    if async_playwright is None:
        return FetchResult(False, None, None, "failed", "playwright_not_available", fetch_method="browser")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=timeout_sec * 1000)
                await asyncio.sleep(cfg.BROWSER_SETTLE_DELAY)
                html = await page.content()
                title = await page.title()
                await browser.close()
                return FetchResult(True, html, title, "ok", None, fetch_method="browser")
            except PlaywrightTimeoutError:
                await browser.close()
                return FetchResult(False, None, None, "failed", "browser_timeout", fetch_method="browser")
            except Exception:
                await browser.close()
                return FetchResult(False, None, None, "failed", "page_navigation_failed", fetch_method="browser")
    except Exception:
        return FetchResult(False, None, None, "failed", "playwright_not_available", fetch_method="browser")


def _trafilatura_extract(html: str, fallback_title: str | None = None) -> tuple[str, str]:
    """Extract text and title using trafilatura (markdown output for richer structure)."""
    if not trafilatura:
        return "", fallback_title or "Untitled"
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        include_links=False,
        output_format="markdown",
        favor_recall=True,
    ) or ""
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else None) or fallback_title or "Untitled"
    return text, title


def _regex_extract(html: str, fallback_title: str | None = None) -> tuple[str, str]:
    """Bare-minimum regex fallback when trafilatura isn't available."""
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
    return text, fallback_title or "Untitled"


async def fetch_jina_reader(url: str, timeout_sec: int = 30) -> tuple[str, str]:
    """
    Jina Reader API — free fallback content extractor.
    Prepend r.jina.ai/ to any URL and get clean markdown.
    Free tier: 20 RPM without API key.
    """
    jina_url = f"https://r.jina.ai/{url}"
    try:
        async with httpx.AsyncClient(timeout=timeout_sec, follow_redirects=True) as client:
            resp = await client.get(jina_url, headers={"Accept": "text/markdown"})
            if resp.status_code == 200 and len(resp.text.strip()) > 100:
                lines = resp.text.strip().splitlines()
                title = "Untitled"
                text_lines = []
                for line in lines:
                    if line.startswith("Title:") and title == "Untitled":
                        title = line[6:].strip() or "Untitled"
                    elif line.startswith("URL:") or line.startswith("Published Time:"):
                        continue
                    else:
                        text_lines.append(line)
                return "\n".join(text_lines).strip(), title
    except Exception:
        pass
    return "", "Untitled"


def _score_extraction(
    text: str, title: str, task_text: str, query_text: str | None,
) -> dict[str, Any]:
    """Score extracted content for relevance and quality."""
    text_len = len(text.strip())
    task_terms = {t.lower() for t in re.findall(r"[a-zA-Z]{4,}", task_text)}
    query_terms = {t.lower() for t in re.findall(r"[a-zA-Z]{4,}", query_text or "")}
    lower = text.lower()
    task_overlap = sum(1 for t in task_terms if t in lower)
    query_overlap = sum(1 for t in query_terms if t in lower)
    repeated_penalty = 1 if len(set(re.findall(r"\w+", lower[:1200]))) < 50 else 0

    score = 0
    if text_len >= 1400:
        score += 2
    elif text_len >= 400:
        score += 1
    if task_overlap >= 2:
        score += 2
    elif task_overlap >= 1:
        score += 1
    if query_overlap >= 1:
        score += 1
    if title and title != "Untitled":
        score += 1
    score -= repeated_penalty

    quality = "good" if score >= cfg.QUALITY_SCORE_GOOD else "medium" if score >= cfg.QUALITY_SCORE_MEDIUM else "poor"
    return {
        "title": title,
        "text": text,
        "quality": quality,
        "metrics": {
            "text_len": text_len,
            "task_overlap": task_overlap,
            "query_overlap": query_overlap,
            "repeated_penalty": repeated_penalty,
            "score": score,
        },
    }


def extract_content(html: str, task_text: str, query_text: str | None, fallback_title: str | None = None) -> dict[str, Any]:
    """
    Primary extraction: trafilatura (markdown mode for richer output).
    Falls back to regex strip if trafilatura isn't installed.
    """
    text, title = _trafilatura_extract(html, fallback_title)
    if not text.strip():
        text, title = _regex_extract(html, fallback_title)
    return _score_extraction(text, title, task_text, query_text)


def should_retry_with_browser(fetch_result: FetchResult, extraction: dict[str, Any] | None) -> tuple[bool, str]:
    if not fetch_result.ok:
        return False, "fetch_not_ok"
    if fetch_result.error_reason in {"non_html", "too_large", "unsupported_scheme"}:
        return False, "ineligible_failure"
    if not extraction:
        return True, "no_extraction"
    if extraction["metrics"]["text_len"] < cfg.MIN_TEXT_LENGTH_FOR_BROWSER_SKIP:
        return True, "extracted_too_short"
    if extraction["quality"] == "poor":
        return True, "quality_poor"
    if extraction["metrics"]["task_overlap"] == 0:
        return True, "low_keyword_overlap"
    return False, "good_enough"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


# ────────────────────────────────────────────────────────
#  Autonomous agent reasoning primitives
# ────────────────────────────────────────────────────────

def build_knowledge_summary(sources: list[dict[str, Any]], max_chars: int = cfg.KNOWLEDGE_SUMMARY_MAX_CHARS) -> str:
    """Compact summary of accumulated knowledge for the thinker prompt."""
    if not sources:
        return "Nothing collected yet."
    lines: list[str] = []
    chars = 0
    for i, s in enumerate(sources, 1):
        title = s.get("title", "Untitled")
        quality = s.get("quality", "?")
        text = (s.get("content_text") or "")[:cfg.KNOWLEDGE_SUMMARY_PER_SOURCE_CHARS].strip().replace("\n", " ")
        entry = f"[{i}] ({quality}) {title}\n    {text}\n"
        if chars + len(entry) > max_chars:
            lines.append(f"    ... and {len(sources) - i + 1} more sources")
            break
        lines.append(entry)
        chars += len(entry)
    return "\n".join(lines)


async def formulate_goal(task_text: str, base_url: str, model: str) -> dict[str, Any]:
    """
    Parse a user's request into a clear research goal with 4-5 specific, searchable
    research angles. Each angle becomes:
      - a search direction for the autonomous loop
      - a fact-extraction question per source
      - a dedicated section in the final report
    So they must be *distinct topics you can actually Google*.
    """
    prompt = (
        "You are a research planner. A user wants you to research something. "
        "Break their request into 4-5 DISTINCT research angles that can be searched on Google.\n\n"
        f"USER REQUEST: {task_text}\n\n"
        "=== EXAMPLE 1 ===\n"
        "Request: 'I want to build wealth. I'm an AI engineer and want to build a business'\n"
        "GOAL: Research practical wealth-building paths for AI engineers — startup ideas, investing, and side businesses.\n"
        "CRITERIA:\n"
        "- AI startup and SaaS business ideas that leverage machine learning engineering skills\n"
        "- Passive income and investment strategies for high-earning tech professionals\n"
        "- Freelancing, consulting, and productized service businesses for AI/ML engineers\n"
        "- Real examples of software engineers who built wealth through side businesses or startups\n"
        "- Financial planning fundamentals: saving rate, compound growth, and tax optimization for tech workers\n\n"
        "=== EXAMPLE 2 ===\n"
        "Request: 'Build a single page digital professional card'\n"
        "GOAL: Find the best tools, templates, and design practices for creating a one-page digital professional card.\n"
        "CRITERIA:\n"
        "- Website builders for single-page portfolio cards: Carrd, Linktree, About.me, Bento comparison\n"
        "- UX and layout best practices for one-page professional profiles\n"
        "- Pre-designed templates and themes for digital business cards or portfolio pages\n"
        "- Interactive and multimedia features that make digital cards stand out: NFC, QR, animations\n\n"
        "=== NOW YOUR TURN ===\n"
        "Write the GOAL and CRITERIA for the actual USER REQUEST above.\n\n"
        "GOAL: [one clear sentence — what does a complete research result look like?]\n"
        "CRITERIA:\n"
        "- [research angle 1 — a specific, searchable topic with named subtopics]\n"
        "- [research angle 2 — different angle, not a rephrasing of angle 1]\n"
        "- [research angle 3 — covers a new aspect]\n"
        "- [research angle 4 — yet another distinct angle]\n\n"
        "RULES:\n"
        "- Each criterion is a TOPIC to search, not a task to do\n"
        "- Name specific subtopics, tools, or concepts in each criterion\n"
        "- Each criterion must be DIFFERENT from the others — cover distinct ground\n"
        "- Do NOT just repeat or rephrase the user's request\n"
        "- Think: what would you actually type into Google to research each angle?\n"
    )
    try:
        output = await ollama_generate(
            prompt, base_url, model,
            temperature=cfg.GOAL_TEMPERATURE,
            max_tokens=600,
        )
        goal = ""
        criteria: list[str] = []
        in_criteria = False
        for line in output.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("GOAL:"):
                goal = stripped[5:].strip()
                in_criteria = False
            elif upper.startswith("CRITERIA") or upper.startswith("CRITERION"):
                in_criteria = True
                continue
            elif in_criteria and stripped:
                cleaned = re.sub(r"^(\d+[.):\-]\s*|[-*•·]\s*)", "", stripped)
                cleaned = cleaned.strip().rstrip(".")
                if len(cleaned) < 15:
                    continue
                # Reject if the criterion is just the raw task echoed back
                if cleaned.lower()[:40] == task_text.lower()[:40]:
                    continue
                criteria.append(cleaned)

        if not goal:
            goal = task_text[:300]

        if not criteria:
            criteria = _generate_fallback_criteria(task_text)

        return {"goal": goal[:400], "criteria": criteria[:6]}
    except Exception:
        return {
            "goal": task_text[:300],
            "criteria": _generate_fallback_criteria(task_text),
        }


def _extract_keywords(text: str, max_words: int = 6) -> str:
    """Pull meaningful keywords from a goal/task for use as fallback search query."""
    stop = {
        "a", "an", "the", "and", "or", "but", "for", "of", "to", "in", "on",
        "is", "are", "was", "were", "be", "been", "that", "this", "it", "its",
        "with", "from", "by", "as", "at", "into", "about", "how", "what",
        "which", "who", "when", "where", "why", "can", "will", "do", "does",
        "not", "no", "so", "if", "while", "without", "also", "i", "my",
        "develop", "create", "write", "make", "provide", "ensure", "clear",
        "professional", "good", "best", "effective", "should", "need",
        "sample", "request", "letter", "potential", "future", "conveying",
        "expressing", "expectation", "immediate", "services", "cooperation",
        "want", "build", "like", "just", "use", "get", "give", "know",
        "think", "need", "currently", "looking", "trying", "help", "going",
        "am", "im", "ve", "re", "ll", "have", "has", "had", "would", "could",
        "some", "more", "any", "all", "other", "own", "same", "such",
    }
    words = re.findall(r"[a-zA-Z]{2,}", text.lower())
    keywords = [w for w in words if w not in stop]
    seen: list[str] = []
    for w in keywords:
        if w not in seen:
            seen.append(w)
        if len(seen) >= max_words:
            break
    return " ".join(seen) if seen else text[:50]


def _generate_fallback_criteria(task_text: str) -> list[str]:
    """
    When the LLM can't generate criteria, build plausible research angles from
    keywords in the task text. These should read like things you'd actually Google.
    """
    kw = _extract_keywords(task_text, max_words=4)
    return [
        f"Best strategies and proven methods for {kw}",
        f"Real-world examples and case studies of {kw}",
        f"Tools, platforms, and resources for {kw}",
        f"Common mistakes and pitfalls to avoid with {kw}",
    ]


def _criteria_fallback_queries(
    criteria: list[str],
    search_history: list[str],
    iteration: int,
    goal: str,
) -> list[str]:
    """
    Build diverse fallback queries from success criteria when the LLM fails.
    Rotates through criteria so each iteration targets a different unsatisfied criterion.
    """
    seen_lower = {h.lower() for h in search_history}
    queries: list[str] = []

    # Try each criterion as a keyword query, rotating start index by iteration
    for offset in range(len(criteria)):
        idx = (iteration - 1 + offset) % len(criteria)
        kw = _extract_keywords(criteria[idx], max_words=5)
        if kw and kw.lower() not in seen_lower:
            queries.append(kw)
        if len(queries) >= 3:
            break

    # If criteria all duped, try topic + angle combos
    if not queries:
        base = _extract_keywords(goal, max_words=3)
        angles = [
            "best tools platforms",
            "design tips examples",
            "how to build",
            "templates comparison",
            "step by step tutorial",
            "features checklist",
        ]
        for angle in angles:
            q = f"{base} {angle}"
            if q.lower() not in seen_lower:
                queries.append(q)
            if queries:
                break

    return queries[:3]


async def autonomous_step(
    goal: str,
    criteria: list[str],
    knowledge_summary: str,
    search_history: list[str],
    last_observation: str,
    iteration: int,
    base_url: str,
    model: str,
    temperature: float = cfg.REASONING_TEMPERATURE,
) -> dict[str, Any]:
    """
    Unified thinking step: reflect on what was found, decide to SEARCH or COMPLETE.
    Uses a simplified 3-section format (ANALYSIS / ACTION / QUERIES) that small
    local models can reliably follow instead of the heavier 5-section layout.
    """
    criteria_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(criteria))
    history_text = "\n".join(f"  - {q}" for q in search_history[-12:]) if search_history else "  (none yet)"

    prompt = (
        "You are an autonomous research agent running an iterative web research loop.\n\n"
        f"RESEARCH GOAL: {goal}\n\n"
        f"CRITERIA TO SATISFY (you must gather evidence for EACH):\n{criteria_text}\n\n"
        f"KNOWLEDGE GATHERED SO FAR:\n{knowledge_summary}\n\n"
        f"SEARCHES ALREADY TRIED (do NOT repeat these):\n{history_text}\n\n"
        f"LAST ROUND RESULTS:\n{last_observation}\n\n"
        "Now respond in EXACTLY this 3-section format:\n\n"
        "ANALYSIS: (2-3 sentences) Which criteria have evidence? Which are still missing? What gaps remain?\n\n"
        "ACTION: SEARCH\n\n"
        "QUERIES:\n"
        "query one here\n"
        "query two here\n"
        "query three here\n\n"
        "RULES FOR QUERIES:\n"
        "- Write 3-4 queries, each on its own line\n"
        "- Each query: 3-6 words, keyword-style (like typing into Google)\n"
        "- Each query must target a DIFFERENT unsatisfied criterion\n"
        "- Do NOT repeat or rephrase any previous search\n"
        "- Do NOT use sentences or instructions — only noun phrases and keywords\n"
        "- Only write ACTION: COMPLETE (instead of SEARCH) when ALL criteria have strong evidence\n"
        "\nExample output for a goal about digital professional cards:\n"
        "ANALYSIS: Found examples of digital cards and design tips. Still missing info on specific website builders and pre-built templates.\n"
        "ACTION: SEARCH\n"
        "QUERIES:\n"
        "Carrd Linktree professional portfolio builder\n"
        "digital business card templates customizable\n"
        "single page portfolio UX best practices\n"
    )

    try:
        output = await ollama_generate(prompt, base_url, model, temperature=temperature)
    except Exception:
        return {
            "thoughts": "Model call failed",
            "reasoning": "", "plan": "", "criticism": "",
            "action": "SEARCH",
            "queries": _criteria_fallback_queries(criteria, search_history, iteration, goal),
            "raw": "",
        }

    sections: dict[str, str] = {}
    current_key = "preamble"
    current_lines: list[str] = []
    header_map = {
        "ANALYSIS": "thoughts",
        "THOUGHTS": "thoughts",
        "THOUGHT": "thoughts",
        "REASONING": "thoughts",
        "REASON": "thoughts",
        "PLAN": "plan",
        "CRITICISM": "criticism",
        "CRITIQUE": "criticism",
        "SELF-CRITICISM": "criticism",
        "SELF CRITICISM": "criticism",
        "ACTION_INPUT": "queries",
        "ACTION INPUT": "queries",
        "QUERIES": "queries",
        "SEARCH QUERIES": "queries",
        "ACTION": "action",
    }

    for line in output.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        matched = False
        for header, section_name in header_map.items():
            if upper.startswith(header + ":") or upper.startswith(header + " :"):
                sections[current_key] = "\n".join(current_lines).strip()
                current_key = section_name
                colon_pos = stripped.find(":")
                rest = stripped[colon_pos + 1:].strip() if colon_pos >= 0 else ""
                current_lines = [rest] if rest else []
                matched = True
                break
        if not matched:
            current_lines.append(line)
    sections[current_key] = "\n".join(current_lines).strip()

    action_raw = sections.get("action", "SEARCH").upper().strip()
    action = "COMPLETE" if "COMPLETE" in action_raw else "SEARCH"

    queries: list[str] = []
    if action == "SEARCH":
        query_text = sections.get("queries", "") or sections.get("action_input", "")

        if not query_text.strip():
            action_section = sections.get("action", "")
            action_lines = action_section.splitlines()
            if len(action_lines) > 1:
                query_text = "\n".join(action_lines[1:])

        for line in query_text.splitlines():
            q = _clean_query(line)
            if 5 <= len(q) <= 200:
                queries.append(q)
        seen = {h.lower() for h in search_history}
        queries = [q for q in queries if q.lower() not in seen]

    if action == "SEARCH" and not queries:
        queries = _criteria_fallback_queries(criteria, search_history, iteration, goal)

    return {
        "thoughts": sections.get("thoughts", "")[:800],
        "reasoning": sections.get("reasoning", sections.get("thoughts", ""))[:400],
        "plan": sections.get("plan", "")[:400],
        "criticism": sections.get("criticism", "")[:400],
        "action": action,
        "queries": queries[:4],
        "raw": output[:1500],
    }


# ────────────────────────────────────────────────────────
#  New autonomous loop agents (modular cognition)
#
#  Each agent is a standalone expert with zero awareness of
#  the pipeline or other agents.  The orchestrator in
#  pipeline.py handles wiring, parsing, and memory updates.
# ────────────────────────────────────────────────────────

def _parse_sections(text: str, header_map: dict[str, str], primary_key: str) -> dict[str, str]:
    """
    Generic section parser reused across all agents.

    Splits LLM output into named sections using flexible header matching.
    Falls back to treating the entire output as the primary section.
    """
    sections: dict[str, str] = {}
    current_key = "_preamble"
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        matched = False
        for header, section_name in header_map.items():
            if upper.startswith(header + ":") or upper.startswith(header + " :"):
                sections[current_key] = "\n".join(current_lines).strip()
                current_key = section_name
                colon_pos = stripped.find(":")
                rest = stripped[colon_pos + 1:].strip() if colon_pos >= 0 else ""
                current_lines = [rest] if rest else []
                matched = True
                break
        if not matched:
            current_lines.append(line)
    sections[current_key] = "\n".join(current_lines).strip()

    if not sections.get(primary_key, "").strip():
        full = text.strip()
        if full:
            sections[primary_key] = full

    return sections


def _extract_queries_from_text(text: str, search_history: list[str] | None = None) -> list[str]:
    """Extract QUERY: lines from agent output, clean and dedup."""
    queries: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        raw = re.sub(r"^QUERY\s*:\s*", "", stripped, flags=re.I)
        raw = re.sub(r"^[-*•·]\s*", "", raw)
        q = _clean_query(raw)
        if 5 <= len(q) <= 200:
            queries.append(q)
    if search_history:
        seen = {h.lower() for h in search_history}
        queries = [q for q in queries if q.lower() not in seen]
    return queries


async def analyze_round(
    task_text: str,
    goal: str,
    criteria: list[str],
    working_thesis: str,
    previous_insights: str,
    new_sources_content: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    """
    The Forensic Analyst — reads new documents and writes up what they mean.

    Returns dict with keys: findings, surprises, contradictions, open_questions, raw.
    """
    criteria_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(criteria))

    prompt = (
        "You are a meticulous research analyst. You have been given a set of documents "
        "to review alongside what is already known about this topic.\n\n"
        f"RESEARCH QUESTION: {task_text}\n"
        f"WHAT WE ARE INVESTIGATING:\n{criteria_text}\n\n"
        f"WHAT IS CURRENTLY BELIEVED:\n{working_thesis or 'No beliefs formed yet.'}\n\n"
        f"WHAT WAS ALREADY KNOWN:\n{previous_insights or 'Nothing yet — this is the first round.'}\n\n"
        f"NEW DOCUMENTS TO ANALYZE:\n{new_sources_content}\n\n"
        "Read every document carefully. Then write your analysis:\n\n"
        "FINDINGS:\n"
        "For each research question these documents address:\n"
        "- Question N: [what specific new facts or evidence you found]\n"
        "  Source: [which document — and rate it PRIMARY/SECONDARY/TERTIARY]\n\n"
        "SURPRISES:\n"
        "Anything that contradicts the current beliefs or introduces something new?\n"
        "- [unexpected findings, or NONE]\n\n"
        "CONTRADICTIONS:\n"
        "Does anything conflict with what was already known? Which source is more credible?\n"
        "- [conflicts with credibility comparison, or NONE]\n\n"
        "OPEN QUESTIONS:\n"
        "What would you want to look into next based on what you just read?\n"
        "- [questions that arose from this analysis]\n"
    )

    header_map = {
        "FINDINGS": "findings", "FINDING": "findings", "RESULTS": "findings",
        "SURPRISES": "surprises", "SURPRISE": "surprises", "UNEXPECTED": "surprises",
        "CONTRADICTIONS": "contradictions", "CONTRADICTION": "contradictions",
        "CONFLICTS": "contradictions", "CONFLICT": "contradictions",
        "OPEN QUESTIONS": "open_questions", "QUESTIONS": "open_questions",
        "OPEN": "open_questions", "NEXT": "open_questions",
    }

    try:
        output = await ollama_generate(
            prompt, base_url, model,
            timeout=cfg.ANALYSIS_ROUND_TIMEOUT,
            max_tokens=cfg.ANALYSIS_ROUND_MAX_TOKENS,
            temperature=cfg.ANALYSIS_ROUND_TEMPERATURE,
        )
    except Exception:
        return {"findings": "", "surprises": "", "contradictions": "", "open_questions": "", "raw": ""}

    sections = _parse_sections(output, header_map, "findings")

    credibility: dict[str, int] = {}
    for m in re.finditer(r"(PRIMARY|SECONDARY|TERTIARY)", sections.get("findings", ""), re.I):
        key = m.group(0).upper()
        credibility[key] = credibility.get(key, 0) + 1

    return {
        "findings": sections.get("findings", ""),
        "surprises": sections.get("surprises", ""),
        "contradictions": sections.get("contradictions", ""),
        "open_questions": sections.get("open_questions", ""),
        "credibility_counts": credibility,
        "raw": output[:2000],
    }


async def reflect_and_reason(
    task_text: str,
    goal: str,
    plan_steps: str,
    criteria_with_confidence: str,
    working_thesis: str,
    all_insights: str,
    credibility_summary: str,
    search_history_text: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    """
    The Senior Scientist — steps back and thinks about the big picture.

    Returns dict with keys: beliefs, confidence, critique, gaps, implications, decision, reasoning, raw.
    """
    prompt = (
        "You are a senior scientist reviewing the current state of a research project. "
        "Step back from the details and think about the big picture.\n\n"
        f"ORIGINAL QUESTION: {task_text}\n"
        f"RESEARCH OBJECTIVE: {goal}\n"
        f"RESEARCH PLAN:\n{plan_steps}\n\n"
        f"RESEARCH QUESTIONS AND CURRENT CONFIDENCE:\n{criteria_with_confidence}\n\n"
        f"CURRENT BELIEFS (what the evidence suggests so far):\n{working_thesis or 'No beliefs formed yet.'}\n\n"
        f"ALL FINDINGS TO DATE:\n{all_insights or 'No findings yet.'}\n\n"
        f"SOURCE QUALITY DISTRIBUTION:\n{credibility_summary or 'No sources rated yet.'}\n\n"
        f"SEARCHES CONDUCTED:\n{search_history_text or '(none yet)'}\n\n"
        "As the senior advisor on this project, provide your assessment:\n\n"
        "UPDATED BELIEFS:\n"
        "For each research question, state in 1-2 sentences what the evidence currently "
        "points to. Be concrete — \"evidence suggests X because Y, though Z remains "
        "uncertain\" — not \"more data needed.\"\n\n"
        "CONFIDENCE:\n"
        "For each research question: LOW / MEDIUM / HIGH\n"
        "Consider: How many independent sources agree? How credible are they? Do we have "
        "ONLY confirming evidence (suspicious) or also disconfirming evidence that we've "
        "addressed? Is the evidence specific or vague?\n\n"
        "HONEST CRITIQUE:\n"
        "Be brutally honest about this research:\n"
        "- Are we only finding what we expect to find?\n"
        "- What is the most obvious thing a domain expert would say we're missing?\n"
        "- Are we relying on weak sources for strong claims?\n"
        "- Are we asking the right questions, or just the easy ones?\n\n"
        "GAPS:\n"
        "What specific information — concrete facts, data points, expert perspectives — "
        "would make this research substantially more valuable?\n\n"
        "IMPLICATIONS:\n"
        f"For each well-supported finding, what does it actually mean for someone who "
        f"asked \"{task_text[:120]}\"? Connect the evidence to practical reality.\n\n"
        "VERDICT: CONTINUE or STOP\n"
        "STOP only if: the majority of questions have HIGH confidence, key claims are "
        "supported by credible sources, and you genuinely believe more searching will "
        "not materially improve the answer.\n"
        "REASONING: [why]\n"
    )

    header_map = {
        "UPDATED BELIEFS": "beliefs", "BELIEFS": "beliefs", "UPDATED": "beliefs",
        "CONFIDENCE": "confidence",
        "HONEST CRITIQUE": "critique", "CRITIQUE": "critique", "CRITICISM": "critique",
        "SELF-CRITIQUE": "critique",
        "GAPS": "gaps", "GAP": "gaps", "MISSING": "gaps", "MISSING PIECES": "gaps",
        "IMPLICATIONS": "implications", "IMPLICATION": "implications",
        "SO WHAT": "implications",
        "VERDICT": "decision", "DECISION": "decision",
        "REASONING": "reasoning", "REASON": "reasoning",
    }

    try:
        output = await ollama_generate(
            prompt, base_url, model,
            timeout=cfg.REFLECT_TIMEOUT,
            max_tokens=cfg.REFLECT_MAX_TOKENS,
            temperature=cfg.REFLECT_TEMPERATURE,
        )
    except Exception:
        return {
            "beliefs": "", "confidence": "", "critique": "", "gaps": "",
            "implications": "", "decision": "CONTINUE", "reasoning": "", "raw": "",
        }

    sections = _parse_sections(output, header_map, "beliefs")

    decision_raw = (sections.get("decision", "") + " " + sections.get("reasoning", "")).upper()
    decision = "STOP" if "STOP" in decision_raw else "CONTINUE"

    return {
        "beliefs": sections.get("beliefs", ""),
        "confidence": sections.get("confidence", ""),
        "critique": sections.get("critique", ""),
        "gaps": sections.get("gaps", ""),
        "implications": sections.get("implications", ""),
        "decision": decision,
        "reasoning": sections.get("reasoning", ""),
        "raw": output[:2000],
    }


async def challenge_beliefs(
    task_text: str,
    working_thesis_and_claims: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    """
    The Skeptical Reviewer — tears apart claims to find weaknesses.

    Returns dict with keys: weaknesses, alternatives, blind_spots, queries, raw.
    """
    prompt = (
        "You are a rigorous peer reviewer. You have been given a set of claims from a "
        "research paper. Your job is to find every weakness, every assumption, every "
        "alternative explanation the authors may have missed.\n\n"
        f"THE PAPER'S RESEARCH QUESTION: {task_text}\n\n"
        f"CLAIMS BEING MADE:\n{working_thesis_and_claims}\n\n"
        "For each claim, provide your review:\n\n"
        "WEAKNESSES:\n"
        "- Claim: [restate it]\n"
        "  Best counter-argument: [the strongest case AGAINST this claim]\n"
        "  What would disprove it: [what evidence, if found, would invalidate this]\n"
        "  Search suggestion: [3-7 word query to find such evidence]\n\n"
        "ALTERNATIVE READINGS:\n"
        "Could the same evidence support a DIFFERENT conclusion? What else could "
        "explain what was observed?\n"
        "- [alternative interpretation of the evidence]\n\n"
        "BLIND SPOTS:\n"
        "Whose perspective is missing from this analysis? What stakeholder, expert "
        "community, or contrarian viewpoint has not been considered?\n"
        "- [missing perspective and why it matters]\n\n"
        "DISCONFIRMING SEARCHES:\n"
        "1-3 search queries specifically designed to find evidence that would weaken "
        "or disprove the strongest claims. Frame these as a skeptic would — look for "
        "failures, criticisms, risks, exceptions, counterexamples.\n"
        "- QUERY: [3-7 words]\n"
        "  CHALLENGES: [which claim this would test]\n"
    )

    header_map = {
        "WEAKNESSES": "weaknesses", "WEAKNESS": "weaknesses",
        "ALTERNATIVE READINGS": "alternatives", "ALTERNATIVES": "alternatives",
        "ALTERNATIVE": "alternatives",
        "BLIND SPOTS": "blind_spots", "BLIND SPOT": "blind_spots",
        "MISSING PERSPECTIVES": "blind_spots",
        "DISCONFIRMING SEARCHES": "queries", "DISCONFIRMING": "queries",
        "SEARCH QUERIES": "queries", "QUERIES": "queries",
    }

    try:
        output = await ollama_generate(
            prompt, base_url, model,
            timeout=cfg.CHALLENGE_TIMEOUT,
            max_tokens=cfg.CHALLENGE_MAX_TOKENS,
            temperature=cfg.CHALLENGE_TEMPERATURE,
        )
    except Exception:
        return {"weaknesses": "", "alternatives": "", "blind_spots": "", "queries": [], "raw": ""}

    sections = _parse_sections(output, header_map, "weaknesses")

    queries_text = sections.get("queries", "")
    parsed_queries: list[dict[str, str]] = []
    current_query = ""
    current_challenge = ""
    for line in queries_text.splitlines():
        stripped = line.strip()
        q_match = re.match(r"^(?:QUERY\s*:\s*|[-*•]\s*QUERY\s*:\s*)(.*)", stripped, re.I)
        c_match = re.match(r"^(?:CHALLENGES?\s*:\s*|[-*•]\s*CHALLENGES?\s*:\s*)(.*)", stripped, re.I)
        if q_match:
            if current_query:
                parsed_queries.append({"query": _clean_query(current_query), "challenges": current_challenge})
            current_query = q_match.group(1).strip()
            current_challenge = ""
        elif c_match:
            current_challenge = c_match.group(1).strip()
        elif not q_match and not c_match and stripped:
            q = _clean_query(stripped)
            if 5 <= len(q) <= 200:
                current_query = q
    if current_query:
        parsed_queries.append({"query": _clean_query(current_query), "challenges": current_challenge})

    parsed_queries = [p for p in parsed_queries if p["query"] and len(p["query"]) >= 5]

    return {
        "weaknesses": sections.get("weaknesses", ""),
        "alternatives": sections.get("alternatives", ""),
        "blind_spots": sections.get("blind_spots", ""),
        "queries": parsed_queries[:3],
        "raw": output[:2000],
    }


async def discover_new_angles(
    task_text: str,
    criteria: list[str],
    all_insights: str,
    all_surprises: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    """
    The Pattern Spotter — finds hidden connections and missing angles.

    Returns dict with keys: patterns, missing, new_directions, raw.
    """
    criteria_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(criteria))

    prompt = (
        "You are an interdisciplinary researcher known for spotting patterns others miss. "
        "You have been given the accumulated findings from a research project and asked "
        "to look at them with fresh eyes.\n\n"
        f"ORIGINAL QUESTION: {task_text}\n"
        f"CURRENT RESEARCH AREAS:\n{criteria_text}\n\n"
        f"ALL FINDINGS:\n{all_insights}\n\n"
        f"UNEXPECTED OBSERVATIONS:\n{all_surprises or 'None so far.'}\n\n"
        "Look at everything together — not one area at a time, but ALL of it at once.\n\n"
        "PATTERNS:\n"
        "What threads run ACROSS different research areas? When findings from Area A "
        "and Area B are placed side by side, what emerges? What topics keep showing up "
        "that are NOT in the research areas?\n"
        "- [pattern and what it implies]\n\n"
        "MISSING:\n"
        "If a leading expert in this domain read these findings, what would they say "
        "is obviously missing? What adjacent fields could shed light?\n"
        "- [what's missing and why it matters]\n\n"
        "NEW DIRECTIONS:\n"
        "0-3 new research areas to explore. Only suggest if genuinely distinct from "
        "existing areas AND supported by patterns in the findings. Write NONE if the "
        "current areas are sufficient.\n"
        "- [new direction and evidence for it]\n"
    )

    header_map = {
        "PATTERNS": "patterns", "PATTERN": "patterns",
        "HIDDEN CONNECTIONS": "patterns", "CONNECTIONS": "patterns",
        "MISSING": "missing", "GAPS": "missing", "BLIND SPOTS": "missing",
        "NEW DIRECTIONS": "new_directions", "NEW DIRECTION": "new_directions",
        "SUGGESTED": "new_directions", "SUGGESTIONS": "new_directions",
    }

    try:
        output = await ollama_generate(
            prompt, base_url, model,
            timeout=cfg.DISCOVER_TIMEOUT,
            max_tokens=cfg.DISCOVER_MAX_TOKENS,
            temperature=cfg.DISCOVER_TEMPERATURE,
        )
    except Exception:
        return {"patterns": [], "missing": "", "new_directions": [], "raw": ""}

    sections = _parse_sections(output, header_map, "patterns")

    patterns = [
        line.strip().lstrip("-*• ") for line in sections.get("patterns", "").splitlines()
        if line.strip() and len(line.strip()) > 15 and not line.strip().upper().startswith("NONE")
    ]

    new_dirs_raw = sections.get("new_directions", "")
    new_directions = []
    if "NONE" not in new_dirs_raw.upper()[:20]:
        for line in new_dirs_raw.splitlines():
            cleaned = line.strip().lstrip("-*• ")
            if len(cleaned) > 15:
                new_directions.append(cleaned)

    return {
        "patterns": patterns[:5],
        "missing": sections.get("missing", ""),
        "new_directions": new_directions[:3],
        "raw": output[:2000],
    }


async def strategize_next_move(
    goal: str,
    task_text: str,
    gaps: str,
    challenge_items: str,
    new_angles: str,
    search_history: list[str],
    phase: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    """
    The Search Strategist — produces the best search queries for the next round.

    Returns dict with keys: queries (list[str]), raw.
    """
    history_text = "\n".join(f"  - {q}" for q in search_history[-40:]) if search_history else "  (none yet)"

    prompt = (
        "You are an intelligence analyst planning the next round of information "
        "gathering. You need to find specific information to fill gaps in an "
        "ongoing investigation.\n\n"
        f"OBJECTIVE: {goal}\n\n"
        f"INFORMATION GAPS (areas where evidence is thin or missing):\n{gaps or '(no specific gaps identified)'}\n\n"
        f"CLAIMS TO STRESS-TEST (look for evidence AGAINST these):\n{challenge_items or '(none)'}\n\n"
        f"NEW LEADS TO FOLLOW:\n{new_angles or '(none)'}\n\n"
        f"ALREADY SEARCHED (do NOT repeat these or close variants):\n{history_text}\n\n"
        f"CURRENT PHASE: {phase}\n"
        "- EARLY (building initial understanding): Broad, diverse queries. Cover as much ground as possible.\n"
        "- MIDDLE (filling specific gaps): Precise, targeted queries. Drill into weak areas.\n"
        "- LATE (stress-testing conclusions): Look for contradictions, critiques, failures, exceptions.\n\n"
        f"The person who needs this information specifically asked: \"{task_text[:150]}\"\n"
        "Every query should ultimately serve their needs.\n\n"
        "Produce 3-5 queries, one per line:\n"
        "QUERY: [3-7 words, suitable for a web search engine]\n"
        "TARGET: [which gap or claim this addresses]\n"
    )

    try:
        output = await ollama_generate(
            prompt, base_url, model,
            timeout=cfg.STRATEGY_TIMEOUT,
            max_tokens=cfg.STRATEGY_MAX_TOKENS,
            temperature=cfg.STRATEGY_TEMPERATURE,
        )
    except Exception:
        return {"queries": [], "raw": ""}

    queries = _extract_queries_from_text(output, search_history)

    if len(queries) < 2:
        for line in output.splitlines():
            q = _clean_query(line.strip())
            if 5 <= len(q) <= 200 and q not in queries:
                queries.append(q)
            if len(queries) >= 4:
                break

    if search_history:
        seen = {h.lower() for h in search_history}
        queries = [q for q in queries if q.lower() not in seen]

    return {"queries": queries[:5], "raw": output[:1500]}


def _criteria_to_initial_queries(criteria: list[str], task_text: str) -> list[str]:
    """Generate initial search queries from plan criteria for iteration 1 bootstrap."""
    queries: list[str] = []
    for criterion in criteria[:5]:
        kw = _extract_keywords(criterion, max_words=5)
        if kw and len(kw) >= 5:
            queries.append(kw)
    if not queries:
        queries = [_extract_keywords(task_text, max_words=5)]
    return queries[:4]
