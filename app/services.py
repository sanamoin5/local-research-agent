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


def extract_content(html: str, task_text: str, query_text: str | None, fallback_title: str | None = None) -> dict[str, Any]:
    if trafilatura:
        text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
        meta = trafilatura.extract_metadata(html)
        title = (meta.title if meta and meta.title else None) or fallback_title or "Untitled"
    else:
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
        title = fallback_title or "Untitled"

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
    """Parse a user's request into a clear goal with measurable success criteria."""
    prompt = (
        "You are a research goal analyst. Convert this request into a clear, measurable research goal.\n\n"
        f"Request: {task_text}\n\n"
        "Write EXACTLY this format:\n"
        "GOAL: (one sentence — what does a complete answer look like?)\n"
        "CRITERIA:\n"
        "- (first specific thing that must be found or confirmed)\n"
        "- (second specific criterion)\n"
        "- (third specific criterion)\n"
        "- (optional fourth criterion)\n\n"
        "Be SPECIFIC. Each criterion should be something you can check.\n"
        "Each criterion must be SHORT (under 30 words) and testable.\n"
    )
    try:
        output = await ollama_generate(prompt, base_url, model, temperature=cfg.GOAL_TEMPERATURE)
        goal = ""
        criteria: list[str] = []
        in_criteria = False
        for line in output.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("GOAL:"):
                goal = stripped[5:].strip()
                in_criteria = False
            elif upper.startswith("CRITERIA"):
                in_criteria = True
                continue
            elif in_criteria:
                # Accept: "- item", "* item", "• item", "1. item", "1) item"
                cleaned = re.sub(r"^(\d+[.)]\s*|[-*•·]\s*)", "", stripped)
                if len(cleaned) > 10:
                    criteria.append(cleaned.rstrip("."))
        if not goal:
            goal = task_text[:300]
        if not criteria:
            criteria = [f"Find comprehensive, detailed information about: {task_text[:100]}"]
        return {"goal": goal[:400], "criteria": criteria[:6]}
    except Exception:
        return {
            "goal": task_text[:300],
            "criteria": [f"Find comprehensive information about: {task_text[:100]}"],
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
    Unified Auto-GPT style thinking step.
    One call produces: THOUGHTS → REASONING → PLAN → CRITICISM → ACTION.
    The model reflects on what it researched, critiques itself, then decides
    whether to SEARCH more or COMPLETE.
    """
    criteria_text = "\n".join(f"  {i}. {c}" for i, c in enumerate(criteria, 1))
    history_text = "\n".join(f"  - {q}" for q in search_history[-10:]) if search_history else "  (none yet)"

    prompt = (
        "You are an autonomous research agent. You operate in a loop: think, search the web, observe results, repeat.\n\n"
        f"GOAL: {goal}\n\n"
        f"SUCCESS CRITERIA:\n{criteria_text}\n\n"
        f"WHAT I HAVE COLLECTED SO FAR:\n{knowledge_summary}\n\n"
        f"SEARCHES ALREADY TRIED:\n{history_text}\n\n"
        f"LAST OBSERVATION:\n{last_observation}\n\n"
        "Now respond in EXACTLY this format (use these exact headers):\n\n"
        "THOUGHTS: Analyze what I found so far. What is useful? What is missing? How close am I to meeting each criterion?\n\n"
        "REASONING: Why am I taking my next action? What angle or gap am I targeting?\n\n"
        "PLAN:\n1. (immediate next step)\n2. (step after that)\n3. (further steps)\n\n"
        "CRITICISM: Be harsh with yourself. What are you missing? What could go wrong? Is your research thorough enough? Are you being lazy?\n\n"
        "ACTION: SEARCH or COMPLETE\n"
        "(choose COMPLETE only if ALL criteria are well-satisfied with strong evidence)\n"
        "(choose SEARCH if ANY criterion still needs more evidence)\n\n"
        "QUERIES:\n"
        "(if SEARCH: write 3-4 SHORT web search queries, one per line)\n"
        "(if COMPLETE: write \"ready to synthesize\")\n\n"
        "CRITICAL RULES FOR QUERIES:\n"
        "- Each query MUST be 3-7 words, keyword-style like a Google search\n"
        "- Queries search FOR INFORMATION, they are NOT instructions or tasks\n"
        "- Do NOT write the goal or task as a query — break it into searchable topics\n"
        "- Do NOT copy or rephrase previous queries\n"
        "- Do NOT use verbs like 'develop', 'write', 'create', 'find', 'search' at the start\n"
        "Examples for good or bad queries when asked to help in creating a letter of intent for belgium for a freelance professional card:"
        "- GOOD queries: noun phrases, topics, terms — things you'd type into Google\n"
        "- GOOD: letter of intent Belgian agency template\n"
        "- GOOD: freelance professional card Belgium requirements\n"
        "- GOOD: neutral tone business correspondence Belgium\n"
        "- BAD: Develop a letter for Belgian agency cooperation\n"
        "- BAD: Write a sample letter that requests letter of intent\n"
        "- BAD: Find information about freelance cards in Belgium\n"
    )

    try:
        output = await ollama_generate(prompt, base_url, model, temperature=temperature)
    except Exception:
        kw = _extract_keywords(goal)
        return {
            "thoughts": "Model call failed",
            "reasoning": "", "plan": "", "criticism": "",
            "action": "SEARCH",
            "queries": [kw],
            "raw": "",
        }

    sections: dict[str, str] = {}
    current_key = "preamble"
    current_lines: list[str] = []
    # Flexible matching: accept common variations of section headers
    header_map = {
        "THOUGHTS": "thoughts",
        "THOUGHT": "thoughts",
        "REASONING": "reasoning",
        "REASON": "reasoning",
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

        # Fallback: if no QUERIES section, try lines after ACTION as implicit queries
        if not query_text.strip():
            action_section = sections.get("action", "")
            action_lines = action_section.splitlines()
            # First line is "SEARCH", remaining lines may be queries
            if len(action_lines) > 1:
                query_text = "\n".join(action_lines[1:])

        for line in query_text.splitlines():
            q = _clean_query(line)
            if 5 <= len(q) <= 200:
                queries.append(q)
        # Dedup against search history
        seen = {h.lower() for h in search_history}
        queries = [q for q in queries if q.lower() not in seen]

    # Fallback: if no usable queries, extract keywords from goal + criteria
    if action == "SEARCH" and not queries:
        kw = _extract_keywords(goal)
        criteria_kw = _extract_keywords(" ".join(criteria[:2]), max_words=4) if criteria else ""
        fallback = []
        if kw:
            fallback.append(kw)
        if criteria_kw and criteria_kw != kw:
            fallback.append(criteria_kw)
        seen_lower = {h.lower() for h in search_history}
        fallback = [q for q in fallback if q.lower() not in seen_lower]
        if not fallback:
            fallback = [f"{kw} guide {iteration}"]
        queries = fallback

    return {
        "thoughts": sections.get("thoughts", "")[:800],
        "reasoning": sections.get("reasoning", "")[:400],
        "plan": sections.get("plan", "")[:400],
        "criticism": sections.get("criticism", "")[:400],
        "action": action,
        "queries": queries[:4],
        "raw": output[:1500],
    }
