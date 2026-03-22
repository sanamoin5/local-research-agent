import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

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


async def ollama_generate(prompt: str, base_url: str, model: str) -> str:
    async def _call() -> str:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{base_url}/api/generate", json={"model": model, "prompt": prompt, "stream": False})
            resp.raise_for_status()
            return resp.json().get("response", "")
    return await run_with_retries(_call, retries=2)


async def generate_queries(task_text: str, base_url: str, model: str) -> list[str]:
    prompt = (
        "Generate exactly 2 to 3 concise web search queries for this task. Return one per line, no numbering.\n\n"
        f"Task: {task_text}"
    )
    try:
        output = await ollama_generate(prompt, base_url, model)
        queries = [line.strip(" -\t") for line in output.splitlines() if line.strip()]
        cleaned = [q for q in queries if 5 <= len(q) <= 200][:3]
        return cleaned or [task_text]
    except Exception:
        return [task_text]


async def tavily_search(api_key: str, query: str, max_results: int) -> list[dict[str, Any]]:
    if not api_key:
        return []

    async def _call() -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": query, "max_results": max_results},
            )
            resp.raise_for_status()
            return resp.json().get("results", [])

    rows = await run_with_retries(_call, retries=2)
    out = []
    for idx, row in enumerate(rows, start=1):
        out.append(
            {
                "query": query,
                "title": row.get("title") or "Untitled",
                "url": row.get("url"),
                "snippet": row.get("content") or "",
                "rank": idx,
                "provider": "tavily",
            }
        )
    return out


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
        return await run_with_retries(_call, retries=2)
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
                await asyncio.sleep(0.6)
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

    quality = "good" if score >= 5 else "medium" if score >= 3 else "poor"
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
    if extraction["metrics"]["text_len"] < 200:
        return True, "extracted_too_short"
    if extraction["quality"] == "poor":
        return True, "quality_poor"
    if extraction["metrics"]["task_overlap"] == 0:
        return True, "low_keyword_overlap"
    return False, "good_enough"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
