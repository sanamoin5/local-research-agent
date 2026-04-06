from dataclasses import dataclass
from typing import Any

from .repository import Repository
from .schemas import Settings, StopDecision, ToolCall
from .services import extract_content, fetch_http, generate_queries, normalize_url, web_search


@dataclass
class ToolContext:
    repo: Repository
    task_id: str
    task_text: str
    settings: Settings
    ollama_base_url: str
    collected_sources: list[dict[str, Any]]


class ToolRouter:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    async def run(self, call: ToolCall) -> dict[str, Any]:
        if call.tool_name == "search_web":
            query = str(call.tool_input.get("query", self.ctx.task_text))
            rows = await web_search(query, self.ctx.settings.max_pages_per_query)
            unique = {}
            for row in rows:
                if not row.get("url"):
                    continue
                row["normalized_url"] = normalize_url(row["url"])
                unique.setdefault(row["normalized_url"], row)
            return {"query": query, "results": list(unique.values())[: self.ctx.settings.max_pages_per_query]}

        if call.tool_name == "fetch_url":
            url = str(call.tool_input["url"])
            query = str(call.tool_input.get("query", self.ctx.task_text))
            fetch = await fetch_http(url, self.ctx.settings.fetch_timeout_sec, self.ctx.settings.max_page_size_bytes)
            if not fetch.ok or not fetch.html:
                return {"ok": False, "reason": fetch.error_reason, "url": url}
            extracted = extract_content(fetch.html, self.ctx.task_text, query)
            out = {
                "ok": extracted["quality"] in {"medium", "good"},
                "url": url,
                "title": extracted["title"],
                "content_text": extracted["text"][:30_000],
                "quality": extracted["quality"],
                "metrics": extracted["metrics"],
                "fetch_status": fetch.fetch_status,
                "content_type": fetch.content_type,
                "content_length_bytes": fetch.content_length_bytes,
                "extraction_method": "http_trafilatura",
            }
            if out["ok"]:
                self.ctx.collected_sources.append(out)
            return out

        if call.tool_name == "refine_search_query":
            seed = str(call.tool_input.get("seed", self.ctx.task_text))
            queries = await generate_queries(seed, self.ctx.ollama_base_url, self.ctx.settings.model_name)
            return {"queries": queries}

        if call.tool_name == "evaluate_stop_condition":
            step_count = int(call.tool_input.get("step_count", 0))
            elapsed_sec = int(call.tool_input.get("elapsed_sec", 0))
            if step_count >= self.ctx.settings.max_steps:
                decision = StopDecision(should_stop=True, reason="max_steps_reached")
            elif elapsed_sec >= self.ctx.settings.max_total_runtime_sec:
                decision = StopDecision(should_stop=True, reason="max_runtime_reached")
            elif len(self.ctx.collected_sources) >= 3:
                decision = StopDecision(should_stop=True, reason="sufficient_sources")
            else:
                decision = StopDecision(should_stop=False, reason="continue")
            return decision.model_dump()

        if call.tool_name == "summarize_sources":
            return {"source_count": len(self.ctx.collected_sources)}

        raise ValueError(f"unknown_tool:{call.tool_name}")
