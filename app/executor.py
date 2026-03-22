import json
from datetime import datetime, timezone
from typing import Any

from .pipeline import RunConfig
from .planner import default_plan
from .repository import Repository
from .schemas import Settings, ToolCall
from .services import ollama_generate
from .reporting import build_limitations_seed, build_preview, build_report_metadata, detect_conflicts, generate_structured_report, render_markdown_report
from .tool_router import ToolContext, ToolRouter


TOOL_DESCRIPTIONS = [
    {"name": "search_web", "input": {"query": "string"}},
    {"name": "fetch_url", "input": {"url": "string", "query": "string"}},
    {"name": "refine_search_query", "input": {"seed": "string"}},
    {"name": "evaluate_stop_condition", "input": {"step_count": "int", "elapsed_sec": "int"}},
    {"name": "summarize_sources", "input": {}},
]


class AgentExecutor:
    def __init__(self, repo: Repository, bus: Any, config: RunConfig):
        self.repo = repo
        self.bus = bus
        self.config = config

    async def _choose_tool_call(self, task_text: str, plan: dict[str, Any], state: dict[str, Any], settings: Settings, constrained: bool) -> ToolCall:
        toolset = TOOL_DESCRIPTIONS[:3] if constrained else TOOL_DESCRIPTIONS
        prompt = (
            "Return ONLY JSON with schema: {\"tool_name\": string, \"tool_input\": object, \"reasoning\": string}.\n"
            f"Allowed tools: {json.dumps(toolset)}\n"
            f"Task: {task_text}\n"
            f"Plan: {json.dumps(plan)}\n"
            f"State: {json.dumps(state)}\n"
            "Pick the single best next tool call."
        )
        raw = await ollama_generate(prompt, self.config.ollama_base_url, settings.model_name)
        return ToolCall(**json.loads(raw))

    async def run(self, task_id: str) -> dict[str, Any]:
        task = self.repo.get_task(task_id)
        if not task:
            return {"status": "failed", "error": "task_not_found"}

        settings = Settings(**json.loads(task["settings_snapshot_json"]))
        plan = json.loads(task["plan_json"]) if task.get("plan_json") else default_plan(task["input_text"]).model_dump()
        start = datetime.now(timezone.utc)
        self.repo.mark_started(task_id)
        self.repo.update_execution_metadata(task_id, execution_mode="agent")

        ctx = ToolContext(
            repo=self.repo,
            task_id=task_id,
            task_text=task["input_text"],
            settings=settings,
            tavily_api_key=self.config.tavily_api_key,
            ollama_base_url=self.config.ollama_base_url,
            collected_sources=[],
        )
        router = ToolRouter(ctx)

        step_idx = 1
        retries = 0
        state: dict[str, Any] = {"queries": [], "candidates": [], "fetched": 0, "usable_sources": 0}

        while step_idx <= settings.max_steps:
            latest = self.repo.get_task(task_id)
            if latest and latest["status"] == "cancelled":
                await self.bus.emit(task_id, "warning", {"message": "Execution stopped due to cancellation"})
                await self.bus.close(task_id)
                return {"status": "cancelled"}

            elapsed = int((datetime.now(timezone.utc) - start).total_seconds())
            if elapsed >= settings.max_total_runtime_sec:
                break

            try:
                call = await self._choose_tool_call(task["input_text"], plan, state, settings, constrained=False)
            except Exception:
                retries += 1
                self.repo.add_step(task_id, step_idx, "agent_parse", "failed", datetime.now(timezone.utc), "Tool call parse failed; constrained retry", {"retry": retries})
                if retries > 1:
                    return {"status": "fallback", "reason": "tool_call_parse_failed"}
                try:
                    call = await self._choose_tool_call(task["input_text"], plan, state, settings, constrained=True)
                except Exception:
                    return {"status": "fallback", "reason": "tool_call_parse_failed_twice"}

            await self.bus.emit(task_id, "step_started", {"step": call.tool_name, "message": call.reasoning or f"Running {call.tool_name}"})
            started = datetime.now(timezone.utc)
            try:
                output = await router.run(call)
                self.repo.add_step(
                    task_id,
                    step_idx,
                    "agent_tool",
                    "completed",
                    started,
                    call.reasoning or f"Executed {call.tool_name}",
                    {"tool_name": call.tool_name, "tool_input": call.tool_input, "tool_output": output},
                    reasoning_text=call.reasoning,
                    tool_name=call.tool_name,
                    tool_input=call.tool_input,
                    tool_output=output,
                )
            except Exception as exc:
                self.repo.add_step(
                    task_id,
                    step_idx,
                    "agent_tool",
                    "failed",
                    started,
                    f"Tool exception: {exc.__class__.__name__}",
                    {"tool_name": call.tool_name},
                    reasoning_text=call.reasoning,
                    tool_name=call.tool_name,
                    tool_input=call.tool_input,
                    tool_output={"error": exc.__class__.__name__},
                )
                await self.bus.emit(task_id, "warning", {"message": f"Tool failed: {call.tool_name}"})
                step_idx += 1
                continue

            if call.tool_name == "search_web":
                state["candidates"] = output.get("results", [])
                if state["candidates"]:
                    state["queries"].append(output.get("query"))
            elif call.tool_name == "fetch_url":
                state["fetched"] += 1
                if output.get("ok"):
                    state["usable_sources"] = len(ctx.collected_sources)
                    self.repo.upsert_source(
                        task_id,
                        {
                            "url": output["url"],
                            "normalized_url": output["url"],
                            "title": output.get("title"),
                            "fetch_status": "ok",
                            "extraction_status": "ok",
                            "quality": output.get("quality"),
                            "extraction_method": output.get("extraction_method"),
                            "content_text": output.get("content_text"),
                            "content_type": output.get("content_type"),
                            "content_length_bytes": output.get("content_length_bytes"),
                            "provider": "tavily",
                            "cache_hit": False,
                        },
                    )
                    await self.bus.emit(task_id, "source_found", {"url": output["url"], "message": "Usable source added"})
                else:
                    self.repo.upsert_source(
                        task_id,
                        {
                            "url": output.get("url", ""),
                            "normalized_url": output.get("url", ""),
                            "fetch_status": "failed",
                            "extraction_status": "skipped",
                            "quality": "poor",
                            "error_reason": output.get("reason", "fetch_failed"),
                            "provider": "tavily",
                            "cache_hit": False,
                        },
                    )
                    await self.bus.emit(task_id, "source_skipped", {"url": output.get("url", "unknown"), "reason": output.get("reason", "fetch_failed")})
            elif call.tool_name == "refine_search_query":
                state["queries"].extend(output.get("queries", []))
            elif call.tool_name == "evaluate_stop_condition":
                decision = output
                if decision.get("should_stop"):
                    break

            await self.bus.emit(task_id, "step_completed", {"step": call.tool_name})

            # deterministic stop heuristic also enforced
            if len(ctx.collected_sources) >= 4:
                break
            step_idx += 1

        await self.bus.emit(task_id, "conflict_analysis_started", {"message": "Analyzing source conflicts"})
        conflicts = detect_conflicts(ctx.collected_sources)
        await self.bus.emit(task_id, "conflict_analysis_completed", {"conflict_count": len(conflicts)})

        skipped = max(0, len(state.get("candidates", [])) - len(ctx.collected_sources))
        domains = len({(s.get("url", "").split("/")[2] if "/" in s.get("url", "") else s.get("url", "")) for s in ctx.collected_sources if s.get("url")})
        limitations = build_limitations_seed(
            usable_count=len(ctx.collected_sources),
            skipped_count=skipped,
            fallback_used=False,
            conflict_count=len(conflicts),
            domain_count=domains,
        )

        await self.bus.emit(task_id, "final_synthesis_started", {"message": "Generating final structured report"})
        structured, synthesis_mode = await generate_structured_report(
            task_text=task["input_text"],
            sources=ctx.collected_sources,
            conflicts=conflicts,
            limitations_seed=limitations,
            ollama_base_url=self.config.ollama_base_url,
            model_name=settings.model_name,
        )
        markdown = render_markdown_report(structured)
        preview = build_preview(markdown)
        meta = build_report_metadata(
            usable_count=len(ctx.collected_sources),
            skipped_count=skipped,
            conflict_count=len(structured.conflicts),
            execution_mode="agent",
            fallback_used=False,
            started_at=task.get("started_at"),
        )
        meta["synthesis_mode"] = synthesis_mode

        self.repo.store_report(
            task_id,
            markdown=markdown,
            structured_output=structured.model_dump(),
            output_preview=preview,
            usable_count=len(ctx.collected_sources),
            skipped_count=skipped,
            conflict_count=len(structured.conflicts),
            report_metadata=meta,
        )
        self.repo.update_task_status(task_id, "completed")
        await self.bus.emit(task_id, "final_synthesis_completed", {"synthesis_mode": synthesis_mode})
        await self.bus.emit(task_id, "output_ready", {"task_id": task_id})
        await self.bus.close(task_id)
        return {"status": "completed", "usable_sources": len(ctx.collected_sources)}
