import asyncio
import json
import os
import subprocess
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .diagnostics import run_diagnostics
from .executor import AgentExecutor
from .logging_utils import configure_logging, log_event, recent_logs
from .pipeline import EventBus, RunConfig, health_report, run_direct_pipeline, set_runtime
from .planner import generate_plan
from .repository import Repository
from .schemas import SettingsUpdate, TaskCreate

DB_PATH = os.getenv("LRA_DB_PATH", "local_research_agent.db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
RECOMMENDED_MODEL = os.getenv("RECOMMENDED_MODEL", "llama3.1:8b")

configure_logging()
repo = Repository(DB_PATH)
bus = EventBus()
config = RunConfig(ollama_base_url=OLLAMA_BASE_URL, tavily_api_key=TAVILY_API_KEY)
set_runtime(repo, bus)
executor = AgentExecutor(repo, bus, config)

app = FastAPI(title="Local Research Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_hook() -> None:
    reconciled = repo.reconcile_interrupted_tasks()
    log_event("info", "startup", "startup reconciliation complete", metadata={"reconciled_tasks": reconciled})


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/api/settings")
async def get_settings() -> dict[str, Any]:
    return repo.get_settings().model_dump()


@app.put("/api/settings")
async def update_settings(payload: SettingsUpdate) -> dict[str, Any]:
    return repo.update_settings(payload.model_dump(exclude_none=True)).model_dump()


@app.post("/api/tasks")
async def create_task(payload: TaskCreate) -> dict[str, Any]:
    settings = repo.get_settings()
    plan, plan_mode = await generate_plan(payload.task, OLLAMA_BASE_URL, settings.model_name)
    task_id = repo.create_task(payload.task, settings, plan.model_dump())
    repo.set_plan(task_id, plan.model_dump(), "pending")
    await bus.emit(task_id, "plan_created", {"plan_mode": plan_mode, "plan": plan.model_dump()})
    await bus.emit(task_id, "plan_waiting_confirmation", {"task_id": task_id})
    return {"task_id": task_id, "status": "planning", "plan": plan.model_dump()}


@app.post("/api/tasks/{task_id}/confirm")
async def confirm_task(task_id: str) -> dict[str, str]:
    task = repo.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    if task.get("plan_status") != "pending":
        return {"task_id": task_id, "status": task["status"]}

    repo.approve_plan(task_id)
    await bus.emit(task_id, "plan_confirmed", {"task_id": task_id})

    async def guarded_run() -> None:
        try:
            result = await executor.run(task_id)
            if result.get("status") == "fallback":
                reason = result.get("reason", "tooling_unreliable")
                repo.update_execution_metadata(task_id, execution_mode="direct_fallback", fallback_reason=reason)
                await bus.emit(task_id, "fallback_triggered", {"reason": reason, "message": "Switched to direct research pipeline for reliability"})
                await run_direct_pipeline(task_id, config, repo, bus)
        except Exception as exc:
            log_event("error", "task", "guarded execution failed", task_id=task_id, metadata={"error": exc.__class__.__name__})
            repo.update_task_status(task_id, "failed", error_message="Task execution failed", terminal_reason=exc.__class__.__name__)
            await bus.emit(task_id, "error", {"message": "Task failed unexpectedly. Please check diagnostics."})
            await bus.close(task_id)
        finally:
            latest = repo.get_task(task_id)
            if latest and latest["status"] in {"planning", "running"}:
                repo.update_task_status(task_id, "failed", error_message="Execution ended without terminal state", terminal_reason="terminal_state_guard")
                await bus.emit(task_id, "error", {"message": "Task stopped unexpectedly and was marked failed."})
                await bus.close(task_id)

    asyncio.create_task(guarded_run())
    return {"task_id": task_id, "status": "running"}


@app.post("/api/tasks/{task_id}/reject")
async def reject_task(task_id: str) -> dict[str, str]:
    task = repo.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    if task.get("plan_status") == "pending":
        repo.reject_plan(task_id)
        await bus.emit(task_id, "warning", {"message": "Plan rejected by user"})
        await bus.close(task_id)
    return {"task_id": task_id, "status": "cancelled"}


@app.get("/api/tasks")
async def list_tasks() -> list[dict[str, Any]]:
    return repo.list_tasks()


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str) -> dict[str, Any]:
    task = repo.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    task["settings_snapshot"] = json.loads(task.get("settings_snapshot_json") or "{}")
    task["plan"] = json.loads(task.get("plan_json") or "null")
    task["structured_output"] = json.loads(task.get("structured_output_json") or "null")
    task["report_metadata"] = json.loads(task.get("report_metadata_json") or "{}")
    return task


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str) -> dict[str, str]:
    task = repo.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    if task["status"] in {"planning", "running"}:
        repo.request_cancel(task_id)
        repo.update_task_status(task_id, "cancelled", terminal_reason="cancelled_by_user")
        await bus.emit(task_id, "warning", {"message": "Task cancelled by user"})
        await bus.close(task_id)
        return {"task_id": task_id, "status": "cancelled"}
    return {"task_id": task_id, "status": task["status"]}


@app.get("/api/tasks/{task_id}/events")
async def task_events(task_id: str) -> StreamingResponse:
    async def stream() -> AsyncGenerator[str, None]:
        q = bus.queue_for(task_id)
        while True:
            e = await q.get()
            if e.type == "__close__":
                break
            yield f"event: {e.type}\ndata: {json.dumps(e.payload)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/tasks/{task_id}/export")
async def export_task(task_id: str, format: str = "md"):
    task = repo.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    if format != "md":
        raise HTTPException(status_code=400, detail="unsupported format")
    return JSONResponse({"filename": f"task_{task_id}.md", "content": task.get("output_markdown") or ""})


@app.get("/api/health")
async def health() -> JSONResponse:
    diag = await run_diagnostics(repo, OLLAMA_BASE_URL, TAVILY_API_KEY, RECOMMENDED_MODEL)
    return JSONResponse(diag)


@app.post("/api/repair/playwright")
async def repair_playwright() -> JSONResponse:
    try:
        proc = subprocess.run(["python", "-m", "playwright", "install", "chromium"], capture_output=True, text=True, timeout=300)
        ok = proc.returncode == 0
        msg = "Chromium installed" if ok else (proc.stderr.strip() or "Playwright install failed")
        return JSONResponse({"ok": ok, "message": msg})
    except Exception as exc:
        return JSONResponse({"ok": False, "message": f"Playwright repair failed: {exc.__class__.__name__}"})


@app.post("/api/repair/model")
async def repair_model() -> JSONResponse:
    try:
        proc = subprocess.run(["ollama", "pull", RECOMMENDED_MODEL], capture_output=True, text=True, timeout=600)
        ok = proc.returncode == 0
        msg = f"Model {RECOMMENDED_MODEL} ready" if ok else (proc.stderr.strip() or "Model pull failed")
        return JSONResponse({"ok": ok, "message": msg})
    except Exception as exc:
        return JSONResponse({"ok": False, "message": f"Model repair failed: {exc.__class__.__name__}"})


@app.post("/api/repair/db")
async def repair_db() -> JSONResponse:
    try:
        repo.recreate_schema()
        return JSONResponse({"ok": True, "message": "Database schema checked/recreated"})
    except Exception as exc:
        return JSONResponse({"ok": False, "message": f"DB repair failed: {exc.__class__.__name__}"})


@app.post("/api/repair/settings/reset")
async def repair_reset_settings() -> JSONResponse:
    settings = repo.reset_settings()
    return JSONResponse({"ok": True, "message": "Settings reset to defaults", "settings": settings.model_dump()})


@app.get("/api/logs/recent")
async def logs_recent(limit: int = 100) -> JSONResponse:
    return JSONResponse({"items": recent_logs(limit)})


@app.get("/api/health/basic")
async def basic_health() -> JSONResponse:
    return JSONResponse(await health_report(OLLAMA_BASE_URL, repo, TAVILY_API_KEY))
