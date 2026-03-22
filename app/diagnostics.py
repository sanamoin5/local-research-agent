import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .repository import Repository


@dataclass
class CheckResult:
    name: str
    status: str  # pass/warn/fail
    message: str


def overall_status(checks: list[CheckResult]) -> str:
    if any(c.status == "fail" for c in checks):
        return "setup_required"
    if any(c.status == "warn" for c in checks):
        return "degraded"
    return "healthy"


async def run_diagnostics(repo: Repository, ollama_base_url: str, tavily_api_key: str, recommended_model: str) -> dict[str, Any]:
    checks: list[CheckResult] = []
    repair_actions: list[str] = []

    # Ollama + model
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            resp = await client.get(f"{ollama_base_url}/api/tags")
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            checks.append(CheckResult("ollama", "pass", "Ollama reachable"))
            if any(recommended_model in m for m in models):
                checks.append(CheckResult("model", "pass", f"Recommended model present: {recommended_model}"))
            elif models:
                checks.append(CheckResult("model", "warn", f"Recommended model missing ({recommended_model})"))
                repair_actions.append("pull_model")
            else:
                checks.append(CheckResult("model", "fail", "No model installed"))
                repair_actions.append("pull_model")
    except Exception:
        checks.append(CheckResult("ollama", "fail", "Ollama unreachable"))
        checks.append(CheckResult("model", "warn", "Model check skipped because Ollama is unreachable"))

    # DB and app dir
    try:
        repo.list_tasks()
        checks.append(CheckResult("database", "pass", "SQLite writable"))
    except Exception as exc:
        checks.append(CheckResult("database", "fail", f"SQLite error: {exc.__class__.__name__}"))
        repair_actions.append("recreate_db")

    app_dir = Path(".")
    if os.access(app_dir, os.W_OK):
        checks.append(CheckResult("app_dir", "pass", "App directory writable"))
    else:
        checks.append(CheckResult("app_dir", "fail", "App directory is not writable"))

    # Playwright chromium
    try:
        proc = subprocess.run(["python", "-m", "playwright", "--help"], capture_output=True, text=True, timeout=8)
        if proc.returncode == 0:
            checks.append(CheckResult("playwright", "pass", "Playwright CLI available"))
        else:
            checks.append(CheckResult("playwright", "warn", "Playwright installed but CLI check failed"))
    except Exception:
        checks.append(CheckResult("playwright", "fail", "Playwright Chromium likely missing"))
        repair_actions.append("install_playwright")

    if tavily_api_key:
        checks.append(CheckResult("search_provider", "pass", "Tavily configured"))
    else:
        checks.append(CheckResult("search_provider", "warn", "Tavily key missing; search may fail"))

    try:
        repo.get_settings()
        checks.append(CheckResult("settings", "pass", "Settings loaded"))
    except Exception:
        checks.append(CheckResult("settings", "fail", "Settings could not be loaded"))
        repair_actions.append("reset_settings")

    return {
        "overall_status": overall_status(checks),
        "checks": [c.__dict__ for c in checks],
        "repair_actions": sorted(set(repair_actions)),
    }
