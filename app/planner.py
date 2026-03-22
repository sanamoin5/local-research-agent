import json
from typing import Any

from .schemas import TaskPlan
from .services import ollama_generate


def default_plan(task_text: str) -> TaskPlan:
    return TaskPlan(
        goal=task_text[:260],
        steps=[
            {"step_number": 1, "step_type": "search", "description": "Search for high-signal sources"},
            {"step_number": 2, "step_type": "fetch", "description": "Fetch and extract candidate pages"},
            {"step_number": 3, "step_type": "summarize", "description": "Summarize findings with limitations"},
        ],
    )


async def generate_plan(task_text: str, ollama_base_url: str, model_name: str) -> tuple[TaskPlan, str]:
    prompt = (
        "Return ONLY valid JSON for a research plan.\n"
        "Schema: {\"goal\": string, \"steps\": [{\"step_number\": int, \"step_type\": one of [search,fetch,summarize,refine_query,stop_check], \"description\": string, \"dependencies\": [int]}]}\n"
        "Create 3-6 steps with sequential step_number starting at 1.\n"
        f"Task: {task_text}"
    )
    try:
        raw = await ollama_generate(prompt, ollama_base_url, model_name)
        data: dict[str, Any] = json.loads(raw)
        return TaskPlan(**data), "model"
    except Exception:
        return default_plan(task_text), "fallback_default"
