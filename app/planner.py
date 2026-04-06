from .schemas import TaskPlan
from .services import formulate_goal


def default_plan(task_text: str) -> TaskPlan:
    return TaskPlan(
        goal=task_text[:380],
        success_criteria=["Find comprehensive information about the topic"],
        steps=[
            {"step_number": 1, "step_type": "search", "description": "Search the web for relevant sources"},
            {"step_number": 2, "step_type": "fetch", "description": "Fetch and extract content from found pages"},
            {"step_number": 3, "step_type": "stop_check", "description": "Verify if goal criteria are met, refine if not"},
            {"step_number": 4, "step_type": "summarize", "description": "Synthesize findings into a comprehensive report"},
        ],
    )


async def generate_plan(task_text: str, ollama_base_url: str, model_name: str) -> tuple[TaskPlan, str]:
    try:
        goal_data = await formulate_goal(task_text, ollama_base_url, model_name)
        goal = goal_data["goal"]
        criteria = goal_data["criteria"]

        steps = [
            {"step_number": 1, "step_type": "search", "description": "Autonomous chain-of-thought: reason about what to search, then execute searches"},
            {"step_number": 2, "step_type": "fetch", "description": "Fetch top candidate pages and extract content"},
        ]
        for i, criterion in enumerate(criteria[:5], 3):
            steps.append({
                "step_number": i,
                "step_type": "stop_check",
                "description": f"Verify: {criterion[:220]}",
            })
        steps.append({
            "step_number": len(steps) + 1,
            "step_type": "summarize",
            "description": "Synthesize verified findings into comprehensive report",
        })

        return TaskPlan(goal=goal[:380], success_criteria=criteria, steps=steps), "model"
    except Exception:
        return default_plan(task_text), "fallback_default"
