from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


StepType = Literal["search", "fetch", "summarize", "refine_query", "stop_check"]
ToolName = Literal["search_web", "fetch_url", "summarize_sources", "refine_search_query", "evaluate_stop_condition"]


class TaskCreate(BaseModel):
    task: str = Field(min_length=8, max_length=3000)

    @field_validator("task")
    @classmethod
    def task_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task cannot be empty or whitespace")
        return value


class PlanStep(BaseModel):
    step_number: int = Field(ge=1, le=20)
    step_type: StepType
    description: str = Field(min_length=4, max_length=240)
    dependencies: list[int] = Field(default_factory=list)


class TaskPlan(BaseModel):
    goal: str = Field(min_length=3, max_length=280)
    steps: list[PlanStep] = Field(min_length=3, max_length=6)

    @field_validator("steps")
    @classmethod
    def step_numbers_ordered(cls, steps: list[PlanStep]) -> list[PlanStep]:
        expected = list(range(1, len(steps) + 1))
        got = [s.step_number for s in steps]
        if got != expected:
            raise ValueError("step numbers must start at 1 and be sequential")
        return steps


class SettingsUpdate(BaseModel):
    max_steps: int | None = Field(default=None, ge=1, le=30)
    max_pages_per_query: int | None = Field(default=None, ge=1, le=20)
    max_total_runtime_sec: int | None = Field(default=None, ge=30, le=3600)
    fetch_timeout_sec: int | None = Field(default=None, ge=1, le=120)
    max_page_size_bytes: int | None = Field(default=None, ge=50_000, le=10_000_000)
    model_name: str | None = Field(default=None, min_length=1, max_length=200)
    search_provider: Literal["tavily"] | None = None
    cache_enabled: bool | None = None


class Settings(BaseModel):
    max_steps: int = 12
    max_pages_per_query: int = 5
    max_total_runtime_sec: int = 300
    fetch_timeout_sec: int = 10
    max_page_size_bytes: int = 2 * 1024 * 1024
    model_name: str = "llama3.1:8b"
    search_provider: Literal["tavily"] = "tavily"
    cache_enabled: bool = True


class Event(BaseModel):
    type: str
    payload: dict[str, Any]


class ToolCall(BaseModel):
    tool_name: ToolName
    tool_input: dict[str, Any]
    reasoning: str = Field(default="", max_length=240)


class StopDecision(BaseModel):
    should_stop: bool
    reason: str
