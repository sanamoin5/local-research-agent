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
    step_number: int = Field(ge=1, le=30)
    step_type: StepType
    description: str = Field(min_length=4, max_length=500)
    dependencies: list[int] = Field(default_factory=list)


class TaskPlan(BaseModel):
    goal: str = Field(min_length=3, max_length=600)
    success_criteria: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(min_length=1, max_length=25)

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
    max_total_runtime_sec: int | None = Field(default=None, ge=30, le=36000)
    max_iterations: int | None = Field(default=None, ge=1, le=50)
    fetch_timeout_sec: int | None = Field(default=None, ge=1, le=120)
    max_page_size_bytes: int | None = Field(default=None, ge=50_000, le=10_000_000)
    model_name: str | None = Field(default=None, min_length=1, max_length=200)
    search_provider: Literal["duckduckgo"] | None = None
    cache_enabled: bool | None = None
    reasoning_temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    synthesis_temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    synthesis_max_tokens: int | None = Field(default=None, ge=512, le=65536)
    top_sources_cap: int | None = Field(default=None, ge=1, le=30)
    inter_query_delay: float | None = Field(default=None, ge=0.0, le=30.0)
    inter_iteration_cooldown: float | None = Field(default=None, ge=0.0, le=60.0)


class Settings(BaseModel):
    max_steps: int = 12
    max_pages_per_query: int = 5
    max_total_runtime_sec: int = 36000
    max_iterations: int = 10
    fetch_timeout_sec: int = 10
    max_page_size_bytes: int = 2 * 1024 * 1024
    model_name: str = "llama3.1:8b"
    search_provider: Literal["duckduckgo"] = "duckduckgo"
    cache_enabled: bool = True
    reasoning_temperature: float = 0.4
    synthesis_temperature: float = 0.4
    synthesis_max_tokens: int = 8192
    top_sources_cap: int = 9
    inter_query_delay: float = 4.0
    inter_iteration_cooldown: float = 5.0


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


class ModelAction(BaseModel):
    model_name: str = Field(min_length=1, max_length=200)
