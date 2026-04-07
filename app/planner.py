"""
Multi-agent planning pipeline.

Four specialized agents collaborate to build a research plan:

  Agent 1 — INTENT:      What does the user actually want?
  Agent 2 — DECOMPOSE:   What major topic areas should we investigate?
  Agent 3 — EXPAND:      What specific questions should we Google per topic?
  Agent 4 — CRITIQUE:    Is the plan complete? Deduplicate, fill gaps, refine.

Each agent has ONE simple job, keeping prompts short enough for small local models.
The output feeds the autonomous search loop, the per-source fact extractors, and
the report synthesizer — so every criterion must be a distinct, searchable topic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import config as cfg
from .schemas import TaskPlan
from .services import _generate_fallback_criteria, ollama_generate


@dataclass
class PlanningTrace:
    agent: str
    label: str
    detail: str


@dataclass
class PlanningResult:
    plan: TaskPlan
    mode: str  # "multi_agent" | "fallback_default"
    traces: list[PlanningTrace] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────
#  Agent 1: INTENT CLARIFIER
# ────────────────────────────────────────────────────────────────

async def _clarify_intent(
    task_text: str, base_url: str, model: str,
) -> dict[str, str]:
    """
    Understand what the user actually wants. Output a clear research intent
    and a complexity assessment that determines how many topics to decompose into.
    """
    prompt = (
        "A user submitted a research request. Your ONLY job is to understand what they want.\n\n"
        f"USER REQUEST: {task_text}\n\n"
        "Respond in EXACTLY this format:\n\n"
        "INTENT: [Rewrite the request as a clear, specific research question — one sentence]\n"
        "COMPLEXITY: [SIMPLE or MODERATE or COMPLEX]\n"
        "CONTEXT: [One sentence about who this person is and what situation they're in, if evident from the request]\n\n"
        "Rules:\n"
        "- SIMPLE = narrow factual lookup (1-3 search angles enough)\n"
        "- MODERATE = multi-faceted topic (4-6 search angles needed)\n"
        "- COMPLEX = broad life/career/strategy question (7+ search angles needed)\n"
        "- INTENT must be a question or goal, NOT a task instruction\n"
    )
    try:
        output = await ollama_generate(
            prompt, base_url, model,
            timeout=cfg.PLANNING_TIMEOUT,
            max_tokens=cfg.PLANNING_MAX_TOKENS,
            temperature=cfg.PLANNING_TEMPERATURE,
        )
        intent = task_text[:300]
        complexity = "MODERATE"
        context = ""
        for line in output.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("INTENT:"):
                val = stripped.split(":", 1)[1].strip()
                if len(val) > 15:
                    intent = val
            elif upper.startswith("COMPLEXITY:"):
                val = stripped.split(":", 1)[1].strip().upper()
                if "SIMPLE" in val:
                    complexity = "SIMPLE"
                elif "COMPLEX" in val:
                    complexity = "COMPLEX"
                else:
                    complexity = "MODERATE"
            elif upper.startswith("CONTEXT:"):
                context = stripped.split(":", 1)[1].strip()
        return {"intent": intent[:500], "complexity": complexity, "context": context[:300]}
    except Exception:
        return {"intent": task_text[:300], "complexity": "MODERATE", "context": ""}


# ────────────────────────────────────────────────────────────────
#  Agent 2: TOPIC DECOMPOSER
# ────────────────────────────────────────────────────────────────

async def _decompose_topics(
    intent: str, complexity: str, context: str,
    base_url: str, model: str,
) -> list[str]:
    """
    Break the research intent into major topic areas.
    Number of topics scales with complexity.
    """
    if complexity == "SIMPLE":
        count = "3-4"
    elif complexity == "COMPLEX":
        count = "6-8"
    else:
        count = "4-6"

    context_line = f"\nCONTEXT: {context}\n" if context else ""

    prompt = (
        f"Break this research question into {count} MAJOR topic areas to investigate.\n\n"
        f"RESEARCH QUESTION: {intent}\n"
        f"{context_line}\n"
        f"List {count} broad but distinct topic areas, one per line. Each topic is a different angle to research.\n\n"
        "Example for 'How can a teacher transition to tech?':\n"
        "1. Tech career paths that value teaching skills: developer advocacy, instructional design, edtech\n"
        "2. Reskilling and bootcamps for career changers: coding bootcamps, certifications, self-study paths\n"
        "3. Financial planning during career transitions: savings, part-time income, timeline\n"
        "4. Success stories of teachers who moved into tech: real examples and what worked\n"
        "5. Networking and job search strategies for non-traditional tech candidates\n\n"
        f"Now list {count} topic areas for the actual research question above:\n"
    )
    try:
        output = await ollama_generate(
            prompt, base_url, model,
            timeout=cfg.PLANNING_TIMEOUT,
            max_tokens=cfg.PLANNING_MAX_TOKENS,
            temperature=cfg.PLANNING_TEMPERATURE,
        )
        topics = []
        for line in output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            cleaned = re.sub(r"^(\d+[.):\-]\s*|[-*•·]\s*)", "", stripped).strip()
            if len(cleaned) > 15:
                topics.append(cleaned.rstrip("."))
        return topics[:10] if topics else [intent[:200]]
    except Exception:
        return [intent[:200]]


# ────────────────────────────────────────────────────────────────
#  Agent 3: RESEARCH QUESTION EXPANDER
# ────────────────────────────────────────────────────────────────

async def _expand_questions(
    intent: str, topics: list[str],
    base_url: str, model: str,
) -> list[str]:
    """
    For each broad topic, generate 2-3 specific, searchable sub-questions.
    These become the success criteria and report sections.
    """
    topics_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(topics))

    prompt = (
        "For each topic area below, write 2-3 SPECIFIC sub-questions that can be searched on Google.\n\n"
        f"RESEARCH QUESTION: {intent}\n\n"
        f"TOPIC AREAS:\n{topics_text}\n\n"
        "For each topic, write specific searchable sub-questions:\n\n"
    )
    for i, topic in enumerate(topics):
        short = topic[:60]
        prompt += (
            f"TOPIC {i+1}: {short}\n"
            f"- [specific searchable question about {short[:30]}]\n"
            f"- [another specific angle]\n\n"
        )

    prompt += (
        "Rules:\n"
        "- Each sub-question must be something you'd type into Google\n"
        "- Name specific tools, platforms, methods, or concepts\n"
        "- Each sub-question must be DIFFERENT from every other\n"
        "- Do NOT write generic questions like 'what is X' — be specific\n"
    )

    try:
        output = await ollama_generate(
            prompt, base_url, model,
            timeout=cfg.PLANNING_TIMEOUT,
            max_tokens=min(cfg.PLANNING_MAX_TOKENS * 2, 2048),
            temperature=cfg.PLANNING_TEMPERATURE,
        )
        questions: list[str] = []
        for line in output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Skip lines that are topic headers (like "TOPIC 1: ...")
            if re.match(r"^(TOPIC\s+\d+|#{1,3}\s)", stripped, re.I):
                continue
            cleaned = re.sub(r"^(\d+[.):\-]\s*|[-*•·]\s*)", "", stripped).strip()
            cleaned = cleaned.rstrip(".")
            if len(cleaned) > 20:
                questions.append(cleaned)
        return questions[:20] if questions else [t[:200] for t in topics]
    except Exception:
        return [t[:200] for t in topics]


# ────────────────────────────────────────────────────────────────
#  Agent 4: PLAN CRITIC
# ────────────────────────────────────────────────────────────────

async def _critique_plan(
    intent: str, questions: list[str],
    base_url: str, model: str,
) -> list[str]:
    """
    Review the research plan: remove duplicates, merge overlaps, fill gaps.
    Returns the final refined list of research criteria.
    """
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    prompt = (
        "Review this research plan. Your job:\n"
        "1. Remove any duplicate or near-duplicate items\n"
        "2. Merge items that overlap significantly\n"
        "3. Add 1-2 important angles that are MISSING\n"
        "4. Make each item specific and searchable\n\n"
        f"RESEARCH QUESTION: {intent}\n\n"
        f"DRAFT PLAN:\n{questions_text}\n\n"
        "Write the FINAL refined list. Keep good items as-is, fix weak ones, add missing angles.\n"
        "One item per line, numbered. Each item must be a specific, searchable research angle.\n\n"
        "FINAL PLAN:\n"
    )
    try:
        output = await ollama_generate(
            prompt, base_url, model,
            timeout=cfg.PLANNING_TIMEOUT,
            max_tokens=min(cfg.PLANNING_MAX_TOKENS * 2, 2048),
            temperature=cfg.PLANNING_TEMPERATURE,
        )
        refined: list[str] = []
        in_plan = False
        for line in output.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("FINAL PLAN"):
                in_plan = True
                continue
            if not stripped:
                continue
            # Accept numbered items or bullet items
            cleaned = re.sub(r"^(\d+[.):\-]\s*|[-*•·]\s*)", "", stripped).strip()
            cleaned = cleaned.rstrip(".")
            if len(cleaned) > 15:
                if in_plan or not any(upper.startswith(h) for h in ("REVIEW", "ANALYSIS", "DUPLICATE", "MISSING", "MERGE")):
                    refined.append(cleaned)
                    in_plan = True  # Once we start seeing items, keep reading

        # Deduplicate by checking for high overlap
        final: list[str] = []
        for item in refined:
            item_words = set(item.lower().split())
            is_dup = False
            for existing in final:
                existing_words = set(existing.lower().split())
                overlap = len(item_words & existing_words) / max(len(item_words | existing_words), 1)
                if overlap > 0.6:
                    is_dup = True
                    break
            if not is_dup:
                final.append(item)

        return final[:20] if final else questions[:15]
    except Exception:
        return questions[:15]


# ────────────────────────────────────────────────────────────────
#  Main entry point
# ────────────────────────────────────────────────────────────────

def _criteria_to_steps(criteria: list[str]) -> list[dict[str, Any]]:
    """Turn each research criterion into a user-visible plan step."""
    steps: list[dict[str, Any]] = []
    for i, criterion in enumerate(criteria, 1):
        desc = criterion.strip()
        if len(desc) > 120:
            desc = desc[:117].rsplit(" ", 1)[0] + "..."
        steps.append({
            "step_number": i,
            "step_type": "search",
            "description": f"Research: {desc}",
        })
    steps.append({
        "step_number": len(steps) + 1,
        "step_type": "summarize",
        "description": "Compile all findings into a structured report with actionable recommendations",
    })
    return steps


def default_plan(task_text: str) -> PlanningResult:
    criteria = _generate_fallback_criteria(task_text)
    steps = _criteria_to_steps(criteria)
    plan = TaskPlan(goal=task_text[:380], success_criteria=criteria, steps=steps)
    return PlanningResult(plan=plan, mode="fallback_default", traces=[
        PlanningTrace("fallback", "Using fallback plan", "LLM planning failed — using keyword-based criteria"),
    ])


async def generate_plan(
    task_text: str, ollama_base_url: str, model_name: str,
) -> PlanningResult:
    """
    Multi-agent planning pipeline.
    4 specialized agents collaborate to produce a thorough research plan.
    """
    traces: list[PlanningTrace] = []

    try:
        # ── Agent 1: INTENT ──
        intent_data = await _clarify_intent(task_text, ollama_base_url, model_name)
        intent = intent_data["intent"]
        complexity = intent_data["complexity"]
        context = intent_data["context"]
        traces.append(PlanningTrace(
            "intent",
            f"[INTENT] Understood request (complexity: {complexity})",
            f"Intent: {intent}\nContext: {context}" if context else f"Intent: {intent}",
        ))

        # ── Agent 2: DECOMPOSE ──
        topics = await _decompose_topics(
            intent, complexity, context, ollama_base_url, model_name,
        )
        traces.append(PlanningTrace(
            "decompose",
            f"[DECOMPOSE] Identified {len(topics)} major topic areas",
            "\n".join(f"  {i+1}. {t}" for i, t in enumerate(topics)),
        ))

        # ── Agent 3: EXPAND ──
        questions = await _expand_questions(
            intent, topics, ollama_base_url, model_name,
        )
        traces.append(PlanningTrace(
            "expand",
            f"[EXPAND] Generated {len(questions)} specific research angles",
            "\n".join(f"  {i+1}. {q}" for i, q in enumerate(questions)),
        ))

        # ── Agent 4: CRITIQUE ──
        criteria = await _critique_plan(
            intent, questions, ollama_base_url, model_name,
        )
        traces.append(PlanningTrace(
            "critique",
            f"[CRITIQUE] Refined to {len(criteria)} final research criteria",
            "\n".join(f"  {i+1}. {q}" for i, q in enumerate(criteria)),
        ))

        if not criteria:
            criteria = _generate_fallback_criteria(task_text)
            traces.append(PlanningTrace(
                "fallback", "[FALLBACK] Critic produced nothing — using keyword criteria", "",
            ))

        steps = _criteria_to_steps(criteria)
        plan = TaskPlan(
            goal=intent[:580],
            success_criteria=criteria,
            steps=steps,
        )
        return PlanningResult(plan=plan, mode="multi_agent", traces=traces)

    except Exception as exc:
        traces.append(PlanningTrace(
            "error", f"[ERROR] Planning pipeline failed: {exc.__class__.__name__}", str(exc)[:300],
        ))
        result = default_plan(task_text)
        result.traces = traces + result.traces
        return result
