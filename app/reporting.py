import json
import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .services import ollama_generate


class ReportFinding(BaseModel):
    id: str
    text: str
    source_ids: list[str] = Field(min_length=1)
    confidence: str


class ReportConflict(BaseModel):
    topic: str
    description: str
    source_ids: list[str]


class ReportSource(BaseModel):
    id: str
    title: str
    url: str
    quality: str
    note: str


class StructuredReport(BaseModel):
    summary: str
    findings: list[ReportFinding]
    conflicts: list[ReportConflict] = []
    limitations: list[str] = Field(min_length=1)
    sources: list[ReportSource]


def _source_ids(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, s in enumerate(sources, start=1):
        copy = dict(s)
        copy["source_id"] = f"src_{i}"
        out.append(copy)
    return out


def build_limitations_seed(*, usable_count: int, skipped_count: int, fallback_used: bool, conflict_count: int, domain_count: int) -> list[str]:
    items = []
    if usable_count < 2:
        items.append("Few usable sources were available, so coverage may be incomplete.")
    if skipped_count > usable_count:
        items.append("Many candidate sources were skipped due to fetch or extraction quality issues.")
    if domain_count <= 1:
        items.append("Most usable evidence comes from a narrow set of domains.")
    if conflict_count > 0:
        items.append("Some claims conflict across sources and remain unresolved.")
    if fallback_used:
        items.append("Execution switched to direct fallback mode for reliability.")
    if not items:
        items.append("This report is based only on currently retrievable web content and may miss newer or inaccessible sources.")
    return items


def detect_conflicts(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    pattern = re.compile(r"\b(supports?|does not support|recommended|not recommended|required|optional)\b", re.I)
    topic_map: dict[str, list[tuple[str, str]]] = {}
    for s in sources:
        text = (s.get("content_text") or "")[:2000]
        hits = pattern.findall(text)
        if not hits:
            continue
        key = (s.get("title") or s.get("url") or "topic").split(" ")[0].lower()
        topic_map.setdefault(key, []).append((s["source_id"], " ".join(sorted(set([h.lower() for h in hits])))))

    for topic, entries in topic_map.items():
        if len(entries) < 2:
            continue
        phrases = {p for _, p in entries}
        if len(phrases) > 1:
            conflicts.append(
                {
                    "topic": topic,
                    "description": "Sources describe materially different claims for this topic.",
                    "source_ids": [sid for sid, _ in entries[:4]],
                }
            )
    return conflicts


def assign_confidence(source_quality: list[str], conflict: bool) -> str:
    good = source_quality.count("good")
    medium = source_quality.count("medium")
    if conflict:
        return "low"
    if good >= 2:
        return "high"
    if good >= 1 or medium >= 2:
        return "medium"
    return "low"


def render_markdown_report(report: StructuredReport) -> str:
    lines = ["## Summary", "", report.summary.strip(), "", "## Key Findings", ""]
    for f in report.findings:
        refs = ", ".join(f.source_ids)
        lines.append(f"- {f.text} ({f.confidence}; sources: {refs})")

    if report.conflicts:
        lines.extend(["", "## Source Conflicts", ""])
        for c in report.conflicts:
            refs = ", ".join(c.source_ids)
            lines.append(f"- **{c.topic}**: {c.description} (sources: {refs})")

    lines.extend(["", "## Sources", ""])
    for s in report.sources:
        lines.append(f"1. **{s.id}** [{s.title}]({s.url}) — {s.note} (quality: {s.quality})")

    lines.extend(["", "## Limitations", ""])
    for lim in report.limitations:
        lines.append(f"- {lim}")
    return "\n".join(lines)


def build_preview(markdown: str) -> str:
    text = re.sub(r"[#*`_\[\]()\-]", " ", markdown)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:220]


def build_report_metadata(*, usable_count: int, skipped_count: int, conflict_count: int, execution_mode: str, fallback_used: bool, started_at: str | None) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    duration_ms = None
    if started_at:
        try:
            duration_ms = int((now - datetime.fromisoformat(started_at)).total_seconds() * 1000)
        except Exception:
            duration_ms = None
    return {
        "usable_source_count": usable_count,
        "skipped_source_count": skipped_count,
        "conflict_count": conflict_count,
        "execution_mode": execution_mode,
        "fallback_used": fallback_used,
        "generated_at": now.isoformat(),
        "duration_ms": duration_ms,
    }


async def generate_structured_report(
    *,
    task_text: str,
    sources: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    limitations_seed: list[str],
    ollama_base_url: str,
    model_name: str,
) -> tuple[StructuredReport, str]:
    sources_with_ids = _source_ids(sources)

    prompt = {
        "task": task_text,
        "sources": [
            {
                "id": s["source_id"],
                "title": s.get("title") or "Untitled",
                "url": s.get("url", ""),
                "quality": s.get("quality", "medium"),
                "snippet": (s.get("content_text") or "")[:1300],
            }
            for s in sources_with_ids
        ],
        "conflicts": conflicts,
        "limitations_seed": limitations_seed,
        "schema": {
            "summary": "string",
            "findings": [{"id": "finding_1", "text": "", "source_ids": ["src_1"], "confidence": "low|medium|high"}],
            "conflicts": [{"topic": "", "description": "", "source_ids": ["src_1"]}],
            "limitations": [""],
            "sources": [{"id": "src_1", "title": "", "url": "", "quality": "", "note": ""}],
        },
    }

    def _fallback_report() -> StructuredReport:
        findings = []
        for i, s in enumerate(sources_with_ids[:5], start=1):
            fid = f"finding_{i}"
            finding_text = (s.get("content_text") or "")[:160].strip() or f"Source {s['source_id']} provides relevant context for the task."
            conf = assign_confidence([s.get("quality", "medium")], False)
            findings.append({"id": fid, "text": finding_text, "source_ids": [s["source_id"]], "confidence": conf})
        source_rows = [
            {
                "id": s["source_id"],
                "title": s.get("title") or "Untitled",
                "url": s.get("url", ""),
                "quality": s.get("quality", "medium"),
                "note": "Used as evidence for synthesis.",
            }
            for s in sources_with_ids
        ]
        summary = "This report summarizes the collected sources and highlights the strongest supported findings for the requested task."
        return StructuredReport(summary=summary, findings=findings or [{"id": "finding_1", "text": "Insufficient usable sources for strong conclusions.", "source_ids": ["src_1"] if source_rows else ["src_0"], "confidence": "low"}], conflicts=conflicts, limitations=limitations_seed, sources=source_rows)

    try:
        raw = await ollama_generate(
            "Return ONLY valid JSON matching the schema.\n" + json.dumps(prompt),
            ollama_base_url,
            model_name,
        )
        report = StructuredReport(**json.loads(raw))
        return report, "model"
    except Exception:
        try:
            constrained_prompt = "Return ONLY strict JSON. No prose. Use confidence values low|medium|high." + json.dumps(prompt)
            raw2 = await ollama_generate(constrained_prompt, ollama_base_url, model_name)
            report = StructuredReport(**json.loads(raw2))
            return report, "model_retry"
        except Exception:
            return _fallback_report(), "fallback_minimal"
