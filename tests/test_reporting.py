from app.reporting import (
    StructuredReport,
    assign_confidence,
    build_limitations_seed,
    build_preview,
    render_markdown_report,
)


def test_confidence_assignment():
    assert assign_confidence(["good", "good"], False) == "high"
    assert assign_confidence(["good"], False) == "medium"
    assert assign_confidence(["medium"], True) == "low"


def test_limitations_seed_always_present():
    out = build_limitations_seed(
        usable_count=1,
        skipped_count=4,
        fallback_used=True,
        conflict_count=1,
        domain_count=1,
    )
    assert out
    assert any("Few usable" in x for x in out)


def test_markdown_renderer_sections():
    report = StructuredReport(
        summary="Short summary",
        findings=[{"id": "finding_1", "text": "Finding", "source_ids": ["src_1"], "confidence": "medium"}],
        conflicts=[],
        limitations=["A limitation"],
        sources=[{"id": "src_1", "title": "Title", "url": "https://example.com", "quality": "good", "note": "Contribution"}],
    )
    md = render_markdown_report(report)
    assert "## Summary" in md
    assert "## Key Findings" in md
    assert "## Sources" in md
    assert "## Limitations" in md
    assert "## Source Conflicts" not in md


def test_preview_generation_plaintext():
    md = "## Summary\n**Hello** [link](https://example.com)"
    preview = build_preview(md)
    assert "#" not in preview
    assert len(preview) <= 220
