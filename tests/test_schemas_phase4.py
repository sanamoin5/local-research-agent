import pytest
from pydantic import ValidationError

from app.reporting import StructuredReport


def test_structured_report_requires_limitations():
    with pytest.raises(ValidationError):
        StructuredReport(
            summary="s",
            findings=[{"id": "f1", "text": "x", "source_ids": ["src_1"], "confidence": "low"}],
            conflicts=[],
            limitations=[],
            sources=[{"id": "src_1", "title": "t", "url": "https://e.com", "quality": "good", "note": "n"}],
        )
