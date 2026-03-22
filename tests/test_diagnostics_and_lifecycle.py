import asyncio
from pathlib import Path

from app.diagnostics import overall_status
from app.repository import Repository
from app.retry_utils import is_retryable_error


class TimeoutLikeError(Exception):
    pass


def test_overall_status_levels():
    checks = [
        type("C", (), {"status": "pass"})(),
        type("C", (), {"status": "warn"})(),
    ]
    assert overall_status(checks) == "degraded"
    checks2 = [type("C", (), {"status": "fail"})()]
    assert overall_status(checks2) == "setup_required"


def test_retry_classifier():
    assert is_retryable_error(TimeoutLikeError("timeout while connecting"))
    assert not is_retryable_error(ValueError("invalid input"))


def test_reconcile_interrupted_tasks(tmp_path: Path):
    db = tmp_path / "test.db"
    repo = Repository(str(db))
    settings = repo.get_settings()
    task_id = repo.create_task("Research OCR tools", settings, None)
    repo.update_task_status(task_id, "running")

    n = repo.reconcile_interrupted_tasks()
    assert n >= 1
    task = repo.get_task(task_id)
    assert task["status"] == "failed"
    assert task["terminal_reason"] == "application_stopped_during_execution"
