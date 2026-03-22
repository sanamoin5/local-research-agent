import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

_recent_logs: deque[dict[str, Any]] = deque(maxlen=300)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def log_event(level: str, category: str, message: str, *, task_id: str | None = None, metadata: dict[str, Any] | None = None) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "category": category,
        "message": message,
        "task_id": task_id,
        "metadata": metadata or {},
    }
    _recent_logs.append(payload)
    line = json.dumps(payload, default=str)
    if level == "error":
        logging.error(line)
    elif level == "warning":
        logging.warning(line)
    else:
        logging.info(line)


def recent_logs(limit: int = 100) -> list[dict[str, Any]]:
    return list(_recent_logs)[-limit:]
