import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Any

from .schemas import Settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:
    def __init__(self, path: str) -> None:
        self.path = path
        self.init_db()

    def conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, coldef: str) -> None:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coldef}")

    def init_db(self) -> None:
        with closing(self.conn()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    input_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    output_markdown TEXT,
                    output_preview TEXT,
                    structured_output_json TEXT,
                    conflict_count INTEGER DEFAULT 0,
                    report_metadata_json TEXT,
                    settings_snapshot_json TEXT,
                    usable_source_count INTEGER DEFAULT 0,
                    skipped_source_count INTEGER DEFAULT 0,
                    plan_json TEXT,
                    plan_status TEXT DEFAULT 'pending',
                    execution_mode TEXT DEFAULT 'agent',
                    fallback_reason TEXT,
                    confirmed_at TEXT,
                    cancelled_at TEXT,
                    terminal_reason TEXT,
                    interrupted_by_restart INTEGER DEFAULT 0,
                    cancel_requested_at TEXT
                );

                CREATE TABLE IF NOT EXISTS task_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    step_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    message TEXT,
                    metadata_json TEXT,
                    reasoning_text TEXT,
                    tool_name TEXT,
                    tool_input_json TEXT,
                    tool_output_json TEXT,
                    FOREIGN KEY(task_id) REFERENCES tasks(id)
                );

                CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    normalized_url TEXT NOT NULL,
                    title TEXT,
                    snippet TEXT,
                    provider TEXT,
                    fetch_status TEXT,
                    http_status INTEGER,
                    content_type TEXT,
                    content_length_bytes INTEGER,
                    extraction_status TEXT,
                    quality TEXT,
                    extraction_method TEXT,
                    content_text TEXT,
                    error_reason TEXT,
                    cache_hit INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    UNIQUE(task_id, normalized_url),
                    FOREIGN KEY(task_id) REFERENCES tasks(id)
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS task_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    trace_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    detail TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(id)
                );

                CREATE TABLE IF NOT EXISTS url_cache (
                    normalized_url TEXT PRIMARY KEY,
                    original_url TEXT,
                    fetched_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    fetch_method TEXT,
                    content_type TEXT,
                    http_status INTEGER,
                    html_text TEXT,
                    extracted_text TEXT,
                    title TEXT,
                    quality TEXT,
                    extraction_method TEXT,
                    fetch_status TEXT,
                    extraction_status TEXT,
                    error_reason TEXT,
                    content_hash TEXT
                );
                """
            )
            for col, cdef in [
                ("structured_output_json", "TEXT"),
                ("conflict_count", "INTEGER DEFAULT 0"),
                ("report_metadata_json", "TEXT"),
                ("plan_json", "TEXT"),
                ("plan_status", "TEXT DEFAULT 'pending'"),
                ("execution_mode", "TEXT DEFAULT 'agent'"),
                ("fallback_reason", "TEXT"),
                ("confirmed_at", "TEXT"),
                ("cancelled_at", "TEXT"),
                ("terminal_reason", "TEXT"),
                ("interrupted_by_restart", "INTEGER DEFAULT 0"),
                ("cancel_requested_at", "TEXT"),
            ]:
                self._ensure_column(conn, "tasks", col, cdef)
            for col, cdef in [
                ("reasoning_text", "TEXT"),
                ("tool_name", "TEXT"),
                ("tool_input_json", "TEXT"),
                ("tool_output_json", "TEXT"),
            ]:
                self._ensure_column(conn, "task_steps", col, cdef)
            conn.commit()
        self._init_default_settings()

    def _init_default_settings(self) -> None:
        with closing(self.conn()) as conn:
            if conn.execute("SELECT 1 FROM settings WHERE key='runtime'").fetchone():
                return
            conn.execute("INSERT INTO settings(key,value_json) VALUES('runtime',?)", (Settings().model_dump_json(),))
            conn.commit()

    def get_settings(self) -> Settings:
        with closing(self.conn()) as conn:
            row = conn.execute("SELECT value_json FROM settings WHERE key='runtime'").fetchone()
        if not row:
            return Settings()
        try:
            loaded = Settings(**json.loads(row["value_json"]))
        except Exception:
            return self.reset_settings()
        # Auto-migrate: old DB rows may have a very low runtime (e.g. legacy 600s default).
        # Silently bump to the current code default so users aren't surprised by quick timeouts.
        if loaded.max_total_runtime_sec < 3600:
            data = loaded.model_dump()
            data["max_total_runtime_sec"] = Settings.model_fields["max_total_runtime_sec"].default
            loaded = Settings(**data)
            with closing(self.conn()) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO settings(key,value_json) VALUES('runtime',?)",
                    (loaded.model_dump_json(),),
                )
                conn.commit()
        return loaded

    def update_settings(self, patch: dict[str, Any]) -> Settings:
        current = self.get_settings().model_dump()
        current.update({k: v for k, v in patch.items() if v is not None})
        updated = Settings(**current)
        with closing(self.conn()) as conn:
            conn.execute("INSERT OR REPLACE INTO settings(key,value_json) VALUES('runtime',?)", (updated.model_dump_json(),))
            conn.commit()
        return updated

    def create_task(self, task_text: str, settings_snapshot: Settings, plan_json: dict[str, Any] | None = None) -> str:
        task_id = str(uuid.uuid4())
        now = utc_now()
        with closing(self.conn()) as conn:
            conn.execute(
                """
                INSERT INTO tasks(id,input_text,status,created_at,updated_at,settings_snapshot_json,plan_json,plan_status,execution_mode)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (task_id, task_text, "planning", now, now, settings_snapshot.model_dump_json(), json.dumps(plan_json) if plan_json else None, "pending", "agent"),
            )
            conn.commit()
        return task_id

    def set_plan(self, task_id: str, plan_json: dict[str, Any], plan_status: str = "pending") -> None:
        with closing(self.conn()) as conn:
            conn.execute("UPDATE tasks SET plan_json=?,plan_status=?,updated_at=? WHERE id=?", (json.dumps(plan_json), plan_status, utc_now(), task_id))
            conn.commit()

    def approve_plan(self, task_id: str) -> None:
        with closing(self.conn()) as conn:
            conn.execute("UPDATE tasks SET plan_status='approved',confirmed_at=?,status='running',updated_at=? WHERE id=?", (utc_now(), utc_now(), task_id))
            conn.commit()

    def reject_plan(self, task_id: str) -> None:
        with closing(self.conn()) as conn:
            conn.execute("UPDATE tasks SET plan_status='rejected',cancelled_at=?,status='cancelled',updated_at=? WHERE id=?", (utc_now(), utc_now(), task_id))
            conn.commit()

    def update_execution_metadata(self, task_id: str, execution_mode: str | None = None, fallback_reason: str | None = None) -> None:
        with closing(self.conn()) as conn:
            if execution_mode is not None:
                conn.execute("UPDATE tasks SET execution_mode=?,updated_at=? WHERE id=?", (execution_mode, utc_now(), task_id))
            if fallback_reason is not None:
                conn.execute("UPDATE tasks SET fallback_reason=?,updated_at=? WHERE id=?", (fallback_reason, utc_now(), task_id))
            conn.commit()

    def update_task_status(self, task_id: str, status: str, error_message: str | None = None, terminal_reason: str | None = None) -> None:
        now = utc_now()
        completed_at = now if status in {"completed", "failed", "cancelled"} else None
        with closing(self.conn()) as conn:
            conn.execute(
                "UPDATE tasks SET status=?,updated_at=?,completed_at=COALESCE(?,completed_at),error_message=?,terminal_reason=COALESCE(?,terminal_reason) WHERE id=?",
                (status, now, completed_at, error_message, terminal_reason, task_id),
            )
            conn.commit()

    def mark_started(self, task_id: str) -> None:
        with closing(self.conn()) as conn:
            conn.execute("UPDATE tasks SET status='running',started_at=?,updated_at=? WHERE id=?", (utc_now(), utc_now(), task_id))
            conn.commit()

    def store_report(self, task_id: str, *, markdown: str, structured_output: dict[str, Any], output_preview: str, usable_count: int, skipped_count: int, conflict_count: int, report_metadata: dict[str, Any]) -> None:
        with closing(self.conn()) as conn:
            conn.execute(
                """
                UPDATE tasks SET output_markdown=?,structured_output_json=?,output_preview=?,usable_source_count=?,
                    skipped_source_count=?,conflict_count=?,report_metadata_json=?,updated_at=?
                WHERE id=?
                """,
                (
                    markdown,
                    json.dumps(structured_output),
                    output_preview,
                    usable_count,
                    skipped_count,
                    conflict_count,
                    json.dumps(report_metadata),
                    utc_now(),
                    task_id,
                ),
            )
            conn.commit()

    def add_step(self, task_id: str, step_index: int, step_type: str, status: str, started_at: datetime, message: str, metadata: dict[str, Any] | None = None, *, reasoning_text: str | None = None, tool_name: str | None = None, tool_input: dict[str, Any] | None = None, tool_output: dict[str, Any] | None = None) -> None:
        ended = datetime.now(timezone.utc)
        with closing(self.conn()) as conn:
            conn.execute(
                """
                INSERT INTO task_steps(task_id,step_index,step_type,status,started_at,completed_at,duration_ms,message,metadata_json,reasoning_text,tool_name,tool_input_json,tool_output_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    task_id,
                    step_index,
                    step_type,
                    status,
                    started_at.isoformat(),
                    ended.isoformat(),
                    int((ended - started_at).total_seconds() * 1000),
                    message,
                    json.dumps(metadata or {}),
                    reasoning_text,
                    tool_name,
                    json.dumps(tool_input or {}),
                    json.dumps(tool_output or {}),
                ),
            )
            conn.commit()

    def upsert_source(self, task_id: str, source: dict[str, Any]) -> None:
        payload = {
            "task_id": task_id,
            "url": source.get("url", ""),
            "normalized_url": source.get("normalized_url", source.get("url", "")),
            "title": source.get("title"),
            "snippet": source.get("snippet"),
            "provider": source.get("provider"),
            "fetch_status": source.get("fetch_status"),
            "http_status": source.get("http_status"),
            "content_type": source.get("content_type"),
            "content_length_bytes": source.get("content_length_bytes"),
            "extraction_status": source.get("extraction_status"),
            "quality": source.get("quality"),
            "extraction_method": source.get("extraction_method"),
            "content_text": source.get("content_text"),
            "error_reason": source.get("error_reason"),
            "cache_hit": 1 if source.get("cache_hit") else 0,
            "created_at": utc_now(),
        }
        with closing(self.conn()) as conn:
            conn.execute(
                """
                INSERT INTO sources(task_id,url,normalized_url,title,snippet,provider,fetch_status,http_status,content_type,content_length_bytes,extraction_status,quality,extraction_method,content_text,error_reason,cache_hit,created_at)
                VALUES(:task_id,:url,:normalized_url,:title,:snippet,:provider,:fetch_status,:http_status,:content_type,:content_length_bytes,:extraction_status,:quality,:extraction_method,:content_text,:error_reason,:cache_hit,:created_at)
                ON CONFLICT(task_id, normalized_url) DO UPDATE SET
                    title=excluded.title,snippet=excluded.snippet,provider=excluded.provider,fetch_status=excluded.fetch_status,http_status=excluded.http_status,
                    content_type=excluded.content_type,content_length_bytes=excluded.content_length_bytes,extraction_status=excluded.extraction_status,
                    quality=excluded.quality,extraction_method=excluded.extraction_method,content_text=excluded.content_text,error_reason=excluded.error_reason,
                    cache_hit=excluded.cache_hit,created_at=excluded.created_at
                """,
                payload,
            )
            conn.commit()

    def list_sources(self, task_id: str) -> list[dict[str, Any]]:
        with closing(self.conn()) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM sources WHERE task_id = ? ORDER BY created_at", (task_id,)).fetchall()
            return [dict(r) for r in rows]

    def list_tasks(self) -> list[dict[str, Any]]:
        with closing(self.conn()) as conn:
            rows = conn.execute(
                """
                SELECT id,input_text,status,created_at,updated_at,COALESCE(output_preview,'') as output_preview,
                       usable_source_count,skipped_source_count,COALESCE(plan_status,'pending') as plan_status,
                       COALESCE(execution_mode,'agent') as execution_mode,COALESCE(conflict_count,0) as conflict_count,fallback_reason
                FROM tasks ORDER BY created_at DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with closing(self.conn()) as conn:
            task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not task:
                return None
            steps = conn.execute("SELECT * FROM task_steps WHERE task_id=? ORDER BY step_index ASC, id ASC", (task_id,)).fetchall()
            sources = conn.execute("SELECT * FROM sources WHERE task_id=? ORDER BY id ASC", (task_id,)).fetchall()
        out = dict(task)
        out["steps"] = [dict(s) for s in steps]
        out["sources"] = [dict(s) for s in sources]
        return out


    def add_trace(self, task_id: str, trace_type: str, label: str, detail: str = "") -> None:
        with closing(self.conn()) as conn:
            conn.execute(
                "INSERT INTO task_traces(task_id, trace_type, label, detail, created_at) VALUES(?,?,?,?,?)",
                (task_id, trace_type, label, detail[:5000], utc_now()),
            )
            conn.commit()

    def list_traces(self, task_id: str) -> list[dict[str, Any]]:
        with closing(self.conn()) as conn:
            rows = conn.execute(
                "SELECT trace_type, label, detail, created_at FROM task_traces WHERE task_id=? ORDER BY id ASC",
                (task_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def request_cancel(self, task_id: str) -> None:
        with closing(self.conn()) as conn:
            conn.execute("UPDATE tasks SET cancel_requested_at=?, updated_at=? WHERE id=?", (utc_now(), utc_now(), task_id))
            conn.commit()

    def reset_settings(self) -> Settings:
        defaults = Settings()
        with closing(self.conn()) as conn:
            conn.execute("INSERT OR REPLACE INTO settings(key,value_json) VALUES('runtime',?)", (defaults.model_dump_json(),))
            conn.commit()
        return defaults

    def reconcile_interrupted_tasks(self) -> int:
        now = utc_now()
        with closing(self.conn()) as conn:
            rows = conn.execute("SELECT id FROM tasks WHERE status IN ('running','planning')").fetchall()
            for r in rows:
                conn.execute(
                    "UPDATE tasks SET status='failed', completed_at=?, updated_at=?, terminal_reason='application_stopped_during_execution', interrupted_by_restart=1 WHERE id=?",
                    (now, now, r['id']),
                )
            conn.commit()
        return len(rows)

    def recreate_schema(self) -> None:
        self.init_db()

    def get_cache(self, normalized_url: str) -> dict[str, Any] | None:
        with closing(self.conn()) as conn:
            row = conn.execute("SELECT * FROM url_cache WHERE normalized_url=?", (normalized_url,)).fetchone()
        return dict(row) if row else None

    def cache_reusable(self, row: dict[str, Any] | None) -> bool:
        if not row:
            return False
        if row.get("quality") == "poor" or row.get("fetch_status") != "ok" or row.get("extraction_status") != "ok":
            return False
        try:
            return datetime.fromisoformat(row["expires_at"]) > datetime.now(timezone.utc)
        except Exception:
            return False

    def upsert_cache(self, normalized_url: str, original_url: str, payload: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=7)
        with closing(self.conn()) as conn:
            conn.execute(
                """
                INSERT INTO url_cache(normalized_url,original_url,fetched_at,expires_at,fetch_method,content_type,http_status,html_text,extracted_text,title,quality,extraction_method,fetch_status,extraction_status,error_reason,content_hash)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(normalized_url) DO UPDATE SET
                    original_url=excluded.original_url,fetched_at=excluded.fetched_at,expires_at=excluded.expires_at,fetch_method=excluded.fetch_method,
                    content_type=excluded.content_type,http_status=excluded.http_status,html_text=excluded.html_text,extracted_text=excluded.extracted_text,
                    title=excluded.title,quality=excluded.quality,extraction_method=excluded.extraction_method,fetch_status=excluded.fetch_status,
                    extraction_status=excluded.extraction_status,error_reason=excluded.error_reason,content_hash=excluded.content_hash
                """,
                (
                    normalized_url,
                    original_url,
                    now.isoformat(),
                    expires.isoformat(),
                    payload.get("fetch_method"),
                    payload.get("content_type"),
                    payload.get("http_status"),
                    payload.get("html_text"),
                    payload.get("extracted_text"),
                    payload.get("title"),
                    payload.get("quality"),
                    payload.get("extraction_method"),
                    payload.get("fetch_status"),
                    payload.get("extraction_status"),
                    payload.get("error_reason"),
                    payload.get("content_hash"),
                ),
            )
            conn.commit()
