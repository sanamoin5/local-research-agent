# Local Research Agent

Local Research Agent is a **local-first AI research worker** that turns a user task into a structured, source-backed deliverable.

It runs locally, stores history in SQLite, supports a web UI, and can execute in either:
- **Agent mode** (plan + confirmation + tool-driven execution)
- **Direct fallback mode** (deterministic resilient pipeline)

---

## Table of Contents

- [What this project is](#what-this-project-is)
- [Core features](#core-features)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
  - [Option A: One command (recommended)](#option-a-one-command-recommended)
  - [Option B: Docker](#option-b-docker)
  - [Option C: Local Python](#option-c-local-python)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Task lifecycle](#task-lifecycle)
- [Health checks, diagnostics, and repair](#health-checks-diagnostics-and-repair)
- [Reporting output](#reporting-output)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)

---

## What this project is

This is **not** a chatbot product. Users submit a research job; the app executes a research workflow and returns a deliverable.

### Product principles

1. **Local-first**: local runtime, local SQLite state, local model integration via Ollama.
2. **Transparent**: live status/events and persisted step history.
3. **Resilient**: skips bad sources, retries bounded operations, and falls back safely.
4. **Task-oriented**: submit task → execute research → receive structured output.

---

## Core features

- Structured plan generation with user confirmation/rejection.
- Agent executor loop with typed tool calls.
- Direct resilient fallback pipeline when structured tool execution is unreliable.
- URL fetch + extraction with guardrails and optional browser fallback (Playwright Chromium).
- Source quality filtering and skip reasoning.
- Dedicated final reporting pipeline:
  - conflict analysis,
  - limitations seed builder,
  - structured JSON report,
  - deterministic markdown rendering,
  - trust metadata.
- SQLite persistence for tasks, steps, sources, cache, settings.
- Diagnostics and repair APIs for ship-readiness.

---

## Architecture

### Backend
- **FastAPI** app with REST + SSE.
- **Pydantic** schemas for typed contracts.
- **SQLite** repository layer for all persistent state.

### Execution modes
1. **Agent mode**:
   - `create task` → `generate plan` → `confirm plan` → `executor loop`
2. **Direct fallback mode**:
   - deterministic pipeline with robust fetch/extract/synthesis

### Reporting subsystem
Final output is generated through a dedicated report stage:
1. conflict detection
2. limitations seed generation
3. structured report synthesis (with retry)
4. markdown rendering from structured data
5. preview + report metadata generation

---

## Project structure

```text
app/
  main.py              # FastAPI app, endpoints, startup hooks
  repository.py        # SQLite schema + persistence operations
  planner.py           # Plan generation + default fallback plan
  executor.py          # Agent execution loop
  tool_router.py       # Typed tool handlers
  pipeline.py          # Direct resilient fallback pipeline
  services.py          # Search/fetch/extract/model utilities
  reporting.py         # Structured report + markdown rendering
  diagnostics.py       # Health checks and readiness diagnostics
  retry_utils.py       # Bounded retry/backoff utility
  logging_utils.py     # Lightweight structured app logging
  schemas.py           # Pydantic request/response models

static/
  index.html           # Single-page local UI

scripts/
  easy_start.sh        # One-command bootstrap/start script

tests/
  test_diagnostics_and_lifecycle.py
  test_reporting.py
  test_schemas_phase4.py
```

---

## Quick start

## Option A: One command (recommended)

```bash
./scripts/easy_start.sh
```

Behavior:
- If Docker exists: runs containerized stack.
- If Docker not found: bootstraps local Python `.venv`, installs deps + Playwright, starts server.

Open: `http://localhost:8000`

---

## Option B: Docker

1. Create env file:
```bash
cp .env.example .env
```
2. Set `TAVILY_API_KEY` (recommended).
3. Start:
```bash
docker compose up --build -d
```
4. Open: `http://localhost:8000`

Stop:
```bash
docker compose down
```

---

## Option C: Local Python

```bash
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium
uvicorn app.main:app --reload
```

---

## Configuration

Use environment variables:

- `LRA_DB_PATH` (default: `local_research_agent.db`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `TAVILY_API_KEY` (optional but recommended for search)
- `RECOMMENDED_MODEL` (default: `llama3.1:8b`)

`.env.example` provides a template for Docker usage.

---

## API reference

### Tasks
- `POST /api/tasks`
- `POST /api/tasks/{id}/confirm`
- `POST /api/tasks/{id}/reject`
- `POST /api/tasks/{id}/cancel`
- `GET /api/tasks`
- `GET /api/tasks/{id}`
- `GET /api/tasks/{id}/events` (SSE)
- `GET /api/tasks/{id}/export?format=md`

### Settings
- `GET /api/settings`
- `PUT /api/settings`

### Diagnostics / Operations
- `GET /api/health` (full diagnostics)
- `GET /api/health/basic` (legacy basic checks)
- `GET /api/logs/recent`

### Repair actions
- `POST /api/repair/playwright`
- `POST /api/repair/model`
- `POST /api/repair/db`
- `POST /api/repair/settings/reset`

---

## Task lifecycle

Typical path:
1. Submit task (`planning`)
2. Plan generated and persisted
3. User confirms plan
4. Task enters `running`
5. Executes via agent loop
6. If needed, falls back to direct pipeline
7. Dedicated final report stage runs
8. Terminal state set (`completed`, `failed`, or `cancelled`)

On restart, stale `planning`/`running` tasks are reconciled and marked interrupted.

---

## Health checks, diagnostics, and repair

`GET /api/health` returns structured status with:
- per-check status (`pass`, `warn`, `fail`)
- overall status (`healthy`, `degraded`, `setup_required`)
- suggested repair actions

UI includes a setup/diagnostics panel with repair buttons.

---

## Reporting output

Each completed task stores:
- markdown output (`output_markdown`)
- structured report JSON (`structured_output_json`)
- output preview (`output_preview`)
- report metadata (`report_metadata_json`)
- conflict count and trust signals

Markdown sections:
- Summary
- Key Findings
- Source Conflicts (when relevant)
- Sources
- Limitations

---

## Testing

Run tests:

```bash
PYTHONPATH=. pytest -q
```

Current tests cover:
- diagnostics/lifecycle behavior
- reporting utility logic
- schema validation constraints

---

## Troubleshooting

### App shows setup required
- Open diagnostics panel in UI.
- Run relevant repair actions.
- Re-check health.

### Ollama unreachable
- Start Ollama locally.
- Verify `OLLAMA_BASE_URL`.

### Browser fallback unavailable
- Install Chromium for Playwright:
```bash
python -m playwright install chromium
```

### No search results
- Set valid `TAVILY_API_KEY`.
- Check quota/limits.

### Task stuck previously in running
- Phase 5 startup reconciliation should mark it interrupted automatically.

---

## Known limitations

- Small models can produce unreliable structured tool calls; fallback mitigates this.
- Search API quotas can reduce coverage.
- Some pages remain non-extractable due to anti-bot / rendering complexity.
- This is local-first; hosted multi-user features are out of scope.