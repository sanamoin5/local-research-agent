# Local Research Agent (Phase 5)

Phase 5 focuses on ship-readiness: diagnostics, repair flows, bounded retries, lifecycle guards, restart reconciliation, and polished degraded-state UX.

## 🚀 Easiest start (non-developers)

### Option A — One command script (recommended)

```bash
./scripts/easy_start.sh
```

What this does:
- Uses Docker automatically if available (`docker compose up --build -d`)
- Otherwise falls back to local Python setup (creates `.venv`, installs deps, installs Playwright Chromium, starts app)

Then open: `http://localhost:8000`

---

### Option B — Docker explicitly

1. Copy env file:
```bash
cp .env.example .env
```
2. Add your `TAVILY_API_KEY` in `.env` (optional but recommended).
3. Start:
```bash
docker compose up --build -d
```
4. Open `http://localhost:8000`.

Stop:
```bash
docker compose down
```

---

### Option C — Local Python explicitly

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium
uvicorn app.main:app --reload
```

## API surface

- `POST /api/tasks`
- `POST /api/tasks/{id}/confirm`
- `POST /api/tasks/{id}/reject`
- `POST /api/tasks/{id}/cancel`
- `GET /api/tasks`
- `GET /api/tasks/{id}`
- `GET /api/tasks/{id}/events`
- `GET /api/tasks/{id}/export?format=md`
- `GET /api/settings`
- `PUT /api/settings`
- `GET /api/health`
- `GET /api/logs/recent`
- `POST /api/repair/playwright`
- `POST /api/repair/model`
- `POST /api/repair/db`
- `POST /api/repair/settings/reset`

## Test

```bash
PYTHONPATH=. pytest -q
```

## Cross-platform notes

- macOS/Linux: ensure `python3`, `pip`, and `ollama` are on PATH.
- Windows: use PowerShell and ensure `ollama.exe` is reachable.
- Docker mode expects Ollama on the host at `http://host.docker.internal:11434`.

## Known limitations

- Small local models may return malformed structured tool calls; fallback logic mitigates this.
- Search provider quotas or missing keys can degrade source coverage.
- Browser fallback requires Playwright Chromium.
- Some websites remain non-extractable due to anti-bot or dynamic rendering behavior.
