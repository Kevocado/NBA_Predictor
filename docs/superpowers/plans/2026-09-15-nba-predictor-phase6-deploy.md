# NBA Predictor — Phase 6: Deploy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the app for production (Dockerfile serving both the built
frontend and the API from one container), add the public-snapshot generator,
and wire up the same GitHub Actions → GHCR → Azure Container App deploy
pattern the three sibling projects already use in the shared
`predictor-hub-rg` resource group.

**Architecture:** A multi-stage Dockerfile builds the Phase 5 frontend with
Node, then copies the static build into a Python 3.13 runtime image
alongside the Phase 1-4 backend; `create_app()` mounts the built frontend as
static files behind the API routes (explicit routes always take precedence
over the catch-all static mount, by FastAPI/Starlette registration order).
`nba_predictor.public_snapshot` is a small module that bundles the current
schedule/hub/manifest/track-record data into one `data/public_snapshot.json`
file, committed to git so the Docker image always ships with *something* to
serve even before the first live retrain. Deploy is GitHub Actions building
and pushing to GHCR, then `az containerapp update` (matching
`PL_Predictor/.github/workflows/deploy-azure.yml` exactly, renamed).

**Tech Stack:** Same as Phases 1-5, plus Docker, GitHub Actions, Azure CLI.

**Spec:** [docs/superpowers/specs/2026-09-15-nba-predictor-design.md](../specs/2026-09-15-nba-predictor-design.md)

## Global Constraints

- Python version floor: **3.13**, Node **20** for the frontend build stage —
  matches every sibling Dockerfile.
- Container App name: `nba-predictor`. Resource group: `predictor-hub-rg`
  (existing, shared with PL/F1/NFL-CFB). GHCR image:
  `ghcr.io/kevocado/nba-predictor`. These exact names come from Phase 0's
  confirmed sibling pattern (`PL_Predictor/.github/workflows/deploy-azure.yml`).
- No deploy or Azure-provisioning command in this plan is executed
  automatically — Task 6's `az containerapp create` is a one-time,
  human-run command documented in the README because it changes shared
  Azure infrastructure outside this repo and needs the user's own Azure
  credentials and explicit go-ahead.
- Any step that would push to a remote GitHub repository or trigger a real
  Azure deploy requires the user's explicit confirmation before running —
  flagged inline at that step.

---

## File Structure

```
NBA_Predictor/
  Dockerfile
  .dockerignore
  src/nba_predictor/
    api/app.py              # MODIFY — mount built frontend as static files
    public_snapshot.py        # NEW — snapshot generator + CLI entrypoint
  data/
    public_snapshot.json        # NEW — committed initial snapshot
  .github/workflows/
    deploy-azure.yml              # NEW
    refresh-public-snapshot.yml     # NEW
  README.md                         # MODIFY — add deploy runbook
```

---

### Task 1: Serve the built frontend from FastAPI

**Files:**
- Modify: `src/nba_predictor/api/app.py`
- Test: `tests/test_api_static_frontend.py`

**Interfaces:**
- Consumes: `config.PROJECT_ROOT` (Phase 1).
- Produces: `create_app()` (Phase 1) now also mounts
  `PROJECT_ROOT / "frontend" / "dist"` at `/` via Starlette's `StaticFiles`
  (`html=True`, so `/` serves `index.html` and unknown paths fall back to
  it for client-side routing) — **only if that directory exists** (local
  dev without a frontend build must not crash the API). API routes
  registered via `include_router` are matched before the mount because
  Starlette checks routes in registration order.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api_static_frontend.py
from fastapi.testclient import TestClient


def test_create_app_serves_frontend_index_when_dist_exists(tmp_path, monkeypatch):
    from nba_predictor import config
    from nba_predictor.api.app import create_app

    dist_dir = tmp_path / "frontend" / "dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<html><body>NBA Predictor App</body></html>")

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    app = create_app()
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "NBA Predictor App" in response.text


def test_create_app_health_route_takes_precedence_over_static_mount(tmp_path, monkeypatch):
    from nba_predictor import config
    from nba_predictor.api.app import create_app

    dist_dir = tmp_path / "frontend" / "dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<html></html>")

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_app_skips_static_mount_when_dist_missing(tmp_path, monkeypatch):
    from nba_predictor import config
    from nba_predictor.api.app import create_app

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)  # no frontend/dist under tmp_path
    app = create_app()
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_static_frontend.py -v`
Expected: FAIL — `/` currently 404s even when a `frontend/dist` directory
exists, since nothing mounts it yet.

- [ ] **Step 3: Modify `src/nba_predictor/api/app.py`**

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from nba_predictor.api.routes import router
from nba_predictor.config import PROJECT_ROOT


def create_app() -> FastAPI:
    app = FastAPI(title="NBA Predictor API")
    app.include_router(router)

    frontend_dist = PROJECT_ROOT / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app


app = create_app()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api_static_frontend.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/api/app.py tests/test_api_static_frontend.py
git commit -m "feat: serve built frontend as static files behind API routes"
```

---

### Task 2: Public snapshot generator

**Files:**
- Create: `src/nba_predictor/public_snapshot.py`
- Test: `tests/test_public_snapshot.py`

**Interfaces:**
- Consumes: `services.schedule_repository.load_schedule` (Phase 4),
  `services.hub_service.load_hub_cache`, `compute_track_record` (Phase 4),
  `config.DATA_DIR`, `config.TRACKING_DB_PATH` (Phase 1).
- Produces:
  - `public_snapshot.generate_snapshot(schedule_path: Path, hub_dir: Path, manifest_path: Path, db_path: Path) -> dict` — returns `{"generated_at": <ISO8601 str>, "schedule": [...], "hub": {"teams": [...], "players": [...], "rankings": [...], "standings": [...]}, "track_record": [...], "manifest": dict | None}`; `manifest` is `None` if `manifest_path` doesn't exist.
  - `public_snapshot.write_snapshot(snapshot: dict, output_path: Path) -> None` — pretty-printed JSON.
  - `public_snapshot.main() -> None` — calls `generate_snapshot` with the real `config`-derived paths (`config.DATA_DIR / "cache" / "schedule" / "games.json"`, `config.DATA_DIR / "cache" / "hub"`, `config.PROJECT_ROOT / "models" / "manifest.json"`, `config.TRACKING_DB_PATH`) and writes to `config.DATA_DIR / "public_snapshot.json"`. Invoked via `python -m nba_predictor.public_snapshot`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_public_snapshot.py
import json


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_generate_snapshot_bundles_all_sources(tmp_path):
    from nba_predictor.public_snapshot import generate_snapshot
    from nba_predictor.tracking import store

    schedule_path = tmp_path / "schedule" / "games.json"
    _write_json(schedule_path, [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}])

    hub_dir = tmp_path / "hub"
    _write_json(hub_dir / "teams.json", [{"abbreviation": "BOS"}])
    _write_json(hub_dir / "players.json", [])
    _write_json(hub_dir / "rankings.json", [])
    _write_json(hub_dir / "standings.json", [])

    manifest_path = tmp_path / "models" / "manifest.json"
    _write_json(manifest_path, {"model_version": "v1", "trained_at": "t", "models": [], "metrics": {}})

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="home", model_probability=0.6,
        market_probability=0.5, edge=0.1, bookmaker="DK", american_odds=-130, created_at="2026-11-01T00:00:00",
    )

    snapshot = generate_snapshot(schedule_path, hub_dir, manifest_path, db_path)

    assert snapshot["schedule"][0]["game_id"] == "g1"
    assert snapshot["hub"]["teams"][0]["abbreviation"] == "BOS"
    assert snapshot["hub"]["players"] == []
    assert snapshot["manifest"]["model_version"] == "v1"
    assert snapshot["track_record"][0]["market"] == "h2h"
    assert "generated_at" in snapshot


def test_generate_snapshot_manifest_none_when_missing(tmp_path):
    from nba_predictor.public_snapshot import generate_snapshot
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    snapshot = generate_snapshot(
        tmp_path / "schedule" / "games.json", tmp_path / "hub", tmp_path / "models" / "manifest.json", db_path
    )
    assert snapshot["manifest"] is None


def test_write_snapshot_creates_valid_json(tmp_path):
    from nba_predictor.public_snapshot import write_snapshot

    output_path = tmp_path / "public_snapshot.json"
    write_snapshot({"generated_at": "2026-11-01T00:00:00", "schedule": []}, output_path)

    with open(output_path) as f:
        loaded = json.load(f)
    assert loaded["schedule"] == []


def test_main_writes_to_configured_output_path(tmp_path, monkeypatch):
    from nba_predictor import config
    from nba_predictor.public_snapshot import main
    from nba_predictor.tracking import store

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    store.init_db(tmp_path / "tracking.db")

    main()

    assert (tmp_path / "public_snapshot.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_public_snapshot.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.public_snapshot'`

- [ ] **Step 3: Write `src/nba_predictor/public_snapshot.py`**

```python
import json
from datetime import datetime, timezone
from pathlib import Path

from nba_predictor import config
from nba_predictor.services.hub_service import compute_track_record, load_hub_cache
from nba_predictor.services.schedule_repository import load_schedule


def generate_snapshot(schedule_path: Path, hub_dir: Path, manifest_path: Path, db_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schedule": load_schedule(schedule_path),
        "hub": {
            "teams": load_hub_cache(hub_dir / "teams.json"),
            "players": load_hub_cache(hub_dir / "players.json"),
            "rankings": load_hub_cache(hub_dir / "rankings.json"),
            "standings": load_hub_cache(hub_dir / "standings.json"),
        },
        "track_record": [record.model_dump() for record in compute_track_record(db_path)],
        "manifest": manifest,
    }


def write_snapshot(snapshot: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2))


def main() -> None:
    snapshot = generate_snapshot(
        schedule_path=config.DATA_DIR / "cache" / "schedule" / "games.json",
        hub_dir=config.DATA_DIR / "cache" / "hub",
        manifest_path=config.PROJECT_ROOT / "models" / "manifest.json",
        db_path=config.TRACKING_DB_PATH,
    )
    write_snapshot(snapshot, config.DATA_DIR / "public_snapshot.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_public_snapshot.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Generate and commit an initial snapshot so the Docker image always has one to copy**

```bash
python -m nba_predictor.public_snapshot
cat data/public_snapshot.json
```

Expected: a JSON file with `schedule: []`, `hub: {...all empty...}`,
`track_record: []`, `manifest: null` — correct for a freshly-scaffolded
project with no data collected yet.

- [ ] **Step 6: Commit**

```bash
git add src/nba_predictor/public_snapshot.py tests/test_public_snapshot.py data/public_snapshot.json
git commit -m "feat: add public snapshot generator"
```

---

### Task 3: Dockerfile

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

**Interfaces:**
- Consumes: `pyproject.toml`, `src/`, `frontend/`, `models/`,
  `data/public_snapshot.json` (Task 2).
- Produces: a runnable container image exposing port 8000, serving the API
  and the built frontend from Task 1's static mount.

- [ ] **Step 1: Write `.dockerignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
node_modules/
frontend/dist/
data/cache/
data/tracking.db
.git/
.github/
docs/
tests/
.superpowers/
AI_Continuity.md
```

- [ ] **Step 2: Write `Dockerfile`**

```dockerfile
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim AS runtime
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

COPY models/ ./models/
COPY data/public_snapshot.json ./data/public_snapshot.json
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "nba_predictor.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Build the image locally, if Docker is available**

```bash
docker --version
```

If this fails (no Docker daemon in this environment), skip to Step 5 and
note in the commit message that the build was not locally verified — the
GitHub Actions workflow in Task 5 will build it in CI on push, which is the
authoritative check.

If Docker is available:

```bash
docker build -t nba-predictor:local .
```

Expected: exits 0.

- [ ] **Step 4: Run the container and hit the health check, if Docker is available**

```bash
docker run -d --name nba-predictor-smoke -p 8000:8000 nba-predictor:local
sleep 2
curl -s http://127.0.0.1:8000/health
docker stop nba-predictor-smoke
docker rm nba-predictor-smoke
```

Expected: `{"status":"ok"}`

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat: add multi-stage Dockerfile for frontend + API"
```

---

### Task 4: GitHub Actions — deploy and public-snapshot refresh

**Files:**
- Create: `.github/workflows/deploy-azure.yml`
- Create: `.github/workflows/refresh-public-snapshot.yml`

**Interfaces:**
- Consumes: repository secrets `GHCR_PAT`, `AZURE_CLIENT_ID`,
  `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` — the same secrets already
  configured on the sibling repos' GitHub organization/account (reused, not
  recreated, per the earlier "Existing keys already cover NBA" decision —
  these are Azure/GHCR credentials, not sport-data API keys, but the same
  reuse principle applies: they're already set up at the account level).
- Produces: on push to `main` (excluding snapshot-only commits), builds and
  pushes `ghcr.io/kevocado/nba-predictor` and deploys it to the
  `nba-predictor` Container App. On a schedule, regenerates
  `data/public_snapshot.json` and commits it directly to `main`.

- [ ] **Step 1: Write `.github/workflows/deploy-azure.yml`**

```yaml
name: Deploy to Azure Container Apps

on:
  push:
    branches: [main]
    paths-ignore:
      - 'data/public_snapshot.json'

permissions:
  id-token: write
  contents: read
  packages: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GHCR_PAT }}

      - name: Build and push image
        run: |
          docker build -t ghcr.io/kevocado/nba-predictor:${{ github.sha }} -t ghcr.io/kevocado/nba-predictor:latest .
          docker push ghcr.io/kevocado/nba-predictor:${{ github.sha }}
          docker push ghcr.io/kevocado/nba-predictor:latest

      - name: Log in to Azure
        uses: azure/login@v2
        with:
          client-id: ${{ secrets.AZURE_CLIENT_ID }}
          tenant-id: ${{ secrets.AZURE_TENANT_ID }}
          subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}

      - name: Deploy new image to Container App
        run: |
          az containerapp update \
            --name nba-predictor \
            --resource-group predictor-hub-rg \
            --image ghcr.io/kevocado/nba-predictor:${{ github.sha }}
```

- [ ] **Step 2: Write `.github/workflows/refresh-public-snapshot.yml`**

```yaml
name: Refresh Public Snapshot

on:
  schedule:
    - cron: '0 */6 * * *'
  workflow_dispatch: {}

permissions:
  contents: write

jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install package
        run: pip install -e .

      - name: Regenerate public snapshot
        run: python -m nba_predictor.public_snapshot

      - name: Commit if changed
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/public_snapshot.json
          git diff --staged --quiet || git commit -m "chore: refresh public snapshot"
          git push
```

- [ ] **Step 3: Validate both workflow files parse as YAML**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/deploy-azure.yml'))" && echo "deploy-azure.yml OK"
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/refresh-public-snapshot.yml'))" && echo "refresh-public-snapshot.yml OK"
```

Expected: both print `OK`. (If `yaml` isn't installed in the active venv,
`pip install pyyaml` first — it's a one-off validation tool, not a project
dependency, so don't add it to `pyproject.toml`.)

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/
git commit -m "ci: add Azure deploy and public snapshot refresh workflows"
```

---

### Task 5: README deploy runbook and one-time Azure provisioning

**Files:**
- Modify: `README.md`

**Interfaces:**
- Produces: documentation only — no code interfaces. This task documents
  the **one-time, human-run** commands to create the `nba-predictor`
  Container App in the existing `predictor-hub-rg`/environment before the
  Task 4 workflow's `az containerapp update` has anything to update. These
  commands are **not** run by this plan — they require the user's own Azure
  login and explicit confirmation, since creating cloud infrastructure is
  exactly the kind of hard-to-reverse, shared-system action this project's
  operating rules require a human to explicitly approve.

- [ ] **Step 1: Add a "Deployment" section to `README.md`**

```markdown
## Deployment

NBA_Predictor deploys the same way as PL_Predictor, F1_Predictor, and the
NFL/CFB predictor: GitHub Actions builds the Docker image, pushes it to
GHCR, and updates an Azure Container App in the shared `predictor-hub-rg`
resource group. See `.github/workflows/deploy-azure.yml`.

### One-time setup (run once, by a human, before the first automated deploy)

1. Find the existing Container Apps environment the other predictors share:
   \`\`\`bash
   az containerapp env list --resource-group predictor-hub-rg --query "[].name" -o tsv
   \`\`\`
2. Create the `nba-predictor` Container App in that environment, pointing at
   the image the first CI run will have pushed to GHCR:
   \`\`\`bash
   az containerapp create \\
     --name nba-predictor \\
     --resource-group predictor-hub-rg \\
     --environment <environment-name-from-step-1> \\
     --image ghcr.io/kevocado/nba-predictor:latest \\
     --target-port 8000 \\
     --ingress external \\
     --registry-server ghcr.io \\
     --registry-username <github-username> \\
     --registry-password <GHCR_PAT>
   \`\`\`
3. Once live, update the "NBA Predictor" card in
   `predictor-hub/index.html` from "In development" to a live link pointing
   at the new Container App's URL (printed by the `create` command above,
   or found via `az containerapp show --name nba-predictor --resource-group
   predictor-hub-rg --query properties.configuration.ingress.fqdn`).

After this one-time setup, every push to `main` redeploys automatically via
`.github/workflows/deploy-azure.yml`.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add deployment runbook"
```

---

### Task 6: Full-project smoke test

**Files:** No new files — verification only.

- [ ] **Step 1: Run the entire backend test suite**

Run: `pytest -v`
Expected: all Phase 1-6 backend tests pass, 0 failed, 0 errors.

- [ ] **Step 2: Run the entire frontend test suite and build**

```bash
cd frontend && npm test && npm run build && cd ..
```

Expected: all frontend tests pass; `frontend/dist/` is produced.

- [ ] **Step 3: Boot the full app (API + built frontend) and verify both are served**

```bash
uvicorn nba_predictor.api.app:app --port 8000 &
sleep 1
curl -s http://127.0.0.1:8000/health
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/
kill %1
```

Expected: `/health` returns `{"status":"ok"}`; `/` returns `200` (serving
`frontend/dist/index.html`).

- [ ] **Step 4: Confirm no secrets are staged for commit**

```bash
git status --short
git diff --staged -- '*.env' 2>/dev/null
```

Expected: `.env` never appears in `git status --short` (it's gitignored
from Phase 1); no output from the `diff` command.

---

## Self-Review Notes

- **Spec coverage:** Docker multi-stage build, GHCR push, Azure Container
  App deploy in `predictor-hub-rg`, `public_snapshot.json` generation and
  scheduled refresh — every bullet in spec §2's "Deploy" and "Public
  snapshot" paragraphs is covered. The predictor-hub card update (spec §9,
  explicitly out of scope for the build) is referenced only as a documented
  step in Task 5's runbook, not performed automatically — consistent with
  the spec's explicit scope boundary.
- **Explicitly deferred / human-gated**: the one-time `az containerapp
  create` command (Task 5) and the first push to a GitHub remote that
  triggers Task 4's workflow are both human-run per this plan's Global
  Constraints — creating cloud infrastructure and pushing to a shared
  remote are exactly the "hard-to-reverse" / "affects shared systems"
  actions that require explicit human confirmation, not something an
  implementing agent should do unattended.
- **Placeholder scan:** no TBD/TODO; every step has runnable code or an
  explicitly-labeled human-run command.
- **Type consistency:** `generate_snapshot`'s return shape matches what
  `write_snapshot` and `main` pass through unchanged; `TrackRecordOut.
  model_dump()` (pydantic v2 method) matches the `TrackRecordOut` schema
  from Phase 4 Task 2.
