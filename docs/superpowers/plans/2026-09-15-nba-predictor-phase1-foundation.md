# NBA Predictor — Phase 1: Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the NBA_Predictor project skeleton — packaging, config, the
static 30-team reference table, the SQLite tracking-database schema, and a
bootable FastAPI app with a health check — so every later phase (data
pipeline, models, API, frontend, deploy) has a working foundation to build on.

**Architecture:** Python 3.13 `src/`-layout package `nba_predictor`, packaged
with hatchling. Config is centralized in one module reading from `.env` via
`python-dotenv`. The team reference table is a plain Python data module (no
DB, no API) since it never changes at runtime. Tracking data lives in a
SQLite file accessed through a small connection-per-call wrapper (no ORM,
matching the sibling projects). FastAPI serves `/health` today and will grow
route-by-route in Phase 4.

**Tech Stack:** Python 3.13, FastAPI, Uvicorn, pytest, httpx (TestClient),
python-dotenv, hatchling (build backend). No frontend work in this phase.

**Spec:** [docs/superpowers/specs/2026-09-15-nba-predictor-design.md](../specs/2026-09-15-nba-predictor-design.md)

## Global Constraints

- Python version floor: **3.13** (matches PL_Predictor; project must not use syntax/stdlib features beyond 3.13).
- Package name: `nba_predictor`, importable from `src/nba_predictor/`.
- No ORM — raw `sqlite3`, matching all three sibling projects' tracking-store pattern.
- No component/DB frameworks beyond what's listed in Tech Stack for this phase.
- Every module that will later read secrets (API keys) must go through `config.py` — never read `os.environ` directly elsewhere.
- Conference values are exactly the strings `"East"` and `"West"` everywhere (API, DB, frontend) — do not introduce `"Eastern"`/`"Western"` variants.

---

## File Structure

```
NBA_Predictor/
  .gitignore
  pyproject.toml
  README.md
  src/
    nba_predictor/
      __init__.py
      config.py
      data/
        __init__.py
        team_reference.py
      tracking/
        __init__.py
        store.py
      api/
        __init__.py
        app.py
        routes.py
  tests/
    __init__.py
    test_config.py
    test_team_reference.py
    test_tracking_store.py
    test_api_health.py
```

- `config.py` — all paths and env-derived settings in one place.
- `data/team_reference.py` — static 30-team table (id, name, abbreviation, conference, division, arena lat/lon, timezone, altitude_ft). Pure data + one lookup helper, no I/O.
- `tracking/store.py` — SQLite schema (`init_db`) and the connection helper + CRUD functions later phases will extend.
- `api/app.py` — FastAPI app factory. `api/routes.py` — router(s), starting with `/health`.

---

### Task 1: Project packaging and scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/nba_predictor/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: an installable package `nba_predictor` (editable install), and a `tests/` package pytest can collect.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "nba-predictor"
version = "0.1.0"
description = "NBA game outcome and player prop prediction system"
requires-python = ">=3.13"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "python-dotenv>=1.0",
    "pandas>=2.2",
    "numpy>=2.0",
    "scipy>=1.14",
    "xgboost>=2.1",
    "scikit-learn>=1.5",
    "optuna>=4.0",
    "nba_api>=1.5",
    "requests>=2.32",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",
]

[tool.hatch.build.targets.wheel]
packages = ["src/nba_predictor"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.env
data/cache/
data/tracking.db
data/tracking.db.bak
node_modules/
frontend/dist/
.pytest_cache/
*.egg-info/
dist/
build/
```

- [ ] **Step 3: Create `README.md`**

```markdown
# NBA Predictor

NBA game outcome, spread/total, and player-prop predictions with a
conference-standings projection frontend. Sibling project to PL_Predictor,
NFL_Predictor, and CFB_Predictor — see
`docs/superpowers/specs/2026-09-15-nba-predictor-design.md` for the full
design.

## Setup

\`\`\`bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
\`\`\`

## Tests

\`\`\`bash
pytest
\`\`\`

## Run the API

\`\`\`bash
uvicorn nba_predictor.api.app:app --reload
\`\`\`
```

- [ ] **Step 4: Create empty package/test `__init__.py` files**

```bash
mkdir -p src/nba_predictor tests
touch src/nba_predictor/__init__.py tests/__init__.py
```

- [ ] **Step 5: Create venv and install editable with dev deps**

```bash
cd "/Users/sigey/Documents/Projects/NBA_Predictor"
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

- [ ] **Step 6: Verify the package imports and pytest collects (even with zero tests yet)**

Run: `pytest --collect-only`
Expected: exits 0, reports "no tests ran" (no errors) — confirms `tests/` and `src/nba_predictor` are both importable.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore README.md src/nba_predictor/__init__.py tests/__init__.py
git commit -m "chore: scaffold nba_predictor package"
```

---

### Task 2: Config module

**Files:**
- Create: `src/nba_predictor/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (first real module).
- Produces:
  - `config.PROJECT_ROOT: Path`
  - `config.DATA_DIR: Path`
  - `config.CACHE_DIR: Path`
  - `config.CACHE_SUBDIRS: list[str]` — `["nba_api", "balldontlie", "odds", "sportsbook", "injuries", "espn"]`
  - `config.TRACKING_DB_PATH: Path`
  - `config.TRACKING_DB_BACKUP_PATH: Path | None`
  - `config.PUBLIC_MODE: bool`
  - `config.PUBLIC_SNAPSHOT_POLL_SECONDS: int`
  - `config.BALLDONTLIE_API_KEY: str | None`
  - `config.SPORTSBOOK_API_KEY: str | None`
  - `config.ODDS_API_KEY: str | None`
  - `config.ensure_cache_dirs() -> None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
import os
from pathlib import Path

import pytest


def test_project_root_is_repo_root():
    from nba_predictor import config

    assert (config.PROJECT_ROOT / "pyproject.toml").exists()


def test_default_paths_are_under_data_dir():
    from nba_predictor import config

    assert config.DATA_DIR == config.PROJECT_ROOT / "data"
    assert config.CACHE_DIR == config.DATA_DIR / "cache"
    assert config.TRACKING_DB_PATH == config.DATA_DIR / "tracking.db"


def test_public_mode_defaults_false(monkeypatch):
    monkeypatch.delenv("PUBLIC_MODE", raising=False)
    import importlib

    from nba_predictor import config

    importlib.reload(config)
    assert config.PUBLIC_MODE is False


def test_public_mode_true_when_env_set(monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "true")
    import importlib

    from nba_predictor import config

    importlib.reload(config)
    assert config.PUBLIC_MODE is True
    monkeypatch.delenv("PUBLIC_MODE", raising=False)
    importlib.reload(config)


def test_ensure_cache_dirs_creates_all_subdirs(tmp_path, monkeypatch):
    import importlib

    from nba_predictor import config

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    config.ensure_cache_dirs()
    for name in config.CACHE_SUBDIRS:
        assert (tmp_path / "cache" / name).is_dir()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.config'`

- [ ] **Step 3: Write `src/nba_predictor/config.py`**

```python
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_SUBDIRS = ["nba_api", "balldontlie", "odds", "sportsbook", "injuries", "espn"]

TRACKING_DB_PATH = Path(os.environ.get("TRACKING_DB_PATH", str(DATA_DIR / "tracking.db")))
_backup = os.environ.get("TRACKING_DB_BACKUP_PATH")
TRACKING_DB_BACKUP_PATH = Path(_backup) if _backup else None

PUBLIC_MODE = os.environ.get("PUBLIC_MODE", "false").lower() == "true"
PUBLIC_SNAPSHOT_POLL_SECONDS = int(os.environ.get("PUBLIC_SNAPSHOT_POLL_SECONDS", "300"))

BALLDONTLIE_API_KEY = os.environ.get("BALLDONTLIE_API_KEY")
SPORTSBOOK_API_KEY = os.environ.get("SPORTSBOOK_API_KEY")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")


def ensure_cache_dirs() -> None:
    for name in CACHE_SUBDIRS:
        (CACHE_DIR / name).mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/config.py tests/test_config.py
git commit -m "feat: add centralized config module"
```

---

### Task 3: Static team reference table

**Files:**
- Create: `src/nba_predictor/data/__init__.py`
- Create: `src/nba_predictor/data/team_reference.py`
- Test: `tests/test_team_reference.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `team_reference.TeamInfo` — a `@dataclass(frozen=True)` with fields: `nba_api_id: int`, `abbreviation: str`, `name: str`, `conference: str` (`"East"` or `"West"`), `division: str`, `arena_lat: float`, `arena_lon: float`, `timezone: str`, `altitude_ft: int`.
  - `team_reference.TEAMS: tuple[TeamInfo, ...]` — all 30 teams.
  - `team_reference.get_team(abbreviation: str) -> TeamInfo` — raises `KeyError` if not found.

This is the exact data every later feature-engineering task (travel mileage,
timezone changes, altitude, conference standings) reads from — get it right
now so nothing downstream has to touch it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_team_reference.py
import pytest


def test_has_exactly_thirty_teams():
    from nba_predictor.data.team_reference import TEAMS

    assert len(TEAMS) == 30


def test_abbreviations_are_unique():
    from nba_predictor.data.team_reference import TEAMS

    abbrs = [t.abbreviation for t in TEAMS]
    assert len(abbrs) == len(set(abbrs))


def test_conferences_are_east_or_west_15_each():
    from nba_predictor.data.team_reference import TEAMS

    conferences = [t.conference for t in TEAMS]
    assert set(conferences) == {"East", "West"}
    assert conferences.count("East") == 15
    assert conferences.count("West") == 15


def test_divisions_are_six_of_five_teams_each():
    from nba_predictor.data.team_reference import TEAMS

    from collections import Counter

    counts = Counter(t.division for t in TEAMS)
    assert len(counts) == 6
    assert all(count == 5 for count in counts.values())


def test_only_denver_has_nonzero_altitude():
    from nba_predictor.data.team_reference import TEAMS

    high_altitude = [t for t in TEAMS if t.altitude_ft > 1000]
    assert len(high_altitude) == 1
    assert high_altitude[0].abbreviation == "DEN"


def test_get_team_returns_matching_team():
    from nba_predictor.data.team_reference import get_team

    team = get_team("BOS")
    assert team.name == "Boston Celtics"
    assert team.conference == "East"
    assert team.division == "Atlantic"


def test_get_team_raises_for_unknown_abbreviation():
    from nba_predictor.data.team_reference import get_team

    with pytest.raises(KeyError):
        get_team("ZZZ")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_team_reference.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.data'`

- [ ] **Step 3: Write `src/nba_predictor/data/__init__.py`** (empty file)

```bash
touch src/nba_predictor/data/__init__.py
```

- [ ] **Step 4: Write `src/nba_predictor/data/team_reference.py`**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TeamInfo:
    nba_api_id: int
    abbreviation: str
    name: str
    conference: str
    division: str
    arena_lat: float
    arena_lon: float
    timezone: str
    altitude_ft: int


TEAMS: tuple[TeamInfo, ...] = (
    # --- Eastern Conference ---
    # Atlantic
    TeamInfo(1610612738, "BOS", "Boston Celtics", "East", "Atlantic", 42.3662, -71.0621, "America/New_York", 0),
    TeamInfo(1610612751, "BKN", "Brooklyn Nets", "East", "Atlantic", 40.6826, -73.9754, "America/New_York", 0),
    TeamInfo(1610612752, "NYK", "New York Knicks", "East", "Atlantic", 40.7505, -73.9934, "America/New_York", 0),
    TeamInfo(1610612755, "PHI", "Philadelphia 76ers", "East", "Atlantic", 39.9012, -75.1720, "America/New_York", 0),
    TeamInfo(1610612761, "TOR", "Toronto Raptors", "East", "Atlantic", 43.6435, -79.3791, "America/Toronto", 0),
    # Central
    TeamInfo(1610612741, "CHI", "Chicago Bulls", "East", "Central", 41.8807, -87.6742, "America/Chicago", 0),
    TeamInfo(1610612739, "CLE", "Cleveland Cavaliers", "East", "Central", 41.4965, -81.6882, "America/New_York", 0),
    TeamInfo(1610612765, "DET", "Detroit Pistons", "East", "Central", 42.3410, -83.0550, "America/Detroit", 0),
    TeamInfo(1610612754, "IND", "Indiana Pacers", "East", "Central", 39.7640, -86.1555, "America/Indiana/Indianapolis", 0),
    TeamInfo(1610612749, "MIL", "Milwaukee Bucks", "East", "Central", 43.0451, -87.9172, "America/Chicago", 0),
    # Southeast
    TeamInfo(1610612737, "ATL", "Atlanta Hawks", "East", "Southeast", 33.7573, -84.3963, "America/New_York", 0),
    TeamInfo(1610612766, "CHA", "Charlotte Hornets", "East", "Southeast", 35.2251, -80.8392, "America/New_York", 0),
    TeamInfo(1610612748, "MIA", "Miami Heat", "East", "Southeast", 25.7814, -80.1870, "America/New_York", 0),
    TeamInfo(1610612753, "ORL", "Orlando Magic", "East", "Southeast", 28.5392, -81.3839, "America/New_York", 0),
    TeamInfo(1610612764, "WAS", "Washington Wizards", "East", "Southeast", 38.8981, -77.0209, "America/New_York", 0),
    # --- Western Conference ---
    # Northwest
    TeamInfo(1610612743, "DEN", "Denver Nuggets", "West", "Northwest", 39.7487, -105.0077, "America/Denver", 5280),
    TeamInfo(1610612750, "MIN", "Minnesota Timberwolves", "West", "Northwest", 44.9795, -93.2760, "America/Chicago", 0),
    TeamInfo(1610612760, "OKC", "Oklahoma City Thunder", "West", "Northwest", 35.4634, -97.5151, "America/Chicago", 0),
    TeamInfo(1610612757, "POR", "Portland Trail Blazers", "West", "Northwest", 45.5316, -122.6668, "America/Los_Angeles", 0),
    TeamInfo(1610612762, "UTA", "Utah Jazz", "West", "Northwest", 40.7683, -111.9011, "America/Denver", 0),
    # Pacific
    TeamInfo(1610612744, "GSW", "Golden State Warriors", "West", "Pacific", 37.7680, -122.3877, "America/Los_Angeles", 0),
    TeamInfo(1610612746, "LAC", "LA Clippers", "West", "Pacific", 33.9450, -118.3410, "America/Los_Angeles", 0),
    TeamInfo(1610612747, "LAL", "Los Angeles Lakers", "West", "Pacific", 34.0430, -118.2673, "America/Los_Angeles", 0),
    TeamInfo(1610612756, "PHX", "Phoenix Suns", "West", "Pacific", 33.4457, -112.0712, "America/Phoenix", 0),
    TeamInfo(1610612758, "SAC", "Sacramento Kings", "West", "Pacific", 38.5802, -121.4997, "America/Los_Angeles", 0),
    # Southwest
    TeamInfo(1610612742, "DAL", "Dallas Mavericks", "West", "Southwest", 32.7905, -96.8103, "America/Chicago", 0),
    TeamInfo(1610612745, "HOU", "Houston Rockets", "West", "Southwest", 29.7508, -95.3621, "America/Chicago", 0),
    TeamInfo(1610612763, "MEM", "Memphis Grizzlies", "West", "Southwest", 35.1382, -90.0505, "America/Chicago", 0),
    TeamInfo(1610612740, "NOP", "New Orleans Pelicans", "West", "Southwest", 29.9490, -90.0821, "America/Chicago", 0),
    TeamInfo(1610612759, "SAS", "San Antonio Spurs", "West", "Southwest", 29.4269, -98.4375, "America/Chicago", 0),
)

_BY_ABBREVIATION = {t.abbreviation: t for t in TEAMS}


def get_team(abbreviation: str) -> TeamInfo:
    return _BY_ABBREVIATION[abbreviation]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_team_reference.py -v`
Expected: PASS (7 passed)

- [ ] **Step 6: Commit**

```bash
git add src/nba_predictor/data/__init__.py src/nba_predictor/data/team_reference.py tests/test_team_reference.py
git commit -m "feat: add static 30-team reference table"
```

---

### Task 4: Tracking database schema and store

**Files:**
- Create: `src/nba_predictor/tracking/__init__.py`
- Create: `src/nba_predictor/tracking/store.py`
- Test: `tests/test_tracking_store.py`

**Interfaces:**
- Consumes: nothing (uses stdlib `sqlite3` directly, takes a `Path` argument rather than reading `config` — keeps it testable with `tmp_path`).
- Produces:
  - `store.init_db(db_path: Path) -> None` — creates all 6 tables if they don't exist (idempotent).
  - `store.get_connection(db_path: Path)` — context manager yielding a `sqlite3.Connection` with `row_factory = sqlite3.Row`.
  - `store.insert_prediction(db_path, *, game_id: str, created_at: str, model_version: str, home_win_prob: float, predicted_margin: float, predicted_total: float) -> int` — returns new row id.
  - `store.get_predictions_for_game(db_path: Path, game_id: str) -> list[sqlite3.Row]`.
  - Tables created: `predictions`, `game_market_predictions`, `game_forecast_snapshots`, `odds_timing_snapshots`, `player_prediction_snapshots`, `game_player_outcomes`.

Later phases (odds tracking, player-prop tracking, snapshotting) will add
their own insert/query functions to this same file — this task only needs
the schema plus the `predictions` table's CRUD to prove the pattern works.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tracking_store.py
import sqlite3

import pytest


def test_init_db_creates_all_six_tables(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    table_names = {row[0] for row in rows}

    expected = {
        "predictions",
        "game_market_predictions",
        "game_forecast_snapshots",
        "odds_timing_snapshots",
        "player_prediction_snapshots",
        "game_player_outcomes",
    }
    assert expected.issubset(table_names)


def test_init_db_is_idempotent(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.init_db(db_path)  # must not raise


def test_insert_and_get_predictions_roundtrip(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    row_id = store.insert_prediction(
        db_path,
        game_id="0022500123",
        created_at="2026-11-01T12:00:00",
        model_version="v1",
        home_win_prob=0.62,
        predicted_margin=3.5,
        predicted_total=224.5,
    )
    assert row_id == 1

    rows = store.get_predictions_for_game(db_path, "0022500123")
    assert len(rows) == 1
    assert rows[0]["game_id"] == "0022500123"
    assert rows[0]["home_win_prob"] == pytest.approx(0.62)
    assert rows[0]["model_version"] == "v1"


def test_get_predictions_for_game_returns_empty_for_unknown_game(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    rows = store.get_predictions_for_game(db_path, "does-not-exist")
    assert rows == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tracking_store.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.tracking'`

- [ ] **Step 3: Write `src/nba_predictor/tracking/__init__.py`** (empty file)

```bash
touch src/nba_predictor/tracking/__init__.py
```

- [ ] **Step 4: Write `src/nba_predictor/tracking/store.py`**

```python
import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model_version TEXT NOT NULL,
    home_win_prob REAL NOT NULL,
    predicted_margin REAL NOT NULL,
    predicted_total REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS game_market_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    model_probability REAL NOT NULL,
    market_probability REAL,
    edge REAL,
    bookmaker TEXT,
    american_odds INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS game_forecast_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    snapshot_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS odds_timing_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    market TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    american_odds INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS player_prediction_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    stat TEXT NOT NULL,
    predicted_value REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS game_player_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    stat TEXT NOT NULL,
    actual_value REAL NOT NULL,
    recorded_at TEXT NOT NULL
);
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_connection(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def insert_prediction(
    db_path: Path,
    *,
    game_id: str,
    created_at: str,
    model_version: str,
    home_win_prob: float,
    predicted_margin: float,
    predicted_total: float,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (game_id, created_at, model_version, home_win_prob, predicted_margin, predicted_total)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (game_id, created_at, model_version, home_win_prob, predicted_margin, predicted_total),
        )
        conn.commit()
        return cur.lastrowid


def get_predictions_for_game(db_path: Path, game_id: str) -> list[sqlite3.Row]:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM predictions WHERE game_id = ? ORDER BY created_at",
            (game_id,),
        )
        return cur.fetchall()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_tracking_store.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add src/nba_predictor/tracking/__init__.py src/nba_predictor/tracking/store.py tests/test_tracking_store.py
git commit -m "feat: add tracking database schema and store"
```

---

### Task 5: FastAPI app skeleton with health check

**Files:**
- Create: `src/nba_predictor/api/__init__.py`
- Create: `src/nba_predictor/api/routes.py`
- Create: `src/nba_predictor/api/app.py`
- Test: `tests/test_api_health.py`

**Interfaces:**
- Consumes: nothing yet (Phase 4 will wire in data/model modules).
- Produces:
  - `api.app.create_app() -> FastAPI`
  - `api.app.app` — module-level `FastAPI` instance (`create_app()` result), the entry point `uvicorn nba_predictor.api.app:app` uses.
  - `api.routes.router` — an `APIRouter` with `GET /health` returning `{"status": "ok"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_health.py
def test_health_endpoint_returns_ok():
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_health.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.api'`

- [ ] **Step 3: Write `src/nba_predictor/api/__init__.py`** (empty file)

```bash
touch src/nba_predictor/api/__init__.py
```

- [ ] **Step 4: Write `src/nba_predictor/api/routes.py`**

```python
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 5: Write `src/nba_predictor/api/app.py`**

```python
from fastapi import FastAPI

from nba_predictor.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="NBA Predictor API")
    app.include_router(router)
    return app


app = create_app()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_api_health.py -v`
Expected: PASS (1 passed)

- [ ] **Step 7: Commit**

```bash
git add src/nba_predictor/api/__init__.py src/nba_predictor/api/routes.py src/nba_predictor/api/app.py tests/test_api_health.py
git commit -m "feat: add FastAPI app skeleton with health check"
```

---

### Task 6: Full-suite smoke test and app boot verification

**Files:**
- No new files — this task verifies Tasks 1-5 integrate correctly.

**Interfaces:**
- Consumes: everything produced by Tasks 1-5.
- Produces: nothing new; this is a verification checkpoint before Phase 2 begins.

- [ ] **Step 1: Run the entire test suite**

Run: `pytest -v`
Expected: All tests from Tasks 2-5 pass (17 passed), 0 failed, 0 errors.

- [ ] **Step 2: Boot the app for real and hit it over HTTP**

```bash
uvicorn nba_predictor.api.app:app --port 8000 &
sleep 1
curl -s http://127.0.0.1:8000/health
kill %1
```

Expected output of the curl: `{"status":"ok"}`

- [ ] **Step 3: Verify `ensure_cache_dirs` produces the expected layout against the real project paths (not tmp_path)**

```bash
python -c "from nba_predictor import config; config.ensure_cache_dirs(); import os; print(sorted(os.listdir(config.CACHE_DIR)))"
```

Expected output: `['balldontlie', 'espn', 'injuries', 'nba_api', 'odds', 'sportsbook']`

- [ ] **Step 4: Commit the now-populated `data/cache/` gitignore confirmation (no tracked files should appear)**

```bash
git status --short
```

Expected: no untracked files under `data/cache/` show up (already covered by `.gitignore`'s `data/cache/` rule from Task 1). If any do appear, the `.gitignore` pattern needs fixing before proceeding — fix it, re-run `git status --short` to confirm clean, then commit the fix.

- [ ] **Step 5: Final commit marking Phase 1 complete**

```bash
git log --oneline
```

Expected: 5 commits from Tasks 1-5 are present (scaffold, config, team reference, tracking store, API health). No further commit needed for this task — it is verification-only.

---

## Self-Review Notes

- **Spec coverage**: This plan covers spec §8 (repo skeleton — package layout, `config.py`, `data/`, `tracking/`, `api/` established), the DB schema portion of §2 (tracking tables), and the static team-reference portion of §3 (arena lat/lon/timezone/altitude table). It intentionally does **not** cover §3's live data sources, §4 (model/features), §5 (markets), §6 (frontend), or §7's non-health routes — those are Phases 2-5. §2's Docker/Azure deploy content is Phase 6.
- **Placeholder scan**: no TBD/TODO markers; every step has runnable code.
- **Type consistency**: `TeamInfo.conference` values (`"East"`/`"West"`) match the spec's Global Constraint; `store.insert_prediction` signature and `get_predictions_for_game` signature are used consistently between Task 4's interface block and its steps.
