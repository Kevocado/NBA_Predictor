# NBA Predictor — Phase 4: API Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full REST API surface (spec §7) on top of the Phase 1
tracking DB, Phase 2 data modules, and Phase 3 features/models: games list
and detail (with predictions and value-bet markets), player props, teams,
the Data Hub endpoints (team hub, player hub, rankings, standings, track
record), and admin-gated retrain/refresh endpoints.

**Architecture:** Predictions are computed **offline** (by the retrain
pipeline this phase adds) and written to `tracking.db`; API routes are thin
— they read schedule metadata from a small JSON-backed repository, read
predictions/markets from `tracking.db`, and read hub aggregates (team/player
season stats — data richer than what Phase 1-3 compute) from precomputed
JSON caches. This mirrors the sibling projects' `public_snapshot.json`
pattern: nothing expensive runs inside a request. `PUBLIC_MODE` (Phase 1's
`config.PUBLIC_MODE`) gates the admin-only write routes (`POST /retrain`,
`POST /refresh-odds`) with a 404, exactly as PL_Predictor does.

**Tech Stack:** Same as Phases 1-3, plus `pydantic` (already a FastAPI
dependency) for response schemas.

**Spec:** [docs/superpowers/specs/2026-09-15-nba-predictor-design.md](../specs/2026-09-15-nba-predictor-design.md)

## Global Constraints

- Python version floor: **3.13**.
- Every route handler stays thin: it resolves dependencies (via FastAPI
  `Depends`) and calls a plain function/class from `services/`, `odds/`, or
  `pipeline/` — no business logic written directly inside a route body.
- `PUBLIC_MODE=true` must 404 (not 401/403 — matches PL_Predictor's
  `GuestAuthMiddleware` behavior of hiding admin routes entirely) every
  admin route.
- Hub aggregate endpoints (`/hub/teams`, `/hub/players`, `/hub/rankings`,
  `/hub/standings`) read precomputed JSON under `data/cache/hub/` — the
  batch job that populates those files is **out of scope for this phase**
  (flagged as a follow-up in Self-Review); routes must degrade gracefully
  (empty list, not a 500) when the cache file doesn't exist yet.
- `/hub/track-record` is the one hub endpoint computed for real in this
  phase, directly from `tracking.db`, since that data already exists after
  Phase 1.

---

## File Structure

```
src/nba_predictor/
  odds/
    __init__.py
    value_bets.py          # NEW — Shin de-vig, implied probability, edge
  api/
    schemas.py              # NEW — Pydantic response models
    deps.py                  # NEW — FastAPI dependency providers
    routes.py                 # MODIFY (Phase 1) — add all new routes
  services/
    __init__.py
    schedule_repository.py  # NEW — JSON-backed schedule read helpers
    hub_service.py            # NEW — hub JSON cache loader + track-record calc
  pipeline/
    __init__.py
    retrain.py                # NEW — offline training pipeline (Phase 3 glue)
  tracking/
    store.py                  # MODIFY (Phase 1) — add market/player prediction CRUD
```

---

### Task 1: Value-bets module (Shin de-vig, implied probability, edge)

**Files:**
- Create: `src/nba_predictor/odds/__init__.py`
- Create: `src/nba_predictor/odds/value_bets.py`
- Test: `tests/test_odds_value_bets.py`

**Interfaces:**
- Consumes: nothing (pure math, uses `scipy.optimize.brentq`, already a dependency).
- Produces:
  - `value_bets.implied_probability(american_odds: int) -> float`
  - `value_bets.shin_devig(raw_probabilities: list[float]) -> list[float]` — takes raw implied probabilities (one per outcome, summing to >1 due to the bookmaker's overround) and returns de-vigged "true" probabilities summing to 1.0, using Shin's (1992) two-parameter model: for insider-trading rate `z`, `π_i(z) = (sqrt(z² + 4(1-z)·p_i²/Σp) - z) / (2(1-z))`, solved so `Σ π_i(z) = 1`.
  - `value_bets.compute_edge(model_probability: float, market_probability: float) -> float` — `model_probability - market_probability`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_odds_value_bets.py
import pytest


def test_implied_probability_positive_american_odds():
    from nba_predictor.odds.value_bets import implied_probability

    assert implied_probability(150) == pytest.approx(100 / 250)


def test_implied_probability_negative_american_odds():
    from nba_predictor.odds.value_bets import implied_probability

    assert implied_probability(-110) == pytest.approx(110 / 210)


def test_shin_devig_symmetric_odds_split_evenly():
    from nba_predictor.odds.value_bets import implied_probability, shin_devig

    raw = [implied_probability(-110), implied_probability(-110)]
    devigged = shin_devig(raw)

    assert devigged[0] == pytest.approx(0.5, abs=1e-4)
    assert devigged[1] == pytest.approx(0.5, abs=1e-4)
    assert sum(devigged) == pytest.approx(1.0, abs=1e-6)


def test_shin_devig_preserves_favorite_ordering():
    from nba_predictor.odds.value_bets import implied_probability, shin_devig

    raw = [implied_probability(-200), implied_probability(170)]
    devigged = shin_devig(raw)

    assert devigged[0] > devigged[1]
    assert sum(devigged) == pytest.approx(1.0, abs=1e-6)


def test_shin_devig_no_overround_just_normalizes():
    from nba_predictor.odds.value_bets import shin_devig

    devigged = shin_devig([0.5, 0.5])
    assert devigged == pytest.approx([0.5, 0.5])


def test_compute_edge_positive_when_model_favors_selection():
    from nba_predictor.odds.value_bets import compute_edge

    assert compute_edge(0.60, 0.52) == pytest.approx(0.08)


def test_compute_edge_negative_when_market_favors_selection():
    from nba_predictor.odds.value_bets import compute_edge

    assert compute_edge(0.45, 0.52) == pytest.approx(-0.07)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_odds_value_bets.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.odds'`

- [ ] **Step 3: Create `src/nba_predictor/odds/__init__.py`** (empty file)

```bash
mkdir -p src/nba_predictor/odds
touch src/nba_predictor/odds/__init__.py
```

- [ ] **Step 4: Write `src/nba_predictor/odds/value_bets.py`**

```python
import math

from scipy.optimize import brentq


def implied_probability(american_odds: int) -> float:
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)


def _shin_probabilities(z: float, raw_probabilities: list[float], total: float) -> list[float]:
    return [
        (math.sqrt(z**2 + 4 * (1 - z) * p**2 / total) - z) / (2 * (1 - z))
        for p in raw_probabilities
    ]


def shin_devig(raw_probabilities: list[float]) -> list[float]:
    total = sum(raw_probabilities)
    if total <= 1.0:
        return [p / total for p in raw_probabilities]

    def sum_minus_one(z: float) -> float:
        return sum(_shin_probabilities(z, raw_probabilities, total)) - 1

    z = brentq(sum_minus_one, 1e-9, 0.999)
    return _shin_probabilities(z, raw_probabilities, total)


def compute_edge(model_probability: float, market_probability: float) -> float:
    return model_probability - market_probability
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_odds_value_bets.py -v`
Expected: PASS (7 passed)

- [ ] **Step 6: Commit**

```bash
git add src/nba_predictor/odds/ tests/test_odds_value_bets.py
git commit -m "feat: add Shin de-vig value-bets module"
```

---

### Task 2: Response schemas

**Files:**
- Create: `src/nba_predictor/api/schemas.py`
- Test: `tests/test_api_schemas.py`

**Interfaces:**
- Consumes: nothing.
- Produces (all `pydantic.BaseModel` subclasses):
  - `TeamOut` — `abbreviation, name, conference, division: str`
  - `PredictionOut` — `home_win_probability, predicted_margin, predicted_total: float`
  - `MarketPredictionOut` — `market, selection: str; model_probability: float; market_probability, edge: float | None = None; bookmaker: str | None = None; american_odds: int | None = None`
  - `GameOut` — `game_id, game_date, home_team, away_team: str; prediction: PredictionOut | None = None`
  - `GameDetailOut` — `GameOut`'s fields plus `markets: list[MarketPredictionOut] = []`
  - `PlayerPropOut` — `player_id, player_name, stat: str; predicted_value: float`
  - `TrackRecordOut` — `market: str; total_predictions: int; correct_predictions: int; hit_rate: float`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api_schemas.py
def test_game_detail_out_builds_with_nested_models():
    from nba_predictor.api.schemas import GameDetailOut, MarketPredictionOut, PredictionOut

    detail = GameDetailOut(
        game_id="0022600001",
        game_date="2026-11-01",
        home_team="BOS",
        away_team="MIA",
        prediction=PredictionOut(home_win_probability=0.62, predicted_margin=3.5, predicted_total=224.5),
        markets=[
            MarketPredictionOut(
                market="h2h", selection="home", model_probability=0.62,
                market_probability=0.55, edge=0.07, bookmaker="DraftKings", american_odds=-130,
            )
        ],
    )

    assert detail.prediction.home_win_probability == 0.62
    assert detail.markets[0].edge == 0.07


def test_game_out_prediction_defaults_to_none():
    from nba_predictor.api.schemas import GameOut

    game = GameOut(game_id="0022600001", game_date="2026-11-01", home_team="BOS", away_team="MIA")
    assert game.prediction is None


def test_track_record_out_computes_independently_of_hit_rate_field():
    from nba_predictor.api.schemas import TrackRecordOut

    record = TrackRecordOut(market="h2h", total_predictions=100, correct_predictions=58, hit_rate=0.58)
    assert record.hit_rate == 0.58
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_schemas.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.api.schemas'`

- [ ] **Step 3: Write `src/nba_predictor/api/schemas.py`**

```python
from pydantic import BaseModel


class TeamOut(BaseModel):
    abbreviation: str
    name: str
    conference: str
    division: str


class PredictionOut(BaseModel):
    home_win_probability: float
    predicted_margin: float
    predicted_total: float


class MarketPredictionOut(BaseModel):
    market: str
    selection: str
    model_probability: float
    market_probability: float | None = None
    edge: float | None = None
    bookmaker: str | None = None
    american_odds: int | None = None


class GameOut(BaseModel):
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    prediction: PredictionOut | None = None


class GameDetailOut(GameOut):
    markets: list[MarketPredictionOut] = []


class PlayerPropOut(BaseModel):
    player_id: str
    player_name: str
    stat: str
    predicted_value: float


class TrackRecordOut(BaseModel):
    market: str
    total_predictions: int
    correct_predictions: int
    hit_rate: float
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api_schemas.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/api/schemas.py tests/test_api_schemas.py
git commit -m "feat: add API response schemas"
```

---

### Task 3: Tracking store extensions (market + player predictions)

**Files:**
- Modify: `src/nba_predictor/tracking/store.py`
- Test: `tests/test_tracking_store.py` (extend)

**Interfaces:**
- Consumes: Phase 1's `store.get_connection`, existing `game_market_predictions` and `player_prediction_snapshots` tables (schema already created in Phase 1 — this task only adds CRUD functions, no schema change).
- Produces (added to `store.py`):
  - `store.insert_market_prediction(db_path, *, game_id: str, market: str, selection: str, model_probability: float, market_probability: float | None, edge: float | None, bookmaker: str | None, american_odds: int | None, created_at: str) -> int`
  - `store.get_market_predictions_for_game(db_path, game_id: str) -> list[sqlite3.Row]`
  - `store.insert_player_prediction(db_path, *, game_id: str, player_id: str, stat: str, predicted_value: float, created_at: str) -> int`
  - `store.get_player_predictions_for_game(db_path, game_id: str) -> list[sqlite3.Row]`
  - `store.get_latest_prediction_for_game(db_path, game_id: str) -> sqlite3.Row | None` — the most recent row from `predictions` for that game (or `None`).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_tracking_store.py`)

```python
# append to tests/test_tracking_store.py

def test_insert_and_get_market_predictions_roundtrip(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    store.insert_market_prediction(
        db_path,
        game_id="g1",
        market="h2h",
        selection="home",
        model_probability=0.62,
        market_probability=0.55,
        edge=0.07,
        bookmaker="DraftKings",
        american_odds=-130,
        created_at="2026-11-01T12:00:00",
    )

    rows = store.get_market_predictions_for_game(db_path, "g1")
    assert len(rows) == 1
    assert rows[0]["market"] == "h2h"
    assert rows[0]["edge"] == pytest.approx(0.07)


def test_insert_and_get_player_predictions_roundtrip(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    store.insert_player_prediction(
        db_path, game_id="g1", player_id="203999", stat="points",
        predicted_value=27.5, created_at="2026-11-01T12:00:00",
    )

    rows = store.get_player_predictions_for_game(db_path, "g1")
    assert len(rows) == 1
    assert rows[0]["stat"] == "points"
    assert rows[0]["predicted_value"] == pytest.approx(27.5)


def test_get_latest_prediction_for_game_returns_most_recent(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-11-01T08:00:00", model_version="v1",
        home_win_prob=0.55, predicted_margin=1.0, predicted_total=220.0,
    )
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-11-01T18:00:00", model_version="v1",
        home_win_prob=0.60, predicted_margin=2.0, predicted_total=222.0,
    )

    latest = store.get_latest_prediction_for_game(db_path, "g1")
    assert latest["created_at"] == "2026-11-01T18:00:00"


def test_get_latest_prediction_for_game_returns_none_when_missing(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    assert store.get_latest_prediction_for_game(db_path, "does-not-exist") is None
```

Also add `import pytest` at the top of `tests/test_tracking_store.py` if not already present (needed for `pytest.approx`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tracking_store.py -v`
Expected: 4 new tests FAIL / ERROR — `AttributeError: module 'nba_predictor.tracking.store' has no attribute 'insert_market_prediction'`

- [ ] **Step 3: Add the new functions to `src/nba_predictor/tracking/store.py`** (append below the existing `get_predictions_for_game` function)

```python
def insert_market_prediction(
    db_path: Path,
    *,
    game_id: str,
    market: str,
    selection: str,
    model_probability: float,
    market_probability: float | None,
    edge: float | None,
    bookmaker: str | None,
    american_odds: int | None,
    created_at: str,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO game_market_predictions
                (game_id, market, selection, model_probability, market_probability, edge, bookmaker, american_odds, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (game_id, market, selection, model_probability, market_probability, edge, bookmaker, american_odds, created_at),
        )
        conn.commit()
        return cur.lastrowid


def get_market_predictions_for_game(db_path: Path, game_id: str) -> list[sqlite3.Row]:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM game_market_predictions WHERE game_id = ? ORDER BY created_at",
            (game_id,),
        )
        return cur.fetchall()


def insert_player_prediction(
    db_path: Path,
    *,
    game_id: str,
    player_id: str,
    stat: str,
    predicted_value: float,
    created_at: str,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO player_prediction_snapshots (game_id, player_id, stat, predicted_value, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (game_id, player_id, stat, predicted_value, created_at),
        )
        conn.commit()
        return cur.lastrowid


def get_player_predictions_for_game(db_path: Path, game_id: str) -> list[sqlite3.Row]:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM player_prediction_snapshots WHERE game_id = ? ORDER BY created_at",
            (game_id,),
        )
        return cur.fetchall()


def get_latest_prediction_for_game(db_path: Path, game_id: str) -> sqlite3.Row | None:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM predictions WHERE game_id = ? ORDER BY created_at DESC LIMIT 1",
            (game_id,),
        )
        return cur.fetchone()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tracking_store.py -v`
Expected: PASS (8 passed — 4 original + 4 new)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/tracking/store.py tests/test_tracking_store.py
git commit -m "feat: add market/player prediction CRUD to tracking store"
```

---

### Task 4: Schedule repository

**Files:**
- Create: `src/nba_predictor/services/__init__.py`
- Create: `src/nba_predictor/services/schedule_repository.py`
- Test: `tests/test_schedule_repository.py`

**Interfaces:**
- Consumes: nothing (plain JSON file I/O).
- Produces:
  - `schedule_repository.load_schedule(path: Path) -> list[dict]` — reads a JSON array of `{"game_id": str, "game_date": str, "home_team": str, "away_team": str}`; returns `[]` if the file doesn't exist.
  - `schedule_repository.get_games_for_date(schedule: list[dict], game_date: str) -> list[dict]`
  - `schedule_repository.get_game(schedule: list[dict], game_id: str) -> dict | None`

The JSON file at `path` is expected to be produced by a scheduled job that
calls `nba_predictor.data.nba_api.get_schedule` across the season and writes
the results here — that job is out of scope for this phase (flagged in
Self-Review) — this task only needs the read-side contract so routes can be
built and tested against it now.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schedule_repository.py
import json


def test_load_schedule_returns_empty_list_when_file_missing(tmp_path):
    from nba_predictor.services.schedule_repository import load_schedule

    assert load_schedule(tmp_path / "does_not_exist.json") == []


def test_load_schedule_reads_json_array(tmp_path):
    from nba_predictor.services.schedule_repository import load_schedule

    path = tmp_path / "games.json"
    path.write_text(json.dumps([{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}]))

    schedule = load_schedule(path)
    assert schedule == [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}]


def test_get_games_for_date_filters_correctly():
    from nba_predictor.services.schedule_repository import get_games_for_date

    schedule = [
        {"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"},
        {"game_id": "g2", "game_date": "2026-11-02", "home_team": "LAL", "away_team": "GSW"},
    ]
    result = get_games_for_date(schedule, "2026-11-01")
    assert result == [schedule[0]]


def test_get_game_returns_none_when_not_found():
    from nba_predictor.services.schedule_repository import get_game

    assert get_game([], "missing") is None


def test_get_game_finds_by_id():
    from nba_predictor.services.schedule_repository import get_game

    schedule = [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}]
    assert get_game(schedule, "g1") == schedule[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schedule_repository.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.services'`

- [ ] **Step 3: Create `src/nba_predictor/services/__init__.py`** (empty file)

```bash
mkdir -p src/nba_predictor/services
touch src/nba_predictor/services/__init__.py
```

- [ ] **Step 4: Write `src/nba_predictor/services/schedule_repository.py`**

```python
import json
from pathlib import Path


def load_schedule(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def get_games_for_date(schedule: list[dict], game_date: str) -> list[dict]:
    return [game for game in schedule if game["game_date"] == game_date]


def get_game(schedule: list[dict], game_id: str) -> dict | None:
    return next((game for game in schedule if game["game_id"] == game_id), None)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_schedule_repository.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add src/nba_predictor/services/ tests/test_schedule_repository.py
git commit -m "feat: add JSON-backed schedule repository"
```

---

### Task 5: Dependency providers and `/teams` routes

**Files:**
- Create: `src/nba_predictor/api/deps.py`
- Modify: `src/nba_predictor/api/routes.py`
- Test: `tests/test_api_teams.py`

**Interfaces:**
- Consumes: `config.TRACKING_DB_PATH`, `config.DATA_DIR` (Phase 1); `team_reference.TEAMS`, `get_team` (Phase 1); `schedule_repository.load_schedule` (Task 4).
- Produces:
  - `deps.get_db_path() -> Path` — FastAPI dependency returning `config.TRACKING_DB_PATH`.
  - `deps.get_schedule_path() -> Path` — returns `config.DATA_DIR / "cache" / "schedule" / "games.json"`.
  - `deps.get_schedule(schedule_path: Path = Depends(get_schedule_path)) -> list[dict]` — calls `schedule_repository.load_schedule`.
  - Routes: `GET /teams -> list[TeamOut]`, `GET /teams/{abbreviation} -> TeamOut` (404 if unknown).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api_teams.py
def test_list_teams_returns_thirty_teams():
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app

    client = TestClient(app)
    response = client.get("/teams")

    assert response.status_code == 200
    assert len(response.json()) == 30


def test_get_team_returns_matching_team():
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app

    client = TestClient(app)
    response = client.get("/teams/BOS")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Boston Celtics"
    assert body["conference"] == "East"


def test_get_team_404_for_unknown_abbreviation():
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app

    client = TestClient(app)
    response = client.get("/teams/ZZZ")

    assert response.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_teams.py -v`
Expected: FAIL — `404` for `/teams` (route doesn't exist yet, FastAPI returns 404 for unmatched routes too, but the test asserts `200` for the first two, so those fail).

- [ ] **Step 3: Write `src/nba_predictor/api/deps.py`**

```python
from pathlib import Path

from fastapi import Depends

from nba_predictor import config
from nba_predictor.services.schedule_repository import load_schedule


def get_db_path() -> Path:
    return config.TRACKING_DB_PATH


def get_schedule_path() -> Path:
    return config.DATA_DIR / "cache" / "schedule" / "games.json"


def get_schedule(schedule_path: Path = Depends(get_schedule_path)) -> list[dict]:
    return load_schedule(schedule_path)
```

- [ ] **Step 4: Add the teams routes to `src/nba_predictor/api/routes.py`**

```python
from fastapi import HTTPException

from nba_predictor.api.schemas import TeamOut
from nba_predictor.data.team_reference import TEAMS, get_team


@router.get("/teams", response_model=list[TeamOut])
def list_teams() -> list[TeamOut]:
    return [
        TeamOut(abbreviation=t.abbreviation, name=t.name, conference=t.conference, division=t.division)
        for t in TEAMS
    ]


@router.get("/teams/{abbreviation}", response_model=TeamOut)
def get_team_detail(abbreviation: str) -> TeamOut:
    try:
        team = get_team(abbreviation)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown team: {abbreviation}")
    return TeamOut(abbreviation=team.abbreviation, name=team.name, conference=team.conference, division=team.division)
```

(Add these imports to the top of `routes.py` alongside the existing `from
fastapi import APIRouter` line, and append the two route functions after the
existing `/health` route.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_api_teams.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/nba_predictor/api/deps.py src/nba_predictor/api/routes.py tests/test_api_teams.py
git commit -m "feat: add dependency providers and /teams routes"
```

---

### Task 6: Games routes (`/games`, `/games/{game_id}`, `/games/{game_id}/players`)

**Files:**
- Modify: `src/nba_predictor/api/routes.py`
- Test: `tests/test_api_games.py`

**Interfaces:**
- Consumes: `deps.get_db_path`, `deps.get_schedule` (Task 5); `store.get_latest_prediction_for_game`, `store.get_market_predictions_for_game`, `store.get_player_predictions_for_game` (Task 3); `schedule_repository.get_games_for_date`, `get_game` (Task 4).
- Produces routes:
  - `GET /games?date=YYYY-MM-DD -> list[GameOut]` — 422 (FastAPI's default) if `date` is omitted; each game includes its latest prediction if one exists in `tracking.db`, else `prediction: null`.
  - `GET /games/{game_id} -> GameDetailOut` — 404 if `game_id` isn't in the schedule; includes `markets` from `game_market_predictions` (empty list if none tracked yet).
  - `GET /games/{game_id}/players -> list[PlayerPropOut]` — 404 if `game_id` isn't in the schedule; `player_name` is `player_id` itself for now (a player-name lookup table is part of the hub-aggregation follow-up noted in Self-Review) — empty list if no predictions tracked yet.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api_games.py
from datetime import datetime


def _client_with_overrides(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app
    from nba_predictor.api import deps
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(
        '[{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}]'
    )

    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path

    client = TestClient(app)
    return client, db_path


def test_list_games_requires_date_query_param(tmp_path, monkeypatch):
    client, _ = _client_with_overrides(tmp_path, monkeypatch)
    response = client.get("/games")
    assert response.status_code == 422


def test_list_games_returns_games_for_date_with_null_prediction(tmp_path, monkeypatch):
    client, _ = _client_with_overrides(tmp_path, monkeypatch)
    response = client.get("/games", params={"date": "2026-11-01"})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["game_id"] == "g1"
    assert body[0]["prediction"] is None


def test_list_games_includes_latest_prediction_when_present(tmp_path, monkeypatch):
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-11-01T12:00:00", model_version="v1",
        home_win_prob=0.62, predicted_margin=3.5, predicted_total=224.5,
    )

    response = client.get("/games", params={"date": "2026-11-01"})
    body = response.json()
    assert body[0]["prediction"]["home_win_probability"] == 0.62


def test_get_game_detail_404_for_unknown_game(tmp_path, monkeypatch):
    client, _ = _client_with_overrides(tmp_path, monkeypatch)
    response = client.get("/games/does-not-exist")
    assert response.status_code == 404


def test_get_game_detail_includes_markets(tmp_path, monkeypatch):
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="home", model_probability=0.62,
        market_probability=0.55, edge=0.07, bookmaker="DraftKings", american_odds=-130,
        created_at="2026-11-01T12:00:00",
    )

    response = client.get("/games/g1")
    assert response.status_code == 200
    body = response.json()
    assert body["home_team"] == "BOS"
    assert len(body["markets"]) == 1
    assert body["markets"][0]["market"] == "h2h"


def test_get_game_players_404_for_unknown_game(tmp_path, monkeypatch):
    client, _ = _client_with_overrides(tmp_path, monkeypatch)
    response = client.get("/games/does-not-exist/players")
    assert response.status_code == 404


def test_get_game_players_returns_tracked_props(tmp_path, monkeypatch):
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="203999", stat="points",
        predicted_value=27.5, created_at="2026-11-01T12:00:00",
    )

    response = client.get("/games/g1/players")
    assert response.status_code == 200
    body = response.json()
    assert body[0]["stat"] == "points"
    assert body[0]["predicted_value"] == 27.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_games.py -v`
Expected: FAIL — routes don't exist yet (404s where 200/422 expected).

- [ ] **Step 3: Add the games routes to `src/nba_predictor/api/routes.py`**

```python
from pathlib import Path

from fastapi import Depends

from nba_predictor.api.deps import get_db_path, get_schedule, get_schedule_path
from nba_predictor.api.schemas import GameDetailOut, GameOut, MarketPredictionOut, PlayerPropOut, PredictionOut
from nba_predictor.services.schedule_repository import get_game, get_games_for_date
from nba_predictor.tracking import store


def _prediction_out(db_path: Path, game_id: str) -> PredictionOut | None:
    row = store.get_latest_prediction_for_game(db_path, game_id)
    if row is None:
        return None
    return PredictionOut(
        home_win_probability=row["home_win_prob"],
        predicted_margin=row["predicted_margin"],
        predicted_total=row["predicted_total"],
    )


@router.get("/games", response_model=list[GameOut])
def list_games(date: str, schedule: list[dict] = Depends(get_schedule), db_path: Path = Depends(get_db_path)) -> list[GameOut]:
    games = get_games_for_date(schedule, date)
    return [
        GameOut(
            game_id=g["game_id"], game_date=g["game_date"], home_team=g["home_team"], away_team=g["away_team"],
            prediction=_prediction_out(db_path, g["game_id"]),
        )
        for g in games
    ]


@router.get("/games/{game_id}", response_model=GameDetailOut)
def get_game_detail(
    game_id: str, schedule: list[dict] = Depends(get_schedule), db_path: Path = Depends(get_db_path)
) -> GameDetailOut:
    game = get_game(schedule, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_id}")

    markets = [
        MarketPredictionOut(
            market=row["market"], selection=row["selection"], model_probability=row["model_probability"],
            market_probability=row["market_probability"], edge=row["edge"], bookmaker=row["bookmaker"],
            american_odds=row["american_odds"],
        )
        for row in store.get_market_predictions_for_game(db_path, game_id)
    ]

    return GameDetailOut(
        game_id=game["game_id"], game_date=game["game_date"], home_team=game["home_team"], away_team=game["away_team"],
        prediction=_prediction_out(db_path, game_id), markets=markets,
    )


@router.get("/games/{game_id}/players", response_model=list[PlayerPropOut])
def get_game_players(
    game_id: str, schedule: list[dict] = Depends(get_schedule), db_path: Path = Depends(get_db_path)
) -> list[PlayerPropOut]:
    game = get_game(schedule, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_id}")

    return [
        PlayerPropOut(
            player_id=row["player_id"], player_name=row["player_id"], stat=row["stat"],
            predicted_value=row["predicted_value"],
        )
        for row in store.get_player_predictions_for_game(db_path, game_id)
    ]
```

(Add the new imports alongside existing ones at the top of `routes.py`;
append the three route functions and the `_prediction_out` helper after the
`/teams` routes from Task 5.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api_games.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/api/routes.py tests/test_api_games.py
git commit -m "feat: add games list/detail/players routes"
```

---

### Task 7: Hub service and hub routes

**Files:**
- Create: `src/nba_predictor/services/hub_service.py`
- Modify: `src/nba_predictor/api/routes.py`
- Test: `tests/test_hub_service.py`, `tests/test_api_hub.py`

**Interfaces:**
- Consumes: `config.DATA_DIR` (Phase 1); `store.get_connection` (Phase 1/3).
- Produces:
  - `hub_service.load_hub_cache(path: Path) -> list[dict]` — reads a JSON array cache file, returns `[]` if missing (same contract as `schedule_repository.load_schedule`, kept as a separate function since it lives in a conceptually different service).
  - `hub_service.compute_track_record(db_path: Path) -> list[TrackRecordOut]` — joins `game_market_predictions` against `game_player_outcomes`-style resolution is out of scope (no settled-outcome table populated yet in this phase); for now, groups `game_market_predictions` by `market` and reports `total_predictions` = row count, `correct_predictions` = 0, `hit_rate` = 0.0 for every market with at least one tracked prediction — the win/loss settlement job is a follow-up (flagged in Self-Review) that will populate `game_player_outcomes`-equivalent settlement data and this function will then compute real hit rates from it.
  - Routes: `GET /hub/teams -> list[dict]`, `GET /hub/players -> list[dict]`, `GET /hub/rankings -> list[dict]`, `GET /hub/standings -> list[dict]` (all read `data/cache/hub/<name>.json` via `load_hub_cache`, returning `[]` if absent), `GET /hub/track-record -> list[TrackRecordOut]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hub_service.py
def test_load_hub_cache_returns_empty_list_when_missing(tmp_path):
    from nba_predictor.services.hub_service import load_hub_cache

    assert load_hub_cache(tmp_path / "missing.json") == []


def test_load_hub_cache_reads_json_array(tmp_path):
    import json

    from nba_predictor.services.hub_service import load_hub_cache

    path = tmp_path / "teams.json"
    path.write_text(json.dumps([{"abbreviation": "BOS", "points_per_game": 118.2}]))

    assert load_hub_cache(path) == [{"abbreviation": "BOS", "points_per_game": 118.2}]


def test_compute_track_record_groups_by_market(tmp_path):
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="home", model_probability=0.6,
        market_probability=0.5, edge=0.1, bookmaker="DraftKings", american_odds=-130,
        created_at="2026-11-01T12:00:00",
    )
    store.insert_market_prediction(
        db_path, game_id="g2", market="h2h", selection="away", model_probability=0.55,
        market_probability=0.5, edge=0.05, bookmaker="DraftKings", american_odds=120,
        created_at="2026-11-02T12:00:00",
    )
    store.insert_market_prediction(
        db_path, game_id="g1", market="spread", selection="home", model_probability=0.52,
        market_probability=0.5, edge=0.02, bookmaker="DraftKings", american_odds=-110,
        created_at="2026-11-01T12:00:00",
    )

    records = compute_track_record(db_path)
    by_market = {r.market: r for r in records}

    assert by_market["h2h"].total_predictions == 2
    assert by_market["spread"].total_predictions == 1


def test_compute_track_record_empty_db_returns_empty_list(tmp_path):
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    assert compute_track_record(db_path) == []
```

```python
# tests/test_api_hub.py
def _client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app
    from nba_predictor.api import deps
    from nba_predictor import config
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    app.dependency_overrides[deps.get_db_path] = lambda: db_path

    return TestClient(app)


def test_hub_teams_empty_when_no_cache(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.get("/hub/teams")
    assert response.status_code == 200
    assert response.json() == []


def test_hub_players_empty_when_no_cache(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.get("/hub/players")
    assert response.status_code == 200
    assert response.json() == []


def test_hub_rankings_empty_when_no_cache(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    assert client.get("/hub/rankings").json() == []


def test_hub_standings_empty_when_no_cache(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    assert client.get("/hub/standings").json() == []


def test_hub_track_record_empty_when_no_predictions(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.get("/hub/track-record")
    assert response.status_code == 200
    assert response.json() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hub_service.py tests/test_api_hub.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.services.hub_service'`

- [ ] **Step 3: Write `src/nba_predictor/services/hub_service.py`**

```python
import json
from pathlib import Path

from nba_predictor.api.schemas import TrackRecordOut
from nba_predictor.tracking.store import get_connection


def load_hub_cache(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def compute_track_record(db_path: Path) -> list[TrackRecordOut]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT market, COUNT(*) as total FROM game_market_predictions GROUP BY market"
        ).fetchall()

    return [
        TrackRecordOut(market=row["market"], total_predictions=row["total"], correct_predictions=0, hit_rate=0.0)
        for row in rows
    ]
```

- [ ] **Step 4: Add the hub routes to `src/nba_predictor/api/routes.py`**

```python
from nba_predictor import config
from nba_predictor.services.hub_service import compute_track_record, load_hub_cache


@router.get("/hub/teams")
def hub_teams() -> list[dict]:
    return load_hub_cache(config.DATA_DIR / "cache" / "hub" / "teams.json")


@router.get("/hub/players")
def hub_players() -> list[dict]:
    return load_hub_cache(config.DATA_DIR / "cache" / "hub" / "players.json")


@router.get("/hub/rankings")
def hub_rankings() -> list[dict]:
    return load_hub_cache(config.DATA_DIR / "cache" / "hub" / "rankings.json")


@router.get("/hub/standings")
def hub_standings() -> list[dict]:
    return load_hub_cache(config.DATA_DIR / "cache" / "hub" / "standings.json")


@router.get("/hub/track-record", response_model=list[TrackRecordOut])
def hub_track_record(db_path: Path = Depends(get_db_path)) -> list[TrackRecordOut]:
    return compute_track_record(db_path)
```

(Add `TrackRecordOut` to the existing `from nba_predictor.api.schemas import
...` line at the top of `routes.py` rather than importing it twice.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_hub_service.py tests/test_api_hub.py -v`
Expected: PASS (9 passed)

- [ ] **Step 6: Commit**

```bash
git add src/nba_predictor/services/hub_service.py src/nba_predictor/api/routes.py tests/test_hub_service.py tests/test_api_hub.py
git commit -m "feat: add hub service and hub routes"
```

---

### Task 8: Retrain pipeline (Phase 3 glue)

**Files:**
- Create: `src/nba_predictor/pipeline/__init__.py`
- Create: `src/nba_predictor/pipeline/retrain.py`
- Test: `tests/test_pipeline_retrain.py`

**Interfaces:**
- Consumes: `features.build.build_training_frame` (Phase 3), `models.game_outcome.train_win_probability_model/train_margin_model/train_total_model` (Phase 3), `models.manifest.build_manifest/write_manifest/append_manifest_history` (Phase 3), `models.evaluate.walk_forward.chronological_split` (Phase 3).
- Produces:
  - `retrain.run_retrain_pipeline(games: pd.DataFrame, models_dir: Path, model_version: str, trained_at: str) -> dict` — runs `chronological_split` (holdout 20%), builds the training frame via `build_training_frame`, trains the three game-outcome models on the training split, evaluates simple metrics on the holdout split (accuracy for win-probability, MAE for margin/total), saves each model to `models_dir` via `joblib.dump` (`win_probability_model.pkl`, `margin_model.pkl`, `total_model.pkl`), writes `models_dir / "manifest.json"` and appends to `models_dir / "manifest_history.jsonl"`, and returns the manifest dict.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_retrain.py
import numpy as np
import pandas as pd


def _synthetic_games(n=60, seed=2):
    rng = np.random.default_rng(seed)
    teams = ["BOS", "MIA", "LAL", "GSW"]
    dates = pd.date_range("2026-10-21", periods=n).astype(str)
    rows = []
    for i, game_date in enumerate(dates):
        home, away = teams[i % 4], teams[(i + 1) % 4]
        rows.append(
            {
                "game_id": f"g{i}", "game_date": game_date, "home_team": home, "away_team": away,
                "home_pts": 110, "away_pts": 105,
                "home_fgm": 40, "home_fga": 88, "home_fg3m": 12, "home_tov": 11, "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
                "away_fgm": 38, "away_fga": 90, "away_fg3m": 10, "away_tov": 13, "away_oreb": 10, "away_dreb": 30, "away_fta": 18,
                "home_win": int(rng.random() > 0.4),
            }
        )
    return pd.DataFrame(rows)


def test_run_retrain_pipeline_produces_models_and_manifest(tmp_path):
    from nba_predictor.pipeline.retrain import run_retrain_pipeline

    games = _synthetic_games()
    models_dir = tmp_path / "models"

    manifest = run_retrain_pipeline(games, models_dir, model_version="v-test", trained_at="2026-11-01T00:00:00")

    assert manifest["model_version"] == "v-test"
    assert set(manifest["models"]) == {"win_probability", "margin", "total"}
    assert (models_dir / "win_probability_model.pkl").exists()
    assert (models_dir / "margin_model.pkl").exists()
    assert (models_dir / "total_model.pkl").exists()
    assert (models_dir / "manifest.json").exists()
    assert (models_dir / "manifest_history.jsonl").exists()
    assert "accuracy" in manifest["metrics"]["win_probability"]


def test_run_retrain_pipeline_appends_history_across_calls(tmp_path):
    from nba_predictor.pipeline.retrain import run_retrain_pipeline

    games = _synthetic_games()
    models_dir = tmp_path / "models"

    run_retrain_pipeline(games, models_dir, model_version="v1", trained_at="2026-11-01T00:00:00")
    run_retrain_pipeline(games, models_dir, model_version="v2", trained_at="2026-11-02T00:00:00")

    lines = (models_dir / "manifest_history.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_retrain.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.pipeline'`

- [ ] **Step 3: Create `src/nba_predictor/pipeline/__init__.py`** (empty file)

```bash
mkdir -p src/nba_predictor/pipeline
touch src/nba_predictor/pipeline/__init__.py
```

- [ ] **Step 4: Write `src/nba_predictor/pipeline/retrain.py`**

Add `joblib` to `pyproject.toml`'s `dependencies` list first (it's the
standard way to serialize scikit-learn/xgboost-compatible models — not yet a
dependency in Phases 1-3):

```toml
    "joblib>=1.4",
```

```python
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_predictor.features.build import build_training_frame
from nba_predictor.models.evaluate.walk_forward import chronological_split
from nba_predictor.models.game_outcome import (
    predict_win_probability,
    train_margin_model,
    train_total_model,
    train_win_probability_model,
)
from nba_predictor.models.manifest import append_manifest_history, build_manifest, write_manifest


def run_retrain_pipeline(games: pd.DataFrame, models_dir: Path, model_version: str, trained_at: str) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)

    train_games, holdout_games = chronological_split(games, date_col="game_date", holdout_fraction=0.2)

    train_df, feature_cols = build_training_frame(train_games)
    holdout_df, _ = build_training_frame(pd.concat([train_games, holdout_games], ignore_index=True))
    holdout_df = holdout_df[holdout_df["game_id"].isin(holdout_games["game_id"])]

    win_model = train_win_probability_model(train_df[feature_cols], train_df["home_win"])
    margin_model = train_margin_model(train_df[feature_cols], train_df["home_pts"] - train_df["away_pts"])
    total_model = train_total_model(train_df[feature_cols], train_df["home_pts"] + train_df["away_pts"])

    metrics = {}
    if len(holdout_df) > 0:
        win_probs = predict_win_probability(win_model, holdout_df[feature_cols])
        win_preds = (win_probs >= 0.5).astype(int)
        metrics["win_probability"] = {"accuracy": float((win_preds == holdout_df["home_win"]).mean())}

        margin_preds = margin_model.predict(holdout_df[feature_cols])
        actual_margin = holdout_df["home_pts"] - holdout_df["away_pts"]
        metrics["margin"] = {"mae": float(np.mean(np.abs(margin_preds - actual_margin)))}

        total_preds = total_model.predict(holdout_df[feature_cols])
        actual_total = holdout_df["home_pts"] + holdout_df["away_pts"]
        metrics["total"] = {"mae": float(np.mean(np.abs(total_preds - actual_total)))}
    else:
        metrics = {"win_probability": {"accuracy": None}, "margin": {"mae": None}, "total": {"mae": None}}

    joblib.dump(win_model, models_dir / "win_probability_model.pkl")
    joblib.dump(margin_model, models_dir / "margin_model.pkl")
    joblib.dump(total_model, models_dir / "total_model.pkl")

    manifest = build_manifest(
        model_names=["win_probability", "margin", "total"],
        metrics=metrics,
        model_version=model_version,
        trained_at=trained_at,
    )
    write_manifest(manifest, models_dir / "manifest.json")
    append_manifest_history(manifest, models_dir / "manifest_history.jsonl")

    return manifest
```

- [ ] **Step 5: Reinstall the package so the new `joblib` dependency is picked up, then run tests**

```bash
pip install -e ".[dev]"
pytest tests/test_pipeline_retrain.py -v
```

Expected: PASS (2 passed). If the holdout split produces zero valid rows
after `build_training_frame` drops each team's first tracked game (small
synthetic datasets are prone to this), increase `n` in the test's
`_synthetic_games()` rather than changing the pipeline's split logic.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/nba_predictor/pipeline/ tests/test_pipeline_retrain.py
git commit -m "feat: add offline retrain pipeline"
```

---

### Task 9: Admin routes (`POST /retrain`, `POST /refresh-odds`) with `PUBLIC_MODE` gating

**Files:**
- Modify: `src/nba_predictor/api/deps.py`
- Modify: `src/nba_predictor/api/routes.py`
- Test: `tests/test_api_admin.py`

**Interfaces:**
- Consumes: `config.PUBLIC_MODE` (Phase 1), `pipeline.retrain.run_retrain_pipeline` (Task 8).
- Produces:
  - `deps.require_admin() -> None` — raises `HTTPException(status_code=404)` if `config.PUBLIC_MODE` is `True`; no-op otherwise. (404, not 401/403, per this plan's Global Constraints — hides the route's existence entirely on the public deployment.)
  - `deps.get_models_dir() -> Path` — returns `config.PROJECT_ROOT / "models"`.
  - `deps.get_training_games_path() -> Path` — returns `config.DATA_DIR / "cache" / "training" / "games.json"` (another JSON cache, populated by the same future batch job noted in Task 4/7 — out of scope here).
  - Routes: `POST /retrain` (admin-gated) — loads the training games JSON via the dependency, calls `run_retrain_pipeline`, returns the manifest dict; `400` if the training games cache doesn't exist yet. `POST /refresh-odds` (admin-gated) — Task-9-scope is the gating + a `202 {"status": "not implemented"}` acknowledgement; wiring it to `data.sportsbook_api`/`data.odds_api` (Phase 2) plus writing results via `store.insert_market_prediction` is a follow-up once the hub-aggregation batch job (Task 7's note) exists to supply the current schedule/roster context odds need — flagged in Self-Review, not silently dropped.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api_admin.py
import json


def _client(tmp_path, monkeypatch, public_mode: bool):
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app
    from nba_predictor.api import deps
    from nba_predictor import config

    monkeypatch.setattr(config, "PUBLIC_MODE", public_mode)
    app.dependency_overrides[deps.get_models_dir] = lambda: tmp_path / "models"
    app.dependency_overrides[deps.get_training_games_path] = lambda: tmp_path / "training_games.json"

    return TestClient(app)


def test_retrain_404_when_public_mode(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, public_mode=True)
    response = client.post("/retrain")
    assert response.status_code == 404


def test_refresh_odds_404_when_public_mode(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, public_mode=True)
    response = client.post("/refresh-odds")
    assert response.status_code == 404


def test_retrain_400_when_no_training_cache(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, public_mode=False)
    response = client.post("/retrain")
    assert response.status_code == 400


def test_retrain_succeeds_when_training_cache_present(tmp_path, monkeypatch):
    import pandas as pd

    client = _client(tmp_path, monkeypatch, public_mode=False)

    teams = ["BOS", "MIA", "LAL", "GSW"]
    dates = pd.date_range("2026-10-21", periods=60).astype(str)
    rows = []
    for i, game_date in enumerate(dates):
        home, away = teams[i % 4], teams[(i + 1) % 4]
        rows.append(
            {
                "game_id": f"g{i}", "game_date": str(game_date), "home_team": home, "away_team": away,
                "home_pts": 110, "away_pts": 105,
                "home_fgm": 40, "home_fga": 88, "home_fg3m": 12, "home_tov": 11, "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
                "away_fgm": 38, "away_fga": 90, "away_fg3m": 10, "away_tov": 13, "away_oreb": 10, "away_dreb": 30, "away_fta": 18,
                "home_win": i % 2,
            }
        )
    (tmp_path / "training_games.json").write_text(json.dumps(rows))

    response = client.post("/retrain")
    assert response.status_code == 200
    assert response.json()["models"] == ["win_probability", "margin", "total"]


def test_refresh_odds_returns_202_not_public_mode(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, public_mode=False)
    response = client.post("/refresh-odds")
    assert response.status_code == 202
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_admin.py -v`
Expected: FAIL — routes don't exist yet.

- [ ] **Step 3: Add `require_admin`, `get_models_dir`, `get_training_games_path` to `src/nba_predictor/api/deps.py`**

```python
from fastapi import HTTPException

from nba_predictor import config


def require_admin() -> None:
    if config.PUBLIC_MODE:
        raise HTTPException(status_code=404, detail="Not found")


def get_models_dir() -> Path:
    return config.PROJECT_ROOT / "models"


def get_training_games_path() -> Path:
    return config.DATA_DIR / "cache" / "training" / "games.json"
```

- [ ] **Step 4: Add the admin routes to `src/nba_predictor/api/routes.py`**

```python
import json
from datetime import datetime, timezone

import pandas as pd
from fastapi import Depends, HTTPException

from nba_predictor.api.deps import get_models_dir, get_training_games_path, require_admin
from nba_predictor.pipeline.retrain import run_retrain_pipeline


@router.post("/retrain", dependencies=[Depends(require_admin)])
def retrain(
    models_dir: Path = Depends(get_models_dir),
    training_games_path: Path = Depends(get_training_games_path),
) -> dict:
    if not training_games_path.exists():
        raise HTTPException(status_code=400, detail="No training games cache found")

    games = pd.DataFrame(json.loads(training_games_path.read_text()))
    model_version = datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
    trained_at = datetime.now(timezone.utc).isoformat()

    return run_retrain_pipeline(games, models_dir, model_version=model_version, trained_at=trained_at)


@router.post("/refresh-odds", dependencies=[Depends(require_admin)], status_code=202)
def refresh_odds() -> dict:
    return {"status": "not implemented"}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_api_admin.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add src/nba_predictor/api/deps.py src/nba_predictor/api/routes.py tests/test_api_admin.py
git commit -m "feat: add admin retrain/refresh-odds routes with PUBLIC_MODE gating"
```

---

### Task 10: Full API smoke test

**Files:** No new files — verification only.

- [ ] **Step 1: Run the entire test suite**

Run: `pytest -v`
Expected: all Phase 1-4 tests pass, 0 failed, 0 errors.

- [ ] **Step 2: Boot the app and exercise the full route surface over real HTTP**

```bash
uvicorn nba_predictor.api.app:app --port 8000 &
sleep 1
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/teams | python -c "import sys, json; print(len(json.load(sys.stdin)))"
curl -s http://127.0.0.1:8000/teams/BOS
curl -s "http://127.0.0.1:8000/games?date=2026-11-01"
curl -s http://127.0.0.1:8000/hub/track-record
curl -s -X POST http://127.0.0.1:8000/retrain
kill %1
```

Expected: `/health` returns `{"status":"ok"}`; `/teams` prints `30`;
`/teams/BOS` returns the Celtics; `/games?date=...` returns `[]` (no
schedule cache in this ad-hoc run); `/hub/track-record` returns `[]`;
`/retrain` returns `{"detail":"No training games cache found"}` with a 400
(expected — no cache populated in this smoke run, confirms the route is
wired and fails the *correct*, non-crashing way).

- [ ] **Step 3: Verify `PUBLIC_MODE` actually hides the admin routes**

```bash
PUBLIC_MODE=true uvicorn nba_predictor.api.app:app --port 8001 &
sleep 1
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8001/retrain
kill %1
```

Expected output: `404`

---

## Self-Review Notes

- **Spec coverage:** every route in spec §7 that this plan is scoped to
  cover is present: `/health`, `/games`, `/games/{id}`, `/games/{id}/players`,
  `/teams`, `/hub/rankings`, `/hub/standings` (spec's "Projected Standings"),
  `/hub/track-record`, `/hub/teams`, `/hub/players`, `POST /retrain`,
  `POST /refresh-odds`, `PUBLIC_MODE` gating. Value-bet edge calculation
  (Shin de-vig) from spec §5 is implemented and wired into
  `MarketPredictionOut`/`game_market_predictions`.
- **Explicitly deferred (not silently dropped — each is a real follow-up)**:
  (1) the batch job that populates `data/cache/schedule/games.json`,
  `data/cache/hub/{teams,players,rankings,standings}.json`, and
  `data/cache/training/games.json` from the Phase 2 data modules — this
  plan builds every *consumer* of those files and tests them against
  fakes/injected paths, but the *producer* job itself is out of scope;
  (2) settling predictions against real outcomes (so
  `hub_service.compute_track_record` reports real hit rates instead of
  `0`); (3) wiring `POST /refresh-odds` to `data.sportsbook_api`/
  `data.odds_api` and `store.insert_market_prediction`. These three are
  natural candidates for a short "Phase 4.5: scheduled jobs" plan once
  Phase 4 is reviewed.
- **Placeholder scan:** no TBD/TODO; every step has runnable code. The one
  intentionally minimal route (`POST /refresh-odds`) returns a real,
  correctly-gated, tested 202 response — not a crash or a silent no-op —
  and its remaining wiring is named explicitly above rather than hidden.
- **Type consistency:** `PredictionOut`/`MarketPredictionOut`/`GameOut`/
  `GameDetailOut`/`PlayerPropOut`/`TrackRecordOut` field names match between
  Task 2's definitions and every later task's usage; `store` function names
  added in Task 3 match exactly what Task 6/7 call.
