# Full Fixture Experience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Games page default to a week that actually has games (the season's first week, not the real-world calendar week), and expand the game-detail modal into a full fixture view — spread and total lines alongside moneyline, bookmaker/odds columns, head-to-head history, and recent form — matching the depth of PL_Predictor's `FixtureModal` within what NBA_Predictor's real data actually supports.

**Architecture:** This is almost entirely additive to code that already exists and works: `GET /games/week`, `GET /games/{id}`, and a modal that already fetches prediction + markets + player props. The gaps are (1) the page picks "today's" Monday as the default week, which is empty because the loaded schedule is a specific past/future season and "today" per the system clock rarely falls inside it; (2) only the `h2h` (moneyline) market is ever fetched/stored — spread and total are available from the same sportsbook API call but never wired up, and there's no `point` (line value) column to store them; (3) the modal never renders `bookmaker` or `american_odds` even though the schema already carries them; (4) there's no head-to-head or recent-form context, and no final-score treatment for completed games; (5) `player_name` in `/games/{id}/players` is a copy-paste bug — it's actually `player_id` again.

We are **not** building PL_Predictor's post-match verdict review, CLV tracking, or scoreline heatmap — those need data this pipeline doesn't collect (historical odds-timing snapshots, reported box-score stats keyed the same way pre/post match). Extending to those is a real follow-up, not something to fake here.

**Tech Stack:** FastAPI + Pydantic v2, raw sqlite3, pandas/scipy (already used for Shin de-vig), React 19 + TypeScript + Vite, vitest + Testing Library, pytest.

**Spec:** No separate spec doc — this plan works directly off the current code, verified by reading it (see file:line references throughout). The original design doc is `docs/superpowers/specs/2026-09-15-nba-predictor-design.md`.

## Global Constraints

- Every new/changed backend function needs a pytest test in the matching `tests/test_*.py` file (existing convention: one test file per module, e.g. `tests/test_pipeline_refresh_odds.py` for `src/nba_predictor/pipeline/refresh_odds.py`).
- Every new/changed frontend component needs a vitest + Testing Library test in its co-located `*.test.tsx`/`*.test.ts` file.
- No fabricated data: every new field must come from a real computation over real stored/cached data (schedule cache, tracking DB, sportsbook API, model MAE from `models/manifest.json`). Where a simplifying statistical assumption is used (e.g. normal-approximation cover probability), document it in a code comment the same way `pipeline/ingest.py`'s player rating/usage_rate already does.
- Follow existing code style: `Path`-typed args, keyword-only params after `*`, FastAPI `Depends()` for DB path/schedule, Pydantic models in `api/schemas.py`, camelCase-free TypeScript matching `frontend/src/api/client.ts`'s existing interfaces.
- Run `pytest` (backend) and `npm test` inside `frontend/` (frontend) after every task — both suites must pass before moving to the next task.

---

### Task 1: Normal-approximation cover-probability helper for spread/total

**Files:**
- Modify: `src/nba_predictor/odds/value_bets.py`
- Test: `tests/test_odds_value_bets.py`

**Interfaces:**
- Consumes: nothing new (pure math, uses `math.erf`/`math.sqrt` from stdlib, no new deps).
- Produces: `normal_cover_probability(mean: float, line: float, std: float) -> float` and `std_from_mae(mae: float) -> float`, both importable from `nba_predictor.odds.value_bets`. Task 3 imports and calls both.

The model doesn't output a distribution for margin/total, only a point estimate (`predicted_margin`, `predicted_total`) plus a stored MAE (`models/manifest.json` → `metrics.margin.mae`, `metrics.total.mae`). Treating the residual as approximately normal with mean 0, `std_from_mae` converts an MAE into the matching normal std (`E[|X|] = std * sqrt(2/pi)` for `X ~ N(0, std)`, so `std = MAE * sqrt(pi/2)`). `normal_cover_probability` then gives `P(actual > line)` under `actual ~ N(mean, std)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_odds_value_bets.py (append)
import math


def test_std_from_mae_converts_via_half_normal_relation():
    from nba_predictor.odds.value_bets import std_from_mae

    mae = 10.0
    std = std_from_mae(mae)

    assert std == pytest.approx(mae * math.sqrt(math.pi / 2))


def test_normal_cover_probability_is_half_when_mean_equals_line():
    from nba_predictor.odds.value_bets import normal_cover_probability

    assert normal_cover_probability(mean=5.0, line=5.0, std=10.0) == pytest.approx(0.5)


def test_normal_cover_probability_increases_with_higher_mean():
    from nba_predictor.odds.value_bets import normal_cover_probability

    low = normal_cover_probability(mean=1.0, line=5.0, std=10.0)
    high = normal_cover_probability(mean=9.0, line=5.0, std=10.0)

    assert high > low
    assert 0.0 < low < 1.0
    assert 0.0 < high < 1.0


def test_normal_cover_probability_zero_std_is_a_step_function():
    from nba_predictor.odds.value_bets import normal_cover_probability

    assert normal_cover_probability(mean=6.0, line=5.0, std=0.0) == 1.0
    assert normal_cover_probability(mean=4.0, line=5.0, std=0.0) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_odds_value_bets.py -v -k "std_from_mae or normal_cover_probability"`
Expected: FAIL with `ImportError` / `AttributeError` — the functions don't exist yet.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/odds/value_bets.py
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


def std_from_mae(mae: float) -> float:
    """Converts a model's mean absolute error into the std of the matching
    normal distribution, assuming residuals are approximately N(0, std):
    E[|X|] = std * sqrt(2/pi) for X ~ N(0, std), so std = mae * sqrt(pi/2).
    This is a simplifying approximation (no true residual distribution is
    stored), used only to turn a point prediction into a cover probability
    for spread/total markets — not a fabricated number, but an explicit
    statistical assumption over a real, stored MAE.
    """
    return mae * math.sqrt(math.pi / 2)


def normal_cover_probability(mean: float, line: float, std: float) -> float:
    """P(actual > line) assuming actual ~ Normal(mean, std)."""
    if std <= 0:
        return 1.0 if mean > line else 0.0
    z = (mean - line) / (std * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_odds_value_bets.py -v`
Expected: PASS (all tests, old and new)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/odds/value_bets.py tests/test_odds_value_bets.py
git commit -m "feat: add normal-approximation cover probability for spread/total markets"
```

---

### Task 2: Add a `point` (line value) column to `game_market_predictions`

**Files:**
- Modify: `src/nba_predictor/tracking/store.py`
- Test: `tests/test_tracking_store.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `insert_market_prediction(..., point: float | None = None, ...)` (new keyword-only param, defaults to `None` so every existing caller keeps working); rows returned by `get_market_predictions_for_game` now include a `point` column. Task 3 passes `point=` on every call; Task 4's `MarketPredictionOut` reads `row["point"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tracking_store.py (append)
def test_insert_market_prediction_stores_point_line_value(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    store.insert_market_prediction(
        db_path, game_id="g1", market="spread", selection="BOS", model_probability=0.55,
        market_probability=0.5, edge=0.05, bookmaker="DraftKings", american_odds=-110,
        point=-4.5, created_at="2026-11-01T12:00:00",
    )

    rows = store.get_market_predictions_for_game(db_path, "g1")
    assert rows[0]["point"] == pytest.approx(-4.5)


def test_insert_market_prediction_point_defaults_to_none(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="BOS", model_probability=0.55,
        market_probability=0.5, edge=0.05, bookmaker="DraftKings", american_odds=-110,
        created_at="2026-11-01T12:00:00",
    )

    rows = store.get_market_predictions_for_game(db_path, "g1")
    assert rows[0]["point"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tracking_store.py -v -k point`
Expected: FAIL with `sqlite3.OperationalError: table game_market_predictions has no column named point` or a `TypeError` on the unexpected keyword.

- [ ] **Step 3: Implement**

Add the column to `SCHEMA`, add a defensive `ALTER TABLE` migration for any DB file created before this change, and thread the new param through `insert_market_prediction`:

```python
# src/nba_predictor/tracking/store.py
# In SCHEMA, change the game_market_predictions CREATE TABLE to:
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
    point REAL,
    created_at TEXT NOT NULL
);
```

```python
def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        _ensure_point_column(conn)


def _ensure_point_column(conn: sqlite3.Connection) -> None:
    """Adds `point` to a game_market_predictions table created before this
    column existed. CREATE TABLE IF NOT EXISTS above won't add it to an
    already-existing table, so this migration covers any DB file left over
    from a prior deploy."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(game_market_predictions)")}
    if "point" not in columns:
        conn.execute("ALTER TABLE game_market_predictions ADD COLUMN point REAL")
        conn.commit()
```

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
    point: float | None = None,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO game_market_predictions
                (game_id, market, selection, model_probability, market_probability, edge, bookmaker, american_odds, point, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (game_id, market, selection, model_probability, market_probability, edge, bookmaker, american_odds, point, created_at),
        )
        conn.commit()
        return cur.lastrowid
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tracking_store.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/tracking/store.py tests/test_tracking_store.py
git commit -m "feat: store spread/total line value alongside market predictions"
```

---

### Task 3: Wire spread and total markets into `refresh_market_predictions`

**Files:**
- Modify: `src/nba_predictor/pipeline/refresh_odds.py`
- Test: `tests/test_pipeline_refresh_odds.py`

**Interfaces:**
- Consumes: `sportsbook_api.get_odds(event_key) -> list[dict]` (already returns `market`/`selection`/`bookmaker`/`american_odds`/`point` — confirmed in `src/nba_predictor/data/sportsbook_api.py:129-157`, no changes needed there); `store.get_latest_prediction_for_game` (existing, row has `home_win_prob`, `predicted_margin`, `predicted_total`); `store.insert_market_prediction(..., point=...)` from Task 2; `implied_probability`, `shin_devig`, `compute_edge`, `normal_cover_probability`, `std_from_mae` from Task 1/existing `odds/value_bets.py`.
- Produces: `refresh_market_predictions(schedule, db_path, margin_std: float = 12.0, total_std: float = 15.0) -> int` — same name/return type, two new optional keyword params with defaults so every existing call site (`api/routes.py:188`) keeps compiling; Task 4 passes explicit `margin_std`/`total_std` computed from the manifest.

The existing h2h-only behavior must not change (existing tests in `tests/test_pipeline_refresh_odds.py` assert exact `model_probability`/`stored` counts for h2h — they must still pass unmodified). Add spread and total handling alongside it in the same loop, grouped by `(market, bookmaker)` instead of just `bookmaker`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_refresh_odds.py (append)
@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.get_odds")
@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.fetch_nba_events_raw")
def test_refresh_market_predictions_stores_devigged_spread_rows(mock_events, mock_odds, tmp_path):
    from nba_predictor.pipeline.refresh_odds import refresh_market_predictions
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-10-30T00:00:00", model_version="v1",
        home_win_prob=0.62, predicted_margin=5.0, predicted_total=220.0,
    )

    mock_events.return_value = [_real_event()]
    mock_odds.return_value = [
        {"market": "spread", "selection": "BOS", "bookmaker": "DRAFT_KINGS", "american_odds": -110, "point": -4.5},
        {"market": "spread", "selection": "MIA", "bookmaker": "DRAFT_KINGS", "american_odds": -110, "point": 4.5},
    ]

    schedule = [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA", "completed": False}]

    stored = refresh_market_predictions(schedule, db_path, margin_std=10.0)

    assert stored == 2
    rows = store.get_market_predictions_for_game(db_path, "g1")
    by_selection = {r["selection"]: r for r in rows}
    assert by_selection["BOS"]["market"] == "spread"
    assert by_selection["BOS"]["point"] == pytest.approx(-4.5)
    # predicted_margin=5.0 > -point(4.5) => BOS is favored to cover, prob > 0.5
    assert by_selection["BOS"]["model_probability"] > 0.5
    assert by_selection["MIA"]["model_probability"] == pytest.approx(1 - by_selection["BOS"]["model_probability"])


@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.get_odds")
@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.fetch_nba_events_raw")
def test_refresh_market_predictions_stores_devigged_total_rows(mock_events, mock_odds, tmp_path):
    from nba_predictor.pipeline.refresh_odds import refresh_market_predictions
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-10-30T00:00:00", model_version="v1",
        home_win_prob=0.62, predicted_margin=5.0, predicted_total=230.0,
    )

    mock_events.return_value = [_real_event()]
    mock_odds.return_value = [
        {"market": "total", "selection": "over", "bookmaker": "DRAFT_KINGS", "american_odds": -110, "point": 220.5},
        {"market": "total", "selection": "under", "bookmaker": "DRAFT_KINGS", "american_odds": -110, "point": 220.5},
    ]

    schedule = [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA", "completed": False}]

    stored = refresh_market_predictions(schedule, db_path, total_std=10.0)

    assert stored == 2
    rows = store.get_market_predictions_for_game(db_path, "g1")
    by_selection = {r["selection"]: r for r in rows}
    assert by_selection["over"]["point"] == pytest.approx(220.5)
    # predicted_total=230.0 > line 220.5 => over favored, prob > 0.5
    assert by_selection["over"]["model_probability"] > 0.5
    assert by_selection["under"]["model_probability"] == pytest.approx(1 - by_selection["over"]["model_probability"])


@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.get_odds")
@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.fetch_nba_events_raw")
def test_refresh_market_predictions_skips_lopsided_market_groups(mock_events, mock_odds, tmp_path):
    """A market/bookmaker group with anything other than exactly 2 rows is skipped
    (mirrors the existing h2h behavior for len(rows) != 2)."""
    from nba_predictor.pipeline.refresh_odds import refresh_market_predictions
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-10-30T00:00:00", model_version="v1",
        home_win_prob=0.62, predicted_margin=5.0, predicted_total=230.0,
    )

    mock_events.return_value = [_real_event()]
    mock_odds.return_value = [
        {"market": "total", "selection": "over", "bookmaker": "DRAFT_KINGS", "american_odds": -110, "point": 220.5},
    ]
    schedule = [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA", "completed": False}]

    assert refresh_market_predictions(schedule, db_path) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline_refresh_odds.py -v -k "spread_rows or total_rows or lopsided"`
Expected: FAIL — spread/total rows are currently dropped (only `market == "h2h"` rows are ever grouped), so `stored == 0` for the new tests.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/pipeline/refresh_odds.py
from datetime import datetime, timezone
from pathlib import Path

from nba_predictor.data import sportsbook_api
from nba_predictor.data.team_reference import TEAMS
from nba_predictor.odds.value_bets import compute_edge, implied_probability, normal_cover_probability, shin_devig
from nba_predictor.tracking import store

_NAME_TO_ABBREVIATION = {team.name: team.abbreviation for team in TEAMS}


def match_schedule_to_sportsbook_events(schedule: list[dict], events: list[dict]) -> dict[str, str]:
    """Maps our game_id -> sportsbook event key, matched by (home, away, date)."""
    by_matchup: dict[tuple[str, str, str], str] = {}
    for event in events:
        participants = event.get("participants", [])
        home_key = event.get("homeParticipantKey")
        if len(participants) != 2 or home_key is None:
            continue
        home = next((p for p in participants if p["key"] == home_key), None)
        away = next((p for p in participants if p["key"] != home_key), None)
        if home is None or away is None:
            continue
        home_abbr = _NAME_TO_ABBREVIATION.get(home.get("name"))
        away_abbr = _NAME_TO_ABBREVIATION.get(away.get("name"))
        if home_abbr is None or away_abbr is None:
            continue
        event_date = event.get("startTime", "")[:10]
        by_matchup[(home_abbr, away_abbr, event_date)] = event["key"]

    mapping = {}
    for game in schedule:
        key = (game["home_team"], game["away_team"], game["game_date"])
        if key in by_matchup:
            mapping[game["game_id"]] = by_matchup[key]
    return mapping


def _model_probability(
    market: str,
    selection: str,
    point: float | None,
    game: dict,
    prediction,
    margin_std: float,
    total_std: float,
) -> float | None:
    """Model's probability that `selection` wins its side of `market`.

    h2h: the model's stored win probability, flipped for the away side.
    spread: treats the model's predicted_margin as the mean of a normal
    distribution over the actual game margin, and asks whether the selected
    team's margin clears the negative of its own posted line (standard
    "covers" convention — a team posted at -4.5 covers if its margin > 4.5,
    a team posted at +4.5 covers if its margin > -4.5).
    total: same idea against predicted_total, over/under a single line.
    """
    if market == "h2h":
        return prediction["home_win_prob"] if selection == game["home_team"] else 1 - prediction["home_win_prob"]

    if market == "spread":
        if point is None:
            return None
        cover_mean = prediction["predicted_margin"] if selection == game["home_team"] else -prediction["predicted_margin"]
        return normal_cover_probability(mean=cover_mean, line=-point, std=margin_std)

    if market == "total":
        if point is None:
            return None
        over_probability = normal_cover_probability(mean=prediction["predicted_total"], line=point, std=total_std)
        return over_probability if selection == "over" else 1 - over_probability

    return None


def refresh_market_predictions(
    schedule: list[dict],
    db_path: Path,
    margin_std: float = 12.0,
    total_std: float = 15.0,
) -> int:
    """Fetches live odds for upcoming (not completed) scheduled games, de-vigs
    each bookmaker's own market against the model's stored prediction (Shin's
    method for the market side, a normal-approximation cover probability for
    the model side on spread/total), and stores value-bet rows per market
    (h2h, spread, total). Returns the number of market-prediction rows
    written.

    margin_std/total_std default to fixed fallbacks but are meant to be
    passed in by the caller from the trained model's own MAE (see
    api/routes.py's refresh_odds route), via odds.value_bets.std_from_mae.
    """
    upcoming = [g for g in schedule if not g.get("completed")]
    if not upcoming:
        return 0

    events = sportsbook_api.fetch_nba_events_raw()
    event_by_game = match_schedule_to_sportsbook_events(upcoming, events)
    if not event_by_game:
        return 0

    created_at = datetime.now(timezone.utc).isoformat()
    stored = 0

    for game in upcoming:
        event_key = event_by_game.get(game["game_id"])
        if event_key is None:
            continue

        prediction = store.get_latest_prediction_for_game(db_path, game["game_id"])
        if prediction is None:
            continue

        odds_rows = sportsbook_api.get_odds(event_key)
        rows_by_market_bookmaker: dict[tuple[str, str], list[dict]] = {}
        for row in odds_rows:
            rows_by_market_bookmaker.setdefault((row["market"], row["bookmaker"]), []).append(row)

        for (market, bookmaker), rows in rows_by_market_bookmaker.items():
            if len(rows) != 2:
                continue

            model_probs = [
                _model_probability(market, row["selection"], row.get("point"), game, prediction, margin_std, total_std)
                for row in rows
            ]
            if any(p is None for p in model_probs):
                continue

            raw_probs = [implied_probability(row["american_odds"]) for row in rows]
            fair_probs = shin_devig(raw_probs)

            for row, model_prob, fair_prob in zip(rows, model_probs, fair_probs):
                store.insert_market_prediction(
                    db_path,
                    game_id=game["game_id"],
                    market=market,
                    selection=row["selection"],
                    model_probability=model_prob,
                    market_probability=fair_prob,
                    edge=compute_edge(model_prob, fair_prob),
                    bookmaker=bookmaker,
                    american_odds=row["american_odds"],
                    point=row.get("point"),
                    created_at=created_at,
                )
                stored += 1

    return stored
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline_refresh_odds.py -v`
Expected: PASS (all tests, including the pre-existing h2h ones — confirm `test_refresh_market_predictions_stores_devigged_h2h_rows` still passes with `model_probability == 0.62`/`0.38` exactly, since `_model_probability` for h2h is unchanged logic)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/pipeline/refresh_odds.py tests/test_pipeline_refresh_odds.py
git commit -m "feat: store spread and total market predictions alongside h2h"
```

---

### Task 4: Expose `point` on the API and derive spread/total std from the model's real MAE

**Files:**
- Modify: `src/nba_predictor/api/schemas.py`
- Modify: `src/nba_predictor/api/routes.py`
- Test: `tests/test_api_games.py`, `tests/test_api_admin.py`

**Interfaces:**
- Consumes: `MarketPredictionOut` (existing, extend with `point`); `store.get_market_predictions_for_game` rows now have `row["point"]` (Task 2); `refresh_market_predictions(schedule, db_path, margin_std, total_std)` (Task 3); `get_models_dir` dependency (existing, `api/deps.py:24`).
- Produces: `MarketPredictionOut.point: float | None`; `GET /games/{game_id}` responses include `"point"` per market row; `POST /refresh-odds` now reads `models/manifest.json` and passes `margin_std=std_from_mae(mae["margin"])`, `total_std=std_from_mae(mae["total"])` into `refresh_market_predictions`, falling back to the function's own defaults if the manifest or those keys are missing.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api_games.py (append)
def test_get_game_detail_includes_point_for_spread_market(tmp_path, monkeypatch):
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_market_prediction(
        db_path, game_id="g1", market="spread", selection="BOS", model_probability=0.55,
        market_probability=0.5, edge=0.05, bookmaker="DraftKings", american_odds=-110,
        point=-4.5, created_at="2026-11-01T12:00:00",
    )

    response = client.get("/games/g1")
    assert response.status_code == 200
    assert response.json()["markets"][0]["point"] == -4.5


def test_get_game_detail_point_is_null_for_h2h(tmp_path, monkeypatch):
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="BOS", model_probability=0.62,
        market_probability=0.55, edge=0.07, bookmaker="DraftKings", american_odds=-130,
        created_at="2026-11-01T12:00:00",
    )

    response = client.get("/games/g1")
    assert response.json()["markets"][0]["point"] is None
```

```python
# tests/test_api_admin.py — read this file first to match its existing
# _client_with_overrides / admin-header pattern, then append:
from unittest.mock import patch


@patch("nba_predictor.api.routes.refresh_market_predictions")
def test_refresh_odds_reads_std_from_manifest_mae(mock_refresh, tmp_path, monkeypatch):
    import json

    from nba_predictor.api.app import app
    from nba_predictor.api import deps
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    schedule_path = tmp_path / "games.json"
    schedule_path.write_text("[]")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "manifest.json").write_text(json.dumps({
        "metrics": {"margin": {"mae": 10.0}, "total": {"mae": 20.0}},
    }))

    monkeypatch.setenv("ADMIN_API_KEY", "secret")
    from nba_predictor import config
    monkeypatch.setattr(config, "PUBLIC_MODE", False)

    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    app.dependency_overrides[deps.get_models_dir] = lambda: models_dir
    mock_refresh.return_value = 0

    from fastapi.testclient import TestClient
    client = TestClient(app)
    client.post("/refresh-odds")

    import math
    _, kwargs = mock_refresh.call_args
    assert kwargs["margin_std"] == pytest.approx(10.0 * math.sqrt(math.pi / 2))
    assert kwargs["total_std"] == pytest.approx(20.0 * math.sqrt(math.pi / 2))


@patch("nba_predictor.api.routes.refresh_market_predictions")
def test_refresh_odds_falls_back_to_defaults_without_manifest(mock_refresh, tmp_path, monkeypatch):
    from nba_predictor.api.app import app
    from nba_predictor.api import deps
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    schedule_path = tmp_path / "games.json"
    schedule_path.write_text("[]")
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    from nba_predictor import config
    monkeypatch.setattr(config, "PUBLIC_MODE", False)

    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    app.dependency_overrides[deps.get_models_dir] = lambda: models_dir
    mock_refresh.return_value = 0

    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.post("/refresh-odds")

    assert response.status_code == 202
    _, kwargs = mock_refresh.call_args
    assert kwargs["margin_std"] == 12.0
    assert kwargs["total_std"] == 15.0
```

Note: read `tests/test_api_admin.py`'s existing setup helper before writing these — reuse whatever fixture/header pattern it already has for admin auth instead of duplicating `monkeypatch.setenv`/`PUBLIC_MODE` wiring if a shared helper already exists.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_games.py tests/test_api_admin.py -v -k "point or manifest_mae or fallback"`
Expected: FAIL — `point` key doesn't exist on the response yet; `refresh_odds` doesn't read the manifest yet.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/api/schemas.py — extend MarketPredictionOut
class MarketPredictionOut(BaseModel):
    market: str
    selection: str
    model_probability: float
    market_probability: float | None = None
    edge: float | None = None
    bookmaker: str | None = None
    american_odds: int | None = None
    point: float | None = None
```

```python
# src/nba_predictor/api/routes.py
# 1. Update the import:
from nba_predictor.odds.value_bets import std_from_mae

# 2. Update get_game_detail's markets list comprehension to add point=row["point"]:
    markets = [
        MarketPredictionOut(
            market=row["market"], selection=row["selection"], model_probability=row["model_probability"],
            market_probability=row["market_probability"], edge=row["edge"], bookmaker=row["bookmaker"],
            american_odds=row["american_odds"], point=row["point"],
        )
        for row in store.get_market_predictions_for_game(db_path, game_id)
    ]

# 3. Replace the refresh_odds route:
@router.post("/refresh-odds", dependencies=[Depends(require_admin)], status_code=202)
def refresh_odds(
    schedule: list[dict] = Depends(get_schedule),
    db_path: Path = Depends(get_db_path),
    models_dir: Path = Depends(get_models_dir),
) -> dict:
    margin_std, total_std = 12.0, 15.0
    manifest_path = models_dir / "manifest.json"
    if manifest_path.exists():
        metrics = json.loads(manifest_path.read_text()).get("metrics", {})
        margin_mae = metrics.get("margin", {}).get("mae")
        total_mae = metrics.get("total", {}).get("mae")
        if margin_mae is not None:
            margin_std = std_from_mae(margin_mae)
        if total_mae is not None:
            total_std = std_from_mae(total_mae)

    stored = refresh_market_predictions(schedule, db_path, margin_std=margin_std, total_std=total_std)
    return {"status": "ok", "market_predictions_stored": stored}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api_games.py tests/test_api_admin.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/api/schemas.py src/nba_predictor/api/routes.py tests/test_api_games.py tests/test_api_admin.py
git commit -m "feat: expose market line value on the API, derive spread/total std from model MAE"
```

---

### Task 5: Head-to-head history and recent form in `schedule_repository`

**Files:**
- Modify: `src/nba_predictor/services/schedule_repository.py`
- Test: `tests/test_schedule_repository.py`

**Interfaces:**
- Consumes: schedule cache entries (`dict` with `game_id`, `game_date`, `home_team`, `away_team`, `completed`, `home_pts`, `away_pts` — confirmed shape from `data/cache/schedule/games.json`).
- Produces: `get_head_to_head(schedule: list[dict], team_a: str, team_b: str, before_date: str, limit: int = 5) -> list[dict]` and `get_recent_form(schedule: list[dict], team: str, before_date: str, limit: int = 5) -> list[str]` (each element `"W"` or `"L"`, most recent first). Task 6 calls both from `routes.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schedule_repository.py (append)
def _completed_game(game_id, date, home, away, home_pts, away_pts):
    return {
        "game_id": game_id, "game_date": date, "home_team": home, "away_team": away,
        "completed": True, "home_pts": home_pts, "away_pts": away_pts,
    }


def test_get_head_to_head_returns_prior_meetings_sorted_most_recent_first():
    from nba_predictor.services.schedule_repository import get_head_to_head

    schedule = [
        _completed_game("g1", "2026-01-01", "BOS", "MIA", 110, 100),
        _completed_game("g2", "2026-02-01", "MIA", "BOS", 95, 105),
        _completed_game("g3", "2026-03-01", "BOS", "LAL", 120, 100),  # different matchup
        _completed_game("g4", "2026-04-01", "BOS", "MIA", 90, 92),
    ]

    result = get_head_to_head(schedule, "BOS", "MIA", before_date="2026-04-01", limit=5)

    assert [g["game_id"] for g in result] == ["g2", "g1"]


def test_get_head_to_head_only_counts_completed_games():
    from nba_predictor.services.schedule_repository import get_head_to_head

    schedule = [
        {"game_id": "g1", "game_date": "2026-01-01", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None},
    ]

    assert get_head_to_head(schedule, "BOS", "MIA", before_date="2026-02-01") == []


def test_get_head_to_head_respects_limit():
    from nba_predictor.services.schedule_repository import get_head_to_head

    schedule = [
        _completed_game(f"g{i}", f"2026-0{i}-01", "BOS", "MIA", 100 + i, 90 + i) for i in range(1, 6)
    ]

    result = get_head_to_head(schedule, "BOS", "MIA", before_date="2026-06-01", limit=2)
    assert len(result) == 2


def test_get_recent_form_returns_win_loss_letters_most_recent_first():
    from nba_predictor.services.schedule_repository import get_recent_form

    schedule = [
        _completed_game("g1", "2026-01-01", "BOS", "MIA", 110, 100),  # BOS won (home)
        _completed_game("g2", "2026-01-05", "LAL", "BOS", 100, 90),   # BOS lost (away)
        _completed_game("g3", "2026-01-10", "BOS", "DEN", 88, 95),    # BOS lost (home)
    ]

    result = get_recent_form(schedule, "BOS", before_date="2026-01-15", limit=5)
    assert result == ["L", "L", "W"]


def test_get_recent_form_excludes_games_on_or_after_before_date():
    from nba_predictor.services.schedule_repository import get_recent_form

    schedule = [
        _completed_game("g1", "2026-01-01", "BOS", "MIA", 110, 100),
        _completed_game("g2", "2026-01-10", "BOS", "MIA", 90, 100),
    ]

    assert get_recent_form(schedule, "BOS", before_date="2026-01-10", limit=5) == ["W"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schedule_repository.py -v -k "head_to_head or recent_form"`
Expected: FAIL with `ImportError` — the functions don't exist yet.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/services/schedule_repository.py (append)
def get_head_to_head(schedule: list[dict], team_a: str, team_b: str, before_date: str, limit: int = 5) -> list[dict]:
    """Prior completed meetings between team_a and team_b, strictly before
    before_date, most recent first."""
    matches = [
        game
        for game in schedule
        if game.get("completed")
        and game["game_date"] < before_date
        and {game["home_team"], game["away_team"]} == {team_a, team_b}
    ]
    matches.sort(key=lambda g: g["game_date"], reverse=True)
    return matches[:limit]


def get_recent_form(schedule: list[dict], team: str, before_date: str, limit: int = 5) -> list[str]:
    """Team's last `limit` completed results strictly before before_date, as
    "W"/"L", most recent first."""
    games = [
        game
        for game in schedule
        if game.get("completed")
        and game["game_date"] < before_date
        and team in (game["home_team"], game["away_team"])
    ]
    games.sort(key=lambda g: g["game_date"], reverse=True)

    results = []
    for game in games[:limit]:
        is_home = game["home_team"] == team
        team_pts = game["home_pts"] if is_home else game["away_pts"]
        opp_pts = game["away_pts"] if is_home else game["home_pts"]
        results.append("W" if team_pts > opp_pts else "L")
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schedule_repository.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/services/schedule_repository.py tests/test_schedule_repository.py
git commit -m "feat: add head-to-head history and recent form lookups"
```

---

### Task 6: Wire head-to-head and recent form into `GET /games/{game_id}`

**Files:**
- Modify: `src/nba_predictor/api/schemas.py`
- Modify: `src/nba_predictor/api/routes.py`
- Test: `tests/test_api_games.py`

**Interfaces:**
- Consumes: `get_head_to_head`, `get_recent_form` (Task 5).
- Produces: `HeadToHeadMeetingOut` schema; `GameDetailOut.head_to_head: list[HeadToHeadMeetingOut]`, `GameDetailOut.home_recent_form: list[str]`, `GameDetailOut.away_recent_form: list[str]`. Task 9's frontend `GameDetail` TS interface mirrors these three fields exactly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_games.py (append)
def test_get_game_detail_includes_head_to_head_and_recent_form(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import json

    from nba_predictor.api.app import app
    from nba_predictor.api import deps
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(json.dumps([
        {"game_id": "g0", "game_date": "2026-01-01", "home_team": "BOS", "away_team": "MIA", "completed": True, "home_pts": 110, "away_pts": 100},
        {"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None},
    ]))

    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path

    client = TestClient(app)
    response = client.get("/games/g1")

    assert response.status_code == 200
    body = response.json()
    assert body["head_to_head"] == [
        {"game_id": "g0", "game_date": "2026-01-01", "home_team": "BOS", "away_team": "MIA", "home_pts": 110, "away_pts": 100}
    ]
    assert body["home_recent_form"] == ["W"]
    assert body["away_recent_form"] == ["L"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_games.py -v -k head_to_head_and_recent_form`
Expected: FAIL with a `KeyError`/`pydantic.ValidationError` — `head_to_head` isn't a field on the response yet.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/api/schemas.py (append, and extend GameDetailOut)
class HeadToHeadMeetingOut(BaseModel):
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    home_pts: int | None = None
    away_pts: int | None = None


class GameDetailOut(GameOut):
    markets: list[MarketPredictionOut] = []
    head_to_head: list[HeadToHeadMeetingOut] = []
    home_recent_form: list[str] = []
    away_recent_form: list[str] = []
```

```python
# src/nba_predictor/api/routes.py
# Update the import:
from nba_predictor.api.schemas import (
    GameDetailOut,
    GameOut,
    HeadToHeadMeetingOut,
    MarketPredictionOut,
    PlayerPropOut,
    PredictionOut,
    TeamOut,
    TrackRecordOut,
)
from nba_predictor.services.schedule_repository import get_game, get_games_for_date, get_games_for_week, get_head_to_head, get_recent_form

# Replace get_game_detail:
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
            american_odds=row["american_odds"], point=row["point"],
        )
        for row in store.get_market_predictions_for_game(db_path, game_id)
    ]

    head_to_head = [
        HeadToHeadMeetingOut(
            game_id=g["game_id"], game_date=g["game_date"], home_team=g["home_team"], away_team=g["away_team"],
            home_pts=g.get("home_pts"), away_pts=g.get("away_pts"),
        )
        for g in get_head_to_head(schedule, game["home_team"], game["away_team"], before_date=game["game_date"])
    ]

    base = _game_out(game, db_path)
    return GameDetailOut(
        **base.model_dump(),
        markets=markets,
        head_to_head=head_to_head,
        home_recent_form=get_recent_form(schedule, game["home_team"], before_date=game["game_date"]),
        away_recent_form=get_recent_form(schedule, game["away_team"], before_date=game["game_date"]),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_games.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/api/schemas.py src/nba_predictor/api/routes.py tests/test_api_games.py
git commit -m "feat: include head-to-head history and recent form in game detail"
```

---

### Task 7: Fix the `player_name` bug in `/games/{id}/players`

**Files:**
- Modify: `src/nba_predictor/services/hub_service.py`
- Modify: `src/nba_predictor/api/routes.py`
- Test: `tests/test_hub_service.py`, `tests/test_api_games.py`

**Interfaces:**
- Consumes: `data/cache/hub/players.json` (existing cache, confirmed shape: `{"player_id": "4251", "player_name": "Paul George", ...}`).
- Produces: `load_player_name_map(path: Path) -> dict[str, str]` in `hub_service.py`, mapping `player_id -> player_name`. Task 12's frontend player list gets real names instead of the raw ID string.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hub_service.py (append)
def test_load_player_name_map_builds_id_to_name_dict(tmp_path):
    import json

    from nba_predictor.services.hub_service import load_player_name_map

    path = tmp_path / "players.json"
    path.write_text(json.dumps([
        {"player_id": "4251", "player_name": "Paul George"},
        {"player_id": "203999", "player_name": "Nikola Jokic"},
    ]))

    assert load_player_name_map(path) == {"4251": "Paul George", "203999": "Nikola Jokic"}


def test_load_player_name_map_missing_file_returns_empty_dict(tmp_path):
    from nba_predictor.services.hub_service import load_player_name_map

    assert load_player_name_map(tmp_path / "does-not-exist.json") == {}
```

```python
# tests/test_api_games.py (append)
def test_get_game_players_uses_real_player_name_from_hub_cache(tmp_path, monkeypatch):
    import json

    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app
    from nba_predictor.api import deps
    from nba_predictor.tracking import store
    from nba_predictor import config

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="203999", stat="points", predicted_value=27.5, created_at="2026-11-01T12:00:00",
    )

    schedule_path = tmp_path / "games.json"
    schedule_path.write_text('[{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}]')

    hub_players_path = tmp_path / "players.json"
    hub_players_path.write_text(json.dumps([{"player_id": "203999", "player_name": "Nikola Jokic"}]))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    (tmp_path / "cache" / "hub").mkdir(parents=True)
    hub_players_path.rename(tmp_path / "cache" / "hub" / "players.json")

    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path

    client = TestClient(app)
    response = client.get("/games/g1/players")

    assert response.json()[0]["player_name"] == "Nikola Jokic"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hub_service.py tests/test_api_games.py -v -k "player_name"`
Expected: FAIL — `load_player_name_map` doesn't exist; the route still echoes `player_id`.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/services/hub_service.py (append, near load_hub_cache)
def load_player_name_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {row["player_id"]: row["player_name"] for row in rows}
```

```python
# src/nba_predictor/api/routes.py
# Update import:
from nba_predictor.services.hub_service import compute_track_record, load_hub_cache, load_player_name_map

# Replace get_game_players:
@router.get("/games/{game_id}/players", response_model=list[PlayerPropOut])
def get_game_players(
    game_id: str, schedule: list[dict] = Depends(get_schedule), db_path: Path = Depends(get_db_path)
) -> list[PlayerPropOut]:
    game = get_game(schedule, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_id}")

    name_by_id = load_player_name_map(config.DATA_DIR / "cache" / "hub" / "players.json")

    return [
        PlayerPropOut(
            player_id=row["player_id"], player_name=name_by_id.get(row["player_id"], row["player_id"]),
            stat=row["stat"], predicted_value=row["predicted_value"],
        )
        for row in store.get_player_predictions_for_game(db_path, game_id)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hub_service.py tests/test_api_games.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/services/hub_service.py src/nba_predictor/api/routes.py tests/test_hub_service.py tests/test_api_games.py
git commit -m "fix: resolve real player names in game player-props endpoint"
```

---

### Task 8: `GET /season/first-week` — a week that actually has games

**Files:**
- Modify: `src/nba_predictor/services/schedule_repository.py`
- Modify: `src/nba_predictor/api/routes.py`
- Modify: `src/nba_predictor/api/schemas.py`
- Test: `tests/test_schedule_repository.py`, `tests/test_api_games.py`

**Interfaces:**
- Consumes: schedule cache (`game_date` strings, sortable as ISO 8601).
- Produces: `first_week_start(schedule: list[dict]) -> str | None` (Monday of the earliest `game_date` in the schedule, or `None` for an empty schedule); `SeasonBoundsOut` schema `{first_week_start: str | None}`; `GET /season/first-week` route. Task 9/10's frontend calls this to pick the default week.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schedule_repository.py (append)
def test_first_week_start_returns_monday_of_earliest_game():
    from nba_predictor.services.schedule_repository import first_week_start

    schedule = [
        {"game_id": "g2", "game_date": "2025-10-22"},
        {"game_id": "g1", "game_date": "2025-10-21"},  # a Tuesday
    ]

    assert first_week_start(schedule) == "2025-10-20"  # the preceding Monday


def test_first_week_start_empty_schedule_returns_none():
    from nba_predictor.services.schedule_repository import first_week_start

    assert first_week_start([]) is None
```

```python
# tests/test_api_games.py (append)
def test_season_first_week_returns_monday_of_earliest_game(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import json

    from nba_predictor.api.app import app
    from nba_predictor.api import deps

    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(json.dumps([{"game_id": "g1", "game_date": "2025-10-21", "home_team": "BOS", "away_team": "MIA"}]))
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path

    client = TestClient(app)
    response = client.get("/season/first-week")

    assert response.status_code == 200
    assert response.json() == {"first_week_start": "2025-10-20"}


def test_season_first_week_null_for_empty_schedule(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app
    from nba_predictor.api import deps

    schedule_path = tmp_path / "games.json"
    schedule_path.write_text("[]")
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path

    client = TestClient(app)
    response = client.get("/season/first-week")

    assert response.json() == {"first_week_start": None}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schedule_repository.py tests/test_api_games.py -v -k "first_week"`
Expected: FAIL — `first_week_start` and `/season/first-week` don't exist yet.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/services/schedule_repository.py (append)
def first_week_start(schedule: list[dict]) -> str | None:
    """The Monday of the week containing the earliest game_date in the
    schedule, or None if the schedule is empty."""
    if not schedule:
        return None
    earliest = min(game["game_date"] for game in schedule)
    return monday_of(earliest)
```

```python
# src/nba_predictor/api/schemas.py (append)
class SeasonBoundsOut(BaseModel):
    first_week_start: str | None = None
```

```python
# src/nba_predictor/api/routes.py
# Update imports:
from nba_predictor.api.schemas import (
    GameDetailOut,
    GameOut,
    HeadToHeadMeetingOut,
    MarketPredictionOut,
    PlayerPropOut,
    PredictionOut,
    SeasonBoundsOut,
    TeamOut,
    TrackRecordOut,
)
from nba_predictor.services.schedule_repository import (
    first_week_start,
    get_game,
    get_games_for_date,
    get_games_for_week,
    get_head_to_head,
    get_recent_form,
)

# New route (place it near the other /games routes):
@router.get("/season/first-week", response_model=SeasonBoundsOut)
def season_first_week(schedule: list[dict] = Depends(get_schedule)) -> SeasonBoundsOut:
    return SeasonBoundsOut(first_week_start=first_week_start(schedule))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schedule_repository.py tests/test_api_games.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/services/schedule_repository.py src/nba_predictor/api/routes.py src/nba_predictor/api/schemas.py tests/test_schedule_repository.py tests/test_api_games.py
git commit -m "feat: add /season/first-week so the games page can default to a week with games"
```

---

### Task 9: Frontend API client — new types and calls

**Files:**
- Modify: `frontend/src/api/client.ts`
- Test: `frontend/src/api/client.test.ts`

**Interfaces:**
- Consumes: the JSON shapes produced by Tasks 4, 6, 8 (`point` on `MarketPrediction`, `head_to_head`/`home_recent_form`/`away_recent_form` on `GameDetail`, `SeasonBounds` from `/season/first-week`).
- Produces: updated `MarketPrediction`, `GameDetail` interfaces; new `HeadToHeadMeeting`, `SeasonBounds` interfaces; `api.getSeasonFirstWeek(): Promise<SeasonBounds>`. Tasks 10 and 12 import these directly.

- [ ] **Step 1: Write the failing test**

```typescript
// frontend/src/api/client.test.ts (append)
it("getSeasonFirstWeek fetches /season/first-week", async () => {
  mockFetchOnce({ first_week_start: "2025-10-20" });

  const bounds = await api.getSeasonFirstWeek();

  expect(fetch).toHaveBeenCalledWith(expect.stringContaining("/season/first-week"));
  expect(bounds.first_week_start).toBe("2025-10-20");
});

it("getGameDetail response includes head_to_head and recent form fields", async () => {
  mockFetchOnce({
    game_id: "g1", game_date: "2026-11-01", home_team: "BOS", away_team: "MIA",
    prediction: null, markets: [], head_to_head: [], home_recent_form: ["W"], away_recent_form: ["L"],
  });

  const detail = await api.getGameDetail("g1");

  expect(detail.home_recent_form).toEqual(["W"]);
  expect(detail.away_recent_form).toEqual(["L"]);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- --run client.test.ts`
Expected: FAIL — `api.getSeasonFirstWeek` is not a function; `GameDetail` type doesn't have those fields (TypeScript compile error surfaces as a test failure via vitest's esbuild transform, or the runtime property is simply `undefined` and the second assertion fails).

- [ ] **Step 3: Implement**

```typescript
// frontend/src/api/client.ts
// Extend MarketPrediction:
export interface MarketPrediction {
  market: string;
  selection: string;
  model_probability: number;
  market_probability: number | null;
  edge: number | null;
  bookmaker: string | null;
  american_odds: number | null;
  point: number | null;
}

// Add near Game/GameDetail:
export interface HeadToHeadMeeting {
  game_id: string;
  game_date: string;
  home_team: string;
  away_team: string;
  home_pts: number | null;
  away_pts: number | null;
}

export interface GameDetail extends Game {
  markets: MarketPrediction[];
  head_to_head: HeadToHeadMeeting[];
  home_recent_form: string[];
  away_recent_form: string[];
}

export interface SeasonBounds {
  first_week_start: string | null;
}

// Extend Game to carry completed/score fields already returned by the API
// but not yet typed on the frontend (confirmed present on GameOut in
// api/schemas.py: completed, home_pts, away_pts):
export interface Game {
  game_id: string;
  game_date: string;
  home_team: string;
  away_team: string;
  prediction: Prediction | null;
  completed: boolean;
  home_pts: number | null;
  away_pts: number | null;
}

// Add to the api object:
export const api = {
  getTeams: () => fetchJson<Team[]>("/teams"),
  getTeam: (abbreviation: string) => fetchJson<Team>(`/teams/${abbreviation}`),
  getGames: (date: string) => fetchJson<Game[]>(`/games?date=${date}`),
  getGamesWeek: (start: string) => fetchJson<Game[]>(`/games/week?start=${start}`),
  getGameDetail: (gameId: string) => fetchJson<GameDetail>(`/games/${gameId}`),
  getGamePlayers: (gameId: string) => fetchJson<PlayerProp[]>(`/games/${gameId}/players`),
  getSeasonFirstWeek: () => fetchJson<SeasonBounds>("/season/first-week"),
  getHubTeams: () => fetchJson<TeamHubRow[]>("/hub/teams"),
  getHubPlayers: () => fetchJson<PlayerHubRow[]>("/hub/players"),
  getHubRankings: () => fetchJson<PowerRankingRow[]>("/hub/rankings"),
  getHubStandings: () => fetchJson<StandingsRow[]>("/hub/standings"),
  getTrackRecord: () => fetchJson<TrackRecord[]>("/hub/track-record"),
  getManifest: () => fetchJson<Manifest>("/manifest"),
  getCalibration: () => fetchJson<CalibrationBin[]>("/calibration"),
};
```

Note: adding required `completed`/`home_pts`/`away_pts` to the `Game` interface (they were already optional-in-practice from the backend but untyped on the frontend) means every existing test literal that builds a `Game`/`GameDetail` object without those three fields will now fail to type-check. Fix each by adding `completed: false, home_pts: null, away_pts: null` to the existing mock objects in `frontend/src/pages/GamesPage.test.tsx` and `frontend/src/components/GameDetailModal.test.tsx` as part of Step 3 (this is a mechanical fixup, not new test logic — do it now so Task 10/12's own test changes land on a compiling baseline).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm test -- --run`
Expected: PASS for `client.test.ts`; `GamesPage.test.tsx` and `GameDetailModal.test.tsx` should still pass too once their mock literals are updated per the note above — if either still fails, fix the literal, don't change the type.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/client.ts frontend/src/api/client.test.ts frontend/src/pages/GamesPage.test.tsx frontend/src/components/GameDetailModal.test.tsx
git commit -m "feat: type season bounds, head-to-head, and completed-game fields on the api client"
```

---

### Task 10: Default the Games page to the season's first week with games

**Files:**
- Modify: `frontend/src/pages/GamesPage.tsx`
- Test: `frontend/src/pages/GamesPage.test.tsx`

**Interfaces:**
- Consumes: `api.getSeasonFirstWeek()` (Task 9).
- Produces: no exported interface change — `GamesPage` remains a default export with no props. Internal behavior change only: `weekStart` is initialized from the season's first week (falling back to today's Monday if the schedule is empty or the call fails) instead of always defaulting to today's Monday.

- [ ] **Step 1: Write the failing tests**

```typescript
// frontend/src/pages/GamesPage.test.tsx
// Update the top-level mock to include the new API call:
vi.mock("../api/client", () => ({
  api: { getGamesWeek: vi.fn(), getSeasonFirstWeek: vi.fn() },
}));

// Add near the other tests:
it("defaults to the season's first week instead of today's week", async () => {
  vi.mocked(api.getSeasonFirstWeek).mockResolvedValue({ first_week_start: "2025-10-20" });
  vi.mocked(api.getGamesWeek).mockResolvedValue([]);

  render(<GamesPage />);

  await waitFor(() => expect(api.getGamesWeek).toHaveBeenCalledWith("2025-10-20"));
});

it("falls back to today's week if the season bounds call fails", async () => {
  vi.mocked(api.getSeasonFirstWeek).mockRejectedValue(new Error("boom"));
  vi.mocked(api.getGamesWeek).mockResolvedValue([]);

  render(<GamesPage />);

  await waitFor(() => expect(api.getGamesWeek).toHaveBeenCalled());
  const [calledWith] = vi.mocked(api.getGamesWeek).mock.calls[0];
  expect(typeof calledWith).toBe("string");
  expect(calledWith).toMatch(/^\d{4}-\d{2}-\d{2}$/);
});
```

Every pre-existing test in this file that calls `render(<GamesPage />)` and then immediately asserts on `api.getGamesWeek`'s result must keep working — since they don't mock `getSeasonFirstWeek` explicitly (it'll be `undefined` from the `vi.fn()` mock returning `undefined` rather than a promise), Step 3's implementation must tolerate that by treating a non-promise/rejected call as "fall back to today," matching the second new test above. Confirm this by running the full file, not just the two new tests.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm test -- --run GamesPage.test.tsx`
Expected: FAIL on the two new tests — `getGamesWeek` is currently called with `mondayOf(new Date())`, not `"2025-10-20"`.

- [ ] **Step 3: Implement**

```tsx
// frontend/src/pages/GamesPage.tsx
export default function GamesPage() {
  const [weekStart, setWeekStart] = useState<string | null>(null);
  const [games, setGames] = useState<Game[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);

  useEffect(() => {
    api
      .getSeasonFirstWeek()
      .then((bounds) => setWeekStart(bounds.first_week_start ?? mondayOf(new Date())))
      .catch(() => setWeekStart(mondayOf(new Date())));
  }, []);

  useEffect(() => {
    if (weekStart === null) return;
    setGames(null);
    setError(null);
    api
      .getGamesWeek(weekStart)
      .then(setGames)
      .catch(() => setError("Couldn't load games."));
  }, [weekStart]);

  const gamesByDay = new Map<string, Game[]>();
  for (const game of games ?? []) {
    const existing = gamesByDay.get(game.game_date) ?? [];
    existing.push(game);
    gamesByDay.set(game.game_date, existing);
  }

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <button
          aria-label="Previous week"
          onClick={() => setWeekStart((w) => addDays(w ?? mondayOf(new Date()), -7))}
          className="text-[var(--color-net-dim)] hover:text-[var(--color-net)]"
        >
          ←
        </button>
        <span className="stat-display text-lg">{weekStart ? formatWeekRange(weekStart) : ""}</span>
        <button
          aria-label="Next week"
          onClick={() => setWeekStart((w) => addDays(w ?? mondayOf(new Date()), 7))}
          className="text-[var(--color-net-dim)] hover:text-[var(--color-net)]"
        >
          →
        </button>
      </div>

      {error && <p className="text-[var(--color-shotclock)]">{error}</p>}
      {!error && (weekStart === null || games === null) && <p>Loading games…</p>}
      {!error && weekStart !== null && games !== null && games.length === 0 && <p>No games scheduled this week.</p>}

      <div className="space-y-6">
        {Array.from(gamesByDay.entries()).map(([day, dayGames]) => (
          <div key={day}>
            <h3 className="mb-2 text-sm text-[var(--color-net-faint)]">{formatDayHeader(day)}</h3>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {dayGames.map((game) => (
                <GameCard key={game.game_id} game={game} onSelect={setSelectedGameId} />
              ))}
            </div>
          </div>
        ))}
      </div>

      {selectedGameId && (
        <GameDetailModal gameId={selectedGameId} onClose={() => setSelectedGameId(null)} />
      )}
    </div>
  );
}
```

The rest of the file (`toISODate`, `mondayOf`, `addDays`, `formatWeekRange`, `formatDayHeader`) is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm test -- --run GamesPage.test.tsx`
Expected: PASS (all tests, old and new — check the pre-existing "shows a loading state" test still passes now that there are two sequential async calls before games render)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/GamesPage.tsx frontend/src/pages/GamesPage.test.tsx
git commit -m "feat: default games page to the season's first week instead of today's week"
```

---

### Task 11: Final score on `GameCard` for completed games

**Files:**
- Modify: `frontend/src/components/GameCard.tsx`
- Create: `frontend/src/components/GameCard.test.tsx`

**Interfaces:**
- Consumes: `Game.completed`, `Game.home_pts`, `Game.away_pts` (Task 9).
- Produces: no new exported interface — `GameCard` keeps its existing `{ game, onSelect }` props.

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/components/GameCard.test.tsx
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import GameCard from "./GameCard";
import type { Game } from "../api/client";

const baseGame: Game = {
  game_id: "g1", game_date: "2026-11-01", home_team: "BOS", away_team: "MIA",
  prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
  completed: false, home_pts: null, away_pts: null,
};

describe("GameCard", () => {
  it("shows the win-probability prediction for an upcoming game", () => {
    render(<GameCard game={baseGame} onSelect={() => {}} />);
    expect(screen.getByText(/62%/)).toBeInTheDocument();
  });

  it("shows the final score instead of a prediction for a completed game", () => {
    const finished: Game = { ...baseGame, completed: true, home_pts: 110, away_pts: 102 };
    render(<GameCard game={finished} onSelect={() => {}} />);

    expect(screen.getByText("110")).toBeInTheDocument();
    expect(screen.getByText("102")).toBeInTheDocument();
    expect(screen.getByText(/final/i)).toBeInTheDocument();
    expect(screen.queryByText(/62%/)).not.toBeInTheDocument();
  });

  it("calls onSelect with the game id when clicked", async () => {
    const onSelect = vi.fn();
    render(<GameCard game={baseGame} onSelect={onSelect} />);

    await userEvent.click(screen.getByTestId("game-card-g1"));
    expect(onSelect).toHaveBeenCalledWith("g1");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- --run GameCard.test.tsx`
Expected: FAIL on the completed-game test — the current component always renders the prediction block (or "Pending"), never a score.

- [ ] **Step 3: Implement**

```tsx
// frontend/src/components/GameCard.tsx
import type { Game } from "../api/client";

interface GameCardProps {
  game: Game;
  onSelect: (gameId: string) => void;
}

export default function GameCard({ game, onSelect }: GameCardProps) {
  return (
    <button
      data-testid={`game-card-${game.game_id}`}
      onClick={() => onSelect(game.game_id)}
      className="w-full rounded border border-[var(--color-line)] bg-[var(--color-court-900)] p-4 text-left transition hover:border-[var(--color-hardwood)]"
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-xs text-[var(--color-net-faint)]">{game.game_date}</div>
          <div className="mt-1 text-base font-semibold">
            {game.away_team} <span className="text-[var(--color-net-faint)] font-normal">at</span> {game.home_team}
          </div>
        </div>
        {game.completed ? (
          <div className="shrink-0 text-right">
            <div className="stat-display text-2xl leading-none">
              {game.away_pts} <span className="text-[var(--color-net-faint)]">–</span> {game.home_pts}
            </div>
            <div className="mt-1 text-xs text-[var(--color-net-faint)]">Final</div>
          </div>
        ) : game.prediction ? (
          <div className="shrink-0 text-right">
            <div className="stat-display text-3xl leading-none text-[var(--color-hardwood-bright)]">
              {Math.round(game.prediction.home_win_probability * 100)}%
            </div>
            <div className="mt-1 text-xs text-[var(--color-net-faint)]">{game.home_team} to win</div>
          </div>
        ) : (
          <div className="text-xs text-[var(--color-net-dim)]">Pending</div>
        )}
      </div>
    </button>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- --run GameCard.test.tsx`
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/GameCard.tsx frontend/src/components/GameCard.test.tsx
git commit -m "feat: show final score on GameCard for completed games"
```

---

### Task 12: Full fixture modal — lines, bookmaker odds, head-to-head, recent form, final score

**Files:**
- Modify: `frontend/src/components/GameDetailModal.tsx`
- Modify: `frontend/src/components/GameDetailModal.test.tsx`

**Interfaces:**
- Consumes: `GameDetail` (with `point`, `head_to_head`, `home_recent_form`, `away_recent_form`, `completed`/`home_pts`/`away_pts` — all from Task 9), `PlayerProp` (existing, now with real `player_name` from Task 7).
- Produces: no new exported interface — same `{ gameId, onClose }` props.

- [ ] **Step 1: Write the failing tests**

```tsx
// frontend/src/components/GameDetailModal.test.tsx
// Replace the file's `detail` fixture and add new tests:
const detail = {
  game_id: "g1", game_date: "2026-11-01", home_team: "BOS", away_team: "MIA",
  completed: false, home_pts: null, away_pts: null,
  prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
  markets: [
    { market: "h2h", selection: "BOS", model_probability: 0.62, market_probability: 0.55, edge: 0.07, bookmaker: "DraftKings", american_odds: -130, point: null },
    { market: "spread", selection: "BOS", model_probability: 0.52, market_probability: 0.5, edge: 0.02, bookmaker: "DraftKings", american_odds: -110, point: -4.5 },
    { market: "total", selection: "over", model_probability: 0.48, market_probability: 0.5, edge: -0.02, bookmaker: "FanDuel", american_odds: -105, point: 224.5 },
  ],
  head_to_head: [
    { game_id: "g0", game_date: "2026-01-01", home_team: "BOS", away_team: "MIA", home_pts: 110, away_pts: 100 },
  ],
  home_recent_form: ["W", "L", "W"],
  away_recent_form: ["L", "L", "W"],
};

const players = [{ player_id: "203999", player_name: "Nikola Jokic", stat: "points", predicted_value: 27.5 }];

// Add:
it("renders bookmaker and american odds for each market row", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(detail);
  vi.mocked(api.getGamePlayers).mockResolvedValue(players);

  render(<GameDetailModal gameId="g1" onClose={() => {}} />);

  await screen.findAllByTestId("market-row");
  expect(screen.getByText("DraftKings")).toBeInTheDocument();
  expect(screen.getByText("FanDuel")).toBeInTheDocument();
  expect(screen.getByText("-130")).toBeInTheDocument();
});

it("renders the spread and total lines", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(detail);
  vi.mocked(api.getGamePlayers).mockResolvedValue(players);

  render(<GameDetailModal gameId="g1" onClose={() => {}} />);

  await screen.findAllByTestId("market-row");
  expect(screen.getByText("-4.5")).toBeInTheDocument();
  expect(screen.getByText("224.5")).toBeInTheDocument();
});

it("renders head-to-head history", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(detail);
  vi.mocked(api.getGamePlayers).mockResolvedValue(players);

  render(<GameDetailModal gameId="g1" onClose={() => {}} />);

  expect(await screen.findByText(/2026-01-01/)).toBeInTheDocument();
  expect(screen.getByText(/110/)).toBeInTheDocument();
});

it("renders recent form for both teams", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(detail);
  vi.mocked(api.getGamePlayers).mockResolvedValue(players);

  render(<GameDetailModal gameId="g1" onClose={() => {}} />);

  expect(await screen.findAllByTestId("form-badge")).toHaveLength(6);
});

it("shows the final score instead of a live prediction for a completed game", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue({
    ...detail, completed: true, home_pts: 108, away_pts: 101, markets: [], head_to_head: [],
  });
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g1" onClose={() => {}} />);

  expect(await screen.findByText(/final/i)).toBeInTheDocument();
  expect(screen.getByText("108")).toBeInTheDocument();
  expect(screen.getByText("101")).toBeInTheDocument();
});

it("closes when Escape is pressed", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(detail);
  vi.mocked(api.getGamePlayers).mockResolvedValue(players);
  const onClose = vi.fn();

  render(<GameDetailModal gameId="g1" onClose={onClose} />);
  await screen.findByRole("heading", { name: /BOS/ });

  await userEvent.keyboard("{Escape}");
  expect(onClose).toHaveBeenCalled();
});
```

Also update the file's existing `player_id: "203999", player_name: "203999"` player fixture to `player_name: "Nikola Jokic"` (it was documenting the pre-fix bug — now stale) and confirm the pre-existing "renders player prop predictions" test switches its name assertion to match.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm test -- --run GameDetailModal.test.tsx`
Expected: FAIL on all the new tests — bookmaker/odds/point/head-to-head/form/final-score/Escape aren't rendered or handled yet.

- [ ] **Step 3: Implement**

```tsx
// frontend/src/components/GameDetailModal.tsx
import { useEffect, useState } from "react";
import { api, type GameDetail, type PlayerProp } from "../api/client";

interface GameDetailModalProps {
  gameId: string;
  onClose: () => void;
}

function FormBadge({ result }: { result: string }) {
  return (
    <span
      data-testid="form-badge"
      className={
        result === "W"
          ? "inline-flex h-5 w-5 items-center justify-center rounded-full bg-[var(--color-win)]/20 text-xs text-[var(--color-win)]"
          : "inline-flex h-5 w-5 items-center justify-center rounded-full bg-[var(--color-shotclock)]/20 text-xs text-[var(--color-shotclock)]"
      }
    >
      {result}
    </span>
  );
}

export default function GameDetailModal({ gameId, onClose }: GameDetailModalProps) {
  const [detail, setDetail] = useState<GameDetail | null>(null);
  const [players, setPlayers] = useState<PlayerProp[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDetail(null);
    setPlayers(null);
    setError(null);
    Promise.all([api.getGameDetail(gameId), api.getGamePlayers(gameId)])
      .then(([detailResult, playersResult]) => {
        setDetail(detailResult);
        setPlayers(playersResult);
      })
      .catch(() => setError("Couldn't load game details."));
  }, [gameId]);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const sortedMarkets = detail ? [...detail.markets].sort((a, b) => (b.edge ?? 0) - (a.edge ?? 0)) : [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70" onClick={onClose}>
      <div
        className="max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded border border-[var(--color-line)] bg-[var(--color-court-900)] p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-bold">{detail ? `${detail.away_team} at ${detail.home_team}` : "Game detail"}</h2>
          <button aria-label="Close" onClick={onClose} className="text-xl leading-none text-[var(--color-net-dim)]">
            ×
          </button>
        </div>

        {error && <p className="text-[var(--color-shotclock)]">{error}</p>}
        {!error && !detail && <p>Loading…</p>}

        {detail?.completed ? (
          <div className="mb-5 flex items-center justify-center gap-6 border-b border-[var(--color-line)] pb-5">
            <div className="text-center">
              <div className="stat-display text-3xl leading-none">{detail.away_pts}</div>
              <div className="mt-1 text-xs text-[var(--color-net-faint)]">{detail.away_team}</div>
            </div>
            <div className="text-xs uppercase text-[var(--color-net-faint)]">Final</div>
            <div className="text-center">
              <div className="stat-display text-3xl leading-none">{detail.home_pts}</div>
              <div className="mt-1 text-xs text-[var(--color-net-faint)]">{detail.home_team}</div>
            </div>
          </div>
        ) : (
          detail?.prediction && (
            <div className="mb-5 grid grid-cols-3 gap-4 border-b border-[var(--color-line)] pb-5">
              <div>
                <div className="stat-display text-2xl leading-none text-[var(--color-hardwood-bright)]">
                  {Math.round(detail.prediction.home_win_probability * 100)}%
                </div>
                <div className="mt-1 text-xs text-[var(--color-net-faint)]">{detail.home_team} win probability</div>
              </div>
              <div>
                <div className="stat-display text-2xl leading-none">{detail.prediction.predicted_margin.toFixed(1)}</div>
                <div className="mt-1 text-xs text-[var(--color-net-faint)]">Predicted margin</div>
              </div>
              <div>
                <div className="stat-display text-2xl leading-none">{detail.prediction.predicted_total.toFixed(1)}</div>
                <div className="mt-1 text-xs text-[var(--color-net-faint)]">Predicted total</div>
              </div>
            </div>
          )
        )}

        {detail && (detail.home_recent_form.length > 0 || detail.away_recent_form.length > 0) && (
          <div className="mb-5 flex items-center justify-between border-b border-[var(--color-line)] pb-5 text-sm">
            <div>
              <div className="mb-1 text-xs text-[var(--color-net-faint)]">{detail.away_team} form</div>
              <div className="flex gap-1">
                {detail.away_recent_form.map((r, i) => (
                  <FormBadge key={i} result={r} />
                ))}
              </div>
            </div>
            <div className="text-right">
              <div className="mb-1 text-xs text-[var(--color-net-faint)]">{detail.home_team} form</div>
              <div className="flex justify-end gap-1">
                {detail.home_recent_form.map((r, i) => (
                  <FormBadge key={i} result={r} />
                ))}
              </div>
            </div>
          </div>
        )}

        {sortedMarkets.length > 0 && (
          <table className="mb-4 w-full text-sm">
            <thead>
              <tr className="text-left text-[var(--color-net-faint)]">
                <th>Market</th>
                <th>Selection</th>
                <th>Line</th>
                <th>Bookmaker</th>
                <th>Odds</th>
                <th>Model %</th>
                <th>Market %</th>
                <th>Edge</th>
              </tr>
            </thead>
            <tbody>
              {sortedMarkets.map((market, i) => (
                <tr key={i} data-testid="market-row">
                  <td>{market.market}</td>
                  <td>{market.selection}</td>
                  <td>{market.point !== null ? market.point : "—"}</td>
                  <td>{market.bookmaker ?? "—"}</td>
                  <td>{market.american_odds !== null ? market.american_odds : "—"}</td>
                  <td>{Math.round(market.model_probability * 100)}%</td>
                  <td>{market.market_probability !== null ? `${Math.round(market.market_probability * 100)}%` : "—"}</td>
                  <td
                    className={
                      market.edge === null
                        ? undefined
                        : market.edge > 0
                          ? "text-[var(--color-win)]"
                          : "text-[var(--color-shotclock)]"
                    }
                  >
                    {market.edge !== null ? `${(market.edge * 100).toFixed(1)}pp` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {detail && detail.head_to_head.length > 0 && (
          <div className="mb-4">
            <h3 className="mb-2 text-sm text-[var(--color-net-faint)]">Head to head</h3>
            <ul className="text-sm">
              {detail.head_to_head.map((meeting) => (
                <li key={meeting.game_id} className="flex justify-between border-b border-[var(--color-line)] py-1">
                  <span>{meeting.game_date}</span>
                  <span>
                    {meeting.away_team} {meeting.away_pts} – {meeting.home_pts} {meeting.home_team}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {players && players.length > 0 && (
          <ul className="text-sm">
            {players.map((player, i) => (
              <li key={i} className="flex justify-between border-b border-[var(--color-line)] py-1">
                <span>{player.player_name}</span>
                <span>
                  <span>{player.stat}</span>: <span>{player.predicted_value}</span>
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm test -- --run GameDetailModal.test.tsx`
Expected: PASS (all tests, old and new)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/GameDetailModal.tsx frontend/src/components/GameDetailModal.test.tsx
git commit -m "feat: full fixture modal with lines, bookmaker odds, head-to-head, and recent form"
```

---

### Task 13: Full-suite verification and manual browser check

**Files:**
- Modify: `AI_Continuity.md` (append a completion entry)
- No other file changes — this task is verification only.

**Interfaces:**
- Consumes: everything from Tasks 1–12.
- Produces: nothing new. This is the plan's final gate.

- [ ] **Step 1: Run the full backend test suite**

Run: `pytest -v`
Expected: PASS, zero failures. Pay particular attention to `tests/test_hub_service.py`'s existing settlement tests (`_settle_h2h_market_predictions`) — Task 2's schema change adds a nullable column, which shouldn't affect them, but confirm.

- [ ] **Step 2: Run the full frontend test suite**

Run: `cd frontend && npm test -- --run`
Expected: PASS, zero failures.

- [ ] **Step 3: Run a real backfill so there's real spread/total data to look at**

```bash
cd /Users/sigey/Documents/Projects/NBA_Predictor
python -m nba_predictor.pipeline.ingest --start 2025-10-01 --end 2026-06-27
```

Then manually trigger odds refresh against whatever's currently upcoming (requires `SPORTSBOOK_API_KEY` set in `.env`, and `ADMIN_API_KEY` if `PUBLIC_MODE` is set):

```bash
curl -s -X POST "http://localhost:8000/refresh-odds" -H "X-Admin-Key: $ADMIN_API_KEY" | python3 -m json.tool
```

Expected: `market_predictions_stored` > 0 if there are upcoming games in the sportsbook API's window (off-season may return 0 — that's fine, it's still exercising the new code path without erroring).

- [ ] **Step 4: Start both servers and manually verify in the browser**

Start the backend (`uvicorn nba_predictor.api.app:app --reload --port 8000` from the project root with the venv active) and the frontend (`cd frontend && npm run dev`), then open the Games page.

Verify:
- The page loads directly into a week that has games (not an empty "No games scheduled this week" state) — confirms Task 8/10.
- Clicking a game opens the modal; if any spread/total rows were stored in Step 3, the market table shows a Line, Bookmaker, and Odds column populated — confirms Task 3/4/12.
- If the game has prior meetings in the loaded schedule, the head-to-head section renders — confirms Task 5/6.
- Recent-form badges render for both teams — confirms Task 5/6/12.
- Player names in the props list are real names, not numeric IDs — confirms Task 7.
- Pressing Escape closes the modal — confirms Task 12.
- Navigating to a week containing a completed game shows the final score on both the `GameCard` and inside the modal — confirms Task 11/12.

If any of these don't hold, fix the specific task before moving on — do not proceed to Step 5 with a known-broken behavior.

- [ ] **Step 5: Record completion in the continuity log**

Append a dated entry to `AI_Continuity.md` (following the file's existing entry format) summarizing what was built in Tasks 1–12, explicitly naming what was deliberately left out of scope (post-match verdict review, CLV tracking, scoreline heatmap) and why (no historical odds-snapshot or reported-box-score data collected for it yet).

- [ ] **Step 6: Commit**

```bash
git add AI_Continuity.md
git commit -m "docs: record full fixture experience completion in continuity log"
```

---

## Self-Review Notes

- **Spec coverage:** week-1 default (Task 8/10), full fixture modal with lines (Tasks 1–4, 12), bookmaker/odds display (Task 4/12), head-to-head (Task 5/6/12), recent form (Task 5/6/12) — all covered. Player-name bug (found during research, not explicitly requested but blocks "everything else" in the modal from being meaningful) fixed in Task 7.
- **Type consistency checked:** `MarketPredictionOut.point` (Task 4) ↔ `MarketPrediction.point` (Task 9) ↔ rendered in Task 12's table — same optional-float-or-null shape throughout. `GameDetailOut.head_to_head`/`home_recent_form`/`away_recent_form` (Task 6) ↔ `GameDetail` fields (Task 9) ↔ consumed in Task 12 — names match exactly. `refresh_market_predictions(schedule, db_path, margin_std, total_std)` signature is identical across Task 3 (definition) and Task 4 (caller).
- **No placeholders:** every step has literal code, not descriptions of code.
