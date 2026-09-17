# Full Season Predictions + Complete Fixture Modal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the missing "predict a game that hasn't been played yet" code path (team outcomes and player props), widen ingestion to cover the live 2026-27 season, make the Games page default to the right season automatically, and turn the fixture modal into a genuine post-match review with explicit correct/incorrect verdicts for every prediction — including player props, which currently don't exist in the running system at all.

**Architecture:** Both new scoring paths (team, player) reuse the same leak-free causal pattern already proven in `features/four_factors.py` (`shift(1).rolling(window).mean()`): append a game/player-game row with no real outcome yet, and the rolling feature at that row is computed purely from real prior rows. Team and player predictions for upcoming games land in the exact same DB tables (`predictions`, `player_prediction_snapshots`) the existing backtest already writes to — no schema change there. `game_player_outcomes` (schema already has it, never had CRUD functions) gets its first reader/writer for settlement. Post-match verdicts are computed client-side in `GameDetailModal.tsx` from data the API already returns (plus one new field, `PlayerPropOut.actual_value`) — no new verdict-specific backend endpoints.

**Tech Stack:** pandas/XGBoost (existing), FastAPI/Pydantic (existing), React/TypeScript (existing), pytest/vitest (existing). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-17-full-season-predictions-design.md`

## Global Constraints

- Every new/changed backend function needs a pytest test in the matching `tests/test_*.py` file, following this repo's one-file-per-module convention.
- Every new/changed frontend component needs a vitest + Testing Library test in its co-located `*.test.tsx` file.
- No fabricated data: every new field is a real computation over real stored/cached data. Where a simplification is used (rolling-average features, last-5-games roster estimate, in-sample player-model metrics), it is documented in a code comment the same way `pipeline/ingest.py`'s existing rating/usage_rate simplifications already are — never silently presented as more precise than it is.
- `diff` values in post-match verdicts are always `abs(predicted - actual)` (a positive number of points/stat units), never a signed delta.
- Follow existing code style: `Path`-typed args, keyword-only params after `*` in `tracking/store.py` functions, FastAPI `Depends()` for injected dependencies, Pydantic models in `api/schemas.py`.
- Run `pytest` (backend) and `npm test` inside `frontend/` (frontend) after every task — both suites must pass before moving to the next task.

---

### Task 1: Generalize the team feature builder to score upcoming games

**Files:**
- Modify: `src/nba_predictor/features/build.py`
- Test: `tests/test_features_build.py`

**Interfaces:**
- Consumes: nothing new (pure pandas).
- Produces: `build_feature_frame(games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]` — same contract as the existing `build_training_frame`, generalized to tolerate rows where `home_win` is `None`/`NaN` (an upcoming game). `build_training_frame` becomes a thin wrapper around it. Task 2 calls `build_feature_frame` directly on a mixed completed+upcoming frame.

The only behavioral change needed: the win/loss streak history walk currently does `home_results.setdefault(home, []).append("W" if row["home_win"] == 1 else "L")` unconditionally — for an upcoming row (`home_win` is `None`), this must be skipped (no result to record), while still recording the row's rest-days feature (which only needs the game date, not an outcome). This is provably safe for the rolling box-score features too: `add_rolling_four_factors` already computes `s.shift(1).rolling(window=window, min_periods=1).mean()` per team, so a row's own value (even `NaN` for an upcoming game with no box score) never enters its own rolling average, and pandas' rolling `.mean()` skips `NaN` values when a later row's window includes this one — confirmed by reading `four_factors.py` directly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features_build.py (append)
def test_build_feature_frame_computes_rolling_features_for_upcoming_game():
    from nba_predictor.features.build import build_feature_frame

    games = _sample_games()  # 4 completed BOS/MIA games, see file header
    upcoming = pd.DataFrame([
        {
            "game_id": "g4", "game_date": "2026-10-29", "home_team": "BOS", "away_team": "MIA",
            "home_win": None,
            "home_fgm": None, "home_fga": None, "home_fg3m": None, "home_tov": None,
            "home_oreb": None, "home_dreb": None, "home_fta": None,
            "away_fgm": None, "away_fga": None, "away_fg3m": None, "away_tov": None,
            "away_oreb": None, "away_dreb": None, "away_fta": None,
        }
    ])
    combined = pd.concat([games, upcoming], ignore_index=True)

    result_df, feature_cols = build_feature_frame(combined)

    upcoming_row = result_df[result_df["game_id"] == "g4"]
    assert len(upcoming_row) == 1
    assert upcoming_row[feature_cols].isna().sum().sum() == 0
    # BOS's rolling efg should reflect real prior games, not be zero/default
    assert upcoming_row["home_efg_pct_roll"].iloc[0] > 0


def test_build_feature_frame_does_not_record_a_result_for_upcoming_games():
    from nba_predictor.features.build import build_feature_frame

    games = _sample_games()
    upcoming = pd.DataFrame([
        {
            "game_id": "g4", "game_date": "2026-10-29", "home_team": "BOS", "away_team": "MIA",
            "home_win": None,
            "home_fgm": None, "home_fga": None, "home_fg3m": None, "home_tov": None,
            "home_oreb": None, "home_dreb": None, "home_fta": None,
            "away_fgm": None, "away_fga": None, "away_fg3m": None, "away_tov": None,
            "away_oreb": None, "away_dreb": None, "away_fta": None,
        },
        {
            "game_id": "g5", "game_date": "2026-10-31", "home_team": "BOS", "away_team": "MIA",
            "home_win": None,
            "home_fgm": None, "home_fga": None, "home_fg3m": None, "home_tov": None,
            "home_oreb": None, "home_dreb": None, "home_fta": None,
            "away_fgm": None, "away_fga": None, "away_fg3m": None, "away_tov": None,
            "away_oreb": None, "away_dreb": None, "away_fta": None,
        },
    ])
    combined = pd.concat([games, upcoming], ignore_index=True)

    result_df, _ = build_feature_frame(combined)

    # g5's streak feature must not have been polluted by g4 (an upcoming
    # game with no real result) — it should equal g4's own streak value
    # (both computed from the same real prior history, since neither g4
    # nor g5 contributes a result).
    g4_streak = result_df.loc[result_df["game_id"] == "g4", "home_streak"].iloc[0]
    g5_streak = result_df.loc[result_df["game_id"] == "g5", "home_streak"].iloc[0]
    assert g4_streak == g5_streak


def test_build_training_frame_still_works_as_a_thin_wrapper():
    from nba_predictor.features.build import FEATURE_COLUMNS, build_training_frame

    games = _sample_games()
    result_df, feature_cols = build_training_frame(games)

    assert feature_cols == FEATURE_COLUMNS
    assert len(result_df) < len(games)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features_build.py -v -k "build_feature_frame"`
Expected: FAIL with `ImportError` — `build_feature_frame` doesn't exist yet.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/features/build.py
# Rename build_training_frame's body to build_feature_frame, with one
# change: guard the results-history append with a null check. Then make
# build_training_frame a one-line wrapper.

def build_feature_frame(games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    games = games.copy()

    for col in _OPTIONAL_COLUMNS_DEFAULT_ZERO:
        if col not in games.columns:
            games[col] = 0.0
    games["power_rating_diff"] = games["home_power_rating"] - games["away_power_rating"]

    long_form = add_rolling_four_factors(_long_format_box_scores(games))
    rolled = long_form.set_index(["game_id", "team"])[
        ["efg_pct_roll", "tov_rate_roll", "orb_pct_roll", "ft_rate_roll"]
    ]

    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        merged = games.merge(
            rolled.reset_index(),
            left_on=["game_id", team_col],
            right_on=["game_id", "team"],
            how="left",
        )
        for factor in ["efg_pct_roll", "tov_rate_roll", "orb_pct_roll", "ft_rate_roll"]:
            games[f"{side}_{factor}"] = merged[factor].values

    games = games.sort_values("game_date").reset_index(drop=True)

    home_last_game: dict[str, str] = {}
    away_last_game: dict[str, str] = {}
    home_results: dict[str, list[str]] = {}
    away_results: dict[str, list[str]] = {}
    rest_days_home, rest_days_away = [], []
    streak_home, streak_away = [], []

    for _, row in games.iterrows():
        home, away, game_date = row["home_team"], row["away_team"], row["game_date"]

        rest_days_home.append(compute_rest_days(game_date, home_last_game.get(home)))
        rest_days_away.append(compute_rest_days(game_date, away_last_game.get(away)))
        streak_home.append(current_streak(home_results.get(home, [])))
        streak_away.append(current_streak(away_results.get(away, [])))

        home_last_game[home] = game_date
        away_last_game[away] = game_date
        # An upcoming game (home_win is None/NaN) has no real result to
        # record — only completed games extend a team's streak history.
        if pd.notna(row["home_win"]):
            home_results.setdefault(home, []).append("W" if row["home_win"] == 1 else "L")
            away_results.setdefault(away, []).append("L" if row["home_win"] == 1 else "W")

    games["home_rest_days"] = rest_days_home
    games["away_rest_days"] = rest_days_away
    games["home_back_to_back"] = [is_back_to_back(d) for d in rest_days_home]
    games["away_back_to_back"] = [is_back_to_back(d) for d in rest_days_away]
    games["home_streak"] = streak_home
    games["away_streak"] = streak_away

    games["is_high_altitude"] = games["home_team"].apply(is_high_altitude)
    flags = games.apply(lambda r: game_flags(r["home_team"], r["away_team"]), axis=1, result_type="expand")
    games["conference_game"] = flags["conference_game"]
    games["division_game"] = flags["division_game"]

    games = games.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return games, FEATURE_COLUMNS


def build_training_frame(games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    return build_feature_frame(games)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features_build.py -v`
Expected: PASS (all tests, old and new)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/features/build.py tests/test_features_build.py
git commit -m "feat: generalize team feature builder to score upcoming games"
```

---

### Task 2: Score upcoming team-outcome predictions

**Files:**
- Modify: `src/nba_predictor/pipeline/ingest.py`
- Test: `tests/test_pipeline_ingest.py`

**Interfaces:**
- Consumes: `build_feature_frame` (Task 1), `predict_win_probability` (existing, `models/game_outcome.py`), `store.insert_prediction` (existing).
- Produces: `to_scoring_frame(games: list[dict]) -> pd.DataFrame` and `score_upcoming_games(games: list[dict], models_dir: Path, db_path: Path, model_version: str) -> int`. Task 9's `main()` wiring calls `score_upcoming_games` after the existing backtest-scoring step.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_ingest.py (append)
def test_to_scoring_frame_includes_upcoming_games_with_null_fields():
    from nba_predictor.pipeline.ingest import to_scoring_frame

    games = _sample_completed_games() + [
        {"game_id": "3", "game_date": "2026-03-05", "home_team": "LAL", "away_team": "GSW", "completed": False, "home_pts": None, "away_pts": None},
    ]

    df = to_scoring_frame(games)

    assert len(df) == 3
    upcoming_row = df[df["game_id"] == "3"].iloc[0]
    assert pd.isna(upcoming_row["home_win"])
    assert pd.isna(upcoming_row["home_fgm"])
    completed_row = df[df["game_id"] == "1"].iloc[0]
    assert completed_row["home_win"] == 1


def test_score_upcoming_games_stores_predictions_for_not_yet_played_games(tmp_path):
    import numpy as np
    import pandas as pd

    from nba_predictor.pipeline.ingest import score_upcoming_games
    from nba_predictor.pipeline.retrain import run_retrain_pipeline
    from nba_predictor.tracking import store

    rng = np.random.default_rng(3)
    teams = ["BOS", "MIA", "LAL", "GSW"]
    dates = pd.date_range("2026-02-01", periods=50).astype(str)
    completed_games = []
    for i, game_date in enumerate(dates):
        home, away = teams[i % 4], teams[(i + 1) % 4]
        completed_games.append(
            {
                "game_id": f"g{i}", "game_date": game_date, "home_team": home, "away_team": away,
                "completed": True, "home_pts": 110, "away_pts": 105,
                "home_fgm": 40, "home_fga": 88, "home_fg3m": 12, "home_tov": 11, "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
                "away_fgm": 38, "away_fga": 90, "away_fg3m": 10, "away_tov": 13, "away_oreb": 10, "away_dreb": 30, "away_fta": 18,
            }
        )
    upcoming_game = {
        "game_id": "g-upcoming", "game_date": "2026-04-01", "home_team": "BOS", "away_team": "MIA",
        "completed": False, "home_pts": None, "away_pts": None,
    }
    games = completed_games + [upcoming_game]

    from nba_predictor.pipeline.ingest import to_training_frame
    train_df = to_training_frame(completed_games)
    for i, row in train_df.iterrows():
        train_df.at[i, "home_win"] = int(rng.random() > 0.4)

    models_dir = tmp_path / "models"
    run_retrain_pipeline(train_df, models_dir, model_version="v-test", trained_at="2026-03-01T00:00:00")

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    stored = score_upcoming_games(games, models_dir, db_path, model_version="v-test")

    assert stored == 1
    rows = store.get_predictions_for_game(db_path, "g-upcoming")
    assert len(rows) == 1
    assert 0.0 <= rows[0]["home_win_prob"] <= 1.0


def test_score_upcoming_games_returns_zero_when_nothing_upcoming(tmp_path):
    from nba_predictor.pipeline.ingest import score_upcoming_games
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    stored = score_upcoming_games(_sample_completed_games(), tmp_path / "models", db_path, model_version="v-test")

    assert stored == 0
```

Note: `_sample_completed_games()` and `pd`/`pandas` are already available in this test file's imports/helpers (see the top of `tests/test_pipeline_ingest.py`) — add `import pandas as pd` at the top of the file if it isn't already imported at module level (it's currently only imported inside individual test functions; adding a module-level import is fine and matches the pattern other test files in this repo use).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline_ingest.py -v -k "scoring_frame or score_upcoming_games"`
Expected: FAIL with `ImportError` — neither function exists yet.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/pipeline/ingest.py
# Add to imports:
from nba_predictor.features.build import build_feature_frame, build_training_frame

# Add after to_training_frame:
def to_scoring_frame(games: list[dict]) -> pd.DataFrame:
    """All games (completed + upcoming), shaped for
    features.build.build_feature_frame. Upcoming games (or completed games
    missing a full box score) get None for home_win and every box-score
    field — build_feature_frame computes their rolling features from real
    prior games only, never touching a nonexistent result."""
    rows = []
    for g in games:
        row = {
            "game_id": g["game_id"],
            "game_date": g["game_date"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
        }
        if g.get("completed") and f"home_{BOX_FIELDS[0]}" in g:
            row["home_win"] = int(g["home_pts"] > g["away_pts"])
            for field in BOX_FIELDS:
                row[f"home_{field}"] = g[f"home_{field}"]
                row[f"away_{field}"] = g[f"away_{field}"]
        else:
            row["home_win"] = None
            for field in BOX_FIELDS:
                row[f"home_{field}"] = None
                row[f"away_{field}"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def score_upcoming_games(games: list[dict], models_dir: Path, db_path: Path, model_version: str) -> int:
    """Scores every not-yet-completed game using real prior-game rolling
    features (via to_scoring_frame + build_feature_frame) and stores the
    result in the same predictions table score_and_store_predictions
    writes to. Only rows for games that are not completed are scored —
    a completed game already gets its backtest prediction from
    score_and_store_predictions."""
    # Trusted artifacts: these .pkl files are written by run_retrain_pipeline
    # (via joblib.dump) in this same pipeline run — not from an external or
    # user-uploaded source (same trust boundary as score_and_store_predictions).
    win_model = joblib.load(models_dir / "win_probability_model.pkl")
    margin_model = joblib.load(models_dir / "margin_model.pkl")
    total_model = joblib.load(models_dir / "total_model.pkl")

    scoring_df = to_scoring_frame(games)
    frame, feature_cols = build_feature_frame(scoring_df)
    upcoming_frame = frame[frame["home_win"].isna()].reset_index(drop=True)
    if len(upcoming_frame) == 0:
        return 0

    win_probs = predict_win_probability(win_model, upcoming_frame[feature_cols])
    margins = margin_model.predict(upcoming_frame[feature_cols])
    totals = total_model.predict(upcoming_frame[feature_cols])

    created_at = datetime.now(timezone.utc).isoformat()
    for i, row in upcoming_frame.iterrows():
        store.insert_prediction(
            db_path,
            game_id=row["game_id"],
            created_at=created_at,
            model_version=model_version,
            home_win_prob=float(win_probs[i]),
            predicted_margin=float(margins[i]),
            predicted_total=float(totals[i]),
        )
    return len(upcoming_frame)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline_ingest.py -v`
Expected: PASS (all tests, old and new)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/pipeline/ingest.py tests/test_pipeline_ingest.py
git commit -m "feat: score real predictions for upcoming (not-yet-played) games"
```

---

### Task 3: Player-level rolling feature builder

**Files:**
- Create: `src/nba_predictor/features/player_stats.py`
- Test: `tests/test_features_player_stats.py`

**Interfaces:**
- Consumes: nothing new (pure pandas), same shift/rolling pattern as `features/four_factors.py::add_rolling_four_factors`.
- Produces: `PLAYER_STATS: list[str]`, `PLAYER_FEATURE_COLUMNS: list[str]`, `build_player_feature_frame(player_games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]`. Task 4's training frame and Task 8's scoring frame both feed into this.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features_player_stats.py
import pandas as pd


def _sample_player_games() -> pd.DataFrame:
    dates = ["2026-10-21", "2026-10-23", "2026-10-25", "2026-10-27"]
    rows = []
    for i, game_date in enumerate(dates):
        rows.append(
            {
                "player_id": "p1", "player_name": "Jayson Tatum", "team": "BOS",
                "game_id": f"g{i}", "game_date": game_date,
                "points": 20.0 + i, "rebounds": 6.0, "assists": 4.0, "fg3m": 3.0, "minutes": 34.0,
            }
        )
    return pd.DataFrame(rows)


def test_build_player_feature_frame_returns_feature_columns():
    from nba_predictor.features.player_stats import PLAYER_FEATURE_COLUMNS, build_player_feature_frame

    player_games = _sample_player_games()
    result_df, feature_cols = build_player_feature_frame(player_games)

    assert feature_cols == PLAYER_FEATURE_COLUMNS
    for col in feature_cols:
        assert col in result_df.columns


def test_build_player_feature_frame_drops_first_game_with_no_rolling_history():
    from nba_predictor.features.player_stats import build_player_feature_frame

    player_games = _sample_player_games()
    result_df, _ = build_player_feature_frame(player_games)

    assert len(result_df) == 3  # first game has no prior history, dropped
    assert "g0" not in set(result_df["game_id"])


def test_build_player_feature_frame_rolling_value_is_leak_free():
    from nba_predictor.features.player_stats import build_player_feature_frame

    player_games = _sample_player_games()
    result_df, _ = build_player_feature_frame(player_games)

    # g1's points_roll should be exactly g0's real points (20.0), not
    # influenced by g1's own points (21.0).
    g1_row = result_df[result_df["game_id"] == "g1"].iloc[0]
    assert g1_row["points_roll"] == 20.0


def test_build_player_feature_frame_computes_features_for_upcoming_player_game():
    from nba_predictor.features.player_stats import build_player_feature_frame

    player_games = _sample_player_games()
    upcoming = pd.DataFrame([
        {
            "player_id": "p1", "player_name": "Jayson Tatum", "team": "BOS",
            "game_id": "g4", "game_date": "2026-10-29",
            "points": None, "rebounds": None, "assists": None, "fg3m": None, "minutes": None,
        }
    ])
    combined = pd.concat([player_games, upcoming], ignore_index=True)

    result_df, feature_cols = build_player_feature_frame(combined)

    upcoming_row = result_df[result_df["game_id"] == "g4"]
    assert len(upcoming_row) == 1
    assert upcoming_row[feature_cols].isna().sum().sum() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features_player_stats.py -v`
Expected: FAIL with `ModuleNotFoundError` — the file doesn't exist yet.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/features/player_stats.py
import pandas as pd

PLAYER_STATS = ["points", "rebounds", "assists", "fg3m", "minutes"]
PLAYER_FEATURE_COLUMNS = [f"{stat}_roll" for stat in PLAYER_STATS]


def build_player_feature_frame(player_games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Causal per-player rolling features: for each (player_id, game_date)
    row, a shift(1).rolling(window=10).mean() of points/rebounds/assists/
    fg3m/minutes over that player's real prior games only — same pattern
    as features/four_factors.py::add_rolling_four_factors, applied per
    player instead of per team. A row with no stat value yet (an upcoming
    game) never enters its own rolling average (shift(1) excludes it) or
    any other row's, since pandas rolling means skip NaN.
    """
    games = player_games.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    for stat in PLAYER_STATS:
        games[f"{stat}_roll"] = games.groupby("player_id")[stat].transform(
            lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
        )
    games = games.dropna(subset=PLAYER_FEATURE_COLUMNS).reset_index(drop=True)
    return games, PLAYER_FEATURE_COLUMNS
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features_player_stats.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/features/player_stats.py tests/test_features_player_stats.py
git commit -m "feat: add causal per-player rolling feature builder"
```

---

### Task 4: Build per-player training rows from real box scores

**Files:**
- Modify: `src/nba_predictor/pipeline/ingest.py`
- Test: `tests/test_pipeline_ingest.py`

**Interfaces:**
- Consumes: the per-player box-score row shape already produced by `espn.get_player_boxscore` and consumed by `compute_player_hub` (`player_id, player_name, team, position, minutes, points, rebounds, assists, fg_made_attempted, three_made_attempted, ft_made_attempted`), `_parse_made_attempted` (existing).
- Produces: `to_player_training_frame(games: list[dict], player_boxscores: dict[str, list[dict]]) -> pd.DataFrame` with columns `player_id, player_name, team, game_id, game_date, points, rebounds, assists, fg3m, minutes` — the exact shape `build_player_feature_frame` (Task 3) consumes. Task 5's training and Task 8's scoring both build on this.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_ingest.py (append)
def test_to_player_training_frame_builds_one_row_per_player_game():
    from nba_predictor.pipeline.ingest import to_player_training_frame

    games = [
        {"game_id": "g1", "game_date": "2026-03-01", "home_team": "BOS", "away_team": "MIA"},
    ]
    player_boxscores = {
        "g1": [
            {
                "player_id": "p1", "player_name": "Jayson Tatum", "team": "BOS", "position": "F",
                "minutes": 34.0, "points": 28.0, "rebounds": 7.0, "assists": 5.0,
                "fg_made_attempted": "10-19", "three_made_attempted": "3-7", "ft_made_attempted": "5-6",
            }
        ]
    }

    df = to_player_training_frame(games, player_boxscores)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["player_id"] == "p1"
    assert row["game_id"] == "g1"
    assert row["game_date"] == "2026-03-01"
    assert row["points"] == 28.0
    assert row["fg3m"] == 3.0
    assert row["minutes"] == 34.0


def test_to_player_training_frame_skips_games_missing_from_schedule():
    from nba_predictor.pipeline.ingest import to_player_training_frame

    player_boxscores = {"g-unknown": [{"player_id": "p1", "player_name": "X", "team": "BOS", "position": "F", "minutes": 30.0, "points": 10.0, "rebounds": 5.0, "assists": 2.0, "fg_made_attempted": "4-8", "three_made_attempted": "1-2", "ft_made_attempted": "1-1"}]}

    df = to_player_training_frame([], player_boxscores)

    assert len(df) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline_ingest.py -v -k player_training_frame`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/pipeline/ingest.py (append, after fetch_player_boxscores)
def to_player_training_frame(games: list[dict], player_boxscores: dict[str, list[dict]]) -> pd.DataFrame:
    """One row per (player, completed game) with real box-score stats,
    shaped for features.player_stats.build_player_feature_frame."""
    games_by_id = {g["game_id"]: g for g in games}
    rows = []
    for game_id, boxscore_rows in player_boxscores.items():
        game = games_by_id.get(game_id)
        if game is None:
            continue
        for row in boxscore_rows:
            fg3m, _ = _parse_made_attempted(row["three_made_attempted"])
            rows.append(
                {
                    "player_id": row["player_id"],
                    "player_name": row["player_name"],
                    "team": row["team"],
                    "game_id": game_id,
                    "game_date": game["game_date"],
                    "points": row["points"],
                    "rebounds": row["rebounds"],
                    "assists": row["assists"],
                    "fg3m": fg3m,
                    "minutes": row["minutes"],
                }
            )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline_ingest.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/pipeline/ingest.py tests/test_pipeline_ingest.py
git commit -m "feat: build per-player training rows from real box scores"
```

---

### Task 5: Train the four player-prop models

**Files:**
- Modify: `src/nba_predictor/pipeline/ingest.py`
- Test: `tests/test_pipeline_ingest.py`

**Interfaces:**
- Consumes: `build_player_feature_frame` (Task 3), `to_player_training_frame` (Task 4), `models/player_props.py::train_player_stat_model`/`predict_player_stat` (existing, unmodified), `models/manifest.py::build_manifest`/`write_manifest` (existing, unmodified).
- Produces: `PLAYER_STAT_TARGET_COLUMNS: dict[str, str]` (maps each of the 4 stat-target names to the training-frame column that holds its real value — `"threes"` maps to `"fg3m"`), `train_player_prop_models(training_df: pd.DataFrame, models_dir: Path, model_version: str, trained_at: str) -> dict`. Writes `player_points_model.pkl`, `player_rebounds_model.pkl`, `player_assists_model.pkl`, `player_threes_model.pkl`, and `player_props_manifest.json` (kept separate from the existing game-outcome `manifest.json` so `GET /manifest`'s current contract is untouched). Tasks 6 and 8 load these same four `.pkl` files by the same names.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_ingest.py (append)
def test_train_player_prop_models_writes_four_model_files_and_manifest(tmp_path):
    import numpy as np
    import pandas as pd

    from nba_predictor.pipeline.ingest import train_player_prop_models

    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-02-01", periods=20).astype(str)
    rows = []
    for i, game_date in enumerate(dates):
        rows.append(
            {
                "player_id": "p1", "player_name": "Jayson Tatum", "team": "BOS",
                "game_id": f"g{i}", "game_date": game_date,
                "points": float(rng.integers(15, 35)), "rebounds": float(rng.integers(3, 10)),
                "assists": float(rng.integers(2, 8)), "fg3m": float(rng.integers(0, 6)),
                "minutes": float(rng.integers(28, 38)),
            }
        )
    training_df = pd.DataFrame(rows)

    models_dir = tmp_path / "models"
    manifest = train_player_prop_models(training_df, models_dir, model_version="v-test", trained_at="2026-03-01T00:00:00")

    assert (models_dir / "player_points_model.pkl").exists()
    assert (models_dir / "player_rebounds_model.pkl").exists()
    assert (models_dir / "player_assists_model.pkl").exists()
    assert (models_dir / "player_threes_model.pkl").exists()
    assert (models_dir / "player_props_manifest.json").exists()
    assert manifest["model_version"] == "v-test"
    assert set(manifest["models"]) == {"points", "rebounds", "assists", "threes"}
    assert manifest["metrics"]["points"]["mae"] is not None


def test_train_player_prop_models_handles_empty_training_frame(tmp_path):
    import pandas as pd

    from nba_predictor.pipeline.ingest import train_player_prop_models

    models_dir = tmp_path / "models"
    empty_df = pd.DataFrame(columns=["player_id", "player_name", "team", "game_id", "game_date", "points", "rebounds", "assists", "fg3m", "minutes"])

    manifest = train_player_prop_models(empty_df, models_dir, model_version="v-test", trained_at="2026-03-01T00:00:00")

    assert manifest["metrics"]["points"]["mae"] is None
    assert not (models_dir / "player_points_model.pkl").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline_ingest.py -v -k train_player_prop_models`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/pipeline/ingest.py
# Add to imports:
import numpy as np

from nba_predictor.features.player_stats import build_player_feature_frame
from nba_predictor.models.manifest import build_manifest, write_manifest
from nba_predictor.models.player_props import predict_player_stat, train_player_stat_model

# Append, after to_player_training_frame:
PLAYER_STAT_TARGET_COLUMNS = {"points": "points", "rebounds": "rebounds", "assists": "assists", "threes": "fg3m"}


def train_player_prop_models(training_df: pd.DataFrame, models_dir: Path, model_version: str, trained_at: str) -> dict:
    """Trains one XGBRegressor per stat target on real per-player rolling
    features. Metrics are in-sample (fit then scored on the same rows) —
    same honesty tradeoff score_and_store_predictions already accepts for
    the team backtest, not a proper chronological holdout, stated here
    rather than hidden."""
    models_dir.mkdir(parents=True, exist_ok=True)
    frame, feature_cols = build_player_feature_frame(training_df)

    metrics: dict[str, dict] = {}
    for stat, target_col in PLAYER_STAT_TARGET_COLUMNS.items():
        if len(frame) == 0:
            metrics[stat] = {"mae": None}
            continue
        model = train_player_stat_model(frame[feature_cols], frame[target_col])
        joblib.dump(model, models_dir / f"player_{stat}_model.pkl")
        preds = predict_player_stat(model, frame[feature_cols])
        metrics[stat] = {"mae": float(np.mean(np.abs(preds - frame[target_col])))}

    manifest = build_manifest(
        model_names=list(PLAYER_STAT_TARGET_COLUMNS.keys()),
        metrics=metrics,
        model_version=model_version,
        trained_at=trained_at,
    )
    write_manifest(manifest, models_dir / "player_props_manifest.json")
    return manifest
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline_ingest.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/pipeline/ingest.py tests/test_pipeline_ingest.py
git commit -m "feat: train real player-prop models (points/rebounds/assists/threes)"
```

---

### Task 6: Store and read real player outcomes

**Files:**
- Modify: `src/nba_predictor/tracking/store.py`
- Test: `tests/test_tracking_store.py`

**Interfaces:**
- Consumes: the existing `game_player_outcomes` table (schema already in `SCHEMA`, unchanged).
- Produces: `insert_player_outcome(db_path, *, game_id, player_id, stat, actual_value, recorded_at) -> int` and `get_player_outcomes_for_game(db_path, game_id) -> list[sqlite3.Row]`. Task 7 writes via the first; Task 12's API route reads via the second.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tracking_store.py (append)
def test_insert_and_get_player_outcomes_roundtrip(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    row_id = store.insert_player_outcome(
        db_path, game_id="g1", player_id="203999", stat="points",
        actual_value=24.0, recorded_at="2026-11-01T22:00:00",
    )
    assert row_id == 1

    rows = store.get_player_outcomes_for_game(db_path, "g1")
    assert len(rows) == 1
    assert rows[0]["player_id"] == "203999"
    assert rows[0]["stat"] == "points"
    assert rows[0]["actual_value"] == pytest.approx(24.0)


def test_get_player_outcomes_for_game_returns_empty_for_unknown_game(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    assert store.get_player_outcomes_for_game(db_path, "does-not-exist") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tracking_store.py -v -k player_outcome`
Expected: FAIL with `AttributeError` — the functions don't exist yet.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/tracking/store.py (append)
def insert_player_outcome(
    db_path: Path,
    *,
    game_id: str,
    player_id: str,
    stat: str,
    actual_value: float,
    recorded_at: str,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO game_player_outcomes (game_id, player_id, stat, actual_value, recorded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (game_id, player_id, stat, actual_value, recorded_at),
        )
        conn.commit()
        return cur.lastrowid


def get_player_outcomes_for_game(db_path: Path, game_id: str) -> list[sqlite3.Row]:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM game_player_outcomes WHERE game_id = ?",
            (game_id,),
        )
        return cur.fetchall()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tracking_store.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/tracking/store.py tests/test_tracking_store.py
git commit -m "feat: store and read real player outcomes for settlement"
```

---

### Task 7: Backtest-score player predictions for completed games

**Files:**
- Modify: `src/nba_predictor/pipeline/ingest.py`
- Test: `tests/test_pipeline_ingest.py`

**Interfaces:**
- Consumes: `build_player_feature_frame` (Task 3), `PLAYER_STAT_TARGET_COLUMNS` (Task 5), `store.insert_player_prediction` (existing).
- Produces: `score_and_store_player_predictions(training_df: pd.DataFrame, models_dir: Path, db_path: Path, model_version: str) -> int`. Task 9's `main()` wiring calls this after `train_player_prop_models`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_ingest.py (append)
def test_score_and_store_player_predictions_stores_real_model_output(tmp_path):
    import numpy as np
    import pandas as pd

    from nba_predictor.pipeline.ingest import score_and_store_player_predictions, train_player_prop_models
    from nba_predictor.tracking import store

    rng = np.random.default_rng(11)
    dates = pd.date_range("2026-02-01", periods=20).astype(str)
    rows = []
    for i, game_date in enumerate(dates):
        rows.append(
            {
                "player_id": "p1", "player_name": "Jayson Tatum", "team": "BOS",
                "game_id": f"g{i}", "game_date": game_date,
                "points": float(rng.integers(15, 35)), "rebounds": float(rng.integers(3, 10)),
                "assists": float(rng.integers(2, 8)), "fg3m": float(rng.integers(0, 6)),
                "minutes": float(rng.integers(28, 38)),
            }
        )
    training_df = pd.DataFrame(rows)

    models_dir = tmp_path / "models"
    train_player_prop_models(training_df, models_dir, model_version="v-test", trained_at="2026-03-01T00:00:00")

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    stored = score_and_store_player_predictions(training_df, models_dir, db_path, model_version="v-test")

    # 19 rows survive feature assembly (first game has no rolling history) x 4 stats
    assert stored == 19 * 4
    rows = store.get_player_predictions_for_game(db_path, "g5")
    assert len(rows) == 4
    stats = {r["stat"] for r in rows}
    assert stats == {"points", "rebounds", "assists", "threes"}


def test_score_and_store_player_predictions_returns_zero_for_empty_frame(tmp_path):
    import pandas as pd

    from nba_predictor.pipeline.ingest import score_and_store_player_predictions
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    empty_df = pd.DataFrame(columns=["player_id", "player_name", "team", "game_id", "game_date", "points", "rebounds", "assists", "fg3m", "minutes"])

    stored = score_and_store_player_predictions(empty_df, tmp_path / "models", db_path, model_version="v-test")

    assert stored == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline_ingest.py -v -k score_and_store_player_predictions`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/pipeline/ingest.py (append)
def score_and_store_player_predictions(training_df: pd.DataFrame, models_dir: Path, db_path: Path, model_version: str) -> int:
    """Backtest: scores every completed player-game row that survived
    feature assembly with the trained per-stat models, stores each as a
    tracked player prediction. Same leak-free contract as
    score_and_store_predictions — build_player_feature_frame only ever
    looks at a player's *prior* games."""
    frame, feature_cols = build_player_feature_frame(training_df)
    if len(frame) == 0:
        return 0

    # Trusted artifacts: written by train_player_prop_models (via joblib.dump)
    # in this same pipeline run — same trust boundary as score_and_store_predictions.
    models = {stat: joblib.load(models_dir / f"player_{stat}_model.pkl") for stat in PLAYER_STAT_TARGET_COLUMNS}
    predictions = {stat: predict_player_stat(model, frame[feature_cols]) for stat, model in models.items()}

    created_at = datetime.now(timezone.utc).isoformat()
    stored = 0
    for i, row in frame.iterrows():
        for stat in PLAYER_STAT_TARGET_COLUMNS:
            store.insert_player_prediction(
                db_path,
                game_id=row["game_id"],
                player_id=row["player_id"],
                stat=stat,
                predicted_value=float(predictions[stat][i]),
                created_at=created_at,
            )
            stored += 1
    return stored
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline_ingest.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/pipeline/ingest.py tests/test_pipeline_ingest.py
git commit -m "feat: backtest-score real player predictions for completed games"
```

---

### Task 8: Score player props for upcoming games (roster estimate + causal scoring)

**Files:**
- Modify: `src/nba_predictor/pipeline/ingest.py`
- Test: `tests/test_pipeline_ingest.py`

**Interfaces:**
- Consumes: `to_player_training_frame` (Task 4), `build_player_feature_frame` (Task 3), `PLAYER_STAT_TARGET_COLUMNS` (Task 5).
- Produces: `to_player_scoring_frame(games: list[dict], player_boxscores: dict[str, list[dict]], roster_window: int = 5) -> pd.DataFrame` and `score_upcoming_player_props(games: list[dict], player_boxscores: dict[str, list[dict]], models_dir: Path, db_path: Path, model_version: str) -> int`. Task 9's `main()` wiring calls the scoring function.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_ingest.py (append)
def test_to_player_scoring_frame_adds_a_row_per_estimated_roster_player():
    from nba_predictor.pipeline.ingest import to_player_scoring_frame

    games = [
        {"game_id": "g1", "game_date": "2026-10-21", "home_team": "BOS", "away_team": "MIA", "completed": True, "home_pts": 110, "away_pts": 100},
        {"game_id": "g2", "game_date": "2026-10-25", "home_team": "BOS", "away_team": "LAL", "completed": False, "home_pts": None, "away_pts": None},
    ]
    player_boxscores = {
        "g1": [{"player_id": "p1", "player_name": "Jayson Tatum", "team": "BOS", "position": "F", "minutes": 34.0, "points": 28.0, "rebounds": 7.0, "assists": 5.0, "fg_made_attempted": "10-19", "three_made_attempted": "3-7", "ft_made_attempted": "5-6"}]
    }

    df = to_player_scoring_frame(games, player_boxscores)

    upcoming_rows = df[df["game_id"] == "g2"]
    assert len(upcoming_rows) == 1
    assert upcoming_rows.iloc[0]["player_id"] == "p1"
    assert pd.isna(upcoming_rows.iloc[0]["points"])


def test_to_player_scoring_frame_only_includes_players_from_the_named_team():
    from nba_predictor.pipeline.ingest import to_player_scoring_frame

    games = [
        {"game_id": "g1", "game_date": "2026-10-21", "home_team": "BOS", "away_team": "MIA", "completed": True, "home_pts": 110, "away_pts": 100},
        {"game_id": "g2", "game_date": "2026-10-25", "home_team": "BOS", "away_team": "LAL", "completed": False, "home_pts": None, "away_pts": None},
    ]
    player_boxscores = {
        "g1": [
            {"player_id": "p1", "player_name": "BOS Player", "team": "BOS", "position": "F", "minutes": 34.0, "points": 28.0, "rebounds": 7.0, "assists": 5.0, "fg_made_attempted": "10-19", "three_made_attempted": "3-7", "ft_made_attempted": "5-6"},
            {"player_id": "p2", "player_name": "MIA Player", "team": "MIA", "position": "G", "minutes": 30.0, "points": 18.0, "rebounds": 3.0, "assists": 6.0, "fg_made_attempted": "7-15", "three_made_attempted": "2-5", "ft_made_attempted": "2-2"},
        ]
    }

    df = to_player_scoring_frame(games, player_boxscores)

    upcoming_players = set(df[df["game_id"] == "g2"]["player_id"])
    # g2 is BOS vs LAL — MIA's player (p2) never appeared for BOS or LAL,
    # so must not show up as an estimated roster player for g2.
    assert "p2" not in upcoming_players


def test_score_upcoming_player_props_stores_predictions(tmp_path):
    import numpy as np
    import pandas as pd

    from nba_predictor.pipeline.ingest import score_upcoming_player_props, train_player_prop_models
    from nba_predictor.tracking import store

    rng = np.random.default_rng(13)
    dates = pd.date_range("2026-10-01", periods=20).astype(str)
    games = []
    player_boxscores = {}
    for i, game_date in enumerate(dates):
        game_id = f"g{i}"
        games.append({"game_id": game_id, "game_date": game_date, "home_team": "BOS", "away_team": "MIA", "completed": True, "home_pts": 110, "away_pts": 105})
        player_boxscores[game_id] = [
            {
                "player_id": "p1", "player_name": "Jayson Tatum", "team": "BOS", "position": "F",
                "minutes": float(rng.integers(28, 38)), "points": float(rng.integers(15, 35)),
                "rebounds": float(rng.integers(3, 10)), "assists": float(rng.integers(2, 8)),
                "fg_made_attempted": "10-19", "three_made_attempted": "3-7", "ft_made_attempted": "5-6",
            }
        ]
    upcoming_game = {"game_id": "g-upcoming", "game_date": "2026-10-25", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None}
    games.append(upcoming_game)

    from nba_predictor.pipeline.ingest import to_player_training_frame
    training_df = to_player_training_frame(games, player_boxscores)

    models_dir = tmp_path / "models"
    train_player_prop_models(training_df, models_dir, model_version="v-test", trained_at="2026-10-20T00:00:00")

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    stored = score_upcoming_player_props(games, player_boxscores, models_dir, db_path, model_version="v-test")

    assert stored == 4  # one player, four stats
    rows = store.get_player_predictions_for_game(db_path, "g-upcoming")
    assert len(rows) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline_ingest.py -v -k "player_scoring_frame or score_upcoming_player_props"`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/pipeline/ingest.py (append)
def to_player_scoring_frame(games: list[dict], player_boxscores: dict[str, list[dict]], roster_window: int = 5) -> pd.DataFrame:
    """Real per-player rows for completed games (via
    to_player_training_frame) plus one NaN-stat row per (upcoming game,
    estimated roster player) — the same causal-anchor trick
    to_scoring_frame uses for team features, so build_player_feature_frame
    can compute each player's real rolling stats as of right before a
    game they haven't played yet.

    Roster for an upcoming game is approximated as the players who
    appeared in that team's last `roster_window` completed games — the
    same recency simplification compute_player_hub already uses, stated
    here rather than hidden: a new signing or an injury won't be
    reflected."""
    training_df = to_player_training_frame(games, player_boxscores)

    completed_by_team: dict[str, list[dict]] = {}
    for g in sorted((g for g in games if g.get("completed")), key=lambda g: g["game_date"]):
        completed_by_team.setdefault(g["home_team"], []).append(g)
        completed_by_team.setdefault(g["away_team"], []).append(g)

    upcoming_rows = []
    for g in games:
        if g.get("completed"):
            continue
        for team in (g["home_team"], g["away_team"]):
            recent_games = completed_by_team.get(team, [])[-roster_window:]
            recent_game_ids = {rg["game_id"] for rg in recent_games}
            if not recent_game_ids:
                continue
            roster = training_df[training_df["game_id"].isin(recent_game_ids) & (training_df["team"] == team)]
            for player_id, name in roster[["player_id", "player_name"]].drop_duplicates().itertuples(index=False):
                upcoming_rows.append(
                    {
                        "player_id": player_id, "player_name": name, "team": team,
                        "game_id": g["game_id"], "game_date": g["game_date"],
                        "points": None, "rebounds": None, "assists": None, "fg3m": None, "minutes": None,
                    }
                )

    if not upcoming_rows:
        return training_df
    return pd.concat([training_df, pd.DataFrame(upcoming_rows)], ignore_index=True)


def score_upcoming_player_props(
    games: list[dict], player_boxscores: dict[str, list[dict]], models_dir: Path, db_path: Path, model_version: str
) -> int:
    """Scores each estimated-roster player for every upcoming game and
    stores the result in the same player_prediction_snapshots table the
    backtest writes to."""
    # Trusted artifacts: written by train_player_prop_models (via joblib.dump)
    # in this same pipeline run — same trust boundary as score_and_store_predictions.
    models = {stat: joblib.load(models_dir / f"player_{stat}_model.pkl") for stat in PLAYER_STAT_TARGET_COLUMNS}

    scoring_df = to_player_scoring_frame(games, player_boxscores)
    frame, feature_cols = build_player_feature_frame(scoring_df)
    upcoming_game_ids = {g["game_id"] for g in games if not g.get("completed")}
    upcoming_frame = frame[frame["game_id"].isin(upcoming_game_ids)].reset_index(drop=True)
    if len(upcoming_frame) == 0:
        return 0

    predictions = {stat: predict_player_stat(model, upcoming_frame[feature_cols]) for stat, model in models.items()}

    created_at = datetime.now(timezone.utc).isoformat()
    stored = 0
    for i, row in upcoming_frame.iterrows():
        for stat in PLAYER_STAT_TARGET_COLUMNS:
            store.insert_player_prediction(
                db_path,
                game_id=row["game_id"],
                player_id=row["player_id"],
                stat=stat,
                predicted_value=float(predictions[stat][i]),
                created_at=created_at,
            )
            stored += 1
    return stored
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline_ingest.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/pipeline/ingest.py tests/test_pipeline_ingest.py
git commit -m "feat: score player props for upcoming games via estimated roster"
```

---

### Task 9: Settlement + widen ingestion + wire everything into main()

**Files:**
- Modify: `src/nba_predictor/pipeline/ingest.py`
- Modify: `.github/workflows/refresh-data.yml`
- Test: `tests/test_pipeline_ingest.py`

**Interfaces:**
- Consumes: `to_player_training_frame` (Task 4), `store.insert_player_outcome` (Task 6), and every pipeline function from Tasks 2/5/7/8.
- Produces: `store_player_outcomes(training_df: pd.DataFrame, db_path: Path) -> int`. This is the last pipeline task — `main()` afterward runs the complete flow described in the spec's Architecture diagram.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_ingest.py (append)
def test_store_player_outcomes_stores_real_actual_values_for_all_four_stats(tmp_path):
    import pandas as pd

    from nba_predictor.pipeline.ingest import store_player_outcomes
    from nba_predictor.tracking import store

    training_df = pd.DataFrame([
        {
            "player_id": "p1", "player_name": "Jayson Tatum", "team": "BOS",
            "game_id": "g1", "game_date": "2026-03-01",
            "points": 28.0, "rebounds": 7.0, "assists": 5.0, "fg3m": 3.0, "minutes": 34.0,
        }
    ])

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    stored = store_player_outcomes(training_df, db_path)

    assert stored == 4
    outcomes = store.get_player_outcomes_for_game(db_path, "g1")
    by_stat = {row["stat"]: row["actual_value"] for row in outcomes}
    assert by_stat == {"points": 28.0, "rebounds": 7.0, "assists": 5.0, "threes": 3.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_ingest.py -v -k store_player_outcomes`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/pipeline/ingest.py (append)
def store_player_outcomes(training_df: pd.DataFrame, db_path: Path) -> int:
    """Stores the real actual stat value for every completed player-game
    row, for each of the four tracked stat targets — settlement data the
    fixture modal uses to show predicted-vs-actual for player props."""
    recorded_at = datetime.now(timezone.utc).isoformat()
    stored = 0
    for _, row in training_df.iterrows():
        for stat, column in PLAYER_STAT_TARGET_COLUMNS.items():
            store.insert_player_outcome(
                db_path,
                game_id=row["game_id"],
                player_id=row["player_id"],
                stat=stat,
                actual_value=float(row[column]),
                recorded_at=recorded_at,
            )
            stored += 1
    return stored
```

Now wire everything into `main()`. Replace the whole function body from the `--start`/`--end` defaults through the end:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest real NBA schedule/box-score data and refresh caches.")
    parser.add_argument("--start", default="2025-10-01")
    parser.add_argument("--end", default="2026-11-30")
    parser.add_argument(
        "--player-hub-days", type=int, default=21,
        help="How many days (most recent, within --start/--end) to fetch per-player box scores for. "
        "0 skips Player Hub and player-prop training/scoring entirely. Also bounds the player-prop "
        "model's training window — pass a value covering the whole --start/--end range (e.g. 9999) "
        "for a real full backfill. Bounded by default since it's a second ESPN request per game.",
    )
    parser.add_argument(
        "--skip-predictions", action="store_true",
        help="Skip scoring/storing predictions into the tracking DB. For CI contexts (e.g. a scheduled "
        "workflow with no persistent tracking.db) that only need fresh schedule/hub caches and a "
        "retrained model committed to git — predictions belong on the deployed server's live DB, "
        "not a stateless CI runner's throwaway one.",
    )
    args = parser.parse_args()

    print(f"Fetching schedule {args.start} to {args.end} from ESPN...")
    games = fetch_schedule_range(args.start, args.end)
    print(f"  {len(games)} games found")

    print("Fetching box scores for completed games...")
    games = enrich_with_boxscores(games)

    schedule_path = config.DATA_DIR / "cache" / "schedule" / "games.json"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(json.dumps(to_schedule_cache(games)))
    print(f"  wrote {schedule_path}")

    hub_dir = config.DATA_DIR / "cache" / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    (hub_dir / "teams.json").write_text(json.dumps(compute_team_hub(games)))
    (hub_dir / "rankings.json").write_text(json.dumps(compute_power_rankings(games)))
    (hub_dir / "standings.json").write_text(json.dumps(compute_standings(games)))

    if args.player_hub_days > 0:
        cutoff = (date.fromisoformat(args.end) - timedelta(days=args.player_hub_days)).isoformat()
        recent_games = [g for g in games if g["game_date"] >= cutoff]
        print(f"Fetching player box scores for {len(recent_games)} games since {cutoff}...")
        player_boxscores = fetch_player_boxscores(recent_games)
        (hub_dir / "players.json").write_text(json.dumps(compute_player_hub(recent_games, player_boxscores)))
    else:
        recent_games = []
        player_boxscores = {}
        (hub_dir / "players.json").write_text(json.dumps([]))
    print(f"  wrote hub caches to {hub_dir}")

    training_df = to_training_frame(games)
    print(f"Training on {len(training_df)} completed games with full box scores...")

    models_dir = config.PROJECT_ROOT / "models"
    model_version = datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
    trained_at = datetime.now(timezone.utc).isoformat()
    manifest = run_retrain_pipeline(training_df, models_dir, model_version=model_version, trained_at=trained_at)
    print(f"  trained {model_version}: {manifest['metrics']}")

    player_training_df = to_player_training_frame(recent_games, player_boxscores)
    print(f"Training player prop models on {len(player_training_df)} real player-game rows...")
    player_manifest = train_player_prop_models(player_training_df, models_dir, model_version=model_version, trained_at=trained_at)
    print(f"  trained player props: {player_manifest['metrics']}")

    if args.skip_predictions:
        print("  --skip-predictions set: not scoring/storing predictions")
    else:
        store.init_db(config.TRACKING_DB_PATH)

        stored = score_and_store_predictions(training_df, models_dir, config.TRACKING_DB_PATH, model_version)
        print(f"  stored {stored} real predictions for browsing in the UI")

        stored_upcoming = score_upcoming_games(games, models_dir, config.TRACKING_DB_PATH, model_version)
        print(f"  stored {stored_upcoming} predictions for upcoming games")

        player_stored = score_and_store_player_predictions(player_training_df, models_dir, config.TRACKING_DB_PATH, model_version)
        print(f"  stored {player_stored} real player predictions")

        player_stored_upcoming = score_upcoming_player_props(
            recent_games, player_boxscores, models_dir, config.TRACKING_DB_PATH, model_version
        )
        print(f"  stored {player_stored_upcoming} player predictions for upcoming games")

        outcomes_stored = store_player_outcomes(player_training_df, config.TRACKING_DB_PATH)
        print(f"  stored {outcomes_stored} real player outcomes for settlement")


if __name__ == "__main__":
    main()
```

Widen the scheduled CI ingest range to match — open `.github/workflows/refresh-data.yml`, find the `python -m nba_predictor.pipeline.ingest` invocation, and update its `--start`/`--end` arguments to a rolling window that always covers from the season start through several weeks ahead (matching the new `main()` defaults' spirit — if the workflow already computes dates dynamically, adjust the offset used for the end date to extend far enough forward to keep picking up newly-announced games; if it uses fixed dates, change them to `2025-10-01`/`2026-11-30` to match the new `main()` defaults).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_ingest.py -v`
Expected: PASS (all tests — this also re-confirms every test from Tasks 1-8 still passes together in the same module)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/pipeline/ingest.py tests/test_pipeline_ingest.py .github/workflows/refresh-data.yml
git commit -m "feat: wire upcoming-game and player-prop scoring into the ingest pipeline"
```

---

### Task 10: Season-aware default week

**Files:**
- Modify: `src/nba_predictor/services/schedule_repository.py`
- Modify: `tests/test_schedule_repository.py`

**Interfaces:**
- Consumes: `monday_of` (existing, unchanged).
- Produces: `default_week_start(schedule: list[dict], today: str) -> str | None`, replacing `first_week_start` (deleted — nothing else in the codebase calls it). Task 11 calls `default_week_start` from the API route.

This task also closes a real, pre-existing gap: `get_head_to_head` and `get_recent_form` (added in a previous session) were never given tests in this file — their tests are added here alongside `default_week_start`'s, since this task is already touching the file.

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
        _completed_game("g3", "2026-03-01", "BOS", "LAL", 120, 100),
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

    schedule = [_completed_game(f"g{i}", f"2026-0{i}-01", "BOS", "MIA", 100 + i, 90 + i) for i in range(1, 6)]

    result = get_head_to_head(schedule, "BOS", "MIA", before_date="2026-06-01", limit=2)
    assert len(result) == 2


def test_get_recent_form_returns_win_loss_letters_most_recent_first():
    from nba_predictor.services.schedule_repository import get_recent_form

    schedule = [
        _completed_game("g1", "2026-01-01", "BOS", "MIA", 110, 100),
        _completed_game("g2", "2026-01-05", "LAL", "BOS", 100, 90),
        _completed_game("g3", "2026-01-10", "BOS", "DEN", 88, 95),
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


def test_default_week_start_empty_schedule_returns_none():
    from nba_predictor.services.schedule_repository import default_week_start

    assert default_week_start([], today="2026-09-17") is None


def test_default_week_start_uses_earliest_game_when_far_from_next_upcoming():
    from nba_predictor.services.schedule_repository import default_week_start, monday_of

    schedule = [
        _completed_game("g0", "2025-10-21", "BOS", "MIA", 110, 100),
        {"game_id": "g1", "game_date": "2026-10-21", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None},
    ]

    # More than 7 days before 2026-10-21 (the only upcoming game).
    result = default_week_start(schedule, today="2026-09-17")

    assert result == monday_of("2025-10-21")


def test_default_week_start_switches_to_upcoming_week_exactly_seven_days_before():
    from nba_predictor.services.schedule_repository import default_week_start, monday_of

    schedule = [
        _completed_game("g0", "2025-10-21", "BOS", "MIA", 110, 100),
        {"game_id": "g1", "game_date": "2026-10-21", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None},
    ]

    result = default_week_start(schedule, today="2026-10-14")  # exactly 7 days before

    assert result == monday_of("2026-10-21")


def test_default_week_start_stays_on_old_season_one_day_before_the_switch():
    from nba_predictor.services.schedule_repository import default_week_start, monday_of

    schedule = [
        _completed_game("g0", "2025-10-21", "BOS", "MIA", 110, 100),
        {"game_id": "g1", "game_date": "2026-10-21", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None},
    ]

    result = default_week_start(schedule, today="2026-10-13")  # 8 days before

    assert result == monday_of("2025-10-21")


def test_default_week_start_uses_today_once_season_is_underway():
    from nba_predictor.services.schedule_repository import default_week_start, monday_of

    schedule = [
        _completed_game("g0", "2025-10-21", "BOS", "MIA", 110, 100),
        {"game_id": "g1", "game_date": "2026-10-21", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None},
    ]

    result = default_week_start(schedule, today="2026-10-25")  # after the next game's date

    assert result == monday_of("2026-10-25")


def test_default_week_start_falls_back_to_earliest_when_nothing_upcoming():
    from nba_predictor.services.schedule_repository import default_week_start, monday_of

    schedule = [_completed_game("g0", "2025-10-21", "BOS", "MIA", 110, 100)]

    result = default_week_start(schedule, today="2026-09-17")

    assert result == monday_of("2025-10-21")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schedule_repository.py -v -k "head_to_head or recent_form or default_week_start"`
Expected: `get_head_to_head`/`get_recent_form` tests PASS immediately (the functions already exist from a prior session — this just adds their missing coverage); `default_week_start` tests FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/services/schedule_repository.py
# Remove first_week_start entirely, replace with:
def default_week_start(schedule: list[dict], today: str) -> str | None:
    """Monday of the earliest game overall, UNLESS today >= (the next
    not-yet-completed game's date - 7 days) — then Monday of the week
    containing max(today, next_game_date), so once the season is
    underway "today" naturally takes over rather than the boundary
    freezing on opening night. Falls back to the earliest-game behavior
    if there is no upcoming game at all. None only if the schedule is
    completely empty."""
    if not schedule:
        return None

    earliest = min(game["game_date"] for game in schedule)
    upcoming_dates = sorted(g["game_date"] for g in schedule if not g.get("completed"))
    if not upcoming_dates:
        return monday_of(earliest)

    next_game_date = upcoming_dates[0]
    threshold = (date.fromisoformat(next_game_date) - timedelta(days=7)).isoformat()
    if today >= threshold:
        return monday_of(max(today, next_game_date))
    return monday_of(earliest)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schedule_repository.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/services/schedule_repository.py tests/test_schedule_repository.py
git commit -m "feat: season-aware default week + close missing test coverage for head-to-head/recent-form"
```

---

### Task 11: Wire the season-aware default into the API

**Files:**
- Modify: `src/nba_predictor/api/deps.py`
- Modify: `src/nba_predictor/api/routes.py`
- Test: `tests/test_api_games.py`

**Interfaces:**
- Consumes: `default_week_start` (Task 10).
- Produces: `deps.get_today() -> str` (new, injectable so the route stays testable without patching the real clock). `GET /season/first-week` now calls `default_week_start(schedule, today)` instead of the deleted `first_week_start(schedule)` — response shape (`SeasonBoundsOut.first_week_start`) is unchanged, so the frontend contract from the previous session's work needs no changes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_games.py (append)
def test_season_first_week_uses_the_injected_today(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import json

    from nba_predictor.api.app import app
    from nba_predictor.api import deps

    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(json.dumps([
        {"game_id": "g0", "game_date": "2025-10-21", "home_team": "BOS", "away_team": "MIA", "completed": True, "home_pts": 110, "away_pts": 100},
        {"game_id": "g1", "game_date": "2026-10-21", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None},
    ]))
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    app.dependency_overrides[deps.get_today] = lambda: "2026-10-25"

    client = TestClient(app)
    response = client.get("/season/first-week")

    assert response.status_code == 200
    from nba_predictor.services.schedule_repository import monday_of
    assert response.json() == {"first_week_start": monday_of("2026-10-25")}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_games.py -v -k season_first_week_uses_the_injected_today`
Expected: FAIL — `deps.get_today` doesn't exist yet, `dependency_overrides` assignment raises `AttributeError`.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/api/deps.py
# Add to imports:
from datetime import date

# Append:
def get_today() -> str:
    return date.today().isoformat()
```

```python
# src/nba_predictor/api/routes.py
# Update import:
from nba_predictor.api.deps import (
    get_db_path,
    get_models_dir,
    get_schedule,
    get_today,
    get_training_games_path,
    require_admin,
)

# Update the schedule_repository import:
from nba_predictor.services.schedule_repository import (
    default_week_start,
    get_game,
    get_games_for_date,
    get_games_for_week,
    get_head_to_head,
    get_recent_form,
)

# Replace the season_first_week route:
@router.get("/season/first-week", response_model=SeasonBoundsOut)
def season_first_week(schedule: list[dict] = Depends(get_schedule), today: str = Depends(get_today)) -> SeasonBoundsOut:
    return SeasonBoundsOut(first_week_start=default_week_start(schedule, today))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_games.py -v`
Expected: PASS (all tests, including the two pre-existing `season_first_week` tests from the previous session — confirm they still pass since `get_today`'s real default, `date.today()`, will put "today" far from any 2025-2026 schedule fixture in those tests too, same as `first_week_start`'s old earliest-game behavior)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/api/deps.py src/nba_predictor/api/routes.py tests/test_api_games.py
git commit -m "feat: wire season-aware default week into GET /season/first-week"
```

---

### Task 12: Expose actual player-prop outcomes on the API

**Files:**
- Modify: `src/nba_predictor/api/schemas.py`
- Modify: `src/nba_predictor/api/routes.py`
- Test: `tests/test_api_games.py`

**Interfaces:**
- Consumes: `store.get_player_outcomes_for_game` (Task 6).
- Produces: `PlayerPropOut.actual_value: float | None = None`. Task 13's frontend `PlayerProp` TypeScript interface mirrors this exactly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_games.py (append)
def test_get_game_players_includes_actual_value_when_settled(tmp_path, monkeypatch):
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="203999", stat="points",
        predicted_value=27.5, created_at="2026-11-01T12:00:00",
    )
    store.insert_player_outcome(
        db_path, game_id="g1", player_id="203999", stat="points",
        actual_value=24.0, recorded_at="2026-11-01T22:00:00",
    )

    response = client.get("/games/g1/players")
    assert response.status_code == 200
    assert response.json()[0]["actual_value"] == 24.0


def test_get_game_players_actual_value_is_null_when_not_settled(tmp_path, monkeypatch):
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="203999", stat="points",
        predicted_value=27.5, created_at="2026-11-01T12:00:00",
    )

    response = client.get("/games/g1/players")
    assert response.json()[0]["actual_value"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_games.py -v -k actual_value`
Expected: FAIL — `actual_value` key absent from the response.

- [ ] **Step 3: Implement**

```python
# src/nba_predictor/api/schemas.py
class PlayerPropOut(BaseModel):
    player_id: str
    player_name: str
    stat: str
    predicted_value: float
    actual_value: float | None = None
```

```python
# src/nba_predictor/api/routes.py
# Replace get_game_players:
@router.get("/games/{game_id}/players", response_model=list[PlayerPropOut])
def get_game_players(
    game_id: str, schedule: list[dict] = Depends(get_schedule), db_path: Path = Depends(get_db_path)
) -> list[PlayerPropOut]:
    game = get_game(schedule, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_id}")

    name_by_id = load_player_name_map(config.DATA_DIR / "cache" / "hub" / "players.json")
    outcomes = store.get_player_outcomes_for_game(db_path, game_id)
    actual_by_key = {(row["player_id"], row["stat"]): row["actual_value"] for row in outcomes}

    return [
        PlayerPropOut(
            player_id=row["player_id"],
            player_name=name_by_id.get(row["player_id"], row["player_id"]),
            stat=row["stat"],
            predicted_value=row["predicted_value"],
            actual_value=actual_by_key.get((row["player_id"], row["stat"])),
        )
        for row in store.get_player_predictions_for_game(db_path, game_id)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api_games.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/api/schemas.py src/nba_predictor/api/routes.py tests/test_api_games.py
git commit -m "feat: expose real actual player-prop outcomes on the API"
```

---

### Task 13: Frontend types for actual player values

**Files:**
- Modify: `frontend/src/api/client.ts`
- Test: `frontend/src/api/client.test.ts`

**Interfaces:**
- Consumes: the `actual_value` field from Task 12.
- Produces: `PlayerProp.actual_value: number | null`. Task 16 renders it.

- [ ] **Step 1: Write the failing test**

```typescript
// frontend/src/api/client.test.ts (append)
it("getGamePlayers response includes actual_value", async () => {
  mockFetchOnce([{ player_id: "203999", player_name: "Nikola Jokic", stat: "points", predicted_value: 27.5, actual_value: 24.0 }]);

  const players = await api.getGamePlayers("g1");

  expect(players[0].actual_value).toBe(24.0);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- --run client.test.ts`
Expected: FAIL — `actual_value` doesn't exist on the `PlayerProp` type / is `undefined` at runtime since the type doesn't declare it (TypeScript would flag it at build time too via `tsc -b`).

- [ ] **Step 3: Implement**

```typescript
// frontend/src/api/client.ts
export interface PlayerProp {
  player_id: string;
  player_name: string;
  stat: string;
  predicted_value: number;
  actual_value: number | null;
}
```

Note: this makes `actual_value` a required field on the TypeScript type (matching how `Game.completed`/`home_pts`/`away_pts` were made required in a previous session, not optional) — fix any existing test literal in `frontend/src/components/GameDetailModal.test.tsx` that builds a `players` array without it by adding `actual_value: null` to each entry, as part of this step.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm test -- --run`
Expected: PASS (all tests, including `GameDetailModal.test.tsx` once its `players` fixture is updated)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/client.ts frontend/src/api/client.test.ts frontend/src/components/GameDetailModal.test.tsx
git commit -m "feat: type actual player-prop outcomes on the api client"
```

---

### Task 14: Post-match verdict — winner, margin, total

**Files:**
- Modify: `frontend/src/components/GameDetailModal.tsx`
- Modify: `frontend/src/components/GameDetailModal.test.tsx`

**Interfaces:**
- Consumes: `GameDetail` (existing — `prediction`, `completed`, `home_pts`, `away_pts`, `home_team`, `away_team`, all already present).
- Produces: `computePostMatchVerdict(detail: GameDetail): PostMatchVerdict | null` (exported for direct testing), rendered in a new section above the markets table when the game is completed and has a stored prediction.

- [ ] **Step 1: Write the failing tests**

```tsx
// frontend/src/components/GameDetailModal.test.tsx
// Update the `detail` fixture at the top of the file to include a
// completed variant, and add:

const completedDetail = {
  game_id: "g2", game_date: "2026-01-05", home_team: "BOS", away_team: "MIA",
  completed: true, home_pts: 113, away_pts: 105,
  prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
  markets: [], head_to_head: [], home_recent_form: [], away_recent_form: [],
};

it("shows a correct winner-call verdict when the favorite actually won", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(completedDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  expect(await screen.findByText(/correct/i)).toBeInTheDocument();
});

it("shows an incorrect winner-call verdict when the underdog actually won", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue({ ...completedDetail, home_pts: 90, away_pts: 100 });
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  expect(await screen.findByText(/incorrect/i)).toBeInTheDocument();
});

it("shows predicted vs actual margin with the absolute difference", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(completedDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  // predicted_margin=3.5 (BOS), actual = 113-105=8 (BOS) -> off by 4.5
  expect(await screen.findByText(/predicted margin/i)).toBeInTheDocument();
  expect(screen.getByText(/off by 4.5/i)).toBeInTheDocument();
});

it("shows predicted vs actual total with the absolute difference", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(completedDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  // predicted_total=224.5, actual = 113+105=218 -> off by 6.5
  expect(await screen.findByText(/predicted total/i)).toBeInTheDocument();
  expect(screen.getByText(/off by 6.5/i)).toBeInTheDocument();
});

it("does not show a post-match verdict for an upcoming game", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(detail); // the existing not-completed fixture
  vi.mocked(api.getGamePlayers).mockResolvedValue(players);

  render(<GameDetailModal gameId="g1" onClose={() => {}} />);

  await screen.findByRole("heading", { name: /BOS/ });
  expect(screen.queryByTestId("post-match-verdict")).not.toBeInTheDocument();
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm test -- --run GameDetailModal.test.tsx`
Expected: FAIL on all five new tests — no verdict text is rendered yet.

- [ ] **Step 3: Implement**

```tsx
// frontend/src/components/GameDetailModal.tsx
// Add near the top, above the component (module scope):

interface FavoredTeam {
  team: string;
  value: number;
}

function favoredTeam(margin: number, homeTeam: string, awayTeam: string): FavoredTeam {
  return margin >= 0 ? { team: homeTeam, value: margin } : { team: awayTeam, value: -margin };
}

interface PostMatchVerdict {
  winnerCorrect: boolean;
  predictedMargin: FavoredTeam;
  actualMargin: FavoredTeam;
  marginDiff: number;
  predictedTotal: number;
  actualTotal: number;
  totalDiff: number;
}

export function computePostMatchVerdict(detail: GameDetail): PostMatchVerdict | null {
  if (!detail.completed || !detail.prediction || detail.home_pts === null || detail.away_pts === null) {
    return null;
  }
  const actualMarginValue = detail.home_pts - detail.away_pts;
  const actualTotal = detail.home_pts + detail.away_pts;
  return {
    winnerCorrect: detail.prediction.home_win_probability >= 0.5 === actualMarginValue > 0,
    predictedMargin: favoredTeam(detail.prediction.predicted_margin, detail.home_team, detail.away_team),
    actualMargin: favoredTeam(actualMarginValue, detail.home_team, detail.away_team),
    marginDiff: Math.abs(detail.prediction.predicted_margin - actualMarginValue),
    predictedTotal: detail.prediction.predicted_total,
    actualTotal,
    totalDiff: Math.abs(detail.prediction.predicted_total - actualTotal),
  };
}
```

Inside the component, after `const sortedMarkets = ...`, add:

```tsx
  const verdict = detail ? computePostMatchVerdict(detail) : null;
```

And in the JSX, replace the existing `{detail?.prediction && (...)}` prediction-stat-grid block (the one showing home win probability / predicted margin / predicted total for a non-completed game) with a version that shows the verdict instead when the game is completed:

```tsx
        {verdict && (
          <div className="mb-5 border-b border-[var(--color-line)] pb-5 text-sm" data-testid="post-match-verdict">
            <div className="mb-2 flex items-center gap-2">
              <span className={verdict.winnerCorrect ? "text-[var(--color-win)]" : "text-[var(--color-shotclock)]"}>
                {verdict.winnerCorrect ? "✓ Correct" : "✗ Incorrect"}
              </span>
              <span className="text-[var(--color-net-faint)]">winner call</span>
            </div>
            <div className="text-[var(--color-net-faint)]">
              Predicted margin: {verdict.predictedMargin.team} +{verdict.predictedMargin.value.toFixed(1)} —{" "}
              Actual: {verdict.actualMargin.team} +{verdict.actualMargin.value.toFixed(1)} (off by {verdict.marginDiff.toFixed(1)})
            </div>
            <div className="text-[var(--color-net-faint)]">
              Predicted total: {verdict.predictedTotal.toFixed(1)} — Actual: {verdict.actualTotal} (off by {verdict.totalDiff.toFixed(1)})
            </div>
          </div>
        )}

        {!verdict && detail?.prediction && (
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
        )}
```

This replaces the modal's existing `detail?.completed ? (...final score header...) : (detail?.prediction && (...))` block from the previous session — the final-score header (away/FINAL/home) stays exactly as it was, placed above this new verdict section; only the non-completed prediction grid gets the added `!verdict &&` guard (it was already conditioned on `detail?.prediction`, this just also excludes the completed case, which now shows the verdict block instead).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm test -- --run GameDetailModal.test.tsx`
Expected: PASS (all tests, old and new)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/GameDetailModal.tsx frontend/src/components/GameDetailModal.test.tsx
git commit -m "feat: post-match verdict for winner call, margin, and total"
```

---

### Task 15: Per-market hit/miss verdict

**Files:**
- Modify: `frontend/src/components/GameDetailModal.tsx`
- Modify: `frontend/src/components/GameDetailModal.test.tsx`

**Interfaces:**
- Consumes: `MarketPrediction` (existing, has `market`, `selection`, `point`), `GameDetail` (existing).
- Produces: `marketVerdict(market: MarketPrediction, detail: GameDetail): boolean | null` (exported), rendered as a "Result" column in the existing markets table.

- [ ] **Step 1: Write the failing tests**

```tsx
// frontend/src/components/GameDetailModal.test.tsx (append)
const settledDetail = {
  ...completedDetail,
  markets: [
    { market: "h2h", selection: "BOS", model_probability: 0.62, market_probability: 0.55, edge: 0.07, bookmaker: "DraftKings", american_odds: -130, point: null },
    { market: "spread", selection: "MIA", model_probability: 0.4, market_probability: 0.45, edge: -0.05, bookmaker: "DraftKings", american_odds: 110, point: 4.5 },
    { market: "total", selection: "over", model_probability: 0.5, market_probability: 0.5, edge: 0.0, bookmaker: "FanDuel", american_odds: -105, point: 224.5 },
  ],
};
// settledDetail: home_pts=113, away_pts=105 (BOS home, MIA away) ->
// h2h "BOS" HIT (BOS won); spread "MIA" at +4.5 MISS (MIA's margin is
// -8, needs > -4.5 to cover, -8 < -4.5); total "over" 224.5 MISS (actual 218 < 224.5).

it("marks an h2h selection that matches the actual winner as a hit", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(settledDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  const rows = await screen.findAllByTestId("market-row");
  expect(rows[0]).toHaveTextContent("✓");
});

it("marks a spread selection that failed to cover as a miss", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(settledDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  const rows = await screen.findAllByTestId("market-row");
  const spreadRow = rows.find((r) => r.textContent?.includes("spread"));
  expect(spreadRow).toHaveTextContent("✗");
});

it("shows no verdict for a market row without a point value on an unsettled market", async () => {
  const noOddsDetail = { ...completedDetail, markets: [] };
  vi.mocked(api.getGameDetail).mockResolvedValue(noOddsDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  await screen.findByRole("heading", { name: /BOS/ });
  expect(screen.queryAllByTestId("market-row")).toHaveLength(0);
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm test -- --run GameDetailModal.test.tsx`
Expected: FAIL on the two new hit/miss tests — no "Result" column exists yet.

- [ ] **Step 3: Implement**

```tsx
// frontend/src/components/GameDetailModal.tsx
// Add near computePostMatchVerdict:

export function marketVerdict(market: MarketPrediction, detail: GameDetail): boolean | null {
  if (!detail.completed || detail.home_pts === null || detail.away_pts === null) return null;

  if (market.market === "h2h") {
    const actualWinner = detail.home_pts > detail.away_pts ? detail.home_team : detail.away_team;
    return market.selection === actualWinner;
  }
  if (market.market === "spread") {
    if (market.point === null) return null;
    const teamMargin =
      market.selection === detail.home_team
        ? detail.home_pts - detail.away_pts
        : detail.away_pts - detail.home_pts;
    return teamMargin > -market.point;
  }
  if (market.market === "total") {
    if (market.point === null) return null;
    const actualTotal = detail.home_pts + detail.away_pts;
    return market.selection === "over" ? actualTotal > market.point : actualTotal < market.point;
  }
  return null;
}
```

Update the markets table: add a `<th>Result</th>` header, and change the `.map()` body from an implicit-return arrow to a block so the verdict is computed once per row:

```tsx
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
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {sortedMarkets.map((market, i) => {
                const verdict = detail ? marketVerdict(market, detail) : null;
                return (
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
                    <td>
                      {verdict === null ? (
                        "—"
                      ) : verdict ? (
                        <span className="text-[var(--color-win)]">✓</span>
                      ) : (
                        <span className="text-[var(--color-shotclock)]">✗</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm test -- --run GameDetailModal.test.tsx`
Expected: PASS (all tests, old and new)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/GameDetailModal.tsx frontend/src/components/GameDetailModal.test.tsx
git commit -m "feat: per-market hit/miss verdict on the fixture modal"
```

---

### Task 16: Player-prop predicted-vs-actual

**Files:**
- Modify: `frontend/src/components/GameDetailModal.tsx`
- Modify: `frontend/src/components/GameDetailModal.test.tsx`

**Interfaces:**
- Consumes: `PlayerProp.actual_value` (Task 13).
- Produces: no new exported function — inline rendering change in the player list.

- [ ] **Step 1: Write the failing tests**

```tsx
// frontend/src/components/GameDetailModal.test.tsx (append)
it("shows predicted vs actual for a settled player prop", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(completedDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([
    { player_id: "203999", player_name: "Nikola Jokic", stat: "points", predicted_value: 27.5, actual_value: 24.0 },
  ]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  expect(await screen.findByText(/predicted: 27.5/i)).toBeInTheDocument();
  expect(screen.getByText(/actual: 24/i)).toBeInTheDocument();
  expect(screen.getByText(/off by 3.5/i)).toBeInTheDocument();
});

it("shows only the predicted value for an unsettled player prop", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(detail); // existing not-completed fixture
  vi.mocked(api.getGamePlayers).mockResolvedValue([
    { player_id: "203999", player_name: "Nikola Jokic", stat: "points", predicted_value: 27.5, actual_value: null },
  ]);

  render(<GameDetailModal gameId="g1" onClose={() => {}} />);

  expect(await screen.findByText("points")).toBeInTheDocument();
  expect(screen.getByText("27.5")).toBeInTheDocument();
  expect(screen.queryByText(/actual/i)).not.toBeInTheDocument();
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm test -- --run GameDetailModal.test.tsx`
Expected: FAIL on the "predicted vs actual" test — the player list currently only ever shows the predicted value.

- [ ] **Step 3: Implement**

```tsx
// frontend/src/components/GameDetailModal.tsx
// Replace the player list rendering:
        {players && players.length > 0 && (
          <ul className="text-sm">
            {players.map((player, i) => (
              <li key={i} className="flex justify-between border-b border-[var(--color-line)] py-1">
                <span>{player.player_name}</span>
                <span>
                  <span>{player.stat}</span>:{" "}
                  {player.actual_value !== null ? (
                    <>
                      Predicted: {player.predicted_value} — Actual: {player.actual_value} (off by{" "}
                      {Math.abs(player.predicted_value - player.actual_value).toFixed(1)})
                    </>
                  ) : (
                    <span>{player.predicted_value}</span>
                  )}
                </span>
              </li>
            ))}
          </ul>
        )}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm test -- --run GameDetailModal.test.tsx`
Expected: PASS (all tests, old and new)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/GameDetailModal.tsx frontend/src/components/GameDetailModal.test.tsx
git commit -m "feat: predicted-vs-actual verdict for player props in the fixture modal"
```

---

### Task 17: Full-suite verification and real re-ingest

**Files:**
- Modify: `AI_Continuity.md` (append a completion entry)
- No other file changes — this task is verification + a real data run.

- [ ] **Step 1: Run the full backend test suite**

Run: `pytest -v`
Expected: PASS, zero failures.

- [ ] **Step 2: Run the full frontend test suite and production build**

Run: `cd frontend && npm test -- --run && npm run build`
Expected: PASS, zero failures; build completes with no TypeScript errors.

- [ ] **Step 3: Run the real widened ingest**

This is a real, long-running operation — full-season player box scores mean roughly one extra ESPN request per completed game across the whole range, so expect this to take significantly longer than a normal run (potentially 30+ minutes depending on retry/backoff behavior). Let it run to completion rather than interrupting it.

```bash
cd /Users/sigey/Documents/Projects/NBA_Predictor
python -m nba_predictor.pipeline.ingest --start 2025-10-01 --end 2026-11-30 --player-hub-days 9999
```

Expected output includes (numbers will vary with live data, but every line should print a non-error count):
- `Fetching schedule 2025-10-01 to 2026-11-30 from ESPN...` followed by a game count that includes both 2025-26 (completed) and 2026-27 (upcoming) games.
- `Training on N completed games...` and a printed metrics dict.
- `Training player prop models on N real player-game rows...` and a printed metrics dict with all four stats present.
- `stored N predictions for upcoming games` with N > 0 (2026-27 season games exist).
- `stored N player predictions for upcoming games` with N > 0.
- `stored N real player outcomes for settlement` with N > 0.

- [ ] **Step 4: Manually verify in the browser**

Start the backend and frontend dev servers (per `.claude/launch.json` from the previous session), open the Games page, and verify:
- The page defaults to the 2025-26 season's first week (today's real date is more than 7 days before the 2026-27 season opener) — confirms Task 10/11 didn't regress the existing behavior.
- Paging forward to a 2025-26 completed game's fixture modal shows the post-match verdict section (winner ✓/✗, margin, total) and, if that game has player predictions in the freshly-ingested range, predicted-vs-actual lines per player — confirms Tasks 14-16.
- Paging forward to a 2026-27 game shows a live prediction (not "Pending") — confirms Task 2.
- That same upcoming game's fixture modal shows player-prop predictions in the player list (predicted value only, no "Actual" line since it hasn't been played) — confirms Task 8.

If any of these don't hold, fix the specific task before moving on.

- [ ] **Step 5: Record completion in the continuity log**

Append a dated entry to `AI_Continuity.md` (following the file's existing entry format) summarizing Tasks 1-16, the real ingest run's actual output numbers from Step 3, and explicitly naming what's still out of scope per the spec (double-double probability, player-level odds, injury-aware rosters).

- [ ] **Step 6: Commit**

```bash
git add AI_Continuity.md
git commit -m "docs: record full season predictions + complete fixture modal completion"
```

---

## Self-Review Notes

- **Spec coverage:** widened ingestion (Task 9), upcoming team-outcome scoring (Tasks 1-2), player-prop subsystem end-to-end — features, training, backtest scoring, upcoming scoring, settlement (Tasks 3-9), season-aware default week (Tasks 10-11), post-match verdict UI — winner/margin/total, per-market, player props (Tasks 14-16) — all covered. The pre-existing missing-test gap for `get_head_to_head`/`get_recent_form` found while planning Task 10 is closed in the same task.
- **Type consistency checked:** `PLAYER_STAT_TARGET_COLUMNS` (Task 5) is the single source of truth for stat-name-to-column mapping, reused identically in Tasks 7, 8, and 9 — no divergent spellings. `build_feature_frame`/`build_training_frame` (Task 1) signatures match every call site in Task 2 and the existing `retrain.py`/`score_and_store_predictions`. `PlayerPropOut.actual_value` (Task 12) matches `PlayerProp.actual_value` (Task 13) exactly in name and nullability, consumed identically in Task 16.
- **No placeholders:** every step has literal, runnable code.
