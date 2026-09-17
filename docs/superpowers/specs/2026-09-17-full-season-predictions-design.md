# Full Season Predictions + Complete Fixture Modal — Design

## Goal

Close the gap between what the app claims to do (predict NBA games) and what
it actually does today (backtest a model against a single already-finished
season, with player predictions never generated at all). Concretely:

1. Ingest and predict the real, live 2026-27 season schedule — not just
   replay 2025-26.
2. Build the missing "score a game that hasn't been played yet" code path
   for both team outcomes and player props — it does not exist today.
3. Make the Games page default to the right season automatically: old
   season while there's nothing to predict yet, current/upcoming week once
   the new season is close.
4. Make the fixture modal a genuine post-match review for completed games —
   explicit hit/miss verdicts, not just two numbers sitting next to each
   other — for the game outcome, every market, and every player prop.

## Current State (verified by reading the code, not assumed)

- `pipeline/ingest.py::main()` only ever fetches a fixed historical date
  range and calls `score_and_store_predictions(training_df, ...)` —
  `training_df` is filtered to completed games with full box scores
  (`to_training_frame`). There is no path that ever scores an upcoming
  game. This is why the Games page shows predictions only for
  already-finished 2025-26 games.
- `features/build.py::build_training_frame` computes rolling team features
  via a proven leak-free pattern (`shift(1).rolling(window).mean()` in
  `features/four_factors.py::add_rolling_four_factors`), but the function
  requires every row to already have `home_win` and full box scores — it
  cannot accept a mixed batch of completed + upcoming games as-is.
- `models/player_props.py` has real, working XGBoost training/prediction
  functions (`train_player_stat_model`, `predict_player_stat`,
  double-double variants) and `tracking/store.py::insert_player_prediction`
  exists — but nothing in the pipeline ever calls either. The
  `player_prediction_snapshots` table is always empty in production today.
  The `game_player_outcomes` table exists in the schema with no
  insert/read functions at all — actual player results are never recorded.
- `services/schedule_repository.py::first_week_start` returns the Monday of
  the *earliest* game in the whole cache, with no notion of "today" or
  "which season is relevant right now."
- `GameDetailModal.tsx` shows the raw prediction and the raw final score
  side by side for a completed game, with no derived correct/incorrect
  verdict, and shows player props as a flat predicted-value list with no
  actual-outcome comparison (moot today since no player predictions exist
  at all).
- Confirmed live against ESPN (2026-09-17): the 2026-27 season schedule
  is already published — Oct 21, 2026 has 11 real scheduled games,
  `completed: false`.

## Architecture

```
ingest.py main()
  ├─ fetch_schedule_range(start, end)      # widened to cover through the new season
  ├─ enrich_with_boxscores(games)          # unchanged; naturally skips upcoming games
  ├─ [existing] team training + backtest scoring on completed games
  ├─ [NEW] score_upcoming_games(...)       # team outcome predictions for not-yet-played games
  ├─ [NEW] train + backtest-score player-stat models
  ├─ [NEW] score_upcoming_player_props(...)
  └─ [NEW] settle_player_outcomes(...)     # real actual stat for completed games with a prediction
```

Both the team-level and player-level upcoming-game predictions land in the
*same* tables (`predictions`, `player_prediction_snapshots`) the backtest
already writes to — no schema change needed there. `game_market_predictions`
and `game_player_outcomes` are the only tables gaining new writers.

## Component 1: Widen ingestion

`pipeline/ingest.py::main()`'s `--start`/`--end` defaults move to cover the
real season boundary — e.g. `--start 2025-10-01 --end 2026-11-30` for the
first run that also pulls in early 2026-27 games (extended further in later
scheduled runs as the season progresses). `fetch_schedule_range` and
`enrich_with_boxscores` need no code changes — ESPN already returns
`completed: false` games in-range, and `enrich_with_boxscores` already only
fetches box scores `if game["completed"]`.

`.github/workflows/refresh-data.yml`'s scheduled run gets the same widened
range so newly-announced upcoming games keep flowing in automatically.

## Component 2: Score upcoming team-outcome predictions

New function in `features/build.py`:

```python
def build_feature_frame(games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
```

Same causal walk as `build_training_frame`, generalized to tolerate rows
with no box score and no `home_win` (upcoming games):
- `_long_format_box_scores` gets box-score columns that may be `NaN` for
  upcoming rows (the caller ensures the columns exist). `add_rolling_four_factors`'s
  `shift(1)` already excludes each row from its own rolling calculation, so
  an all-`NaN` row never appears in another row's rolling window and does
  not corrupt anyone's average (pandas rolling `.mean()` skips `NaN`
  automatically, verified by inspecting `four_factors.py` directly).
- The win/loss streak walk only appends a result to a team's history when
  the row is actually completed (`row["home_win"]` is not null) — an
  upcoming game contributes no result, only consumes the history built so
  far.
- Rest-days/back-to-back logic is unaffected — it only needs game dates,
  which upcoming games have.
- `build_training_frame` becomes a thin wrapper: `to_training_frame`
  already filters to completed-with-box-scores before calling in, so
  training behavior is byte-for-byte unchanged.

New function in `pipeline/ingest.py`:

```python
def to_scoring_frame(games: list[dict]) -> pd.DataFrame:
    """All games (completed + upcoming), box-score columns NaN where absent."""

def score_upcoming_games(games_df, models_dir, db_path, model_version) -> int:
    """Scores every not-yet-completed game via build_feature_frame + the
    trained models, stores via store.insert_prediction. Returns count."""
```

`main()` calls this after the existing backtest-scoring step, over the full
`games` list (not `training_df`).

## Component 3: Player props — the missing subsystem

New file `features/player_stats.py`:

```python
def build_player_feature_frame(player_games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
```

Same `shift(1).rolling(window=10, min_periods=1)` pattern as team four
factors, applied per `player_id` instead of per `team`, over
points/rebounds/assists/`fg3m`/minutes. Rows lacking sufficient rolling
history (a player's first tracked game) are dropped, same as the team model.

`pipeline/ingest.py` additions:

```python
def to_player_training_frame(games, player_boxscores) -> pd.DataFrame:
    """One row per (player, completed game) with real box-score stats."""

def train_player_prop_models(training_df, models_dir, model_version, trained_at) -> dict:
    """Trains one XGBRegressor per stat (points/rebounds/assists/threes)
    via models/player_props.py, writes player_points_model.pkl etc. plus
    models/player_props_manifest.json (build_manifest reused, separate
    file so it doesn't collide with the existing game-outcome manifest)."""

def score_and_store_player_predictions(scoring_df, models_dir, db_path, model_version) -> int:
    """Backtest: scores every completed player-game row, stores via
    store.insert_player_prediction. Same leak-free contract as the team
    backtest — features only ever look at prior games."""

def score_upcoming_player_props(games, player_boxscores, models_dir, db_path, model_version) -> int:
    """For each upcoming game, roster = players who appeared in that
    team's last 5 completed games (same recency simplification
    compute_player_hub already uses and documents). Scores each such
    player's current rolling stats, stores predictions against the
    upcoming game_id."""
```

**Real cost, stated plainly:** training this for real needs player box
scores for every completed game in the ingest range, not just the last 21
days the Player Hub currently fetches. That's roughly one extra ESPN
request per completed game across the whole range (~1,200+ requests for a
full season) — the next full ingest run will take meaningfully longer than
the ~10 minutes the last one took. No way around that without fabricating
data, so it's accepted as a real, known cost of this feature.

**Settlement** (new, `tracking/store.py`):

```python
def insert_player_outcome(db_path, *, game_id, player_id, stat, actual_value, recorded_at) -> int
def get_player_outcomes_for_game(db_path, game_id) -> list[sqlite3.Row]
```

Writing to the existing `game_player_outcomes` table (schema already has
it, just never had CRUD functions — same situation as the `point` column
before last session's fix). `pipeline/ingest.py` calls
`insert_player_outcome` for every completed game that has a stored
prediction, using the real box-score value for that stat.

## Component 4: Season-aware default week

New function in `services/schedule_repository.py`:

```python
def default_week_start(schedule: list[dict], today: str) -> str:
    """Monday of the earliest game overall, UNLESS today >= (next
    not-yet-completed game's date - 7 days) — then Monday of the week
    containing max(today, next_game_date) (so once the season is
    underway, "today" naturally takes over rather than the boundary
    freezing on opening night). Falls back to the earliest-game behavior
    if there is no upcoming game at all."""
```

`GET /season/first-week` (`api/routes.py::season_first_week`) switches from
calling `first_week_start` to `default_week_start(schedule, date.today().isoformat())`
— response field name (`first_week_start`) stays the same, no frontend
contract change needed. `today` is passed as a parameter specifically so
the function stays a pure, easily-testable unit rather than reaching for
the real clock internally.

## Component 5: Post-match verdict UI

All of this is derivable from data `GameDetailOut`/`PlayerPropOut` already
carry (plus the new `actual_value` field below) — computed client-side in
`GameDetailModal.tsx`, no new backend fields needed for the game-level
verdicts:

- **Winner call**: `(prediction.home_win_probability >= 0.5)` compared
  against `home_pts > away_pts`. Rendered as a ✓/✗ badge.
- **Margin**: `Predicted margin: {team} +{predicted_margin} — Actual: {team} +{actual_margin} (off by {diff})`,
  where predicted/actual margin each resolve to whichever team is favored
  (positive convention) rather than always "home", and `diff` is
  `abs(predicted_margin - actual_margin)` (always shown as a positive
  number of points, not a signed delta).
- **Total**: `Predicted total: {predicted_total} — Actual: {actual_total} (off by {diff})`,
  `diff = abs(predicted_total - actual_total)`.
- **Per-market hit/miss**: added as a column to the existing markets table,
  shown only when `detail.completed` and the row has enough data (h2h:
  selection equals actual winner; spread: selection's actual margin clears
  `-point`; total: actual total clears `point` for over/under). Rows
  without a `point`/insufficient data show no verdict rather than a
  fabricated one.
- **Player props**: `PlayerPropOut` gains `actual_value: float | None`
  (populated from the new settlement step, via `get_player_outcomes_for_game`
  merged into the existing `get_game_players` route by `(player_id, stat)`).
  The modal shows `Predicted: {predicted_value} — Actual: {actual_value} (off by {diff})`
  (`diff = abs(predicted_value - actual_value)`) when `actual_value` is
  present, otherwise the current predicted-only line (upcoming games).

## API Changes Summary

- `PlayerPropOut.actual_value: float | None = None` (new field).
- `GET /season/first-week` — same route/response shape, different
  (date-aware) computation underneath.
- No other route signatures change. `score_upcoming_games` and the player
  pipeline functions are invoked from `ingest.py`'s `main()`, not exposed
  as new API routes (consistent with how the existing backtest-scoring
  step already isn't route-exposed).

## Testing Strategy

- `features/build.py::build_feature_frame`: unit tests with a small
  synthetic team history (3-4 completed games + 1 upcoming game) asserting
  the upcoming row's rolling features match a hand-computed average of the
  team's real prior games, and that it doesn't raise on missing box-score
  columns.
- `features/player_stats.py::build_player_feature_frame`: same shape of
  test, per-player instead of per-team.
- `pipeline/ingest.py`: unit tests for `to_scoring_frame`,
  `score_upcoming_games`, `to_player_training_frame`,
  `score_upcoming_player_props` — mocking model files the same way
  existing pipeline tests do (real function logic, fixture model objects).
- `tracking/store.py`: roundtrip tests for `insert_player_outcome` /
  `get_player_outcomes_for_game`, matching the existing store test style.
- `services/schedule_repository.py::default_week_start`: tests at the
  exact 7/8-day boundary and the no-upcoming-game fallback.
- `GameDetailModal.tsx`: tests asserting the verdict badge/text for a
  correct call, an incorrect call, a market with a point value, a market
  without one, and a player prop with/without an `actual_value`.

## Explicitly Out of Scope

- Double-double probability (the classifier already exists in
  `player_props.py` but predicting it needs the same rolling-feature
  infrastructure this spec builds — a natural, separable follow-up once
  this lands, not bundled in now).
- Player-level odds/lines (no player-market data has ever been observed
  live from the sportsbook API per `sportsbook_api.py`'s own docstring —
  unchanged from before).
- Roster/injury awareness for the "who plays in an upcoming game" estimate
  beyond the last-5-games recency heuristic — a real injury-report signal
  (`espn.get_injuries()` already exists and is fetched elsewhere) could
  refine this later but isn't required for this pass to be honest, since
  the heuristic is clearly documented as an approximation, not represented
  as certain.

## Rollout

After this lands, a one-time full re-ingest is run manually (not just the
scheduled CI job) to backfill the widened range and populate player
predictions for the first time:

```bash
python -m nba_predictor.pipeline.ingest --start 2025-10-01 --end 2026-11-30 --player-hub-days 9999
```

(`--player-hub-days 9999` effectively means "the whole range" for the
Player Hub cache too, matching the training window — the flag's existing
`0 skips entirely` behavior is unchanged, this just uses a large bound
instead of adding a new flag.)
