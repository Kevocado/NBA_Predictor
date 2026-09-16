# NBA Predictor — Phase 3: Features & Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the feature-engineering pipeline (Four Factors, ratings/Elo,
rest/travel/fatigue, injuries, game context) and the XGBoost models (game
outcome, spread, total, player props) with a training manifest and
walk-forward/calibration evaluation, matching spec §4 and §5.

**Architecture:** Each feature category is a small, pure-function module
under `src/nba_predictor/features/` that operates on plain pandas
DataFrames/dicts — no I/O, no API calls (those already exist in
`src/nba_predictor/data/` from Phase 2). `features/build.py` assembles a raw
per-game DataFrame into a model-ready training frame by calling each feature
module and merging the results. `src/nba_predictor/models/` holds one
training function per model (thin wrappers around `xgboost`), plus a
manifest writer and evaluation helpers. Every function here is testable with
small synthetic DataFrames — no network calls anywhere in this phase.

**Tech Stack:** Same as Phase 1/2, plus `xgboost` and `scikit-learn` (already
in `pyproject.toml`) for models; pandas/numpy for feature computation.

**Spec:** [docs/superpowers/specs/2026-09-15-nba-predictor-design.md](../specs/2026-09-15-nba-predictor-design.md)

## Global Constraints

- Python version floor: **3.13**.
- No network I/O in `features/` or `models/` — they consume data already
  fetched by Phase 2's `data/` modules, passed in as DataFrames/dicts by the
  caller (Phase 4's API layer will do the fetching-and-assembling).
- Feature/label frames must never be shuffled before a chronological
  train/holdout split — this is the leakage rule from the spec's Four
  Factors/training section.
- All rolling features must be computed with `.shift(1)` (or equivalent)
  before the rolling window so a game's own result never leaks into its own
  pre-game features.
- Conference values are exactly `"East"`/`"West"` (per Phase 1's
  `team_reference.TeamInfo.conference`) — reuse `team_reference.get_team`
  rather than re-deriving conference/division/altitude data.

---

## File Structure

```
src/nba_predictor/
  features/
    __init__.py
    four_factors.py     # NEW — eFG%, TOV%, ORB%, FT rate (raw + rolling)
    ratings.py           # NEW — possessions, ORTG/DRTG/net rating/pace, Elo
    rest_travel.py        # NEW — rest days, b2b/congestion, travel miles, fatigue index
    context.py             # NEW — head-to-head, streaks, altitude/conference flags
    build.py                 # NEW — assembles all of the above into a training frame
  models/
    __init__.py
    game_outcome.py       # NEW — win-probability classifier, margin/total regressors
    player_props.py        # NEW — points/rebounds/assists/threes regressors, double-double classifier
    manifest.py             # NEW — manifest dict builder + JSON writer + history appender
    evaluate/
      __init__.py
      walk_forward.py        # NEW — chronological train/holdout split
      calibration.py          # NEW — reliability-curve bin computation
```

---

### Task 1: Four Factors feature module

**Files:**
- Create: `src/nba_predictor/features/__init__.py`
- Create: `src/nba_predictor/features/four_factors.py`
- Test: `tests/test_features_four_factors.py`

**Interfaces:**
- Consumes: nothing (pure functions over plain numbers/DataFrames).
- Produces:
  - `four_factors.compute_four_factors(fgm: float, fga: float, fg3m: float, tov: float, oreb: float, opp_dreb: float, fta: float) -> dict` — returns `{"efg_pct": float, "tov_rate": float, "orb_pct": float, "ft_rate": float}`.
  - `four_factors.add_rolling_four_factors(games: pd.DataFrame, window: int = 10) -> pd.DataFrame` — takes a long-format DataFrame (one row per team-game) with columns `team, game_date, fgm, fga, fg3m, tov, oreb, opp_dreb, fta`, sorted by `team, game_date`, and returns it with four new columns `efg_pct_roll, tov_rate_roll, orb_pct_roll, ft_rate_roll` — the trailing `window`-game rolling mean of each factor, computed from **prior** games only (shifted by 1), `NaN` for a team's first game.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features_four_factors.py
import math

import pandas as pd
import pytest


def test_compute_four_factors_known_values():
    from nba_predictor.features.four_factors import compute_four_factors

    result = compute_four_factors(
        fgm=40, fga=90, fg3m=10, tov=12, oreb=10, opp_dreb=30, fta=20
    )

    assert result["efg_pct"] == pytest.approx((40 + 0.5 * 10) / 90)
    assert result["tov_rate"] == pytest.approx(12 / (90 + 0.44 * 20 + 12))
    assert result["orb_pct"] == pytest.approx(10 / (10 + 30))
    assert result["ft_rate"] == pytest.approx(20 / 90)


def test_compute_four_factors_zero_fga_does_not_raise():
    from nba_predictor.features.four_factors import compute_four_factors

    result = compute_four_factors(
        fgm=0, fga=0, fg3m=0, tov=0, oreb=0, opp_dreb=0, fta=0
    )
    assert result["efg_pct"] == 0.0
    assert result["ft_rate"] == 0.0
    assert result["orb_pct"] == 0.0


def test_add_rolling_four_factors_first_game_is_nan():
    from nba_predictor.features.four_factors import add_rolling_four_factors

    games = pd.DataFrame(
        {
            "team": ["BOS", "BOS"],
            "game_date": ["2026-10-01", "2026-10-03"],
            "fgm": [40, 42],
            "fga": [90, 88],
            "fg3m": [10, 12],
            "tov": [12, 10],
            "oreb": [10, 11],
            "opp_dreb": [30, 28],
            "fta": [20, 18],
        }
    )

    result = four_factors_result = add_rolling_four_factors(games, window=10)

    assert math.isnan(result.loc[0, "efg_pct_roll"])
    assert not math.isnan(result.loc[1, "efg_pct_roll"])


def test_add_rolling_four_factors_uses_only_prior_games():
    from nba_predictor.features.four_factors import (
        add_rolling_four_factors,
        compute_four_factors,
    )

    games = pd.DataFrame(
        {
            "team": ["BOS", "BOS", "BOS"],
            "game_date": ["2026-10-01", "2026-10-03", "2026-10-05"],
            "fgm": [40, 42, 44],
            "fga": [90, 88, 86],
            "fg3m": [10, 12, 14],
            "tov": [12, 10, 8],
            "oreb": [10, 11, 12],
            "opp_dreb": [30, 28, 26],
            "fta": [20, 18, 16],
        }
    )

    result = add_rolling_four_factors(games, window=10)

    game1_factors = compute_four_factors(40, 90, 10, 12, 10, 30, 20)
    game2_factors = compute_four_factors(42, 88, 12, 10, 11, 28, 18)
    expected_third_row = (game1_factors["efg_pct"] + game2_factors["efg_pct"]) / 2

    assert result.loc[2, "efg_pct_roll"] == pytest.approx(expected_third_row)


def test_add_rolling_four_factors_keeps_teams_independent():
    from nba_predictor.features.four_factors import add_rolling_four_factors

    games = pd.DataFrame(
        {
            "team": ["BOS", "MIA"],
            "game_date": ["2026-10-01", "2026-10-01"],
            "fgm": [40, 35],
            "fga": [90, 85],
            "fg3m": [10, 8],
            "tov": [12, 15],
            "oreb": [10, 9],
            "opp_dreb": [30, 32],
            "fta": [20, 19],
        }
    )

    result = add_rolling_four_factors(games, window=10)
    assert math.isnan(result.loc[0, "efg_pct_roll"])
    assert math.isnan(result.loc[1, "efg_pct_roll"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features_four_factors.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.features'`

- [ ] **Step 3: Create `src/nba_predictor/features/__init__.py`** (empty file)

```bash
touch src/nba_predictor/features/__init__.py
```

- [ ] **Step 4: Write `src/nba_predictor/features/four_factors.py`**

```python
import pandas as pd


def compute_four_factors(
    fgm: float,
    fga: float,
    fg3m: float,
    tov: float,
    oreb: float,
    opp_dreb: float,
    fta: float,
) -> dict:
    efg_pct = (fgm + 0.5 * fg3m) / fga if fga else 0.0
    tov_rate = tov / (fga + 0.44 * fta + tov) if (fga + 0.44 * fta + tov) else 0.0
    orb_pct = oreb / (oreb + opp_dreb) if (oreb + opp_dreb) else 0.0
    ft_rate = fta / fga if fga else 0.0
    return {
        "efg_pct": efg_pct,
        "tov_rate": tov_rate,
        "orb_pct": orb_pct,
        "ft_rate": ft_rate,
    }


def add_rolling_four_factors(games: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    games = games.sort_values(["team", "game_date"]).reset_index(drop=True)

    factor_rows = games.apply(
        lambda row: compute_four_factors(
            row["fgm"], row["fga"], row["fg3m"], row["tov"], row["oreb"], row["opp_dreb"], row["fta"]
        ),
        axis=1,
        result_type="expand",
    )
    games = pd.concat([games, factor_rows], axis=1)

    for factor in ["efg_pct", "tov_rate", "orb_pct", "ft_rate"]:
        games[f"{factor}_roll"] = games.groupby("team")[factor].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )

    return games
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_features_four_factors.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add src/nba_predictor/features/__init__.py src/nba_predictor/features/four_factors.py tests/test_features_four_factors.py
git commit -m "feat: add Four Factors feature module"
```

---

### Task 2: Ratings module (possessions, ORTG/DRTG/pace, Elo)

**Files:**
- Create: `src/nba_predictor/features/ratings.py`
- Test: `tests/test_features_ratings.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ratings.compute_possessions(fga: float, fta: float, oreb: float, tov: float) -> float`
  - `ratings.compute_offensive_rating(points: float, possessions: float) -> float` — points per 100 possessions.
  - `ratings.compute_pace(team_possessions: float, opp_possessions: float, minutes: float = 48.0) -> float` — average possessions per 48 minutes.
  - `ratings.elo_expected(rating_a: float, rating_b: float, home_adjustment: float = 0.0) -> float` — probability A beats B.
  - `ratings.elo_update(rating: float, expected: float, actual: float, k: float = 20.0) -> float` — new rating after a result (`actual` is 1.0/0.0).
  - `ratings.init_elo_ratings(team_abbreviations: list[str], base_rating: float = 1500.0) -> dict[str, float]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features_ratings.py
import pytest


def test_compute_possessions_known_value():
    from nba_predictor.features.ratings import compute_possessions

    result = compute_possessions(fga=90, fta=20, oreb=10, tov=12)
    assert result == pytest.approx(90 - 10 + 12 + 0.44 * 20)


def test_compute_offensive_rating_known_value():
    from nba_predictor.features.ratings import compute_offensive_rating

    assert compute_offensive_rating(points=110, possessions=100) == pytest.approx(110.0)


def test_compute_offensive_rating_zero_possessions_does_not_raise():
    from nba_predictor.features.ratings import compute_offensive_rating

    assert compute_offensive_rating(points=0, possessions=0) == 0.0


def test_compute_pace_averages_both_teams():
    from nba_predictor.features.ratings import compute_pace

    result = compute_pace(team_possessions=98, opp_possessions=102, minutes=48.0)
    assert result == pytest.approx(100.0)


def test_elo_expected_equal_ratings_is_half():
    from nba_predictor.features.ratings import elo_expected

    assert elo_expected(1500, 1500) == pytest.approx(0.5)


def test_elo_expected_higher_rating_favored():
    from nba_predictor.features.ratings import elo_expected

    assert elo_expected(1600, 1500) > 0.5


def test_elo_expected_home_adjustment_increases_home_probability():
    from nba_predictor.features.ratings import elo_expected

    no_adjustment = elo_expected(1500, 1500, home_adjustment=0)
    with_adjustment = elo_expected(1500, 1500, home_adjustment=100)
    assert with_adjustment > no_adjustment


def test_elo_update_win_increases_rating():
    from nba_predictor.features.ratings import elo_update

    new_rating = elo_update(rating=1500, expected=0.5, actual=1.0, k=20)
    assert new_rating == pytest.approx(1510.0)


def test_elo_update_loss_decreases_rating():
    from nba_predictor.features.ratings import elo_update

    new_rating = elo_update(rating=1500, expected=0.5, actual=0.0, k=20)
    assert new_rating == pytest.approx(1490.0)


def test_init_elo_ratings_sets_base_for_all_teams():
    from nba_predictor.features.ratings import init_elo_ratings

    ratings = init_elo_ratings(["BOS", "MIA", "LAL"], base_rating=1500.0)
    assert ratings == {"BOS": 1500.0, "MIA": 1500.0, "LAL": 1500.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features_ratings.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.features.ratings'`

- [ ] **Step 3: Write `src/nba_predictor/features/ratings.py`**

```python
def compute_possessions(fga: float, fta: float, oreb: float, tov: float) -> float:
    return fga - oreb + tov + 0.44 * fta


def compute_offensive_rating(points: float, possessions: float) -> float:
    return (points / possessions) * 100 if possessions else 0.0


def compute_pace(team_possessions: float, opp_possessions: float, minutes: float = 48.0) -> float:
    return (team_possessions + opp_possessions) / 2 * (48.0 / minutes) if minutes else 0.0


def elo_expected(rating_a: float, rating_b: float, home_adjustment: float = 0.0) -> float:
    adjusted_a = rating_a + home_adjustment
    return 1.0 / (1.0 + 10 ** ((rating_b - adjusted_a) / 400))


def elo_update(rating: float, expected: float, actual: float, k: float = 20.0) -> float:
    return rating + k * (actual - expected)


def init_elo_ratings(team_abbreviations: list[str], base_rating: float = 1500.0) -> dict[str, float]:
    return {team: base_rating for team in team_abbreviations}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features_ratings.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/features/ratings.py tests/test_features_ratings.py
git commit -m "feat: add ratings module (possessions, ORTG, pace, Elo)"
```

---

### Task 3: Rest/travel/fatigue module

**Files:**
- Create: `src/nba_predictor/features/rest_travel.py`
- Test: `tests/test_features_rest_travel.py`

**Interfaces:**
- Consumes: `nba_predictor.data.team_reference.get_team` (Phase 1) for arena lat/lon and timezone lookups.
- Produces:
  - `rest_travel.haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float`
  - `rest_travel.compute_rest_days(current_date: str, previous_game_date: str | None) -> int` — `date.fromisoformat` difference in days; returns a large number (`99`) if `previous_game_date` is `None` (season opener, no fatigue penalty).
  - `rest_travel.is_back_to_back(rest_days: int) -> bool`
  - `rest_travel.congestion_flags(prior_game_dates: list[str], current_date: str) -> dict` — `{"three_in_four": bool, "four_in_six": bool}`, where `prior_game_dates` is that team's games strictly before `current_date`, using the 3 most recent dates within a 4-day window (inclusive of `current_date`) for the first flag and 4 most recent within a 6-day window for the second.
  - `rest_travel.rolling_travel_miles(game_locations: list[str], window_days: int = 7) -> float` — `game_locations` is an ordered list of team abbreviations (the *venue*, i.e. home team of each game played) the team traveled to, most recent last; sums haversine distance between consecutive venues. (Caller is responsible for pre-filtering to the trailing `window_days`.)
  - `rest_travel.timezone_change_count(game_locations: list[str]) -> int` — count of consecutive venue pairs whose `team_reference` timezones differ.
  - `rest_travel.fatigue_index(travel_miles: float, timezone_changes: int, rest_days: int) -> float` — composite: `travel_miles / 500 + timezone_changes * 1.5 + max(0, 2 - rest_days)`, higher = more fatigued.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features_rest_travel.py
import pytest


def test_haversine_miles_same_point_is_zero():
    from nba_predictor.features.rest_travel import haversine_miles

    assert haversine_miles(40.0, -75.0, 40.0, -75.0) == pytest.approx(0.0, abs=1e-6)


def test_haversine_miles_boston_to_lakers_is_roughly_2600_miles():
    from nba_predictor.features.rest_travel import haversine_miles

    distance = haversine_miles(42.3662, -71.0621, 34.0430, -118.2673)
    assert 2550 < distance < 2650


def test_compute_rest_days_counts_calendar_days():
    from nba_predictor.features.rest_travel import compute_rest_days

    assert compute_rest_days("2026-11-05", "2026-11-03") == 2


def test_compute_rest_days_zero_for_back_to_back():
    from nba_predictor.features.rest_travel import compute_rest_days

    assert compute_rest_days("2026-11-05", "2026-11-04") == 1


def test_compute_rest_days_large_for_season_opener():
    from nba_predictor.features.rest_travel import compute_rest_days

    assert compute_rest_days("2026-10-21", None) == 99


def test_is_back_to_back_true_for_one_day_rest():
    from nba_predictor.features.rest_travel import is_back_to_back

    assert is_back_to_back(1) is True
    assert is_back_to_back(2) is False


def test_congestion_flags_three_in_four():
    from nba_predictor.features.rest_travel import congestion_flags

    flags = congestion_flags(
        prior_game_dates=["2026-11-01", "2026-11-03"], current_date="2026-11-04"
    )
    assert flags["three_in_four"] is True
    assert flags["four_in_six"] is False


def test_congestion_flags_four_in_six():
    from nba_predictor.features.rest_travel import congestion_flags

    flags = congestion_flags(
        prior_game_dates=["2026-11-01", "2026-11-02", "2026-11-04"],
        current_date="2026-11-06",
    )
    assert flags["four_in_six"] is True


def test_congestion_flags_false_when_well_rested():
    from nba_predictor.features.rest_travel import congestion_flags

    flags = congestion_flags(prior_game_dates=["2026-10-28"], current_date="2026-11-04")
    assert flags["three_in_four"] is False
    assert flags["four_in_six"] is False


def test_rolling_travel_miles_sums_consecutive_hops():
    from nba_predictor.features.rest_travel import rolling_travel_miles

    total = rolling_travel_miles(["BOS", "MIA", "LAL"])
    assert total > 0


def test_rolling_travel_miles_zero_for_single_location():
    from nba_predictor.features.rest_travel import rolling_travel_miles

    assert rolling_travel_miles(["BOS"]) == 0.0


def test_timezone_change_count_counts_differences():
    from nba_predictor.features.rest_travel import timezone_change_count

    assert timezone_change_count(["BOS", "MIA", "LAL"]) == 1  # NY->NY (0), NY->LA (1)
    assert timezone_change_count(["BOS", "LAL", "GSW"]) == 1  # NY->LA (1), LA->LA (0)


def test_fatigue_index_increases_with_travel_and_fewer_rest_days():
    from nba_predictor.features.rest_travel import fatigue_index

    low_fatigue = fatigue_index(travel_miles=0, timezone_changes=0, rest_days=3)
    high_fatigue = fatigue_index(travel_miles=2500, timezone_changes=3, rest_days=0)
    assert high_fatigue > low_fatigue
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features_rest_travel.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.features.rest_travel'`

- [ ] **Step 3: Write `src/nba_predictor/features/rest_travel.py`**

```python
import math
from datetime import date

from nba_predictor.data.team_reference import get_team


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_miles = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_miles * c


def compute_rest_days(current_date: str, previous_game_date: str | None) -> int:
    if previous_game_date is None:
        return 99
    current = date.fromisoformat(current_date)
    previous = date.fromisoformat(previous_game_date)
    return (current - previous).days - 1


def is_back_to_back(rest_days: int) -> bool:
    return rest_days == 0


def congestion_flags(prior_game_dates: list[str], current_date: str) -> dict:
    current = date.fromisoformat(current_date)
    all_dates = sorted(date.fromisoformat(d) for d in prior_game_dates) + [current]

    def games_within(days: int) -> int:
        cutoff = current.toordinal() - days
        return sum(1 for d in all_dates if d.toordinal() > cutoff)

    return {
        "three_in_four": games_within(3) >= 3,
        "four_in_six": games_within(5) >= 4,
    }


def rolling_travel_miles(game_locations: list[str], window_days: int = 7) -> float:
    if len(game_locations) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(game_locations, game_locations[1:]):
        team_a, team_b = get_team(a), get_team(b)
        total += haversine_miles(team_a.arena_lat, team_a.arena_lon, team_b.arena_lat, team_b.arena_lon)
    return total


def timezone_change_count(game_locations: list[str]) -> int:
    if len(game_locations) < 2:
        return 0
    changes = 0
    for a, b in zip(game_locations, game_locations[1:]):
        if get_team(a).timezone != get_team(b).timezone:
            changes += 1
    return changes


def fatigue_index(travel_miles: float, timezone_changes: int, rest_days: int) -> float:
    return travel_miles / 500 + timezone_changes * 1.5 + max(0, 2 - rest_days)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features_rest_travel.py -v`
Expected: PASS (13 passed)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/features/rest_travel.py tests/test_features_rest_travel.py
git commit -m "feat: add rest/travel/fatigue feature module"
```

---

### Task 4: Injuries feature module

**Files:**
- Create: `src/nba_predictor/features/injuries.py`
- Test: `tests/test_features_injuries.py`

**Interfaces:**
- Consumes: `nba_predictor.data.injuries.get_missing_player_value(player_id: int, season: int) -> float` (Phase 2).
- Produces:
  - `injuries.missing_players_value(injured_player_ids: list[int], season: int, value_fn=None) -> float` — sums `get_missing_player_value` (or the injected `value_fn`, for testability) across all given player ids; returns `0.0` for an empty list.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features_injuries.py
def test_missing_players_value_sums_across_players():
    from nba_predictor.features.injuries import missing_players_value

    def fake_value_fn(player_id, season):
        return {101: 5.0, 102: 3.0}[player_id]

    total = missing_players_value([101, 102], season=2026, value_fn=fake_value_fn)
    assert total == 8.0


def test_missing_players_value_empty_list_is_zero():
    from nba_predictor.features.injuries import missing_players_value

    assert missing_players_value([], season=2026) == 0.0


def test_missing_players_value_uses_real_data_module_by_default(monkeypatch):
    from nba_predictor.features import injuries as injuries_feature
    from nba_predictor.data import injuries as injuries_data

    monkeypatch.setattr(injuries_data, "get_missing_player_value", lambda player_id, season: 2.5)

    total = injuries_feature.missing_players_value([1, 2, 3], season=2026)
    assert total == 7.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features_injuries.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.features.injuries'`

- [ ] **Step 3: Write `src/nba_predictor/features/injuries.py`**

```python
from nba_predictor.data import injuries as injuries_data


def missing_players_value(injured_player_ids: list[int], season: int, value_fn=None) -> float:
    fn = value_fn or injuries_data.get_missing_player_value
    return sum(fn(player_id, season) for player_id in injured_player_ids)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features_injuries.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/features/injuries.py tests/test_features_injuries.py
git commit -m "feat: add injuries feature module"
```

---

### Task 5: Game context module (head-to-head, streaks, altitude, conference/division)

**Files:**
- Create: `src/nba_predictor/features/context.py`
- Test: `tests/test_features_context.py`

**Interfaces:**
- Consumes: `nba_predictor.data.team_reference.get_team` (Phase 1).
- Produces:
  - `context.head_to_head_record(results: list[str], perspective: str = "team_a") -> dict` — `results` is an ordered list of `"team_a"`/`"team_b"` (the winner of each past meeting, oldest first); returns `{"team_a_wins": int, "team_b_wins": int}`.
  - `context.current_streak(results: list[str]) -> int` — `results` is an ordered list of `"W"`/`"L"` for one team, oldest first; returns a positive int for an active win streak, negative for an active loss streak, `0` for an empty list.
  - `context.is_high_altitude(home_team_abbr: str) -> bool` — `True` when `team_reference.get_team(home_team_abbr).altitude_ft > 1000`.
  - `context.game_flags(home_team_abbr: str, away_team_abbr: str) -> dict` — `{"conference_game": bool, "division_game": bool}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features_context.py
def test_head_to_head_record_counts_wins():
    from nba_predictor.features.context import head_to_head_record

    result = head_to_head_record(["team_a", "team_b", "team_a"])
    assert result == {"team_a_wins": 2, "team_b_wins": 1}


def test_head_to_head_record_empty_history():
    from nba_predictor.features.context import head_to_head_record

    assert head_to_head_record([]) == {"team_a_wins": 0, "team_b_wins": 0}


def test_current_streak_positive_for_win_streak():
    from nba_predictor.features.context import current_streak

    assert current_streak(["L", "W", "W", "W"]) == 3


def test_current_streak_negative_for_loss_streak():
    from nba_predictor.features.context import current_streak

    assert current_streak(["W", "L", "L"]) == -2


def test_current_streak_empty_is_zero():
    from nba_predictor.features.context import current_streak

    assert current_streak([]) == 0


def test_is_high_altitude_true_for_denver():
    from nba_predictor.features.context import is_high_altitude

    assert is_high_altitude("DEN") is True


def test_is_high_altitude_false_for_boston():
    from nba_predictor.features.context import is_high_altitude

    assert is_high_altitude("BOS") is False


def test_game_flags_conference_and_division_game():
    from nba_predictor.features.context import game_flags

    flags = game_flags("BOS", "NYK")  # both Atlantic, both East
    assert flags == {"conference_game": True, "division_game": True}


def test_game_flags_conference_only():
    from nba_predictor.features.context import game_flags

    flags = game_flags("BOS", "MIA")  # both East, different division
    assert flags == {"conference_game": True, "division_game": False}


def test_game_flags_non_conference_game():
    from nba_predictor.features.context import game_flags

    flags = game_flags("BOS", "LAL")  # East vs West
    assert flags == {"conference_game": False, "division_game": False}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features_context.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.features.context'`

- [ ] **Step 3: Write `src/nba_predictor/features/context.py`**

```python
from nba_predictor.data.team_reference import get_team


def head_to_head_record(results: list[str], perspective: str = "team_a") -> dict:
    return {
        "team_a_wins": results.count("team_a"),
        "team_b_wins": results.count("team_b"),
    }


def current_streak(results: list[str]) -> int:
    if not results:
        return 0
    streak_char = results[-1]
    streak = 0
    for result in reversed(results):
        if result != streak_char:
            break
        streak += 1
    return streak if streak_char == "W" else -streak


def is_high_altitude(home_team_abbr: str) -> bool:
    return get_team(home_team_abbr).altitude_ft > 1000


def game_flags(home_team_abbr: str, away_team_abbr: str) -> dict:
    home, away = get_team(home_team_abbr), get_team(away_team_abbr)
    conference_game = home.conference == away.conference
    division_game = conference_game and home.division == away.division
    return {"conference_game": conference_game, "division_game": division_game}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features_context.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/features/context.py tests/test_features_context.py
git commit -m "feat: add game context feature module"
```

---

### Task 6: Feature assembly (`build.py`)

**Files:**
- Create: `src/nba_predictor/features/build.py`
- Test: `tests/test_features_build.py`

**Interfaces:**
- Consumes: `four_factors.add_rolling_four_factors`, all pure functions from `ratings.py`, `rest_travel.py`, `context.py` (Tasks 1-5).
- Produces:
  - `build.FEATURE_COLUMNS: list[str]` — the exact ordered list of engineered feature column names a model consumes (see Step 3).
  - `build.build_training_frame(games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]` — `games` is one row per game with columns: `game_id, game_date, home_team, away_team, home_pts, away_pts, home_fgm, home_fga, home_fg3m, home_tov, home_oreb, home_dreb, home_fta, away_fgm, away_fga, away_fg3m, away_tov, away_oreb, away_dreb, away_fta, home_win` (int 0/1, the label). Returns `(games_with_features, FEATURE_COLUMNS)` — the input frame with all engineered columns appended, plus the column-name list for model training. Rows with any `NaN` in `FEATURE_COLUMNS` (i.e. each team's first tracked game) are dropped before returning.

**Note on scope:** this task computes the Four-Factors and rest-day features
directly (both only need this game's own box score / date history, already
present in `games`). Elo power rating, travel/fatigue, and injury features
require running state across the whole season (an Elo ladder that updates
game-by-game, a travel log) or external data (injury reports) that Phase 4's
API layer will maintain and pass in separately — `build_training_frame`
accepts them as **optional** pre-computed columns on the input `games` frame
(`home_power_rating`, `away_power_rating`, `home_fatigue_index`,
`away_fatigue_index`, `home_missing_value`, `away_missing_value`) and, if
absent, fills them with `0.0` so the frame is always trainable end-to-end
even before those pipelines exist.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features_build.py
import pandas as pd
import pytest


def _sample_games() -> pd.DataFrame:
    rows = []
    teams = ["BOS", "MIA"]
    dates = ["2026-10-21", "2026-10-23", "2026-10-25", "2026-10-27"]
    for i, game_date in enumerate(dates):
        home, away = (teams[0], teams[1]) if i % 2 == 0 else (teams[1], teams[0])
        rows.append(
            {
                "game_id": f"g{i}",
                "game_date": game_date,
                "home_team": home,
                "away_team": away,
                "home_pts": 110 + i,
                "away_pts": 105 + i,
                "home_fgm": 40, "home_fga": 88, "home_fg3m": 12, "home_tov": 11,
                "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
                "away_fgm": 38, "away_fga": 90, "away_fg3m": 10, "away_tov": 13,
                "away_oreb": 10, "away_dreb": 30, "away_fta": 18,
                "home_win": 1,
            }
        )
    return pd.DataFrame(rows)


def test_build_training_frame_returns_feature_columns():
    from nba_predictor.features.build import FEATURE_COLUMNS, build_training_frame

    games = _sample_games()
    result_df, feature_cols = build_training_frame(games)

    assert feature_cols == FEATURE_COLUMNS
    for col in feature_cols:
        assert col in result_df.columns


def test_build_training_frame_drops_rows_with_no_rolling_history():
    from nba_predictor.features.build import build_training_frame

    games = _sample_games()
    result_df, feature_cols = build_training_frame(games)

    assert len(result_df) < len(games)
    assert result_df[feature_cols].isna().sum().sum() == 0


def test_build_training_frame_fills_missing_optional_columns_with_zero():
    from nba_predictor.features.build import build_training_frame

    games = _sample_games()
    result_df, _ = build_training_frame(games)

    assert (result_df["home_power_rating"] == 0.0).all()
    assert (result_df["home_fatigue_index"] == 0.0).all()
    assert (result_df["home_missing_value"] == 0.0).all()


def test_build_training_frame_respects_precomputed_optional_columns():
    from nba_predictor.features.build import build_training_frame

    games = _sample_games()
    games["home_power_rating"] = 1550.0
    result_df, _ = build_training_frame(games)

    assert (result_df["home_power_rating"] == 1550.0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features_build.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.features.build'`

- [ ] **Step 3: Write `src/nba_predictor/features/build.py`**

```python
import pandas as pd

from nba_predictor.features.context import current_streak, game_flags, is_high_altitude
from nba_predictor.features.four_factors import add_rolling_four_factors
from nba_predictor.features.rest_travel import compute_rest_days, is_back_to_back

FEATURE_COLUMNS = [
    "home_efg_pct_roll", "home_tov_rate_roll", "home_orb_pct_roll", "home_ft_rate_roll",
    "away_efg_pct_roll", "away_tov_rate_roll", "away_orb_pct_roll", "away_ft_rate_roll",
    "home_power_rating", "away_power_rating", "power_rating_diff",
    "home_rest_days", "away_rest_days", "home_back_to_back", "away_back_to_back",
    "home_fatigue_index", "away_fatigue_index",
    "home_missing_value", "away_missing_value",
    "home_streak", "away_streak",
    "is_high_altitude", "conference_game", "division_game",
]

_OPTIONAL_COLUMNS_DEFAULT_ZERO = [
    "home_power_rating", "away_power_rating",
    "home_fatigue_index", "away_fatigue_index",
    "home_missing_value", "away_missing_value",
]


def _long_format_box_scores(games: pd.DataFrame) -> pd.DataFrame:
    home_rows = games.rename(
        columns={
            "home_team": "team", "home_fgm": "fgm", "home_fga": "fga", "home_fg3m": "fg3m",
            "home_tov": "tov", "home_oreb": "oreb", "away_dreb": "opp_dreb", "home_fta": "fta",
        }
    )[["game_id", "game_date", "team", "fgm", "fga", "fg3m", "tov", "oreb", "opp_dreb", "fta"]]

    away_rows = games.rename(
        columns={
            "away_team": "team", "away_fgm": "fgm", "away_fga": "fga", "away_fg3m": "fg3m",
            "away_tov": "tov", "away_oreb": "oreb", "home_dreb": "opp_dreb", "away_fta": "fta",
        }
    )[["game_id", "game_date", "team", "fgm", "fga", "fg3m", "tov", "oreb", "opp_dreb", "fta"]]

    return pd.concat([home_rows, away_rows], ignore_index=True)


def build_training_frame(games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features_build.py -v`
Expected: PASS (4 passed). If column-naming mismatches surface (e.g. the
`home_{factor}_roll` merge producing `home_efg_pct_roll_roll`), fix the
`rename`/column-selection logic in `build_training_frame` until
`FEATURE_COLUMNS` all resolve with no `NaN`-only columns — do not change
`FEATURE_COLUMNS` itself, since Task 7-8's models are written against that
exact list.

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/features/build.py tests/test_features_build.py
git commit -m "feat: add feature assembly pipeline (build_training_frame)"
```

---

### Task 7: Game outcome models

**Files:**
- Create: `src/nba_predictor/models/__init__.py`
- Create: `src/nba_predictor/models/game_outcome.py`
- Test: `tests/test_models_game_outcome.py`

**Interfaces:**
- Consumes: `features.build.FEATURE_COLUMNS` (Task 6) as the expected column
  set of `X`.
- Produces:
  - `game_outcome.train_win_probability_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgboost.XGBClassifier`
  - `game_outcome.train_margin_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgboost.XGBRegressor`
  - `game_outcome.train_total_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgboost.XGBRegressor`
  - `game_outcome.predict_win_probability(model: "xgboost.XGBClassifier", X: pd.DataFrame) -> "numpy.ndarray"` — probability of the positive class (home win).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models_game_outcome.py
import numpy as np
import pandas as pd
import pytest


def _synthetic_dataset(n=120, seed=0):
    rng = np.random.default_rng(seed)
    power_diff = rng.normal(0, 100, n)
    X = pd.DataFrame(
        {
            "power_rating_diff": power_diff,
            "home_rest_days": rng.integers(0, 4, n),
            "away_rest_days": rng.integers(0, 4, n),
        }
    )
    win_prob = 1 / (1 + np.exp(-power_diff / 100))
    y_win = (rng.random(n) < win_prob).astype(int)
    y_margin = power_diff / 20 + rng.normal(0, 5, n)
    y_total = 220 + rng.normal(0, 10, n)
    return X, pd.Series(y_win), pd.Series(y_margin), pd.Series(y_total)


def test_train_win_probability_model_predicts_valid_probabilities():
    from nba_predictor.models.game_outcome import (
        predict_win_probability,
        train_win_probability_model,
    )

    X, y_win, _, _ = _synthetic_dataset()
    model = train_win_probability_model(X, y_win, n_estimators=20, max_depth=2)
    probs = predict_win_probability(model, X)

    assert len(probs) == len(X)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_train_margin_model_returns_numeric_predictions():
    from nba_predictor.models.game_outcome import train_margin_model

    X, _, y_margin, _ = _synthetic_dataset()
    model = train_margin_model(X, y_margin, n_estimators=20, max_depth=2)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert np.isfinite(predictions).all()


def test_train_total_model_returns_numeric_predictions():
    from nba_predictor.models.game_outcome import train_total_model

    X, _, _, y_total = _synthetic_dataset()
    model = train_total_model(X, y_total, n_estimators=20, max_depth=2)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert np.isfinite(predictions).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models_game_outcome.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.models'`

- [ ] **Step 3: Create `src/nba_predictor/models/__init__.py`** (empty file)

```bash
touch src/nba_predictor/models/__init__.py
```

- [ ] **Step 4: Write `src/nba_predictor/models/game_outcome.py`**

```python
import numpy as np
import pandas as pd
import xgboost as xgb


def train_win_probability_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgb.XGBClassifier:
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, **xgb_params}
    model = xgb.XGBClassifier(**params, eval_metric="logloss")
    model.fit(X, y)
    return model


def train_margin_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgb.XGBRegressor:
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, **xgb_params}
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def train_total_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgb.XGBRegressor:
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, **xgb_params}
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def predict_win_probability(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_models_game_outcome.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/nba_predictor/models/__init__.py src/nba_predictor/models/game_outcome.py tests/test_models_game_outcome.py
git commit -m "feat: add game outcome models (win prob, margin, total)"
```

---

### Task 8: Player prop models

**Files:**
- Create: `src/nba_predictor/models/player_props.py`
- Test: `tests/test_models_player_props.py`

**Interfaces:**
- Consumes: nothing new (same `xgboost` pattern as Task 7).
- Produces:
  - `player_props.STAT_TARGETS: list[str]` — `["points", "rebounds", "assists", "threes"]`.
  - `player_props.train_player_stat_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgboost.XGBRegressor`
  - `player_props.train_double_double_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgboost.XGBClassifier`
  - `player_props.predict_player_stat(model: "xgboost.XGBRegressor", X: pd.DataFrame) -> "numpy.ndarray"`
  - `player_props.predict_double_double_probability(model: "xgboost.XGBClassifier", X: pd.DataFrame) -> "numpy.ndarray"`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models_player_props.py
import numpy as np
import pandas as pd


def _synthetic_player_dataset(n=100, seed=1):
    rng = np.random.default_rng(seed)
    minutes = rng.uniform(15, 38, n)
    usage = rng.uniform(0.12, 0.32, n)
    X = pd.DataFrame({"minutes_roll": minutes, "usage_rate_roll": usage})
    y_points = minutes * usage * 40 + rng.normal(0, 3, n)
    y_double_double = (minutes > 30).astype(int)
    return X, pd.Series(y_points), pd.Series(y_double_double)


def test_stat_targets_lists_four_stats():
    from nba_predictor.models.player_props import STAT_TARGETS

    assert STAT_TARGETS == ["points", "rebounds", "assists", "threes"]


def test_train_player_stat_model_predicts_numeric_values():
    from nba_predictor.models.player_props import (
        predict_player_stat,
        train_player_stat_model,
    )

    X, y_points, _ = _synthetic_player_dataset()
    model = train_player_stat_model(X, y_points, n_estimators=20, max_depth=2)
    predictions = predict_player_stat(model, X)

    assert len(predictions) == len(X)
    assert np.isfinite(predictions).all()


def test_train_double_double_model_predicts_valid_probabilities():
    from nba_predictor.models.player_props import (
        predict_double_double_probability,
        train_double_double_model,
    )

    X, _, y_double_double = _synthetic_player_dataset()
    model = train_double_double_model(X, y_double_double, n_estimators=20, max_depth=2)
    probs = predict_double_double_probability(model, X)

    assert len(probs) == len(X)
    assert ((probs >= 0) & (probs <= 1)).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models_player_props.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.models.player_props'`

- [ ] **Step 3: Write `src/nba_predictor/models/player_props.py`**

```python
import numpy as np
import pandas as pd
import xgboost as xgb

STAT_TARGETS = ["points", "rebounds", "assists", "threes"]


def train_player_stat_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgb.XGBRegressor:
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, **xgb_params}
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def train_double_double_model(X: pd.DataFrame, y: pd.Series, **xgb_params) -> xgb.XGBClassifier:
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, **xgb_params}
    model = xgb.XGBClassifier(**params, eval_metric="logloss")
    model.fit(X, y)
    return model


def predict_player_stat(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def predict_double_double_probability(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models_player_props.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/models/player_props.py tests/test_models_player_props.py
git commit -m "feat: add player prop models"
```

---

### Task 9: Training manifest

**Files:**
- Create: `src/nba_predictor/models/manifest.py`
- Test: `tests/test_models_manifest.py`

**Interfaces:**
- Consumes: nothing (pure dict/JSON I/O).
- Produces:
  - `manifest.build_manifest(model_names: list[str], metrics: dict, model_version: str, trained_at: str) -> dict` — `{"model_version": ..., "trained_at": ..., "models": [...], "metrics": {...}}`.
  - `manifest.write_manifest(manifest: dict, path: Path) -> None` — writes pretty-printed JSON.
  - `manifest.append_manifest_history(manifest: dict, history_path: Path) -> None` — appends one JSON line (JSONL) to `history_path`, creating it if absent.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models_manifest.py
import json


def test_build_manifest_shape():
    from nba_predictor.models.manifest import build_manifest

    manifest = build_manifest(
        model_names=["win_probability", "margin", "total"],
        metrics={"win_probability": {"accuracy": 0.65}},
        model_version="v1",
        trained_at="2026-11-01T00:00:00",
    )

    assert manifest["model_version"] == "v1"
    assert manifest["trained_at"] == "2026-11-01T00:00:00"
    assert manifest["models"] == ["win_probability", "margin", "total"]
    assert manifest["metrics"]["win_probability"]["accuracy"] == 0.65


def test_write_manifest_creates_valid_json(tmp_path):
    from nba_predictor.models.manifest import build_manifest, write_manifest

    manifest = build_manifest(["win_probability"], {}, "v1", "2026-11-01T00:00:00")
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)

    with open(path) as f:
        loaded = json.load(f)
    assert loaded == manifest


def test_append_manifest_history_adds_one_line_per_call(tmp_path):
    from nba_predictor.models.manifest import append_manifest_history, build_manifest

    history_path = tmp_path / "manifest_history.jsonl"
    manifest_v1 = build_manifest(["win_probability"], {}, "v1", "2026-11-01T00:00:00")
    manifest_v2 = build_manifest(["win_probability"], {}, "v2", "2026-11-02T00:00:00")

    append_manifest_history(manifest_v1, history_path)
    append_manifest_history(manifest_v2, history_path)

    lines = history_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["model_version"] == "v1"
    assert json.loads(lines[1])["model_version"] == "v2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models_manifest.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.models.manifest'`

- [ ] **Step 3: Write `src/nba_predictor/models/manifest.py`**

```python
import json
from pathlib import Path


def build_manifest(model_names: list[str], metrics: dict, model_version: str, trained_at: str) -> dict:
    return {
        "model_version": model_version,
        "trained_at": trained_at,
        "models": model_names,
        "metrics": metrics,
    }


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def append_manifest_history(manifest: dict, history_path: Path) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a") as f:
        f.write(json.dumps(manifest) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models_manifest.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/nba_predictor/models/manifest.py tests/test_models_manifest.py
git commit -m "feat: add training manifest builder and writer"
```

---

### Task 10: Walk-forward split and calibration evaluation

**Files:**
- Create: `src/nba_predictor/models/evaluate/__init__.py`
- Create: `src/nba_predictor/models/evaluate/walk_forward.py`
- Create: `src/nba_predictor/models/evaluate/calibration.py`
- Test: `tests/test_models_evaluate.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `walk_forward.chronological_split(df: pd.DataFrame, date_col: str, holdout_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]` — sorts by `date_col` ascending, returns `(train_df, holdout_df)` where `holdout_df` is the most recent `holdout_fraction` of rows (rounded down, minimum 1 row if `len(df) > 1`). Raises `ValueError` if `df` has fewer than 2 rows.
  - `calibration.compute_calibration_bins(y_true: "numpy.ndarray", y_prob: "numpy.ndarray", n_bins: int = 10) -> list[dict]` — bins predictions into `n_bins` equal-width `[0,1]` buckets; each dict is `{"bin_start": float, "bin_end": float, "predicted_rate": float, "actual_rate": float, "count": int}`; bins with zero predictions are omitted.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models_evaluate.py
import numpy as np
import pandas as pd
import pytest


def test_chronological_split_holdout_is_most_recent():
    from nba_predictor.models.evaluate.walk_forward import chronological_split

    df = pd.DataFrame(
        {
            "game_date": pd.date_range("2026-10-01", periods=10).astype(str),
            "value": range(10),
        }
    )
    train, holdout = chronological_split(df, date_col="game_date", holdout_fraction=0.2)

    assert len(train) == 8
    assert len(holdout) == 2
    assert train["game_date"].max() < holdout["game_date"].min()


def test_chronological_split_handles_unsorted_input():
    from nba_predictor.models.evaluate.walk_forward import chronological_split

    df = pd.DataFrame(
        {
            "game_date": ["2026-11-03", "2026-11-01", "2026-11-02"],
            "value": [3, 1, 2],
        }
    )
    train, holdout = chronological_split(df, date_col="game_date", holdout_fraction=0.34)

    assert holdout["value"].tolist() == [3]


def test_chronological_split_raises_on_too_few_rows():
    from nba_predictor.models.evaluate.walk_forward import chronological_split

    df = pd.DataFrame({"game_date": ["2026-11-01"], "value": [1]})
    with pytest.raises(ValueError):
        chronological_split(df, date_col="game_date")


def test_compute_calibration_bins_perfect_calibration():
    from nba_predictor.models.evaluate.calibration import compute_calibration_bins

    y_prob = np.array([0.05] * 20 + [0.95] * 20)
    y_true = np.array([0] * 19 + [1] + [1] * 19 + [0])

    bins = compute_calibration_bins(y_true, y_prob, n_bins=10)

    assert len(bins) == 2
    low_bin = next(b for b in bins if b["bin_start"] < 0.5)
    high_bin = next(b for b in bins if b["bin_start"] >= 0.5)
    assert low_bin["count"] == 20
    assert high_bin["count"] == 20
    assert low_bin["actual_rate"] == pytest.approx(0.05)
    assert high_bin["actual_rate"] == pytest.approx(0.95)


def test_compute_calibration_bins_omits_empty_bins():
    from nba_predictor.models.evaluate.calibration import compute_calibration_bins

    y_prob = np.array([0.5, 0.5])
    y_true = np.array([1, 0])

    bins = compute_calibration_bins(y_true, y_prob, n_bins=10)
    assert len(bins) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models_evaluate.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'nba_predictor.models.evaluate'`

- [ ] **Step 3: Create `src/nba_predictor/models/evaluate/__init__.py`** (empty file)

```bash
mkdir -p src/nba_predictor/models/evaluate
touch src/nba_predictor/models/evaluate/__init__.py
```

- [ ] **Step 4: Write `src/nba_predictor/models/evaluate/walk_forward.py`**

```python
import pandas as pd


def chronological_split(df: pd.DataFrame, date_col: str, holdout_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 2:
        raise ValueError("Need at least 2 rows to perform a chronological split")

    sorted_df = df.sort_values(date_col).reset_index(drop=True)
    holdout_size = max(1, int(len(sorted_df) * holdout_fraction))
    split_index = len(sorted_df) - holdout_size

    train = sorted_df.iloc[:split_index].reset_index(drop=True)
    holdout = sorted_df.iloc[split_index:].reset_index(drop=True)
    return train, holdout
```

- [ ] **Step 5: Write `src/nba_predictor/models/evaluate/calibration.py`**

```python
import numpy as np


def compute_calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []

    for i in range(n_bins):
        bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
        is_last_bin = i == n_bins - 1
        mask = (y_prob >= bin_start) & (y_prob < bin_end if not is_last_bin else y_prob <= bin_end)

        count = int(mask.sum())
        if count == 0:
            continue

        bins.append(
            {
                "bin_start": float(bin_start),
                "bin_end": float(bin_end),
                "predicted_rate": float(y_prob[mask].mean()),
                "actual_rate": float(y_true[mask].mean()),
                "count": count,
            }
        )

    return bins
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_models_evaluate.py -v`
Expected: PASS (5 passed)

- [ ] **Step 7: Commit**

```bash
git add src/nba_predictor/models/evaluate/ tests/test_models_evaluate.py
git commit -m "feat: add walk-forward split and calibration evaluation"
```

---

### Task 11: Full-suite smoke test

**Files:** No new files — verification only.

- [ ] **Step 1: Run the entire test suite**

Run: `pytest -v`
Expected: all Phase 1, 2, and 3 tests pass, 0 failed, 0 errors.

- [ ] **Step 2: End-to-end smoke test — build a training frame and train all models on it**

```bash
python - <<'EOF'
import pandas as pd
import numpy as np

from nba_predictor.features.build import build_training_frame
from nba_predictor.models.game_outcome import train_win_probability_model, predict_win_probability
from nba_predictor.models.manifest import build_manifest, write_manifest
from pathlib import Path

rng = np.random.default_rng(0)
teams = ["BOS", "MIA", "LAL", "GSW"]
dates = pd.date_range("2026-10-21", periods=40).astype(str)
rows = []
for i, game_date in enumerate(dates):
    home, away = teams[i % 4], teams[(i + 1) % 4]
    rows.append({
        "game_id": f"g{i}", "game_date": game_date, "home_team": home, "away_team": away,
        "home_pts": 110, "away_pts": 105,
        "home_fgm": 40, "home_fga": 88, "home_fg3m": 12, "home_tov": 11, "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
        "away_fgm": 38, "away_fga": 90, "away_fg3m": 10, "away_tov": 13, "away_oreb": 10, "away_dreb": 30, "away_fta": 18,
        "home_win": int(rng.random() > 0.4),
    })
games = pd.DataFrame(rows)

df, feature_cols = build_training_frame(games)
assert len(df) > 0, "no rows survived feature assembly"

model = train_win_probability_model(df[feature_cols], df["home_win"], n_estimators=20, max_depth=2)
probs = predict_win_probability(model, df[feature_cols])
assert ((probs >= 0) & (probs <= 1)).all()

manifest = build_manifest(["win_probability"], {"win_probability": {"n_samples": len(df)}}, "v0-smoke", "2026-11-01T00:00:00")
write_manifest(manifest, Path("/tmp/nba_predictor_smoke_manifest.json"))
print("Smoke test OK:", len(df), "training rows,", len(feature_cols), "features")
EOF
```

Expected output: `Smoke test OK: <N> training rows, 23 features` with no
traceback. If it errors on a `KeyError`/`NaN`-shape mismatch inside
`build_training_frame`, fix that function (not this smoke script) before
continuing — this is exactly the integration bug class Task 6's Step 4 note
warns about.

- [ ] **Step 3: Commit** (only if Step 2 required a fix to `build.py`)

```bash
git add src/nba_predictor/features/build.py
git commit -m "fix: correct feature assembly integration bug found in smoke test"
```

---

## Self-Review Notes

- **Spec coverage:** Four Factors, efficiency/pace, Elo power rating, rest/back-to-back/congestion, travel mileage/timezone/fatigue index, injuries, head-to-head/streaks, altitude, conference/division — every feature bullet in spec §4 has a task. Game outcome classifier + margin/total regressors and all four player-prop regressors + double-double classifier from spec §4/§5 are covered by Tasks 7-8. Manifest and walk-forward/calibration evaluation from spec §4's "Training/retraining" paragraph are covered by Tasks 9-10.
- **Placeholder scan:** no TBD/TODO; every step has runnable code.
- **Type consistency:** `FEATURE_COLUMNS` (Task 6) is the single source of truth every model task's `X` is built from; `predict_win_probability`/`predict_player_stat`/`predict_double_double_probability` names and signatures are consistent between their interface blocks and implementations.
- **Known risk flagged explicitly:** `build_training_frame`'s column-merge logic (Task 6) is the most complex code in this phase and is more likely than the rest to need a small fix once actually run — Task 11 Step 2 is a real integration smoke test specifically to catch that before Phase 4 depends on it.
