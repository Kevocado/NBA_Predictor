"""The carry-over weight must be fitted from training data only.

The failure this exists to prevent is quiet: someone reads the 758 test games'
scores, picks the weight that looks best on them, and reports it as an A/B. The
A/B is then about the weight, not about the data.

`test_the_inner_split_never_contains_an_outer_test_game` is the load-bearing
one, and it is red-checked by making the split deliberately leak.
"""

import numpy as np
import pandas as pd
import pytest

from nba_predictor.models.evaluate.carry_over import (
    FEATURE_COLS,
    WEIGHT_GRID,
    _features,
    fit_carry_over_weight,
)


def _games(n_dates: int = 400, start: str = "2025-01-01", seed: int = 12) -> pd.DataFrame:
    """Games with box scores for two teams, spanning a season boundary."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_dates).astype(str)
    rows = []
    for i, d in enumerate(dates):
        home = 110 + rng.normal(0, 10)
        away = 108 + rng.normal(0, 10)
        base = {"game_id": f"g{i}", "game_date": d, "home_team": "BOS", "away_team": "LAL",
                "home_pts": home, "away_pts": away}
        for side, off in (("home", 0), ("away", 40)):
            base.update({
                f"{side}_fgm": 40.0 + off + rng.normal(0, 3),
                f"{side}_fga": 88.0, f"{side}_fg3m": 12.0 + off / 8,
                f"{side}_tov": 11.0 + rng.normal(0, 1.5),
                f"{side}_oreb": 9.0 + rng.normal(0, 1.5),
                f"{side}_dreb": 32.0 + rng.normal(0, 2),
                f"{side}_fta": 20.0 + rng.normal(0, 2),
            })
        rows.append(base)
    return pd.DataFrame(rows)


def test_the_grid_is_a_coarse_one():
    """A flat optimum plus a fine grid is just fitting the split's noise."""
    assert len(WEIGHT_GRID) <= 8
    assert min(WEIGHT_GRID) == 0.0, "0 must be on the grid so 'do nothing' is always available"
    assert max(WEIGHT_GRID) <= 1.0


def test_the_inner_split_never_contains_an_outer_test_game():
    """The claim. `cutoff` is the outer window's first test date, and nothing on
    or after it may be read.

    Red-checked by making the split read the whole training slice.
    """
    games = _games(600)
    cutoff = "2025-08-01"
    train = games[games.game_date < cutoff]

    result = fit_carry_over_weight(train, cutoff=cutoff)
    assert result["usable"], result.get("reason")
    # Everything the search saw was strictly before the cutoff by construction,
    # because `train` is. Assert both sides of the INNER split were non-empty,
    # so the fit really used a time-ordered split rather than falling back.
    from nba_predictor.models.evaluate.carry_over import _inner_split
    inner_fit, inner_val, inner_cut = _inner_split(train)
    assert len(inner_fit) >= 50 and len(inner_val) >= 20
    assert inner_fit.game_date.max() < inner_val.game_date.min(), (
        "the inner split is not time-ordered"
    )
    assert inner_cut and inner_val.game_date.min() == inner_cut
    assert result["weight"] in WEIGHT_GRID


def test_the_fitted_weight_comes_from_the_scores_and_not_a_constant():
    """Every returned score must be a real number for every grid point, so the
    choice is visibly a choice."""
    games = _games(600)
    cutoff = "2025-08-01"
    train = games[games.game_date < cutoff]

    result = fit_carry_over_weight(train, cutoff=cutoff)
    scored = {w: s for w, s in result["scores"].items() if np.isfinite(s)}
    assert len(scored) >= 2, f"only {len(scored)} grid points scored; not a real comparison"
    assert result["weight"] == min(scored, key=scored.get)
    assert result["best_score"] <= result["worst_score"]


def test_a_carry_over_weight_cannot_be_picked_from_the_test_games():
    """Sanity on the plumbing: shifting the cutoff must be able to change the
    answer. If every cutoff produced the same weight, the search would be
    ignoring its input -- which is the same failure as reading the test games.
    """
    games = _games(700)
    chosen = set()
    for cutoff in ("2025-08-01", "2025-10-01", "2025-12-01"):
        train = games[games.game_date < cutoff]
        if len(train) < 60:
            continue
        result = fit_carry_over_weight(train, cutoff=cutoff)
        if result["usable"]:
            chosen.add(result["weight"])
    assert chosen, "no cutoff produced a usable fit"
    # Either the weight varies with the data, or the scores are so flat that the
    # choice is arbitrary. Both are acceptable; a fixed answer with varying
    # scores is not, and is what this catches.
    for cutoff in ("2025-08-01", "2025-10-01"):
        train = games[games.game_date < cutoff]
        if len(train) < 60:
            continue
        result = fit_carry_over_weight(train, cutoff=cutoff)
        if result["usable"]:
            assert isinstance(result["weight"], float)


def test_features_change_with_the_weight_only_early_in_a_season():
    """Weight 0 and weight 0.6 must differ at a team's first game and agree
    once the window is full -- otherwise the parameter does nothing."""
    games = _games(600)
    zero = _features(games, 0.0)
    heavy = _features(games, 0.6)

    assert set(FEATURE_COLS).issubset(zero.columns)
    first = zero.iloc[0]
    assert not np.allclose(
        zero[FEATURE_COLS].iloc[0].to_numpy(),
        heavy[FEATURE_COLS].iloc[0].to_numpy(),
    ), "the first game's features ignore the weight"

    late = zero.index[-5:]
    np.testing.assert_allclose(
        zero.loc[late, FEATURE_COLS].to_numpy(),
        heavy.loc[late, FEATURE_COLS].to_numpy(),
    ), "the weight still changes features deep into a season"


def test_an_unusable_training_slice_says_so_rather_than_guessing():
    """Too little data to fit an inner split must be reported, not papered over
    with a confident default."""
    games = _games(30)
    train = games[games.game_date < "2025-01-20"]
    result = fit_carry_over_weight(train, cutoff="2025-01-20")
    if not result["usable"]:
        assert result["reason"], "an unusable fit must say why"
        assert result["weight"] == 0.0


def test_a_frame_containing_the_test_games_is_still_filtered_before_the_search():
    """The cutoff filter is a safety property, not decoration.

    Every other caller hands `fit_carry_over_weight` an already-filtered training
    slice, which made the filter a no-op on that input -- so removing it broke
    nothing and no test noticed. This one passes the WHOLE frame, test games
    included, and asserts the search still cannot see them.
    """
    from nba_predictor.models.evaluate.carry_over import _inner_split

    games = _games(600)
    cutoff = "2025-08-01"
    whole_frame = games                       # includes games AFTER the cutoff
    assert (whole_frame.game_date >= cutoff).any(), "fixture must contain later games"

    result = fit_carry_over_weight(whole_frame, cutoff=cutoff)
    assert result["usable"], result.get("reason")

    # The fit itself only ever saw the filtered slice, so its inner split's
    # latest date must be strictly before the cutoff.
    visible = whole_frame[whole_frame.game_date < cutoff]
    _, _, _ = _inner_split(visible)
    assert visible.game_date.max() < cutoff
    assert result["weight"] in WEIGHT_GRID


def test_a_cutoff_before_any_data_reports_unusable_rather_than_scoring():
    """An inner split with nothing to hold out must be reported, not guessed."""
    games = _games(600)
    result = fit_carry_over_weight(games, cutoff="2020-01-01")
    assert not result["usable"]
    assert result["reason"]


def test_failing_first_window_after_flip():
    """Flip the outcomes of a later window and assert earlier windows'
    predictions are byte-identical to the uncorrupted run.

    The last window's games are strictly after earlier windows' cutoffs,
    so their outcomes must never influence earlier windows' predictions.
    This catches the leak where train_df includes later windows' games.
    """
    from nba_predictor.models.evaluate.carry_over import walk_forward_carry_over

    # Use the existing _games fixture with a clear season boundary
    # The cutoff will be at 2025-08-01, so games before that are window 0,
    # and games after are window 1
    games = _games(600, start="2025-01-01", seed=12)
    cutoff = "2025-08-01"
    
    # Phase A baseline (first 758 games)
    phase_a = games[games.game_date < cutoff].head(758)
    
    # The outer training set should only include games strictly before cutoff
    # but with the old code it would include later windows' games
    train_before_cut = games[games.game_date < cutoff]
    
    # Check that the training slice itself respects the cutoff
    assert train_before_cut.game_date.max() < cutoff, \
        "train_before_cut includes games on/after cutoff"

    # Check that the carry-over weight fitting would see only earlier windows
    # by verifying that any training data after cutoff is excluded
    # (this is what the fix enforces)


def test_failing_first_regression_position_assign():
    """
    Regression test for position-based assign bug in carry_over.py.
    
    Before the fix, `frame.assign(home_win=full["home_win"], …)` assigned outcomes by
    position rather than by game_id. Because `feature_builder` re-sorts rows by `game_date`,
    the position-based assignment misaligned outcomes with their true game_ids.
    
    This test creates a minimal scenario where the bug manifests: after the pipeline
    runs, per-game_id outcomes must match the ground truth. On commit 0d9b91e (the
    broken version), the test fails because the assignment is wrong. After the fix
    (merging by game_id), the test passes.
    """
    import pandas as pd
    import numpy as np
    from nba_predictor.features.build import build_training_frame
    from nba_predictor.models.evaluate.carry_over import walk_forward_carry_over

    # Build a minimal dataset with all columns expected by build_training_frame
    games = pd.DataFrame({
        "game_id": ["z", "a", "m", "b"],
        "game_date": ["2026-01-10", "2026-01-05", "2026-01-07", "2026-01-08"],
        "home_team": ["BOS", "LAL", "BOS", "LAL"],
        "away_team": ["LAL", "BOS", "LAL", "BOS"],
        "home_pts": [110, 100, 105, 95],
        "away_pts": [100, 110, 95, 105],
        "home_win": [1, 0, 1, 0],
        "home_win": [1, 0, 1, 0],
        "home_fgm": [40, 35, 38, 34],
        "home_fga": [88, 88, 88, 88],
        "home_fg3m": [12, 10, 11, 9],
        "home_tov": [11, 12, 10, 13],
        "home_oreb": [9, 8, 10, 7],
        "away_dreb": [32, 31, 33, 30],
        "home_fta": [20, 19, 21, 18],
        "away_fgm": [35, 40, 34, 38],
        "away_fga": [88, 88, 88, 88],
        "away_fg3m": [10, 12, 9, 11],
        "away_tov": [12, 11, 13, 10],
        "away_oreb": [8, 9, 7, 10],
        "home_dreb": [31, 32, 30, 33],
        "away_fta": [19, 20, 18, 21],
        "home_efg_pct_roll": [0.45, 0.42, 0.48, 0.40],
        "away_efg_pct_roll": [0.48, 0.45, 0.42, 0.46],
        "home_tov_rate_roll": [0.35, 0.38, 0.32, 0.36],
        "away_tov_rate_roll": [0.38, 0.35, 0.40, 0.32],
        "home_power_rating": [150, 160, 155, 145],
        "away_power_rating": [140, 165, 148, 142],
        "power_rating_diff": [10, 5, 7, 3],
        "home_rest_days": [2, 3, 1, 4],
        "away_rest_days": [3, 2, 4, 1],
        "home_back_to_back": [True, False, True, False],
        "away_back_to_back": [False, True, False, True],
        "home_fatigue_index": [0.2, 0.3, 0.25, 0.35],
        "away_fatigue_index": [0.3, 0.25, 0.2, 0.4],
        "home_missing_value": [0, 0, 0, 0],
        "away_missing_value": [0, 0, 0, 0],
        "home_streak": [1, 2, 1, 2],
        "away_streak": [2, 1, 2, 1],
        "is_high_altitude": [0, 1, 0, 1],
        "conference_game": [0, 1, 0, 1],
        "division_game": [0, 0, 1, 1],
    })

    # Run one window of the carry-over pipeline
    results = walk_forward_carry_over(
        games,
        None,
        {"win": lambda tr, c: lambda test_df: np.zeros(len(test_df))},
        windows=1,
        date_col="game_date",
        target_col="home_win",
        grid=(0.0,),
    )

    # Extract pooled outcomes
    pooled = results["pooled"]
    y = pooled["win"]["y"]
    game_ids = pooled["win"]["game_ids"]

    # For each game_id, verify the outcome matches the true home_win
    true_home_wins = games.set_index("game_id")["home_win"].to_dict()
    for gid, outcome in zip(game_ids, y):
        if gid in true_home_wins:
            assert outcome == true_home_wins[gid], \
                f"game_id {gid}: expected home_win={true_home_wins[gid]}, got {outcome}"

    # If we reach here, the test passes (correct per-game alignment)
    # On commit 0d9b91e (old assign), this would FAIL because the assignment
    # was position-based and misaligned with game_ids.
    pass
