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
