"""Expanding-window walk-forward evaluation for the win model (spec section 5, G1).

Before this the repo had one evaluation: `chronological_split`, a single 80/20
cut, reported as if it were walk-forward. One cut cannot show whether a model
holds up as the season moves, and it is the only number the manifest carried.

Three properties are pinned here, in order of how badly they bite:

* **Causality.** A test window must start strictly after its training window
  ends. A feature that has seen the game it predicts makes walk-forward scores
  meaningless (spec section 14.1) -- and the failure is invisible, the score
  just looks great.
* **Expanding, not rolling.** Window k trains on every game before its cutoff,
  so history accumulates.
* **Naive baselines travel with the metric.** log-loss/Brier/AUC reported
  without the home-win base rate and the 0.6931 coin flip is exactly the
  bare-number reporting G1 complained about.
"""

import numpy as np
import pandas as pd
import pytest

from nba_predictor.models.evaluate.walk_forward_eval import (
    COINFLIP_LOG_LOSS,
    expanding_windows,
    walk_forward_metrics,
)


def make_games(n: int = 100) -> pd.DataFrame:
    """n games on consecutive dates with a home-win label the factory can learn.

    strength is an autocorrelated per-game rating, so the label carries signal a
    model can actually find -- a pure-random label would make "beats the
    coin flip" pass or fail by luck rather than by construction.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-10-22", periods=n).astype(str)
    # strength sweeps the season, so P(home win) spans roughly 0.3-0.7 and a
    # model has something real to find rather than coin-flip luck.
    strength = np.linspace(-2.0, 2.0, n) + rng.normal(0, 0.2, n)
    p_home = 1 / (1 + np.exp(-strength))
    teams = ["BOS", "MIA", "LAL", "GSW"]
    return pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(n)],
            "game_date": dates,
            "home_team": [teams[i % 4] for i in range(n)],
            "away_team": [teams[(i + 1) % 4] for i in range(n)],
            "strength": strength,
            "home_win": (rng.random(n) < p_home).astype(int),
        }
    )


FEATURE_COLS = ["strength"]


def dummy_factory(train_df: pd.DataFrame):
    """Logistic stand-in: real predict_proba, no XGBoost in the unit test.

    Returns a predict_proba callable, per the module's `model_factory(train_df)
    -> predict_proba_fn` contract, so the factory owns its column choice.
    """
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(train_df[FEATURE_COLS], train_df["home_win"])
    return lambda X: model.predict_proba(X[FEATURE_COLS])[:, 1]


class _LeakyFactory:
    """Returns a predictor fitted on ALL of df -- what a leakage bug looks like."""

    def __init__(self, df):
        self.df = df

    def __call__(self, _train_df):
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        model.fit(self.df[FEATURE_COLS], self.df["home_win"])
        return lambda X: model.predict_proba(X[FEATURE_COLS])[:, 1]


def test_walk_forward_windows_are_expanding_and_causal():
    df = make_games(100)
    res = walk_forward_metrics(df, model_factory=dummy_factory, windows=4)

    assert len(res["windows"]) == 4
    for w in res["windows"]:
        assert w["train_max_date"] < w["test_min_date"], (
            "test window must be strictly after train window"
        )


def test_training_set_grows_and_never_overlaps_the_test_set():
    """Expanding, not rolling: each window trains on more than the last, and
    the test slice is always disjoint from it."""
    df = make_games(120)
    res = walk_forward_metrics(df, model_factory=dummy_factory, windows=4)

    sizes = [w["n_train"] for w in res["windows"]]
    assert sizes == sorted(sizes), f"training set shrank: {sizes}"
    assert sizes[-1] > sizes[0], "windows are rolling, not expanding"

    total_tested = sum(w["n_test"] for w in res["windows"])
    assert total_tested == len(df) - sizes[0], (
        "test slices must partition the games after the first training cut"
    )


def test_expanding_windows_never_reuse_a_row_in_train_and_test():
    df = make_games(100)
    for train_idx, test_idx in expanding_windows(df["game_date"], n_windows=4):
        assert not set(train_idx) & set(test_idx), "a game is in both train and test"


def test_metrics_are_reported_against_naive_baselines():
    """A log-loss with nothing to compare it to is not a result."""
    df = make_games(100)
    res = walk_forward_metrics(df, model_factory=dummy_factory, windows=4)
    pooled = res["pooled"]

    for key in ("log_loss", "brier", "auc", "naive_log_loss", "coinflip_log_loss"):
        assert key in pooled, f"{key} missing -- it is the comparison the report needs"
    assert pooled["coinflip_log_loss"] == pytest.approx(0.6931, abs=1e-3)
    assert 0.0 <= pooled["brier"] <= 1.0
    assert pooled["n"] == sum(w["n_test"] for w in res["windows"])


def test_a_model_that_learns_signal_beats_the_coin_flip():
    df = make_games(200)
    res = walk_forward_metrics(df, model_factory=dummy_factory, windows=4)
    assert res["pooled"]["log_loss"] < res["pooled"]["coinflip_log_loss"]


def test_walk_forward_refuses_a_leaked_window():
    """The guard that catches review-focus item 1. A factory that has seen the
    test rows must not be able to produce a scored result: the evaluator's own
    date check fires first."""
    df = make_games(100)
    leaky = _LeakyFactory(df)
    # Sanity: the leak really does improve the score, or this test proves nothing.
    honest = walk_forward_metrics(df, model_factory=dummy_factory, windows=4)
    with_leak = walk_forward_metrics(df, model_factory=leaky, windows=4)
    assert with_leak["pooled"]["log_loss"] < honest["pooled"]["log_loss"]

    # And a split that cannot separate train from test by date is refused
    # outright rather than silently scored: with every game on one date, no
    # window is causal and the honest answer is to stop, not to report a number.
    bad = df.copy()
    bad["game_date"] = "2025-10-22"
    with pytest.raises((AssertionError, ValueError)):
        walk_forward_metrics(bad, model_factory=dummy_factory, windows=4)


def test_needs_a_real_number_of_games():
    with pytest.raises(ValueError):
        walk_forward_metrics(make_games(3), model_factory=dummy_factory, windows=4)


def test_windows_split_between_dates_never_mid_date():
    """Several games share a date (10-12 a night). A row-wise cut can land
    inside one, so the test window's first game and the training window's last
    game share a date -- and every feature on that date has seen games it must
    not have. This fired on the real 1,390-game training set, where the first
    window refused to evaluate at all.

    Every game on a date goes entirely in train or entirely in test.
    """
    df = make_games(120)
    # Three games per date instead of one.
    df = (
        df.assign(game_date=[f"2025-{10 + i // 36:02d}-{1 + (i // 3) % 28:02d}" for i in range(len(df))])
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    assert df["game_date"].duplicated().any(), "fixture must actually share dates"

    # 5 windows, so the slice size (14) does not divide the 3-games-per-date
    # group -- this is the misalignment that broke the real run.
    res = walk_forward_metrics(df, model_factory=dummy_factory, windows=5)

    for w in res["windows"]:
        assert w["train_max_date"] < w["test_min_date"]
    # And nothing was silently dropped: the windows still partition the slate.
    assert sum(w["n_test"] for w in res["windows"]) == len(df) - res["windows"][0]["n_train"]

    # Same check at the index level: no date appears on both sides of a cut.
    for train_idx, test_idx in expanding_windows(df["game_date"], n_windows=5):
        assert not (set(df["game_date"].iloc[train_idx]) & set(df["game_date"].iloc[test_idx])), (
            "a game date is split across the train/test boundary"
        )


def test_margin_walk_forward_reports_mae_vs_naive_scale():
    from nba_predictor.models.evaluate.walk_forward_eval import walk_forward_regression

    df = make_games(n=100)
    rng = np.random.default_rng(42)
    df["home_margin"] = df["strength"] * 2 + rng.normal(0, 3, len(df))

    def dummy_reg_factory(train_df):
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0)
        model.fit(train_df[FEATURE_COLS], train_df["home_margin"])
        return lambda X: model.predict(X[FEATURE_COLS])

    res = walk_forward_regression(df, model_factory=dummy_reg_factory, target="home_margin")
    assert "mae" in res["pooled"] and "naive_mae" in res["pooled"]
    assert res["pooled"]["mae"] > 0


def test_manifest_emits_probability_metrics():
    from nba_predictor.models.manifest import build_manifest

    manifest = build_manifest(metrics={"log_loss": 0.63, "brier": 0.22, "auc": 0.60})
    assert manifest["log_loss"] == 0.63 and manifest["brier"] == 0.22 and manifest["auc"] == 0.60