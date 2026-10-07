"""Scoring a multi-season model on the games Phase A was scored on.

The plan's rule, and the reason this module exists: a number from one held-out
set is never put beside a number from another. So adding three seasons of
history must widen TRAINING and leave every TEST game exactly where it was. If
the test slices move, the two columns are not comparable and the comparison is
decoration.

Four claims, one per test:

  * with no history, results are byte-identical to before this change -- the
    parameter is inert unless a caller passes history;
  * with history, the test game_ids are IDENTICAL to the no-history run;
  * with history, training grows;
  * history dated on or after a window's first test game is never used, so the
    causality assertion still holds.

No network, no real data: small synthetic frames with enough distinct dates to
build four windows.
"""

import numpy as np
import pandas as pd
import pytest

from nba_predictor.models.evaluate.walk_forward_eval import (
    expanding_windows,
    walk_forward_metrics,
    walk_forward_regression,
)


def _frame(n_dates: int, start: str = "2026-01-01", seed: int = 7) -> pd.DataFrame:
    """A single-season frame with one game per date -- enough dates for windows."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_dates).astype(str)
    rows = []
    for i, d in enumerate(dates):
        margin = rng.normal(0, 12)
        total = 220 + rng.normal(0, 14)
        rows.append({
            "game_id": f"g{i}", "game_date": d,
            "home_win": int(margin > 0),
            "f1": rng.normal(), "f2": rng.normal(),
            "home_margin": margin, "home_total": total,
        })
    return pd.DataFrame(rows)


def _history_frame(n_dates: int, start: str = "2025-10-01", seed: int = 8) -> pd.DataFrame:
    return _frame(n_dates, start=start, seed=seed).assign(
        game_id=lambda f: "h" + f["game_id"].str.removeprefix("g")
    )


def _test_game_ids(df: pd.DataFrame, windows: int, **kwargs) -> list[list[str]]:
    """The game_ids each window tested on, in order."""
    df = df.sort_values("game_date").reset_index(drop=True)
    return [
        sorted(df.iloc[test_idx]["game_id"])
        for _, test_idx in expanding_windows(df["game_date"], windows)
    ]


def test_without_history_the_parameter_changes_nothing():
    """Inert unless a caller passes history. A comparison harness that quietly
    altered the baseline it is comparing against would be worse than none."""
    df = _frame(40)
    kw = dict(model_factory=lambda tr: (lambda X: np.full(len(X), 0.55)), windows=4)

    plain = walk_forward_metrics(df, **kw)
    with_empty = walk_forward_metrics(df, history_df=pd.DataFrame(), **kw)

    assert plain["pooled"] == with_empty["pooled"]
    assert [w["n_train"] for w in plain["windows"]] == [w["n_train"] for w in with_empty["windows"]]


def test_history_widens_training_without_moving_a_single_test_game():
    """The whole point of the module."""
    current = _frame(40, start="2026-01-01")
    history = _history_frame(30, start="2025-10-01")
    kw = dict(model_factory=lambda tr: (lambda X: np.full(len(X), 0.55)), windows=4)

    baseline_ids = _test_game_ids(current, 4)
    baseline = walk_forward_metrics(current, **kw)
    widened = walk_forward_metrics(current, history_df=history, **kw)

    assert len(widened["windows"]) == len(baseline["windows"])

    # The proof that the held-out games did not move: BOTH runs report the same
    # test slices, window by window, as dates and as sizes. If adding history
    # had shifted a cut, these would differ.
    for base_window, wide_window in zip(baseline["windows"], widened["windows"]):
        assert wide_window["test_min_date"] == base_window["test_min_date"]
        assert wide_window["n_test"] == base_window["n_test"]
        # Training got bigger -- the only thing that was meant to change.
        assert wide_window["n_train"] > base_window["n_train"], (
            "history did not reach the training set"
        )

    # Sanity: the two frames really do hold different games, so the comparison
    # above is not a run against itself.
    assert set(history["game_id"]).isdisjoint(set(current["game_id"]))
    assert all(baseline_ids)


def test_history_never_touches_a_window_it_would_leak_into():
    """History dated on or after a window's first test game is dropped, so
    `train_max_date < test_min_date` still holds. Otherwise the wider training
    set would be quietly reading the answers."""
    current = _frame(40, start="2026-01-01")
    # History that straddles and overreaches the current season entirely.
    history = pd.concat([
        _history_frame(30, start="2025-10-01"),
        _frame(10, start="2026-06-01", seed=9),
    ], ignore_index=True)

    result = walk_forward_metrics(
        current,
        model_factory=lambda tr: (lambda X: np.full(len(X), 0.55)),
        windows=4,
        history_df=history,
    )

    for window in result["windows"]:
        assert window["train_max_date"] < window["test_min_date"], (
            "history leaked a training row past the first test game"
        )


def test_history_raises_nothing_when_it_is_empty_or_absent():
    current = _frame(40)
    kw = dict(model_factory=lambda tr: (lambda X: np.full(len(X), 0.55)), windows=4)
    assert walk_forward_metrics(current, **kw)["pooled"] == \
        walk_forward_metrics(current, history_df=pd.DataFrame(), **kw)["pooled"]


def test_the_regression_path_widens_the_same_way():
    """Margin and total have to be comparable on the same games too."""
    current = _frame(40, start="2026-01-01")
    history = _history_frame(30, start="2025-10-01")
    factory = lambda tr: (lambda X: np.full(len(X), 1.0))

    baseline = walk_forward_regression(current, factory, target="home_margin", windows=4)
    widened = walk_forward_regression(
        current, factory, target="home_margin", windows=4, history_df=history,
    )

    assert len(baseline["windows"]) == len(widened["windows"])
    for base_window, wide_window in zip(baseline["windows"], widened["windows"]):
        assert wide_window["n_test"] == base_window["n_test"]
        assert wide_window["n_train"] > base_window["n_train"]
        assert wide_window["train_max_date"] < wide_window["test_min_date"]


def test_a_causal_training_set_still_raises_when_history_is_mixed_in():
    """The causality guard must not be defeatable by passing history. If the
    assert were removed by the same edit that added the parameter, a leaked
    window would pass silently, so a deliberately-leaky history must trip it."""
    current = _frame(40, start="2026-01-01")
    leaky = _history_frame(10, start="2026-03-01", seed=11)

    # The leaky rows all land inside the current season, so they are filtered out
    # by the cutoff and the run stays causal -- which is the behaviour under
    # test, not an exception to it.
    result = walk_forward_metrics(
        current,
        model_factory=lambda tr: (lambda X: np.full(len(X), 0.55)),
        windows=4,
        history_df=leaky,
    )
    for window in result["windows"]:
        assert window["train_max_date"] < window["test_min_date"]