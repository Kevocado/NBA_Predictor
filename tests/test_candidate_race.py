"""The candidate race: which model actually serves (spec section 6, gap G2).

Two attempts at this were rejected. Both produced code that looked finished and
measured nothing: one scored a candidate that threw as `inf` and declared the
survivor the winner, the other returned a constant 0.5 under the name "Elo".
The tests below are aimed at exactly that failure mode -- a race that cannot
fail is not a race.

Pinned here:
* a candidate that raises FAILS the race rather than losing it quietly;
* every candidate refits on the window it is given (no closure over a model
  already fitted on the full frame);
* Elo is a real rating loop, is scored, and is never selected to serve;
* the winner is decided per target, by that target's metric;
* two identical runs give an identical answer.
"""

import numpy as np
import pandas as pd
import pytest

from nba_predictor.features.build import FEATURE_COLUMNS
from nba_predictor.models.candidate_race import (
    elo_probability_factory,
    run_candidate_race,
)


def make_real_games(n: int = 200) -> pd.DataFrame:
    """A frame shaped like the real one: the 24 FEATURE_COLUMNS plus the target
    columns. Built from the real column set on purpose -- the rejected attempts
    grew `strength` branches that only ever fire for a synthetic fixture."""
    rng = np.random.default_rng(0)
    n = max(n, len(FEATURE_COLUMNS))
    dates = pd.date_range("2025-10-22", periods=n).astype(str)
    teams = ["BOS", "MIA", "LAL", "GSW"]
    data = {
        "game_id": [f"g{i}" for i in range(n)],
        "game_date": dates,
        "home_team": [teams[i % 4] for i in range(n)],
        "away_team": [teams[(i + 1) % 4] for i in range(n)],
    }
    for j, col in enumerate(FEATURE_COLUMNS):
        data[col] = rng.normal(0, 1, n) * (1 + j % 3)
    strength = np.linspace(-2.0, 2.0, n)
    # margin and total must both vary, or every constant baseline scores MAE 0.0
    # and the two baselines become indistinguishable.
    margin = 3.0 + strength * 6.0
    total = 218.0 + rng.normal(0, 12, n)
    data["home_pts"] = ((total + margin) / 2).round()
    data["away_pts"] = ((total - margin) / 2).round()
    data["home_win"] = (rng.random(n) < 1 / (1 + np.exp(-strength))).astype(int)
    return pd.DataFrame(data)


def test_race_scores_every_servable_candidate_with_a_finite_metric():
    """No `inf` winners. A candidate that throws fails the race (see next
    test); a candidate that runs is scored, and every score is finite."""
    res = run_candidate_race(make_real_games(), windows=3)

    for name in ("logistic", "xgboost"):
        pooled = res["results"][name]["win"]["pooled"]
        assert np.isfinite(pooled["log_loss"]), f"{name} scored a non-finite log-loss"
        assert 0.0 <= pooled["brier"] <= 1.0
    for name in ("ridge", "xgboost"):
        for target in ("margin", "total"):
            assert np.isfinite(res["results"][name][target]["pooled"]["mae"])


def test_a_candidate_that_raises_fails_the_race_instead_of_losing_it_silently():
    """The defect that produced a fake result: a broken candidate was caught,
    scored `inf`, and the survivor was declared the winner. The winner was an
    artifact of the breakage."""
    def exploding_factory(_train_df):
        raise RuntimeError("this candidate cannot be scored")

    with pytest.raises(RuntimeError, match="cannot be scored"):
        run_candidate_race(
            make_real_games(),
            candidates={"broken": {"win": exploding_factory}},
            windows=3,
        )


def test_factories_refit_on_the_window_they_are_given():
    """Requirement: no closure over a model already fitted on the full frame.
    Task 4 shipped that bug and it invalidated the whole evaluation."""
    seen: list[int] = []

    def recording_factory(train_df):
        seen.append(len(train_df))
        model = train_df["home_win"].mean()

        def predict(X):
            return np.full(len(X), model)

        return predict

    run_candidate_race(
        make_real_games(),
        candidates={"recorder": {"win": recording_factory}},
        windows=3,
    )

    assert len(seen) == 3, "the factory must be called once per window"
    assert seen == sorted(seen) and seen[0] < seen[-1], (
        f"training slices must grow across windows, got {seen}"
    )


def test_elo_is_a_real_rating_loop_not_a_constant():
    """The rejected attempt returned `np.full(len(X), 0.5)`. A constant is not
    a rating system, and it would have WON the win race at 0.6931 against
    XGBoost's 0.7059 -- a coin flip beating a model, reported as Elo."""
    rng = np.random.default_rng(1)
    df = make_real_games(120)
    # Make one team clearly dominant so ratings have something to find.
    df["home_team"] = "BOS"
    df["away_team"] = "MIA"
    df["home_win"] = 1
    factory = elo_probability_factory()
    train_df = df.iloc[:80]
    preds = factory(train_df)(df.iloc[80:])
    preds = np.asarray(preds, dtype=float)

    assert not np.allclose(preds, 0.5), "Elo predicted a constant 0.5"
    assert preds.mean() > 0.8, (
        f"an 80-0 team should rate well above even money, got {preds.mean():.3f}"
    )
    assert ((preds >= 0) & (preds <= 1)).all(), "probabilities must be in [0, 1]"


def test_elo_is_scored_but_never_selected_to_serve():
    """Elo exists to answer "did this beat something simple". It is a reference,
    never a served model (spec section 3.1)."""
    res = run_candidate_race(make_real_games(), windows=3)

    assert "elo" in res["results"], "Elo must appear in the output as a reference"
    assert np.isfinite(res["results"]["elo"]["win"]["pooled"]["log_loss"])
    assert res["winner"]["win"] != "elo", "Elo must never be selected to serve"
    assert res["winner"]["margin"] != "elo"
    assert res["winner"]["total"] != "elo"


def test_winner_is_decided_per_target_by_that_targets_metric():
    """The rejected attempt reported XGBoost as the margin and total winner
    while printing a better Ridge MAE for both."""
    res = run_candidate_race(make_real_games(), windows=3)

    for target, metric in (("win", "log_loss"), ("margin", "mae"), ("total", "mae")):
        contenders = {
            name: r[target]["pooled"][metric]
            for name, r in res["results"].items()
            if target in r and name != "elo"
        }
        assert contenders, f"no servable contender for {target}"
        best = min(contenders.values())
        assert res["winner"][target] == min(contenders, key=contenders.get), (
            f"{target}: winner is not the minimum {metric} among "
            f"{ {k: round(v, 4) for k, v in contenders.items()} }"
        )
        assert contenders[res["winner"][target]] == pytest.approx(best)


def test_race_is_deterministic():
    """A race that returns a different winner on two identical runs is a coin
    toss. Two runs, one answer."""
    df = make_real_games()
    a = run_candidate_race(df, windows=3)
    b = run_candidate_race(df, windows=3)
    assert a["winner"] == b["winner"]
    for name in a["results"]:
        for target in a["results"][name]:
            assert _dicts_equal(a["results"][name][target]["pooled"], b["results"][name][target]["pooled"]), (
                f"{name}/{target} differed between identical runs"
            )


def _dicts_equal(a: dict, b: dict) -> bool:
    """Compare two dicts that may contain numpy arrays."""
    if a.keys() != b.keys():
        return False
    for k in a:
        av, bv = a[k], b[k]
        if isinstance(av, np.ndarray) and isinstance(bv, np.ndarray):
            if av.shape != bv.shape:
                return False
            if not np.array_equal(av, bv):
                return False
        elif isinstance(av, dict) and isinstance(bv, dict):
            if not _dicts_equal(av, bv):
                return False
        elif isinstance(av, list) and isinstance(bv, list):
            if len(av) != len(bv):
                return False
            for av_i, bv_i in zip(av, bv):
                if isinstance(av_i, (np.ndarray, dict)) or isinstance(bv_i, (np.ndarray, dict)):
                    if not _dicts_equal(av_i, bv_i) if isinstance(av_i, dict) else np.array_equal(av_i, bv_i):
                        return False
                elif av_i != bv_i:
                    return False
        elif av != bv:
            return False
    return True


def test_naive_baselines_are_reported_for_all_three_targets():
    """Kevin's explicit ask: the race compares margin against naive and total
    against nothing otherwise, which is the same 'honest but incomplete' failure
    this workstream exists to stop. Three targets, three baselines."""
    res = run_candidate_race(make_real_games(), windows=3)
    pooled = res["results"]["xgboost"]

    assert "coinflip_log_loss" in pooled["win"]["pooled"]
    assert "naive_log_loss" in pooled["win"]["pooled"]
    assert "naive_mae" in pooled["margin"]["pooled"]
    assert "naive_mae_fixed" in pooled["margin"]["pooled"]
    assert "naive_mae" in pooled["total"]["pooled"]
    assert "naive_mae_fixed" in pooled["total"]["pooled"]


def test_total_fixed_baseline_is_distinct_from_the_mean_baseline():
    """The rejected attempt emitted the training mean under both names for
    total, so 'fixed' and 'league' were one number wearing two labels."""
    res = run_candidate_race(make_real_games(), windows=3)
    pooled = res["results"]["xgboost"]["total"]["pooled"]
    assert pooled["naive_mae"] != pytest.approx(pooled["naive_mae_fixed"]), (
        "the total target's two baselines are the same number"
    )


def test_race_does_not_rebuild_the_feature_frame_under_the_caller():
    """`run_candidate_race` must score the frame it is handed. Rebuilding the
    feature frame inside the race silently re-derives rolling features and makes
    the race's data differ from the model's training data."""
    df = make_real_games()
    res = run_candidate_race(df, windows=3)
    assert res["n_games"] == len(df)
    assert res["feature_cols"], "the race must report the features it scored on"
    assert "strength" not in res["feature_cols"]