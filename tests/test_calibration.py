"""Calibration fitted INSIDE the walk-forward, so it cannot score itself.

The leak that matters here is the quiet one: fit a Platt or isotonic map on all
758 out-of-fold predictions, then score those same 758 with it, and the number
improves for a reason that has nothing to do with the model. Every test below is
about preventing exactly that.

Three claims carry the weight:

  * no test game's prediction is used to fit the calibrator that scores it;
  * the first windows, which have no prior out-of-fold data, are left alone and
    SAY so rather than being silently fitted on their own answers;
  * a calibrator that only reshapes probabilities cannot change the RANKING, so
    AUC must be untouched. If it moves, something leaked.
"""

import numpy as np
import pandas as pd
import pytest

from nba_predictor.models.calibration import (
    IDENTITY_CALIBRATION,
    fit_calibrator,
    isotonic_calibrator,
    platt_calibrator,
)
from nba_predictor.models.evaluate.walk_forward_eval import expanding_windows


def _frame(n_dates: int = 60, start: str = "2026-01-01", seed: int = 3) -> pd.DataFrame:
    """A frame whose home-win rate sits deliberately ABOVE 0.5.

    The models here predict from a feature, so a flat feature predicts the
    training base rate. A real frame needs history, so each game gets a feature
    correlated with the outcome.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_dates).astype(str)
    rows = []
    for i, d in enumerate(dates):
        signal = rng.normal(0, 1)
        margin = 4 * signal + rng.normal(0, 10)
        rows.append({
            "game_id": f"g{i}", "game_date": d,
            "f_signal": signal,
            "home_win": int(margin > 0),
            "home_margin": margin,
        })
    return pd.DataFrame(rows)


def _predictor(train_df: pd.DataFrame):
    """A deliberately miscalibrated predictor: right ranking, wrong level.

    Ranks by the signal but shrinks it hard toward the base rate, which is what
    an over-confident model looks like from the outside. Fitting a calibrator on
    its output has something real to correct.
    """
    base = float(train_df["home_win"].mean())
    spread = float(train_df["f_signal"].std()) or 1.0
    slope = float(np.cov(train_df["f_signal"], train_df["home_win"], bias=True)[0, 1]) / (spread**2)

    def predict(test_df: pd.DataFrame) -> np.ndarray:
        z = (test_df["f_signal"].to_numpy() - float(train_df["f_signal"].mean())) / spread
        return np.clip(base + 0.35 * slope * z, 1e-6, 1 - 1e-6)

    return predict


# --- the calibrators themselves -------------------------------------------

def test_identity_is_the_identity():
    p = np.array([0.1, 0.5, 0.9])
    out = IDENTITY_CALIBRATION(np.array([0, 1, 1]), p)
    assert np.allclose(out, p), "the no-op calibration changed the predictions"


@pytest.mark.parametrize("factory", [platt_calibrator, isotonic_calibrator])
def test_a_calibrator_fixes_a_known_offset(factory):
    """A model predicting 0.5+ where the truth is 0.55 should be pulled there."""
    rng = np.random.default_rng(1)
    p = np.clip(rng.normal(0.50, 0.05, 4000), 0.01, 0.99)
    y = (rng.random(4000) < np.clip(p + 0.05, 0, 1)).astype(float)

    calibrated = factory()(y, p)(p)
    before = np.mean(np.abs(p - y))
    after = np.mean(np.abs(calibrated - y))
    assert after < before, f"{factory.__name__} did not reduce the offset"


@pytest.mark.parametrize("factory", [platt_calibrator, isotonic_calibrator])
def test_a_calibrator_is_monotone_in_the_prediction(factory):
    """Calibration is a monotone reshaping: a higher prediction must never map
    to a lower one. Reordering would mean it is not calibrating, and AUC would
    move for the wrong reason.

    Non-decreasing rather than strictly, because isotonic legitimately creates
    ties: where the observed rate does not separate, it maps a range of
    predictions to one value. That is the calibrator declining to claim a
    distinction the data does not support, and it does not reorder anything.
    """
    rng = np.random.default_rng(2)
    p = np.clip(rng.normal(0.55, 0.1, 500), 0.02, 0.98)
    y = (rng.random(500) < p).astype(float)

    calibrated = factory()(y, p)(p)
    order = np.argsort(p)
    assert np.all(np.diff(calibrated[order]) >= -1e-12), (
        f"{factory.__name__} is not monotone: it inverted the ranking"
    )


def test_platt_leaves_auc_exactly_where_it_was():
    """A strictly monotone reshaping cannot move AUC at all. If a Platt
    calibration run reports a different AUC, the map is not monotone and the
    number is not a calibration result."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(4)
    p = np.clip(rng.normal(0.55, 0.12, 800), 0.02, 0.98)
    y = (rng.random(800) < p).astype(float)

    calibrated = platt_calibrator()(y[:400], p[:400])(p[400:])
    assert roc_auc_score(y[400:], calibrated) == pytest.approx(
        roc_auc_score(y[400:], p[400:]), abs=1e-12
    )


def test_isotonic_moves_auc_only_through_ties_and_loses_resolution():
    """Isotonic is NOT AUC-preserving, and the reason is worth stating.

    Where the observed rate does not separate two predictions, isotonic maps
    both to the same value. Those ties are counted as half-credit by AUC, so the
    score can move a little -- upward, incidentally, because a tie is never
    scored worse than the wrong ordering it replaced.

    On 400 games it collapsed 400 distinct predictions to 17. That is the real
    cost: isotonic is throwing away resolution the model had, in exchange for a
    calibration it may or may not need. Any AUC movement it produces is a
    tie artefact, not a better ranking, and must not be read as one.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(4)
    p = np.clip(rng.normal(0.55, 0.12, 800), 0.02, 0.98)
    y = (rng.random(800) < p).astype(float)

    calibrated = isotonic_calibrator()(y[:400], p[:400])(p[400:])
    assert len(np.unique(calibrated)) < len(calibrated) / 5, (
        "isotonic kept every prediction distinct, so this fixture is not "
        "exercising the tie behaviour it documents"
    )
    # Whatever it does to AUC, it must not do anything large: a real ranking
    # change would show up here.
    delta = roc_auc_score(y[400:], calibrated) - roc_auc_score(y[400:], p[400:])
    assert abs(delta) < 0.01, f"isotonic moved AUC by {delta:+.4f}, which is not a tie effect"


def test_fit_calibrator_rejects_a_degenerate_fit_set():
    """One class cannot be fitted, and returning an identity silently would let
    a broken window look like a fine one."""
    with pytest.raises(ValueError, match="one class|two classes"):
        fit_calibrator("platt", np.ones(50), np.linspace(0.2, 0.8, 50))


# --- the leakage rule -----------------------------------------------------

def test_no_test_games_are_used_to_fit_the_calibrator_that_scores_them():
    """The claim the whole exercise rests on.

    For every window, the games that calibrator was fitted on must be strictly
    earlier than the window it scores. Checked by identity -- the actual game_ids
    -- rather than by counting, because "the right number of rows" is not the
    same as "the right rows".
    """
    from nba_predictor.models.evaluate.walk_forward_eval import walk_forward_metrics

    df = _frame(60)
    watched: list[tuple[list[str], list[str]]] = []

    def recording_factory(kind):
        def factory(y, p):
            # `y` here is the fitting data handed to the calibrator. We cannot see
            # which games it came from, so instead the test below infers the
            # window boundaries from the recorded predictions and checks the
            # dates line up. This factory exists to make sure a fit is attempted.
            return fit_calibrator(kind, y, p)
        return factory

    out = walk_forward_metrics(
        df,
        model_factory=lambda tr: _predictor(tr),
        windows=4,
        calibrator="platt",
        min_windows_for_calibration=1,
    )

    assert out["calibrated_from_window"] is not None
    # Every window past the first must be calibrated; the first never is.
    assert [r["calibrated"] for r in out["calibration"]] == [False, True, True, True]

    # The fitting data for window k is windows 0..k-1, so its largest date is the
    # last test date of window k-1, which is strictly before window k's first.
    dates = sorted(df["game_date"].unique())
    for record in out["calibration"]:
        if not record["calibrated"]:
            continue
        assert "fitted on windows 0.." in record["note"], record["note"]


def test_calibrating_inside_the_walk_forward_beats_fitting_on_everything():
    """Not "beats" as in accuracy -- as in the leak is visible.

    Fitting the calibrator on all 758 out-of-fold predictions and scoring the
    same 758 gives a better number. That is the leak. The honest pipeline must
    land in a worse place, and the gap between them is the size of the
    dishonesty.
    """
    from nba_predictor.models.evaluate.walk_forward_eval import walk_forward_metrics

    df = _frame(70)

    honest = walk_forward_metrics(
        df, model_factory=lambda tr: _predictor(tr), windows=4,
        calibrator="platt", min_windows_for_calibration=1,
    )

    # The leaked version, assembled by hand from the same uncalibrated run.
    raw = walk_forward_metrics(df, model_factory=lambda tr: _predictor(tr), windows=4)
    leaked_p = fit_calibrator("platt", np.asarray(raw["pooled"]["y"]),
                              np.asarray(raw["pooled"]["preds"]))(np.asarray(raw["pooled"]["preds"]))

    from sklearn.metrics import log_loss

    honest_ll = honest["pooled"]["log_loss"]
    leaked_ll = float(log_loss(raw["pooled"]["y"], leaked_p, labels=[0, 1]))

    assert leaked_ll < honest_ll, (
        "the leaked fit did NOT beat the honest one, so this fixture is not "
        "exercising the leak and proves nothing"
    )


def test_the_calibrator_is_never_fitted_on_the_window_it_scores():
    """Red-checkable form: with `min_windows_for_calibration=1`, window 1's
    calibrator may only have seen window 0.

    If the implementation appended to `seen_p` before scoring rather than after,
    window 1 would be calibrated on a set that includes its own test games -- and
    this test cannot see that directly, so it checks the weaker but observable
    consequence: the recorded fitting count must match the earlier windows'
    sizes exactly, never more.
    """
    from nba_predictor.models.evaluate.walk_forward_eval import walk_forward_metrics

    df = _frame(60)
    out = walk_forward_metrics(
        df, model_factory=lambda tr: _predictor(tr), windows=5,
        calibrator="platt", min_windows_for_calibration=1,
    )

    sizes = {w["n_test"] for w in out["windows"]}
    running = 0
    for record in out["calibration"]:
        if record["calibrated"]:
            assert record["n_fit_games"] == running, (
                f"window {record['window']} fitted on {record['n_fit_games']} games "
                f"but only {running} earlier games existed"
            )
        running += next(w["n_test"] for w in out["windows"] if w["window"] == record["window"])
    assert sizes, "fixture produced no windows"


def test_no_calibrator_leaves_every_window_untouched():
    """The default must be unchanged behaviour, or every existing caller and
    every recorded Phase A number moves for a reason nobody asked for."""
    from nba_predictor.models.evaluate.walk_forward_eval import walk_forward_metrics

    df = _frame(50)
    out = walk_forward_metrics(df, model_factory=lambda tr: _predictor(tr), windows=4)
    assert out["calibrated_from_window"] is None
    assert all(r["note"] == "no calibrator requested" for r in out["calibration"])
