"""One residual-σ probability model, shared by every market (spec section 3.2, §6).

Win prob currently comes straight out of `predict_proba`, and cover/over are
independent regressors with no shared error distribution. That means the three
numbers a visitor reads do not have to agree with each other: a game can be
picked at 60% to win, favoured to cover by a different margin model, and
under a different total model, with nothing tying them together.

The fix is one fitted Normal residual σ per output, taken from pooled
walk-forward residuals, and every probability derived from a CDF:

    win   = norm.cdf(predicted_margin / sigma_margin)
    cover = norm.cdf((predicted_margin - line) / sigma_margin)
    over  = norm.cdf((predicted_total - line) / sigma_total)

`cover_prob` returns **None** when there is no real line. CFB's bug was
fabricating a 0.5 spread to cover against; a placeholder line is the same lie
one level up, so the honest answer is "no line, no cover chance".

Pinned here: the CDF is used rather than a logistic approximation, cover is
None without a line, probabilities are monotone in the prediction, a zero σ
cannot divide by zero, and the derived probabilities are coherent with each
other (a 50% cover line must read 0.5 whatever the predicted margin).
"""

import numpy as np
import pandas as pd
import pytest

from nba_predictor.models.probability import (
    cover_prob,
    fit_residual_sigma,
    over_prob,
    residual_sigmas,
    win_prob,
)


def test_cover_prob_is_none_without_a_real_line():
    """The CFB bug, one level up. A placeholder spread is a fabricated input to
    a real formula, and it is still fabricated."""
    assert cover_prob(predicted_margin=5.0, line=None, sigma=13.0) is None
    assert cover_prob(predicted_margin=5.0, line=float("nan"), sigma=13.0) is None


def test_win_prob_is_monotone_in_predicted_margin():
    assert win_prob(10.0, 13.0) > win_prob(2.0, 13.0) > win_prob(-5.0, 13.0)
    assert 0.0 <= win_prob(-100.0, 13.0) <= 1.0
    assert win_prob(100.0, 13.0) <= 1.0


def test_a_pick_em_line_reads_exactly_even():
    """Symmetry, and the property that makes the number trustworthy: predicting
    the line exactly must give 0.5, not 'roughly even'."""
    assert cover_prob(5.0, 5.0, 13.0) == pytest.approx(0.5, abs=1e-9)
    assert over_prob(220.0, 220.0, 15.0) == pytest.approx(0.5, abs=1e-9)
    assert win_prob(0.0, 13.0) == pytest.approx(0.5, abs=1e-9)


def test_cover_and_over_are_coherent_with_win():
    """The point of one residual model: the three numbers a visitor reads come
    from the same error distribution, so they cannot disagree.

    A pick'em line is `line = 0` — covering 0 *is* winning — so cover at 0 must
    reproduce the win probability exactly. And predicting a margin equal to the
    line is a genuine coin flip, not a confident 0.5.
    """
    sigma = 13.0
    assert cover_prob(8.0, 0.0, sigma) == pytest.approx(win_prob(8.0, sigma))
    assert cover_prob(8.0, 8.0, sigma) == pytest.approx(0.5, abs=1e-9)
    assert cover_prob(8.0, 3.0, sigma) > cover_prob(8.0, 12.0, sigma)
    assert over_prob(230.0, 220.0, 15.0) > 0.5
    assert over_prob(210.0, 220.0, 15.0) < 0.5


def test_a_tighter_sigma_sharpens_the_call():
    """Same prediction, more confident model: the probability moves further from
    even. If σ had no effect, the number would be a fixed-shape guess."""
    loose, tight = win_prob(6.0, 20.0), win_prob(6.0, 8.0)
    assert tight > loose > 0.5


def test_a_nonpositive_sigma_is_refused_rather_than_dividing_by_zero():
    """σ comes from a fitted residual spread. Zero or negative means the fit
    produced nothing usable, and `x / 0` would silently become ±inf."""
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="sigma"):
            win_prob(5.0, bad)


def test_fit_residual_sigma_is_the_actual_spread_of_the_errors():
    """σ must be measured, not configured. A model's MAE is not its σ either --
    for Normal errors, σ = MAE * sqrt(pi/2), and using MAE directly would
    understate the spread and make every probability too confident."""
    rng = np.random.default_rng(0)
    errors = rng.normal(0, 7.0, 20000)
    sigma = fit_residual_sigma(errors)
    assert sigma == pytest.approx(7.0, rel=0.05)


def test_fit_residual_sigma_refuses_an_empty_or_degenerate_fit():
    with pytest.raises(ValueError):
        fit_residual_sigma(np.array([]))
    # Zero spread means the model was perfect, which means the fit is broken.
    with pytest.raises(ValueError):
        fit_residual_sigma(np.zeros(50))


def test_residual_sigmas_pulls_one_sigma_per_output():
    rng = np.random.default_rng(1)
    margin_errors = rng.normal(0, 12.0, 5000)
    total_errors = rng.normal(0, 15.0, 5000)
    sigmas = residual_sigmas(margin_errors=margin_errors, total_errors=total_errors)
    assert sigmas["margin_sigma"] == pytest.approx(12.0, rel=0.05)
    assert sigmas["total_sigma"] == pytest.approx(15.0, rel=0.05)


def test_probabilities_are_computed_from_a_real_normal_cdf():
    """Not a logistic eyeball fit of the CDF. At x/σ = 0.674 the standard
    normal CDF is 0.75; a logistic shortcut would miss that."""
    from scipy.stats import norm

    for margin in (-8.0, 0.0, 3.5, 20.0):
        assert win_prob(margin, 13.0) == pytest.approx(
            float(norm.cdf(margin / 13.0)), abs=1e-12
        )


def test_an_all_nan_residual_set_refuses_to_fit_rather_than_returning_nan():
    """The all-NaN column class of bug (spec section 12.2): a broken pull reads
    as "no signal" unless it is made to fail. An all-NaN residual set means the
    walk-forward fit produced nothing, so it must raise — a NaN σ would flow on
    into a NaN probability that renders on screen as if it were a real number."""
    with pytest.raises(ValueError, match="no finite residuals"):
        residual_sigmas(
            margin_errors=np.array([np.nan, np.nan, np.nan]),
            total_errors=np.array([1.0, 2.0, 3.0]),
        )
    with pytest.raises(ValueError):
        win_prob(5.0, float("nan"))

def test_sigmas_fitted_from_real_walk_forward_residuals_land_in_a_plausible_range():
    """σ must come from the race-winning model's actual out-of-fold errors on
    the real data, and land near the spread the spec quotes (margin σ≈12,
    total σ≈15-16). A σ fitted on in-sample errors would be far too small and
    every probability would read as confident."""
    import json

    from nba_predictor.features.build import build_training_frame
    from nba_predictor.models.candidate_race import default_candidates
    from nba_predictor.models.evaluate.walk_forward_eval import expanding_windows

    frame, cols = build_training_frame(
        pd.DataFrame(json.load(open("data/cache/training/games.json")))
    )
    frame = frame.sort_values("game_date").reset_index(drop=True)
    frame = frame.assign(
        home_margin=frame["home_pts"] - frame["away_pts"],
        home_total=frame["home_pts"] + frame["away_pts"],
    )
    candidates = default_candidates(cols)
    ridge = candidates["ridge"]

    margin_errors, total_errors = [], []
    for train_idx, test_idx in expanding_windows(frame["game_date"], n_windows=4):
        train_df, test_df = frame.iloc[train_idx], frame.iloc[test_idx]
        for target, col, bucket in (
            ("margin", "home_margin", margin_errors),
            ("total", "home_total", total_errors),
        ):
            preds = ridge[target](train_df)(test_df)
            bucket.extend((test_df[col].to_numpy() - np.asarray(preds)).tolist())

    sigmas = residual_sigmas(
        margin_errors=margin_errors, total_errors=total_errors
    )
    assert 8.0 < sigmas["margin_sigma"] < 20.0, sigmas
    assert 10.0 < sigmas["total_sigma"] < 25.0, sigmas
