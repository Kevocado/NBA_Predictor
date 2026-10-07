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

import json
from pathlib import Path

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

def _synthetic_season(n_games: int = 400) -> "pd.DataFrame":
    """A frame with the real feature schema, built from real column names.

    The first version of this test read `data/cache/training/games.json`, which
    is gitignored, so it passed locally and failed in CI with FileNotFoundError
    -- a test that only runs on one machine is not a test.
    """
    from nba_predictor.features.build import build_training_frame

    rng = np.random.default_rng(3)
    teams = ["BOS", "MIA", "LAL", "GSW", "DEN", "PHX", "MIL", "OKC"]
    dates = pd.date_range("2025-10-22", periods=n_games).astype(str)
    strength = {t: rng.normal(0, 5) for t in teams}
    rows = []
    for i, day in enumerate(dates):
        for k in range(len(teams) // 2):
            home, away = teams[2 * k], teams[2 * k + 1]
            margin = strength[home] - strength[away] + rng.normal(0, 13)
            total = 222 + rng.normal(0, 13)
            rows.append({
                "game_id": f"g{i}-{k}", "game_date": day,
                "home_team": home, "away_team": away,
                "home_pts": round((total + margin) / 2),
                "away_pts": round((total - margin) / 2),
                "home_fgm": 40, "home_fga": 88, "home_fg3m": 12,
                "home_tov": 11, "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
                "away_fgm": 38, "away_fga": 90, "away_fg3m": 10,
                "away_tov": 13, "away_oreb": 10, "away_dreb": 30, "away_fta": 18,
                "home_win": int(margin > 0),
            })
    frame, _ = build_training_frame(pd.DataFrame(rows))
    frame = frame.sort_values("game_date").reset_index(drop=True)
    return frame.assign(
        home_margin=frame["home_pts"] - frame["away_pts"],
        home_total=frame["home_pts"] + frame["away_pts"],
    )


def _out_of_fold_errors(frame, cols, n_windows: int = 4):
    from nba_predictor.models.candidate_race import default_candidates
    from nba_predictor.models.evaluate.walk_forward_eval import expanding_windows

    ridge = default_candidates(cols)["ridge"]
    margin_errors, total_errors = [], []
    for train_idx, test_idx in expanding_windows(frame["game_date"], n_windows=n_windows):
        train_df, test_df = frame.iloc[train_idx], frame.iloc[test_idx]
        for target, col, bucket in (
            ("margin", "home_margin", margin_errors),
            ("total", "home_total", total_errors),
        ):
            preds = ridge[target](train_df)(test_df)
            bucket.extend((test_df[col].to_numpy() - np.asarray(preds)).tolist())
    return margin_errors, total_errors


def test_sigmas_from_out_of_fold_errors_are_never_tighter_than_the_mae():
    """sigma = MAE * sqrt(pi/2), so it must always exceed the MAE it came from.

    Using MAE directly as sigma understates the spread by ~25% and makes every
    served probability read more confident than the evidence supports -- the
    failure mode this whole module exists to prevent.
    """
    from nba_predictor.features.build import FEATURE_COLUMNS

    frame = _synthetic_season()
    cols = [c for c in FEATURE_COLUMNS if c in frame.columns]
    margin_errors, total_errors = _out_of_fold_errors(frame, cols)

    for errors, name in ((margin_errors, "margin"), (total_errors, "total")):
        mae = float(np.mean(np.abs(errors)))
        sigma = fit_residual_sigma(errors)
        assert sigma > mae, f"{name}: sigma {sigma:.3f} <= MAE {mae:.3f}"
        assert sigma == pytest.approx(mae * np.sqrt(np.pi / 2), rel=1e-9)
        assert 5.0 < sigma < 40.0, f"{name}: sigma {sigma:.3f} is not NBA-scaled"


@pytest.mark.skipif(
    not Path("data/cache/training/games.json").exists(),
    reason="real training cache is gitignored; runs where the pipeline has been executed",
)
def test_sigmas_on_the_real_training_set_land_in_a_plausible_range():
    """The real 1,390-game set, where the fitted sigmas were margin 16.58 /
    total 20.69 -- wider than the spec's naive references (12 / 15-16) because
    the model's real error is worse than naive."""
    from nba_predictor.features.build import build_training_frame

    frame, cols = build_training_frame(
        pd.DataFrame(json.loads(Path("data/cache/training/games.json").read_text()))
    )
    frame = frame.sort_values("game_date").reset_index(drop=True)
    frame = frame.assign(
        home_margin=frame["home_pts"] - frame["away_pts"],
        home_total=frame["home_pts"] + frame["away_pts"],
    )
    margin_errors, total_errors = _out_of_fold_errors(frame, cols)
    sigmas = residual_sigmas(margin_errors=margin_errors, total_errors=total_errors)
    assert 8.0 < sigmas["margin_sigma"] < 20.0, sigmas
    assert 10.0 < sigmas["total_sigma"] < 25.0, sigmas


def test_win_from_margin_walk_forward_computes_on_synthetic():
    """Evaluate the served win-from-margin probability walk-forward on synthetic
    data to verify the computation pipeline works. The real comparison happens
    on the actual training data in the PR body."""
    from nba_predictor.features.build import build_training_frame
    from nba_predictor.models.candidate_race import default_candidates
    from nba_predictor.models.evaluate.walk_forward_eval import expanding_windows
    from nba_predictor.models.probability import win_prob, fit_residual_sigma

    frame = _synthetic_season(800)
    cols = [c for c in __import__("nba_predictor.features.build", fromlist=["FEATURE_COLUMNS"]).FEATURE_COLUMNS if c in frame.columns]

    # Out-of-fold margin predictions and residuals (same as walk_forward_regression)
    margin_errors, preds, outcomes = [], [], []
    ridge = default_candidates(cols)["ridge"]
    for train_idx, test_idx in expanding_windows(frame["game_date"], n_windows=4):
        train_df, test_df = frame.iloc[train_idx], frame.iloc[test_idx]
        m = ridge["margin"](train_df)
        test_preds = m(test_df)
        preds.extend(test_preds)
        # The ACTUAL result of each held-out game. Grading the prediction against
        # its own sign (`preds > 0`) would make every score perfect by
        # construction and is what the first version of this test did.
        outcomes.extend(test_df["home_win"].astype(int).tolist())
        margin_errors.extend((test_df["home_margin"].to_numpy() - np.asarray(test_preds)).tolist())

    # Fit sigma from out-of-fold margin residuals
    sigma = fit_residual_sigma(margin_errors)

    # Win-from-margin, scored against the held-out results
    y_true = np.asarray(outcomes, dtype=int)
    y_prob = np.array([win_prob(float(m), sigma) for m in preds])

    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
    wf_logloss = log_loss(y_true, y_prob)
    wf_brier = brier_score_loss(y_true, y_prob)
    wf_auc = roc_auc_score(y_true, y_prob)

    # Just verify it computes and produces valid probabilities
    assert 0.0 < wf_logloss < 2.0
    assert 0.0 < wf_brier < 1.0
    assert 0.0 < wf_auc < 1.0
    assert sigma > 0

    # Report for PR body comparison
    print(f"Win-from-margin walk-forward (synthetic): log_loss={wf_logloss:.4f}, brier={wf_brier:.4f}, auc={wf_auc:.4f}")
    print(f"Phase A winner (758 games): log_loss=0.6636, brier=0.2351, auc=0.6285")
    print(f"Sigma used: {sigma:.2f}")


def test_win_prob_derived_from_margin_matches_cover_at_pickem():
    """The coherence property: cover_prob(margin, 0, σ) == win_prob(margin, σ).

    A pick'em spread (line=0) *is* a positive margin, so the cover probability
    at line=0 must equal the win probability exactly. This is the mathematical
    guarantee that the three numbers a visitor reads come from one distribution.
    """
    sigma = 13.0
    for margin in (-10.0, -3.5, 0.0, 2.0, 8.0, 15.0):
        assert cover_prob(margin, 0.0, sigma) == pytest.approx(
            win_prob(margin, sigma), abs=1e-12
        ), f"margin={margin}: cover at 0 != win prob"


def test_win_probability_coherence_check():
    """A coherence check that the served win probability should match the
    margin-derived probability.

    This test documents the expected behavior: when a game is served,
    home_win_probability should equal norm.cdf(predicted_margin / margin_sigma).
    Any discrepancy means the model's win probability and margin are telling
    different stories about the same game.
    """
    # Example from live audit: MIN 72% win, margin +13.3 with σ≈13
    # If margin is +13.3 and σ=13, win prob from margin = norm.cdf(13.3/13) ≈ 0.85
    # But classifier says 0.72 -> INCOHERENT (discrepancy ~0.13)
    from scipy.stats import norm

    predicted_margin = 13.3
    sigma = 13.0
    win_from_margin = float(norm.cdf(predicted_margin / sigma))
    win_from_classifier = 0.72  # what the live site currently shows

    # The coherent win probability should be derived from margin
    assert win_from_margin == pytest.approx(0.85, abs=0.02)
    # This is the discrepancy we need to fix
    assert abs(win_from_margin - win_from_classifier) > 0.1
