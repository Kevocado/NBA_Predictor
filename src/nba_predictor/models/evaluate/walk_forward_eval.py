"""Expanding-window walk-forward evaluation for the win model (spec section 5, G1).

The repo had one evaluation before this: `chronological_split`, a single 80/20
cut, reported as if it were walk-forward. One cut cannot show whether a model
holds up as the season moves.

Two properties are load-bearing:

* **Causality.** A test window starts strictly after its training window ends.
  A feature that has seen the game it predicts inflates every score here and
  the failure is invisible -- the number just looks good. Checked per window,
  not assumed.
* **Naive baselines ride along.** log-loss / Brier / AUC reported without the
  home-win base rate and the 0.6931 coin flip is exactly the bare-number
  reporting G1 was about.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score

from nba_predictor.models.calibration import (
    MIN_WINDOWS_FOR_CALIBRATION,
    fit_calibrator,
)

#: log(2) -- what a model that knows nothing scores. The NFL repo's 0.6227 is
#: quoted against this (spec section 5).
COINFLIP_LOG_LOSS = math.log(2)


def expanding_windows(
    dates: pd.Series,
    n_windows: int = 4,
    min_train_fraction: float = 0.4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_idx, test_idx) position arrays per window, in date order.

    Window 0 trains on the oldest `min_train_fraction` and tests on the next
    slice; each later window folds the previous test slice into training and
    takes a fresh slice after it. Expanding, not rolling -- history
    accumulates, which is what a season of games actually looks like.

    Cuts land on **date** boundaries, not row boundaries. The NBA plays 10-12
    games a night, so a row-wise cut lands mid-date often enough to matter:
    every feature on that date would have seen games in the same window it is
    being scored in. This refused to evaluate the real 1,390-game training set
    before the cut was moved onto dates.
    """
    n = len(dates)
    if n < 4:
        raise ValueError("Need at least 4 rows for walk-forward windows")

    unique_dates = pd.Index(dates).unique()  # order of first appearance
    n_dates = len(unique_dates)
    if n_dates < 3:
        raise ValueError("Need at least 3 distinct dates for walk-forward windows")

    first_train_dates = max(1, int(n_dates * min_train_fraction))
    dates_per_window = max(1, (n_dates - first_train_dates) // n_windows)
    if n_dates - first_train_dates < 1:
        raise ValueError("Not enough distinct dates to build a walk-forward window")

    # date value -> positions, so each window is a set of whole dates.
    by_date = {d: np.flatnonzero(np.asarray(dates == d)) for d in unique_dates}

    windows, train_end = [], first_train_dates
    while train_end < n_dates:
        test_end = min(n_dates, train_end + dates_per_window)
        train_idx = np.concatenate([by_date[d] for d in unique_dates[:train_end]])
        test_idx = np.concatenate([by_date[d] for d in unique_dates[train_end:test_end]])
        windows.append((train_idx, test_idx))
        train_end = test_end
    return windows


def classification_metrics(
    y_true: np.ndarray,
    p: np.ndarray,
    naive_base_rate: float | None = None,
) -> dict:
    """log-loss / Brier / AUC over out-of-fold predictions, plus both baselines.

    `naive_base_rate` is the home-win rate of the games *before* this window --
    what a naive model in production could actually know. Accepts a scalar (one
    window) or a per-observation array (pooled windows, each carrying the rate
    that was knowable then). Defaults to the pooled set's own rate only when a
    caller has no training window to hand.

    AUC is None when the pooled set has one class (a window with no home wins),
    because a number there would be fabricated.
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)
    if naive_base_rate is None:
        naive_p = np.full(len(y_true), np.clip(np.mean(y_true), 1e-12, 1 - 1e-12))
    else:
        naive_p = np.broadcast_to(np.asarray(naive_base_rate, dtype=float), y_true.shape)
        naive_p = np.clip(naive_p, 1e-12, 1 - 1e-12)
    return {
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p)),
        "accuracy": float(((p >= 0.5).astype(int) == y_true).mean()),
        "auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else None,
        "n": int(len(y_true)),
        # The bars. An evaluator that omits these is the thing G1 was about.
        "coinflip_log_loss": COINFLIP_LOG_LOSS,
        "naive_base_rate": float(np.mean(naive_p)),
        "naive_log_loss": float(log_loss(y_true, naive_p, labels=[0, 1])),
    }


def _history(history_df: pd.DataFrame | None, date_col: str) -> pd.DataFrame | None:
    """The extra-history frame, sorted, or None when there is nothing to add.

    Sorted for the same reason `df` is: the causality check reads the maximum
    training date as a string and compares it to the minimum test date, and that
    only means anything if both are in the same order.
    """
    if history_df is None or not len(history_df):
        return None
    return history_df.sort_values(date_col).reset_index(drop=True)


def walk_forward_metrics(
    df: pd.DataFrame,
    model_factory: Callable[[pd.DataFrame], object],
    windows: int = 4,
    *,
    date_col: str = "game_date",
    target_col: str = "home_win",
    history_df: pd.DataFrame | None = None,
    calibrator: str | None = None,
    min_windows_for_calibration: int = MIN_WINDOWS_FOR_CALIBRATION,
) -> dict:
    """Expanding-window walk-forward for the win model.

    `model_factory(train_df)` returns a `predict_proba(X) -> P(home win)`
    callable. Handing back the callable rather than a bare fitted model lets
    each candidate own its own feature columns -- which is what lets an Elo
    baseline, a Ridge and an XGBoost race unchanged in Task 5.

    Returns per-window metrics plus `pooled`: every out-of-fold prediction
    concatenated, the only figure computed entirely on games no window trained on.

    **`history_df` widens training without moving the tests.** Pass older games
    here and they join every window's TRAINING set; the test slices come from
    `df` alone and are therefore byte-identical to a run without history. That
    is the whole point: a multi-season model and a single-season model have to be
    scored on the same games, or the two numbers are not comparable and the
    comparison means nothing. History dated on or after a window's first test
    game is dropped, so the causality assertion below still holds.

    **`calibrator` is fitted inside the walk-forward, never outside it.** Pass
    "platt" or "isotonic" and window k's probabilities are mapped by a calibrator
    fitted only on out-of-fold predictions from windows *strictly before* k. A
    test game's own prediction is therefore never part of the data that
    calibrates it, which is the only version of this that measures anything
    rather than flattering the model.

    The first `MIN_WINDOWS_FOR_CALIBRATION` windows have too little earlier
    out-of-fold data to fit anything trustworthy, and are left uncalibrated. The
    threshold is reported per window in `calibrated` and as
    `calibrated_from_window`, so a pooled figure never silently mixes the two
    without saying which games were which.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    history = _history(history_df, date_col)
    per_window, ys, ps, naive_ps = [], [], [], []
    # Collect game_ids for explicit alignment
    game_ids_collected = []
    # Everything an earlier window scored out-of-fold, which is the only data a
    # later window's calibrator may see.
    seen_y: list[np.ndarray] = []
    seen_p: list[np.ndarray] = []
    calibration_record: list[dict] = []

    for i, (train_idx, test_idx) in enumerate(expanding_windows(df[date_col], windows)):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        if history is not None:
            cutoff = str(test_df[date_col].min())
            extra = history[history[date_col].astype(str) < cutoff]
            if len(extra):
                train_df = pd.concat([extra, train_df], ignore_index=True)
        train_df = train_df.sort_values(date_col).reset_index(drop=True)
        y_test = test_df[target_col]
        game_ids_collected.append(test_df["game_id"].to_numpy())

        train_max, test_min = str(train_df[date_col].max()), str(test_df[date_col].min())
        if train_max >= test_min:
            raise AssertionError(
                f"window {i}: train_max_date {train_max} >= test_min_date {test_min}"
            )

        p_raw = np.asarray(model_factory(train_df)(test_df), dtype=float)

        # Calibrate against STRICTLY earlier windows. `seen_*` is appended at the
        # bottom of this loop, so at this point it holds windows 0..i-1 and cannot
        # contain anything from window i, let alone from window i's test games.
        n_seen_windows = len(calibration_record)
        if calibrator is not None and n_seen_windows >= min_windows_for_calibration:
            prior_y = np.concatenate(seen_y)
            prior_p = np.concatenate(seen_p)
            try:
                p = np.asarray(fit_calibrator(calibrator, prior_y, prior_p)(p_raw), dtype=float)
                note = f"fitted on windows 0..{n_seen_windows - 1} (n={len(prior_y)})"
                did_calibrate = True
            except ValueError as exc:
                # A prior window with one class cannot support a fit. Score this
                # window uncalibrated and say so, rather than dropping the games.
                p = p_raw
                note = f"skipped: {exc}"
                did_calibrate = False
        else:
            p = p_raw
            did_calibrate = False
            if calibrator is not None:
                need = min_windows_for_calibration - n_seen_windows
                note = f"not calibrated: {need} more earlier window(s) needed"
            else:
                note = "no calibrator requested"

        calibration_record.append({
            "window": i,
            "calibrated": did_calibrate,
            # How many games the calibrator actually saw, cumulatively -- not the
            # size of the most recent window. A reader checking for leakage wants
            # the total that was in scope.
            "n_fit_games": int(sum(len(x) for x in seen_p)),
            "n_fit_windows": n_seen_windows,
            "note": note,
        })

        # The naive comparator uses the TRAINING window's home-win rate, never
        # the test window's. A rate read off the answers would quietly
        # strengthen the baseline with hindsight, by an amount that grows as
        # the season's home rate drifts -- so the headline number would move
        # for reasons that have nothing to do with the model.
        train_rate = float(train_df[target_col].mean())
        y_true = y_test.to_numpy()
        metrics = classification_metrics(y_true, p, naive_base_rate=train_rate)
        metrics.update(
            window=i,
            train_max_date=train_max,
            test_min_date=test_min,
            n_train=len(train_df),
            n_test=len(test_df),
        )
        per_window.append(metrics)
        ys.append(y_true)
        ps.append(p)
        naive_ps.append(np.full(len(y_true), np.clip(train_rate, 1e-12, 1 - 1e-12)))
        # Only now, after this window has been scored: these become the fitting
        # data for LATER windows and never for this one.
        seen_y.append(y_true)
        seen_p.append(p_raw)

    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    game_ids_all = np.concatenate(game_ids_collected)
    pooled = classification_metrics(
        y_all, p_all, naive_base_rate=np.concatenate(naive_ps)
    )
    # The out-of-fold predictions themselves, aligned with `y`. A summary of the
    # error is not enough to compare two models on the SAME games: a paired
    # bootstrap and a reliability table both need per-game values. Mirrors what
    # `walk_forward_regression` already returns for margin and total.
    pooled["preds"] = p_all.tolist()
    pooled["y"] = y_all.tolist()
    pooled["game_ids"] = game_ids_all.tolist()
    calibrated_windows = [r["window"] for r in calibration_record if r["calibrated"]]
    return {
        "windows": per_window,
        "calibration": calibration_record,
        # The first window that actually got calibrated, or None when none did.
        # A reader comparing two runs needs to know the pooled figure is a mix
        # of calibrated and uncalibrated games, and where the mix starts.
        "calibrated_from_window": calibrated_windows[0] if calibrated_windows else None,
        # Pooled naive uses each window's own training base rate, so the
        # comparator is honest about what was knowable at each point in time.
        "pooled": pooled,
    }


def walk_forward_regression(
    df: pd.DataFrame,
    model_factory: Callable[[pd.DataFrame], object],
    target: str,
    windows: int = 4,
    *,
    date_col: str = "game_date",
    fixed_baseline: float | None = None,
    history_df: pd.DataFrame | None = None,
) -> dict:
    """Expanding-window walk-forward for a regression target.

    `model_factory(train_df)` returns a `predict(X) -> np.ndarray` callable
    taking the full test frame and picking its own columns (mirrors the win
    path contract).

    Two naive baselines, deliberately distinct:
      * `naive_mae` scores the target's **training-window mean** -- a constant
        that adapts to each window.
      * `naive_mae_fixed` scores a **fixed constant** (e.g. margin −3, league
        average total) that does not adapt, and is only emitted when the caller
        supplies one. Reporting the training mean under both names would be one
        number wearing two labels.

    `history_df` widens training without moving the tests, exactly as
    `walk_forward_metrics` documents it.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    history = _history(history_df, date_col)
    windows_list = expanding_windows(df[date_col], windows)
    per_window, ys, yhs = [], [], []

    for i, (train_idx, test_idx) in enumerate(windows_list):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        if history is not None:
            cutoff = str(test_df[date_col].min())
            extra = history[history[date_col].astype(str) < cutoff]
            if len(extra):
                train_df = pd.concat([extra, train_df], ignore_index=True)
        train_df = train_df.sort_values(date_col).reset_index(drop=True)
        y_test = test_df[target].to_numpy()

        train_max, test_min = str(train_df[date_col].max()), str(test_df[date_col].min())
        if train_max >= test_min:
            raise AssertionError(
                f"window {i}: train_max_date {train_max} >= test_min_date {test_min}"
            )

        pred_fn = model_factory(train_df)
        y_pred = np.asarray(pred_fn(test_df), dtype=float)
        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        mae = float(mean_absolute_error(y_test, y_pred))
        naive_mean = float(train_df[target].mean()) if len(train_df) > 0 else 0.0
        naive_mae_train = float(mean_absolute_error(y_test, np.full_like(y_test, naive_mean)))
        metrics = {
            "mae": mae,
            "naive_mae": naive_mae_train,  # train-window mean baseline
            "naive_mean": naive_mean,
            "n": int(len(y_test)),
            "window": i,
            "train_max_date": train_max,
            "test_min_date": test_min,
            "n_train": len(train_df),
            "n_test": len(test_df),
        }
        if fixed_baseline is not None:
            metrics["naive_mae_fixed"] = float(
                mean_absolute_error(y_test, np.full_like(y_test, fixed_baseline))
            )
        per_window.append(metrics)
        ys.append(y_test)
        yhs.append(y_pred)

    y_all = np.concatenate(ys)
    yh_all = np.concatenate(yhs)
    residuals = y_all - yh_all
    # Pooled naive: per-window training means (leak-free), mirroring classification
    naive_preds = []
    for i, (train_idx, test_idx) in enumerate(windows_list):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        m = float(train_df[target].mean()) if len(train_df) > 0 else 0.0
        naive_preds.append(np.full(len(test_df), m))
    naive_all = np.concatenate(naive_preds) if naive_preds else np.full_like(y_all, 0.0)
    pooled = {
        "mae": float(mean_absolute_error(y_all, yh_all)),
        "naive_mae": float(mean_absolute_error(y_all, naive_all)),
        "naive_mean": float(naive_all.mean()) if len(naive_all) else 0.0,
        "n": int(len(y_all)),
        "residuals": residuals,  # for fitting proper sigma
        # The out-of-fold predictions themselves, aligned with `residuals`
        # and with `y`. The served win probability is a function of the
        # predicted margin (norm.cdf(margin / sigma)), so evaluating what is
        # actually served needs the margins and the real outcomes, not only
        # their error summary.
        "preds": yh_all.tolist(),
        "y": y_all.tolist(),
    }
    if fixed_baseline is not None:
        pooled["naive_mae_fixed"] = float(
            mean_absolute_error(y_all, np.full_like(y_all, fixed_baseline))
        )
    return {"windows": per_window, "pooled": pooled}