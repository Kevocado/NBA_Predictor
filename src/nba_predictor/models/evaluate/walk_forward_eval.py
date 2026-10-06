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


def walk_forward_metrics(
    df: pd.DataFrame,
    model_factory: Callable[[pd.DataFrame], object],
    windows: int = 4,
    *,
    date_col: str = "game_date",
    target_col: str = "home_win",
) -> dict:
    """Expanding-window walk-forward for the win model.

    `model_factory(train_df)` returns a `predict_proba(X) -> P(home win)`
    callable. Handing back the callable rather than a bare fitted model lets
    each candidate own its own feature columns -- which is what lets an Elo
    baseline, a Ridge and an XGBoost race unchanged in Task 5.

    Returns per-window metrics plus `pooled`: every out-of-fold prediction
    concatenated, the only figure computed entirely on games no window trained on.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    per_window, ys, ps, naive_ps = [], [], [], []

    for i, (train_idx, test_idx) in enumerate(expanding_windows(df[date_col], windows)):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        y_test = test_df[target_col]

        train_max, test_min = str(train_df[date_col].max()), str(test_df[date_col].min())
        if train_max >= test_min:
            raise AssertionError(
                f"window {i}: train_max_date {train_max} >= test_min_date {test_min}"
            )

        p = np.asarray(model_factory(train_df)(test_df), dtype=float)
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

    y_all = np.concatenate(ys)
    return {
        "windows": per_window,
        # Pooled naive uses each window's own training base rate, so the
        # comparator is honest about what was knowable at each point in time.
        "pooled": classification_metrics(
            y_all, np.concatenate(ps), naive_base_rate=np.concatenate(naive_ps)
        ),
    }


def walk_forward_regression(
    df: pd.DataFrame,
    model_factory: Callable[[pd.DataFrame], object],
    target: str,
    windows: int = 4,
    *,
    date_col: str = "game_date",
    fixed_baseline: float | None = None,
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
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    windows_list = expanding_windows(df[date_col], windows)
    per_window, ys, yhs = [], [], []

    for i, (train_idx, test_idx) in enumerate(windows_list):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
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
    }
    if fixed_baseline is not None:
        pooled["naive_mae_fixed"] = float(
            mean_absolute_error(y_all, np.full_like(y_all, fixed_baseline))
        )
    return {"windows": per_window, "pooled": pooled}