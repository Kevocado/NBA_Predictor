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
    """(train_idx, test_idx) position arrays per window, in date order."""
    n = len(dates)
    if n < 4:
        raise ValueError("Need at least 4 rows for walk-forward windows")

    unique_dates = pd.Index(dates).unique()
    n_dates = len(unique_dates)
    if n_dates < 3:
        raise ValueError("Need at least 3 distinct dates for walk-forward windows")

    first_train_dates = max(1, int(n_dates * min_train_fraction))
    dates_per_window = max(1, (n_dates - first_train_dates) // n_windows)
    if n_dates - first_train_dates < 1:
        raise ValueError("Not enough distinct dates to build a walk-forward window")

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
) -> dict:
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
            "naive_mae": naive_mae_train,
            "naive_mean": naive_mean,
            "n": int(len(y_test)),
            "window": i,
            "train_max_date": train_max,
            "test_min_date": test_min,
            "n_train": len(train_df),
            "n_test": len(test_df),
        }
        if target == "home_margin":
            naive_fixed = 3.0
            metrics["naive_mae_fixed"] = float(mean_absolute_error(y_test, np.full_like(y_test, naive_fixed)))
        elif target == "home_total":
            naive_fixed = float(train_df[target].mean()) if len(train_df) > 0 else 0.0
            metrics["naive_mae_fixed"] = float(mean_absolute_error(y_test, np.full_like(y_test, naive_fixed)))
        per_window.append(metrics)
        ys.append(y_test)
        yhs.append(y_pred)

    y_all = np.concatenate(ys)
    yh_all = np.concatenate(yhs)
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
    if target == "home_margin":
        naive_fixed_all = np.full_like(y_all, 3.0)
        pooled["naive_mae_fixed"] = float(mean_absolute_error(y_all, naive_fixed_all))
    elif target == "home_total":
        naive_preds_fixed = []
        for i, (train_idx, test_idx) in enumerate(windows_list):
            train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
            m = float(train_df[target].mean()) if len(train_df) > 0 else 0.0
            naive_preds_fixed.append(np.full(len(test_df), m))
        naive_fixed_all = np.concatenate(naive_preds_fixed) if naive_preds_fixed else np.full_like(y_all, 0.0)
        pooled["naive_mae_fixed"] = float(mean_absolute_error(y_all, naive_fixed_all))
    return {"windows": per_window, "pooled": pooled}
