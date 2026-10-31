"""Selection and holdout procedure for the pre-registered ratio-feature search.

The rule is fixed in docs/nba-prop-ratio-preregistration-2026-10.md. Summary:

* Selection: walk-forward over labels 2020, 2021, 2022 (fold f trains on every label < f). For each
  (candidate, market), the per-row paired difference d = |e_baseline| - |e_candidate| is pooled over the
  folds. A candidate passes if d is positive in every fold and its one-sided paired-t p-value survives
  Holm across all candidate x market tests.
* Holdout: labels 2023 and 2024, opened only for the top K selected pairs. Train on labels < 2023. The
  target market needs its paired-bootstrap MAE interval, at alpha/K, to exclude zero in the improving
  direction. Every other market must have a point-estimate MAE no higher than baseline.

Baseline and candidate use the same rows, the same XGBoost settings (models.player_props), and the same
production feature set (features.prop_model_frame.PROP_FEATURE_COLUMNS). Only the candidate column differs.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nba_predictor.features.prop_model_frame import PROP_FEATURE_COLUMNS
from nba_predictor.models.evaluate.paired_bootstrap import paired_bootstrap
from nba_predictor.models.player_props import predict_player_stat, train_player_stat_model
from nba_predictor.models.prop_holdout import MARKET_TARGETS

SELECTION_FOLDS = (2020, 2021, 2022)
HOLDOUT_LABELS = (2023, 2024)
TOP_K = 3
ALPHA = 0.05


def season_label(game_date: pd.Series) -> pd.Series:
    """Label = calendar year of the season's October start: 2019-04-10 is label 2018, 2024-10-22 is label 2024."""
    dt = pd.to_datetime(game_date)
    return dt.dt.year - (dt.dt.month < 9).astype(int)


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def one_sided_t_p(total: float, total_sq: float, n: int) -> tuple[float, float]:
    """(mean, p) for H1: mean > 0, from sums of d and d^2 (normal approximation to the t distribution)."""
    if n < 2:
        return float("nan"), 1.0
    mean = total / n
    var = max((total_sq - n * mean * mean) / (n - 1), 0.0)
    if var == 0.0:
        return mean, (0.0 if mean > 0 else 1.0)
    t = mean / math.sqrt(var / n)
    p = 0.5 * math.erfc(t / math.sqrt(2.0))
    return mean, p


def dedupe_candidates(cands: dict[str, pd.Series], frame: pd.DataFrame) -> tuple[dict[str, pd.Series], list[str]]:
    """Drop candidates numerically identical to an earlier candidate or to a production feature. Returns (kept, dropped names)."""
    kept: dict[str, pd.Series] = {}
    dropped: list[str] = []
    seen: list[np.ndarray] = [np.nan_to_num(frame[c].to_numpy(float), nan=-1e18) for c in PROP_FEATURE_COLUMNS if c in frame]
    for name, series in cands.items():
        arr = np.nan_to_num(series.to_numpy(float), nan=-1e18)
        if any(np.array_equal(arr, s) for s in seen):
            dropped.append(name)
            continue
        seen.append(arr)
        kept[name] = series
    return kept, dropped


def _fit_predict(frame: pd.DataFrame, feature_cols: list[str], train_mask, test_mask, target: str) -> np.ndarray:
    model = train_player_stat_model(frame.loc[train_mask, feature_cols], frame.loc[train_mask, target])
    return np.asarray(predict_player_stat(model, frame.loc[test_mask, feature_cols]), dtype=float)


def selection_stage(frame: pd.DataFrame, cands: dict[str, pd.Series], markets=None, folds=SELECTION_FOLDS, log=print) -> list[dict]:
    """Score every candidate x market on the selection folds. Returns one record per (candidate, market)."""
    markets = list(markets or MARKET_TARGETS)
    label = season_label(frame["game_date"])
    work = frame.copy()
    records: list[dict] = []
    for market in markets:
        target = MARKET_TARGETS[market]
        y_all = pd.to_numeric(work[target], errors="coerce").astype(float)
        base_err = {}
        for f in folds:
            tr, te = label < f, label == f
            pred = _fit_predict(work, PROP_FEATURE_COLUMNS, tr, te, target)
            base_err[f] = np.abs(pred - y_all[te].to_numpy())
        for k, (name, series) in enumerate(cands.items()):
            work["_cand"] = series.to_numpy(float)
            cols = PROP_FEATURE_COLUMNS + ["_cand"]
            per_fold = {}
            total, total_sq, n = 0.0, 0.0, 0
            for f in folds:
                tr, te = label < f, label == f
                pred = _fit_predict(work, cols, tr, te, target)
                d = base_err[f] - np.abs(pred - y_all[te].to_numpy())
                per_fold[str(f)] = float(d.mean())
                total += float(d.sum())
                total_sq += float((d * d).sum())
                n += int(d.size)
            mean, p = one_sided_t_p(total, total_sq, n)
            records.append({
                "candidate": name, "market": market, "n_test_rows": n,
                "mean_improvement_mae": mean, "per_fold_improvement_mae": per_fold,
                "stable_all_folds": all(v > 0 for v in per_fold.values()), "p_raw": p,
            })
        log(f"selection: market={market} candidates={len(cands)} done")

    adj = holm([r["p_raw"] for r in records])
    for r, a in zip(records, adj):
        r["p_holm"] = a
        r["selected_pool"] = bool(r["stable_all_folds"] and a < ALPHA)
    return records


def pick_top(records: list[dict], k: int = TOP_K) -> list[dict]:
    chosen = [r for r in records if r["selected_pool"]]
    chosen.sort(key=lambda r: (r["p_holm"], -r["mean_improvement_mae"]))
    return chosen[:k]


def holdout_stage(frame: pd.DataFrame, cands: dict[str, pd.Series], selected: list[dict], log=print) -> list[dict]:
    """Open the holdout once for the selected (candidate, market) pairs. Each candidate is added to all four markets."""
    label = season_label(frame["game_date"])
    train = label < min(HOLDOUT_LABELS)
    test = label.isin(HOLDOUT_LABELS)
    k = len(selected)
    alpha_k = ALPHA / max(k, 1)
    out: list[dict] = []
    for sel in selected:
        name = sel["candidate"]
        target_market = sel["market"]
        work = frame.copy()
        work["_cand"] = cands[name].to_numpy(float)
        cols = PROP_FEATURE_COLUMNS + ["_cand"]
        markets_out = {}
        for market in MARKET_TARGETS:
            target = MARKET_TARGETS[market]
            y = pd.to_numeric(work.loc[test, target], errors="coerce").astype(float).to_numpy()
            base_pred = _fit_predict(work, PROP_FEATURE_COLUMNS, train, test, target)
            cand_pred = _fit_predict(work, cols, train, test, target)
            naive = pd.to_numeric(work.loc[test, f"{target}_roll"], errors="coerce").astype(float).to_numpy()
            boot = paired_bootstrap(y, base_pred, cand_pred, "mae", alpha=alpha_k)
            markets_out[market] = {
                "n_holdout_rows": int(len(y)),
                "naive_mae": float(np.nanmean(np.abs(naive - y))),
                "baseline_mae": boot["baseline"], "candidate_mae": boot["candidate"],
                "difference_cand_minus_base": boot["difference"],
                "ci_low": boot["ci_low"], "ci_high": boot["ci_high"], "alpha_used": alpha_k,
                "improved_bootstrap": boot["improved"],
            }
        target_ok = markets_out[target_market]["improved_bootstrap"]
        others_ok = all(markets_out[m]["difference_cand_minus_base"] <= 0 for m in MARKET_TARGETS if m != target_market)
        out.append({
            "candidate": name, "target_market": target_market, "selection": sel,
            "markets": markets_out, "target_passes": bool(target_ok), "other_markets_not_worse": bool(others_ok),
            "passes": bool(target_ok and others_ok),
        })
        log(f"holdout: {name} target={target_market} passes={target_ok and others_ok}")
    return out
