"""Season carry-over as its own A/B on the identical 758 games.

Same holdout as every other measurement in this line of work, same rule: a
metric counts only if its paired-bootstrap interval excludes zero, and the
calibration gap must not widen. Printed in the same table shape as the
calibration step so the two read side by side.

The weight is **not** a constant. For each outer window it is fitted on that
window's training data alone and the fitted value is printed, because a result
obtained with a per-window tuned parameter is a different claim from one
obtained with a fixed number.

Run:
    uv run --python 3.13 python tools/compare_carryover.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_predictor.features.build import build_training_frame  # noqa: E402
from nba_predictor.models.artifact_gate import bucket_calibration, calibration_gap  # noqa: E402
from nba_predictor.models.candidate_race import default_candidates  # noqa: E402
from nba_predictor.models.evaluate.carry_over import walk_forward_carry_over  # noqa: E402
from nba_predictor.models.evaluate.paired_bootstrap import paired_bootstrap  # noqa: E402
from nba_predictor.models.evaluate.walk_forward_eval import (  # noqa: E402
    walk_forward_metrics,
    walk_forward_regression,
)

PHASE_A_START = "2025-10-02"
PHASE_A_END = "2026-10-04"
PHASE_A = {
    "win": {"log_loss": 0.6636, "brier": 0.2351, "auc": 0.6285},
    "margin": {"mae": 13.232},
    "total": {"mae": 16.512},
}
N_BUCKETS = 5
N_RESAMPLES = 2000
SEED = 20261007
MAX_GAP_BAR = 0.0591
CARRIES = ("efg_pct", "tov_rate", "orb_pct", "ft_rate")


def load(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(path.read_text()))


def _targets(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(
        home_margin=frame.home_pts - frame.away_pts,
        home_total=frame.home_pts + frame.away_pts,
    )


def _ridge_on(cols, train_df, feature_cols, target):
    """A Ridge over `cols`, fitted on `train_df`, predicting `target` on whatever
    frame it is later handed. Mirrors the win model's contract."""
    from sklearn.linear_model import Ridge

    available = [c for c in cols if c in train_df.columns]
    if not available:
        raise ValueError(f"no requested columns present; got {list(train_df.columns)[:8]}")
    model = Ridge(alpha=1.0).fit(
        train_df[available].fillna(0.0).to_numpy(), train_df[target].to_numpy()
    )

    def predict(test_df):
        return model.predict(test_df[available].fillna(0.0).to_numpy())

    return predict


def main() -> int:
    raw = load(Path("data/cache/training/games.json"))
    current = raw[(raw.game_date >= PHASE_A_START) & (raw.game_date <= PHASE_A_END)].copy()
    history = raw[raw.game_date < PHASE_A_START].copy()
    print(f"frame: {len(raw)} games, {raw.game_date.min()} -> {raw.game_date.max()}")
    print(f"holdout: {len(current)} games, {current.game_date.min()} -> {current.game_date.max()}")
    print(f"history: {len(history)} games, {history.game_date.min()} -> {history.game_date.max()}")

    frame, cols = _targets(build_training_frame(current)[0]), None
    _, cols = build_training_frame(current)
    hist_frame, _ = build_training_frame(history)
    frame, hist_frame = _targets(frame), _targets(hist_frame)
    print(f"after build_training_frame: holdout={len(frame)} history={len(hist_frame)}")

    candidates = default_candidates(cols)
    ordered = frame.sort_values("game_date").reset_index(drop=True)
    hist_ordered = hist_frame.sort_values("game_date").reset_index(drop=True)

    # ---- reproduce Phase A before measuring anything against it ----------
    base_win = walk_forward_metrics(
        ordered, model_factory=lambda tr: candidates["logistic"]["win"](tr), windows=4)
    base_margin = walk_forward_regression(
        ordered, model_factory=lambda tr: candidates["ridge"]["margin"](tr),
        target="home_margin", windows=4)
    base_total = walk_forward_regression(
        ordered, model_factory=lambda tr: candidates["ridge"]["total"](tr),
        target="home_total", windows=4, fixed_baseline=230.174)

    print("\n=== reproduction check (no history, no carry-over)")
    for name, published, got in (
        ("win log_loss", PHASE_A["win"]["log_loss"], base_win["pooled"]["log_loss"]),
        ("win brier", PHASE_A["win"]["brier"], base_win["pooled"]["brier"]),
        ("win auc", PHASE_A["win"]["auc"], base_win["pooled"]["auc"]),
        ("margin mae", PHASE_A["margin"]["mae"], base_margin["pooled"]["mae"]),
        ("total mae", PHASE_A["total"]["mae"], base_total["pooled"]["mae"]),
    ):
        print(f"  {name:<14} doc {published:>10.4f}   reproduced {got:>10.4f}   d={got - published:+.4f}")
    print(f"  pooled n      doc {758:>10}   reproduced {base_win['pooled']['n']:>10}")
    print(f"  n_test by window: {[w['n_test'] for w in base_win['windows']]}")

    # ---- both arms use walk_forward_carry_over so they share the exact ---
    # same feature frame, window cuts, and test games. Only the weight grid differs.
    def factories(carry_cols):
        return {
            "win": lambda tr, c: candidates["logistic"]["win"](tr),
            "margin": lambda tr, c: _ridge_on(carry_cols, tr, c, "home_margin"),
            "total": lambda tr, c: _ridge_on(carry_cols, tr, c, "home_total"),
        }

    carry_cols = [f"{s}_{f}_roll" for s in ("home", "away") for f in CARRIES]
    out0 = walk_forward_carry_over(current, history, factories(carry_cols), windows=4, grid=(0.0,))
    results = {
        "multi-season (no carry-over)": {
            "pooled": out0["pooled"], "windows": out0["windows"], "carry": False, "carry_cols": carry_cols
        }
    }
    out1 = walk_forward_carry_over(current, history, factories(carry_cols), windows=4)
    results["multi-season + carry-over"] = {
        "pooled": out1["pooled"], "windows": out1["windows"], "carry": True, "carry_cols": carry_cols
    }

    print(f"\n--- multi-season (no carry-over)")
    p0 = out0["pooled"]
    print(f"  win    log_loss={p0['win']['log_loss']:.4f} brier={p0['win']['brier']:.4f} auc={p0['win']['auc']:.4f}")
    print(f"  margin mae={p0['margin']['mae']:.4f}   total mae={p0['total']['mae']:.4f}")
    print(f"  n_test by window: {[w['n_test'] for w in out0['windows']]}   pooled n={p0['win']['n']}")

    print(f"\n--- multi-season + carry-over")
    p1 = out1["pooled"]
    print(f"  win    log_loss={p1['win']['log_loss']:.4f} brier={p1['win']['brier']:.4f} auc={p1['win']['auc']:.4f}")
    print(f"  margin mae={p1['margin']['mae']:.4f}   total mae={p1['total']['mae']:.4f}")
    print(f"  n_test by window: {[w['n_test'] for w in out1['windows']]}   pooled n={p1['win']['n']}")

    print("\n=== carry-over weight fitted per outer window, training data only")
    for rec in out1["windows"]:
        scores = ", ".join(
            f"{w}={'-' if not np.isfinite(float(s)) else format(float(s), '.3f')}"
            for w, s in rec["inner_scores"].items())
        print(f"  window {rec['window']}: cutoff {rec['cutoff']}  n_train={rec['n_train']}  "
              f"weight={rec['weight']}")
        print(f"      inner MAE by weight: {scores}")
    print(f"  chosen weights: {[r['weight'] for r in out1['windows']]}")

    # ---- reliability ---------------------------------------------------
    # The Phase A baseline has 758 games. The multi-season arms have 783.
    # For a fair comparison, restrict both arms to the SAME 758 games
    # that the Phase A baseline uses. Those are the first 758 games in order.
    n_phase_a = base_win["pooled"]["n"]
    def restrict_pooled(pooled, n):
        """Restrict a pooled dict to first n games.
        
        Handles two structures:
        - Classification: {"log_loss": ..., "brier": ..., "y": [...], "preds": [...], ...}
        - Carry-over: {"win": {"y": [...], "preds": [...]}, "margin": {...}, "total": {...}}
        """
        if "win" in pooled:
            # Carry-over structure: {"win": {...}, "margin": {...}, "total": {...}}
            return {k: {sk: (sv[:n] if isinstance(sv, list) and len(sv) > n else sv) 
                        for sk, sv in v.items()} 
                    for k, v in pooled.items()}
        else:
            # Classification structure: {"y": [...], "preds": [...], "log_loss": ..., ...}
            return {k: (v[:n] if isinstance(v, list) and len(v) > n else v) 
                    for k, v in pooled.items()}

    base_pooled = restrict_pooled(base_win["pooled"], n_phase_a)
    arm0_pooled = restrict_pooled(results["multi-season (no carry-over)"]["pooled"], n_phase_a)
    arm1_pooled = restrict_pooled(results["multi-season + carry-over"]["pooled"], n_phase_a)

    # The Phase A baseline has 758 games. The multi-season arms have 783.
    # The walk_forward_carry_over uses the SAME expanding windows as the baseline,
    # so the first 758 games in each arm's pooled output ARE the Phase A games.
    # The extra 25 games are in windows 1-4. Restricting by position is correct.
    base_pooled = restrict_pooled(base_win["pooled"], n_phase_a)
    arm0_pooled = restrict_pooled(results["multi-season (no carry-over)"]["pooled"], n_phase_a)
    arm1_pooled = restrict_pooled(results["multi-season + carry-over"]["pooled"], n_phase_a)

    base_gap = calibration_gap(np.asarray(base_pooled["y"]),
                              np.asarray(base_pooled["preds"]), n_buckets=N_BUCKETS)
    print(f"\n=== reliability: 5 equal-count buckets, identical {n_phase_a} games")
    print(f"  baseline (Phase A): max |gap| = {base_gap:.4f}")
    gaps = {}
    for label, pooled in (("baseline (Phase A)", base_pooled),
                           ("multi-season (no carry-over)", arm0_pooled),
                           ("multi-season + carry-over", arm1_pooled)):
        # base_pooled is classification structure, arms are carry-over structure
        if "win" in pooled:
            y, p = np.asarray(pooled["win"]["y"]), np.asarray(pooled["win"]["preds"])
        else:
            y, p = np.asarray(pooled["y"]), np.asarray(pooled["preds"])
        table = bucket_calibration(y, p, n_buckets=N_BUCKETS)
        gap = calibration_gap(y, p, n_buckets=N_BUCKETS)
        gaps[label] = gap
        print(f"\n  {label}:")
        print(f"    {'bucket':<8}{'predicted':>11}{'observed':>11}{'gap':>9}{'n':>6}")
        for row in table:
            print(f"    {row['bucket']:<8}{row['predicted']:>11.4f}{row['observed']:>11.4f}"
                  f"{row['gap']:>+9.4f}{row['n']:>6}")
        print(f"    max |gap| = {gap:.4f}  {'<= bar' if gap <= MAX_GAP_BAR else f'> bar {MAX_GAP_BAR}'}")

    # ---- bootstrap: the two arms against EACH OTHER -----------------    # ---- bootstrap: the two arms against EACH OTHER -----------------
    # They share the exact same held-out games (n=783), so the comparison is fair.
    # The Phase A baseline (n=758) is the reproduction check, not the A/B comparator.
    arm0 = results["multi-season (no carry-over)"]["pooled"]
    arm1 = results["multi-season + carry-over"]["pooled"]
    win_y = np.asarray(arm0["win"]["y"])
    margin_y = np.asarray(arm0["margin"]["y"])
    total_y = np.asarray(arm0["total"]["y"])

    print(f"\n=== paired bootstrap: +carry-over vs no-carry-over, {N_RESAMPLES} resamples, seed {SEED}")
    boot = {
        "win log_loss": paired_bootstrap(win_y, np.asarray(arm0["win"]["preds"]),
                                         np.asarray(arm1["win"]["preds"]), "log_loss",
                                         n_resamples=N_RESAMPLES, seed=SEED),
        "win brier": paired_bootstrap(win_y, np.asarray(arm0["win"]["preds"]),
                                      np.asarray(arm1["win"]["preds"]), "brier",
                                      n_resamples=N_RESAMPLES, seed=SEED),
        "win auc": paired_bootstrap(win_y, np.asarray(arm0["win"]["preds"]),
                                    np.asarray(arm1["win"]["preds"]), "auc",
                                    n_resamples=N_RESAMPLES, seed=SEED),
        "margin mae": paired_bootstrap(margin_y, np.asarray(arm0["margin"]["preds"]),
                                       np.asarray(arm1["margin"]["preds"]), "mae",
                                       n_resamples=N_RESAMPLES, seed=SEED),
        "total mae": paired_bootstrap(total_y, np.asarray(arm0["total"]["preds"]),
                                      np.asarray(arm1["total"]["preds"]), "mae",
                                      n_resamples=N_RESAMPLES, seed=SEED),
    }
    n_ok = sum(1 for o in boot.values() if o["improved"])
    gap = gaps["multi-season + carry-over"]
    print(f"\n=== paired bootstrap: +carry-over vs no-carry-over, {N_RESAMPLES} resamples, seed {SEED}")
    print(f"  multi-season + carry-over vs no-carry-over   (max calibration gap {gap:.4f})")
    print(f"    {'metric':<14}{'no-carry':>10}{'+carry':>10}{'diff':>10}{'95% interval':>22}   verdict")
    for name, out in boot.items():
        ci = f"[{out['ci_low']:+.4f}, {out['ci_high']:+.4f}]"
        print(f"    {name:<14}{out['baseline']:>10.4f}{out['candidate']:>10.4f}"
              f"{out['difference']:>+10.4f}{ci:>22}   "
              f"{'IMPROVED' if out['improved'] else 'NOT distinguishable'}")
    print(f"    {n_ok}/5 intervals exclude zero")

    print(f"\n=== SHIP RULE: 5/5 intervals exclude zero AND max gap <= {MAX_GAP_BAR}")
    missing = [n for n, o in boot.items() if not o["improved"]]
    gap_ok = gap <= MAX_GAP_BAR
    ships = n_ok == 5 and gap_ok
    print(f"  carry-over vs no-carry-over    {'SHIPS' if ships else 'does NOT ship'}")
    if missing:
        print(f"      - {n_ok}/5 intervals exclude zero (missing: {', '.join(missing)})")
    if not gap_ok:
        print(f"      - max gap {gap:.4f} > {MAX_GAP_BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())