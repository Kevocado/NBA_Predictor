"""Phase A vs multi-season, on the identical held-out games.

The gate: a number from one held-out set is never put beside a number from
another. So `df` is Phase A's own frame -- the games it was evaluated on, same
dates -- and `history_df` is the earlier seasons, which reach TRAINING only.

Run:
    uv run --python 3.13 python tools/compare_phase_a.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_predictor.features.build import FEATURE_COLUMNS, build_training_frame  # noqa: E402
from nba_predictor.models.candidate_race import default_candidates  # noqa: E402
from nba_predictor.models.artifact_gate import bucket_calibration, calibration_gap  # noqa: E402
from nba_predictor.models.evaluate.paired_bootstrap import paired_bootstrap  # noqa: E402
from nba_predictor.models.evaluate.walk_forward_eval import (  # noqa: E402
    walk_forward_metrics,
    walk_forward_regression,
)

# Phase A's frame, as recorded in docs/nba-parity-evaluation-2026-10.md:
# "1,368 games from data/cache/training/games.json (2025-10-02 -> 2026-10-04)".
PHASE_A_START = "2025-10-02"
PHASE_A_END = "2026-10-04"

#: Reliability buckets, bootstrap resamples and seed. All three are stated here
#: rather than left implicit so a reader can reproduce the exact intervals, and
#: so a re-run with different numbers is visibly a change of method.
N_BUCKETS = 5
N_RESAMPLES = 2000
SEED = 20261007

# Phase A's published figures, to reproduce first and then beat.
PHASE_A = {
    "win": {"log_loss": 0.6636, "brier": 0.2351, "auc": 0.6285},
    "margin": {"mae": 13.232, "naive": 14.018},
    "total": {"mae": 16.512, "naive_fixed": 16.375},
}


def load(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(path.read_text()))


def main() -> int:
    training = load(Path("data/cache/training/games.json"))
    print(f"training cache: {len(training)} games, "
          f"{training['game_date'].min()} -> {training['game_date'].max()}")

    current = training[(training["game_date"] >= PHASE_A_START)
                       & (training["game_date"] <= PHASE_A_END)].copy()
    history = training[training["game_date"] < PHASE_A_START].copy()

    print(f"Phase A frame:  {len(current)} games, "
          f"{current['game_date'].min()} -> {current['game_date'].max()}")
    print(f"history frame:  {len(history)} games, "
          f"{history['game_date'].min()} -> {history['game_date'].max()}")

    # Features are built over each frame independently. That is deliberate: the
    # features for a Phase A game must be exactly what Phase A saw, or the
    # baseline is not the baseline.
    frame, cols = build_training_frame(current)
    hist_frame, _ = build_training_frame(history)
    print(f"after build_training_frame: phase_a={len(frame)} history={len(hist_frame)}")

    # The regression targets are derived columns. They have to be derived on the
    # history frame too, or the joined training set carries NaN targets for
    # every historical row and the fit refuses the data.
    frame = frame.assign(
        home_margin=frame["home_pts"] - frame["away_pts"],
        home_total=frame["home_pts"] + frame["away_pts"],
    )
    hist_frame = hist_frame.assign(
        home_margin=hist_frame["home_pts"] - hist_frame["away_pts"],
        home_total=hist_frame["home_pts"] + hist_frame["away_pts"],
    )

    candidates = default_candidates(cols)
    results = {}

    for label, hist in (("phase_a_only", None), ("with_history", hist_frame)):
        win = walk_forward_metrics(
            frame,
            model_factory=lambda tr: candidates["logistic"]["win"](tr),
            windows=4,
            history_df=hist,
        )
        margin = walk_forward_regression(
            frame,
            model_factory=lambda tr: candidates["ridge"]["margin"](tr),
            target="home_margin",
            windows=4,
            history_df=hist,
        )
        total = walk_forward_regression(
            frame,
            model_factory=lambda tr: candidates["ridge"]["total"](tr),
            target="home_total",
            windows=4,
            # The league-average total of the Phase A frame, so the naive column
            # reproduces the 16.375 in the Phase A doc rather than a constant
            # I picked.
            fixed_baseline=230.174,
            history_df=hist,
        )
        results[label] = {"win": win, "margin": margin, "total": total}
        print(f"\n--- {label}")
        print(f"  win    log_loss={win['pooled']['log_loss']:.4f} "
              f"brier={win['pooled']['brier']:.4f} auc={win['pooled']['auc']:.4f}")
        print(f"  margin mae={margin['pooled']['mae']:.4f} "
              f"naive={margin['pooled']['naive_mae']:.4f}")
        print(f"  total  mae={total['pooled']['mae']:.4f} "
              f"naive_fixed={total['pooled'].get('naive_mae_fixed')}")
        print(f"  n_train by window: {[w['n_train'] for w in win['windows']]}")
        print(f"  n_test  by window: {[w['n_test'] for w in win['windows']]}")
        print(f"  pooled n: {win['pooled'].get('n', 'n/a')}")

    print("\n=== against Phase A (published 2026-10-05)")
    base = results["phase_a_only"]
    hist = results["with_history"]
    rows = [
        ("win log_loss", PHASE_A["win"]["log_loss"], base["win"]["pooled"]["log_loss"], hist["win"]["pooled"]["log_loss"], "lower"),
        ("win brier", PHASE_A["win"]["brier"], base["win"]["pooled"]["brier"], hist["win"]["pooled"]["brier"], "lower"),
        ("win auc", PHASE_A["win"]["auc"], base["win"]["pooled"]["auc"], hist["win"]["pooled"]["auc"], "higher"),
        ("margin mae", PHASE_A["margin"]["mae"], base["margin"]["pooled"]["mae"], hist["margin"]["pooled"]["mae"], "lower"),
        ("total mae", PHASE_A["total"]["mae"], base["total"]["pooled"]["mae"], hist["total"]["pooled"]["mae"], "lower"),
    ]
    print(f"{'metric':<14}{'phase A doc':>12}{'reproduced':>13}{'+history':>12}   verdict")
    for name, published, reproduced, with_history, better in rows:
        if better == "lower":
            verdict = "history WINS" if with_history < reproduced else ("tie" if with_history == reproduced else "history LOSES")
        else:
            verdict = "history WINS" if with_history > reproduced else ("tie" if with_history == reproduced else "history LOSES")
        print(f"{name:<14}{published:>12.4f}{reproduced:>13.4f}{with_history:>12.4f}   {verdict}")

    # --- (a) reliability, 5 quantile buckets, on the identical 758 ---------
    print("\n=== reliability: observed vs predicted, 5 equal-count buckets")
    print("(identical 758 out-of-fold games; equal-COUNT buckets because NBA")
    print(" win probabilities cluster near 0.56 and equal-width would fill one)")
    for label, res in (("baseline", base), ("multi-season", hist)):
        table = bucket_calibration(
            np.asarray(res["win"]["pooled"]["y"]),
            np.asarray(res["win"]["pooled"]["preds"]),
            n_buckets=N_BUCKETS,
        )
        print(f"\n  {label}:")
        print(f"    {'bucket':<8}{'predicted':>11}{'observed':>11}{'gap':>9}{'n':>6}")
        for row in table:
            print(f"    {row['bucket']:<8}{row['predicted']:>11.4f}{row['observed']:>11.4f}"
                  f"{row['gap']:>+9.4f}{row['n']:>6}")
        worst = max(table, key=lambda r: abs(r["gap"]))
        gap = calibration_gap(
            np.asarray(res["win"]["pooled"]["y"]),
            np.asarray(res["win"]["pooled"]["preds"]),
            n_buckets=N_BUCKETS,
        )
        results[f"{label}_calibration"] = {"table": table, "max_gap": gap}
        print(f"    max |gap| = {gap:.4f}  (bucket {worst['bucket']})")
    base_gap = results["baseline_calibration"]["max_gap"]
    hist_gap = results["multi-season_calibration"]["max_gap"]
    verdict = ("NOT WIDENED" if hist_gap <= base_gap
               else f"WIDENED by {hist_gap - base_gap:+.4f}")
    print(f"\n  calibration gap: baseline {base_gap:.4f} -> multi-season {hist_gap:.4f}  {verdict}")

    # --- (b) paired bootstrap on the identical 758 -------------------------
    print(f"\n=== paired bootstrap, {N_RESAMPLES} resamples, seed {SEED}")
    print("one draw of game indices applied to BOTH models; 95% percentile interval")
    print("on the DIFFERENCE. A metric counts as improved only if the interval")
    print("excludes zero in the direction that is better.")
    win_y = np.asarray(base["win"]["pooled"]["y"])
    margin_y = np.asarray(base["margin"]["pooled"]["y"])
    total_y = np.asarray(base["total"]["pooled"]["y"])

    boot = {
        "win log_loss": paired_bootstrap(
            win_y, np.asarray(base["win"]["pooled"]["preds"]),
            np.asarray(hist["win"]["pooled"]["preds"]), "log_loss",
            n_resamples=N_RESAMPLES, seed=SEED),
        "win brier": paired_bootstrap(
            win_y, np.asarray(base["win"]["pooled"]["preds"]),
            np.asarray(hist["win"]["pooled"]["preds"]), "brier",
            n_resamples=N_RESAMPLES, seed=SEED),
        "win auc": paired_bootstrap(
            win_y, np.asarray(base["win"]["pooled"]["preds"]),
            np.asarray(hist["win"]["pooled"]["preds"]), "auc",
            n_resamples=N_RESAMPLES, seed=SEED),
        "margin mae": paired_bootstrap(
            margin_y, np.asarray(base["margin"]["pooled"]["preds"]),
            np.asarray(hist["margin"]["pooled"]["preds"]), "mae",
            n_resamples=N_RESAMPLES, seed=SEED),
        "total mae": paired_bootstrap(
            total_y, np.asarray(base["total"]["pooled"]["preds"]),
            np.asarray(hist["total"]["pooled"]["preds"]), "mae",
            n_resamples=N_RESAMPLES, seed=SEED),
    }
    print(f"\n  {'metric':<14}{'baseline':>10}{'+history':>10}{'diff':>10}"
          f"{'95% interval':>22}   verdict")
    n_improved = 0
    for name, out in boot.items():
        ci = f"[{out['ci_low']:+.4f}, {out['ci_high']:+.4f}]"
        if out["improved"]:
            verdict = "IMPROVED (CI excludes 0)"
            n_improved += 1
        else:
            verdict = "NOT distinguishable from 0"
        print(f"  {name:<14}{out['baseline']:>10.4f}{out['candidate']:>10.4f}"
              f"{out['difference']:>+10.4f}{ci:>22}   {verdict}")
    print(f"\n  {n_improved}/5 metrics have intervals excluding zero")

    # --- (c) the verdict, stated as a gate --------------------------------
    print("\n=== verdict")
    not_widened = hist_gap <= base_gap
    print(f"  metrics improved with a 95% interval excluding zero: {n_improved}/5")
    if not [n for n in boot if not boot[n]["improved"]]:
        print("  calibration gap: not widened")
        print("  VERDICT: all five metrics clear the bar and calibration holds.")
    else:
        short = [n for n in boot if not boot[n]["improved"]]
        print(f"  calibration gap: baseline {base_gap:.4f} -> {hist_gap:.4f} "
              f"({'not widened' if not_widened else 'WIDENED'})")
        print(f"  VERDICT: NOT cleared. {len(short)} metric(s) do not: {', '.join(short)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())