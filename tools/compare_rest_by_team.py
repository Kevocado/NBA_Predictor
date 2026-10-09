"""Does measuring rest from the team's last game (any role) beat the role-split default?

Adapted from compare_travel_fatigue.py. The baseline arm is the production default (rest from the last game in the SAME
role); the candidate arm passes rest_by_team=True. Same holdout, same rule: a metric counts only if its paired-bootstrap
interval excludes zero in the better direction (AUC: excludes a decline), and the calibration gap must not widen.
Nothing ships from a run of this tool: switching the default is its own reviewed PR with these numbers in it.

(Original header, for the shared machinery below:) Task 4(b): do travel, congestion and fatigue earn their place?

Audit followup `2026-10-06-nba-audit-followup.md` Task 4(b). `home_fatigue_index`
and `away_fatigue_index` were in `FEATURE_COLUMNS` and zero-filled by
`_OPTIONAL_COLUMNS_DEFAULT_ZERO`, so they existed and measured nothing. This
computes them, plus travel miles, timezone changes and the congestion flags, and
measures whether the extra columns help.

Same holdout as every other measurement in this line of work, and the same rule:
a metric counts only if its paired-bootstrap interval excludes zero in the better
direction (AUC: excludes a decline), and the calibration gap must not widen.

This is a **tool**, not a test. It reads `data/cache/training/games.json`, which is
gitignored, so a test reading it would pass locally and error in CI -- which is how
a green run stops meaning anything. `tools/compare_carryover.py` is the same shape.

Run:
    uv run --python 3.13 python tools/compare_rest_by_team.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_predictor.features.build import build_training_frame  # noqa: E402
from nba_predictor.models.candidate_race import default_candidates  # noqa: E402
from nba_predictor.models.evaluate.paired_bootstrap import paired_bootstrap  # noqa: E402
from nba_predictor.models.evaluate.walk_forward_eval import (  # noqa: E402
    walk_forward_metrics,
    walk_forward_regression,
)

PHASE_A_START = "2025-10-02"
PHASE_A_END = "2026-10-04"
N_BUCKETS = 5
N_RESAMPLES = 2000
SEED = 20261009
MAX_GAP_BAR = 0.0591
CACHE = Path("data/cache/training/games.json")

#: The columns this PR adds. Kept as a list so the doc and the tool agree.
NEW_COLUMNS = [
    "home_travel_miles", "away_travel_miles",
    "home_timezone_changes", "away_timezone_changes",
    "home_three_in_four", "away_three_in_four",
    "home_four_in_six", "away_four_in_six",
    "home_fatigue_index", "away_fatigue_index",
]


def _factories(cols, margin_targets):
    return {
        "win": lambda tr, c: default_candidates(cols)["logistic"]["win"](tr),
        "margin": lambda tr, c: default_candidates(cols)["ridge"]["margin"](tr),
        "total": lambda tr, c: default_candidates(cols)["ridge"]["total"](tr),
    }


def _run(cols, current, history, label):
    out_win = walk_forward_metrics(
        current.sort_values("game_date").reset_index(drop=True),
        model_factory=lambda tr: default_candidates(cols)["logistic"]["win"](tr),
        windows=4, history_df=history,
    )
    out_margin = walk_forward_regression(
        current.sort_values("game_date").reset_index(drop=True),
        model_factory=lambda tr: default_candidates(cols)["ridge"]["margin"](tr),
        target="home_margin", windows=4, history_df=history,
    )
    out_total = walk_forward_regression(
        current.sort_values("game_date").reset_index(drop=True),
        model_factory=lambda tr: default_candidates(cols)["ridge"]["total"](tr),
        target="home_total", windows=4, fixed_baseline=230.174, history_df=history,
    )
    return {
        "label": label, "n": out_win["pooled"]["n"],
        "win": out_win["pooled"], "margin": out_margin["pooled"], "total": out_total["pooled"],
    }


def main() -> int:
    if not CACHE.exists():
        print(f"no training cache at {CACHE}; this is a local measurement, not a test")
        return 1

    from nba_predictor.models.artifact_gate import calibration_gap

    raw = pd.DataFrame(json.loads(CACHE.read_text()))
    current = raw[(raw.game_date >= PHASE_A_START) & (raw.game_date <= PHASE_A_END)].copy()
    history = raw[raw.game_date < PHASE_A_START].copy()

    # Build each arm from the CHRONOLOGICAL history + holdout together and split the Phase A rows out AFTER feature
    # construction. Building from the holdout alone started every team's game history at PHASE_A_START (a team's first
    # holdout game read 99 rest days) and dropped the first games without rolling factors; and the pre-holdout rows
    # are passed to the walk-forward evaluators as history_df so each model trains on everything before its window.
    full = pd.concat([history, current], ignore_index=True).sort_values("game_date").reset_index(drop=True)
    arms = {}
    for label, candidate in (("baseline", False), ("+ travel/fatigue", True)):
        frame_all, cols = build_training_frame(full, rest_by_team=candidate)
        frame_all = frame_all.assign(home_margin=frame_all.home_pts - frame_all.away_pts,
                                     home_total=frame_all.home_pts + frame_all.away_pts)
        in_holdout = (frame_all.game_date >= PHASE_A_START) & (frame_all.game_date <= PHASE_A_END)
        arms[label] = {"frame": frame_all[in_holdout].reset_index(drop=True),
                       "history": frame_all[frame_all.game_date < PHASE_A_START].reset_index(drop=True), "cols": cols}

    rows = {}
    for label, arm in arms.items():
        frame = arm["frame"]
        rows[label] = _run(arm["cols"], frame, arm["history"], label)

    a, b = rows["baseline"], rows["+ travel/fatigue"]
    print(f"\nframe: holdout {len(current)} games, pooled out-of-fold n={a['n']}")
    print("candidate: rest_by_team=True (no columns added, the two rest columns and back-to-back flags change)")

    print(f"\n=== paired bootstrap: + travel/fatigue vs baseline, "
          f"{N_RESAMPLES} resamples, seed {SEED}")

    win_y = np.asarray(a["win"]["y"])
    margin_y = np.asarray(a["margin"]["y"])
    total_y = np.asarray(a["total"]["y"])

    metrics = [
        ("win log_loss", win_y, np.asarray(a["win"]["preds"]), np.asarray(b["win"]["preds"]), "log_loss"),
        ("win brier", win_y, np.asarray(a["win"]["preds"]), np.asarray(b["win"]["preds"]), "brier"),
        ("win auc", win_y, np.asarray(a["win"]["preds"]), np.asarray(b["win"]["preds"]), "auc"),
        ("margin mae", margin_y, np.asarray(a["margin"]["preds"]), np.asarray(b["margin"]["preds"]), "mae"),
        ("total mae", total_y, np.asarray(a["total"]["preds"]), np.asarray(b["total"]["preds"]), "mae"),
    ]
    boot = {}
    for name, y, base_p, cand_p, kind in metrics:
        boot[name] = paired_bootstrap(y, base_p, cand_p, kind,
                                      n_resamples=N_RESAMPLES, seed=SEED)

    print(f"  {'metric':<14}{'baseline':>10}{'+travel':>10}{'diff':>10}{'95% interval':>22}   verdict")
    n_ok = 0
    n_worse = 0
    for name, out in boot.items():
        ci = f"[{out['ci_low']:+.4f}, {out['ci_high']:+.4f}]"
        crosses_zero = out["ci_low"] <= 0 <= out["ci_high"]
        if out["improved"]:
            verdict, n_ok = "IMPROVED", n_ok + 1
        elif not crosses_zero:
            # The interval excludes zero, but in the WRONG direction. This is a
            # finding, not noise, and "NOT distinguishable" would hide it.
            verdict, n_worse = "WORSE (interval excludes zero)", n_worse + 1
        else:
            verdict = "NOT distinguishable"
        print(f"  {name:<14}{out['baseline']:>10.4f}{out['candidate']:>10.4f}"
              f"{out['difference']:>+10.4f}{ci:>22}   {verdict}")
    print(f"  {n_ok}/5 improved, {n_worse}/5 measurably worse")

    gaps = {}
    for label in ("baseline", "+ travel/fatigue"):
        y, p = np.asarray(rows[label]["win"]["y"]), np.asarray(rows[label]["win"]["preds"])
        gaps[label] = calibration_gap(y, p, n_buckets=N_BUCKETS)
        print(f"\n  {label}: max |gap| = {gaps[label]:.4f}")

    from nba_predictor.models.evaluate.ship_rule import MAX_GAP_BAR, ship_decision

    gap = gaps["+ travel/fatigue"]
    decision = ship_decision(boot, gap=gap, baseline_gap=gaps["baseline"])
    print(f"\n=== SHIP RULE: AUC interval excludes a decline; the other four metrics each improve; "
          f"max gap <= {MAX_GAP_BAR} and not wider than the baseline's {gaps['baseline']:.4f}")
    print(f"  {'SHIPS' if decision['ships'] else 'does NOT ship'}")
    print(f"      - AUC non-decline: {'yes' if decision['auc_ok'] else 'NO (interval includes a decline)'}")
    if decision["missing"]:
        print(f"      - not improved: {', '.join(decision['missing'])}")
    if not decision["gap_ok"]:
        print(f"      - calibration gap {gap:.4f} (bar {MAX_GAP_BAR}, baseline {gaps['baseline']:.4f})")
    if n_worse:
        print(f"      - {n_worse}/5 metrics are measurably WORSE, not noise")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
