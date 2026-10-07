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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_predictor.features.build import FEATURE_COLUMNS, build_training_frame  # noqa: E402
from nba_predictor.models.candidate_race import default_candidates  # noqa: E402
from nba_predictor.models.evaluate.walk_forward_eval import (  # noqa: E402
    walk_forward_metrics,
    walk_forward_regression,
)

# Phase A's frame, as recorded in docs/nba-parity-evaluation-2026-10.md:
# "1,368 games from data/cache/training/games.json (2025-10-02 -> 2026-10-04)".
PHASE_A_START = "2025-10-02"
PHASE_A_END = "2026-10-04"

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())