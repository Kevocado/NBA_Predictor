"""Run the pre-registered ratio-feature search (docs/nba-prop-ratio-preregistration-2026-10.md).

Usage (from the repo root):  PYTHONPATH=src python scripts/prop_ratio_search.py

Writes docs/nba-prop-ratio-search-2026-10.json (every candidate, selection stage first, then holdout if
the pre-registered gate opened it) and docs/nba-prop-ratio-search-2026-10.md (the table).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from nba_predictor.features.prop_model_frame import PROP_FEATURE_COLUMNS, build_prop_model_frame
from nba_predictor.features.prop_ratio import add_base_quantities, ratio_candidates, team_game_table
from nba_predictor.models.evaluate import prop_ratio_search as rs
from nba_predictor.pipeline.history import PLAYER_COLS, load_history

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data" / "history" / "nba_history.json.gz"
OUT_JSON = ROOT / "docs" / "nba-prop-ratio-search-2026-10.json"
OUT_MD = ROOT / "docs" / "nba-prop-ratio-search-2026-10.md"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write(result: dict) -> None:
    OUT_JSON.write_text(json.dumps(result, indent=2, default=float) + "\n")


def _md(result: dict) -> str:
    lines = ["# NBA prop ratio-feature search: results", "",
             "Pre-registered in `docs/nba-prop-ratio-preregistration-2026-10.md`. Raw numbers: `docs/nba-prop-ratio-search-2026-10.json`.", "",
             f"Candidates generated: **{result['pool']['generated']}**, dropped as duplicates: **{result['pool']['dropped_duplicates']}**, "
             f"tested: **{result['pool']['tested']}** x {len(rs.MARKET_TARGETS)} markets = **{result['pool']['tests']}** selection tests (Holm family).", "",
             f"Selection tests passing stability and Holm < {rs.ALPHA}: **{sum(r['selected_pool'] for r in result['selection'])}**.", ""]
    if not result["holdout_opened"]:
        lines += ["**HOLDOUT NOT OPENED.** No candidate passed selection, so the pre-registered rule did not open it.", ""]
    else:
        lines += ["## Holdout (labels 2023 and 2024, opened once)", "",
                  "| candidate | target | target MAE base -> cand | diff | CI (alpha/K) | other markets not worse | passes |",
                  "|---|---|---|---|---|---|---|"]
        for h in result["holdout"]:
            t = h["markets"][h["target_market"]]
            lines.append(f"| `{h['candidate']}` | {h['target_market']} | {t['baseline_mae']:.4f} -> {t['candidate_mae']:.4f} | "
                         f"{t['difference_cand_minus_base']:+.4f} | [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] | "
                         f"{h['other_markets_not_worse']} | {h['passes']} |")
        lines.append("")
        lines.append(f"Any candidate passing: **{any(h['passes'] for h in result['holdout'])}**.")
    lines += ["", "## Selection, top candidates by Holm-adjusted p", "",
              "| candidate | market | mean MAE gain (selection) | per-fold gains 2020/2021/2022 | p raw | p Holm | stable | selected |",
              "|---|---|---|---|---|---|---|---|"]
    for r in sorted(result["selection"], key=lambda r: r["p_holm"])[:25]:
        f = r["per_fold_improvement_mae"]
        lines.append(f"| `{r['candidate']}` | {r['market']} | {r['mean_improvement_mae']:+.4f} | "
                     f"{f['2020']:+.4f} / {f['2021']:+.4f} / {f['2022']:+.4f} | {r['p_raw']:.3g} | {r['p_holm']:.3g} | "
                     f"{r['stable_all_folds']} | {r['selected_pool']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    games, players = load_history("2100-01-01", HISTORY)
    players = players[players["minutes"].fillna(0) > 0].reset_index(drop=True)
    _log(f"history: {len(games)} games, {len(players)} player rows with minutes > 0")

    frame, _ = build_prop_model_frame(players)
    base = add_base_quantities(frame, team_game_table(games))
    cands_all = ratio_candidates(base)
    cands, dropped = rs.dedupe_candidates(cands_all, base)
    _log(f"candidates generated {len(cands_all)}, dropped duplicates {len(dropped)}, tested {len(cands)}")

    label = rs.season_label(base["game_date"])
    split_rows = {str(int(k)): int(v) for k, v in label.value_counts().sort_index().items()}

    result = {
        "preregistration": "docs/nba-prop-ratio-preregistration-2026-10.md",
        "data": {"file": str(HISTORY.relative_to(ROOT)), "games": len(games), "player_rows_minutes_gt_0": len(players),
                 "model_rows": len(base), "rows_by_season_label": split_rows},
        "production_features": PROP_FEATURE_COLUMNS,
        "pool": {"generated": len(cands_all), "dropped_duplicates": len(dropped), "dropped_names": dropped,
                 "tested": len(cands), "tests": len(cands) * len(rs.MARKET_TARGETS)},
        "candidates": sorted(cands),
        "selection_folds": list(rs.SELECTION_FOLDS), "holdout_labels": list(rs.HOLDOUT_LABELS), "top_k": rs.TOP_K,
        "selection": [], "selected": [], "holdout_opened": False, "holdout": [],
    }

    t0 = time.time()
    records = rs.selection_stage(base, cands, log=_log)
    result["selection"] = records
    result["selected"] = rs.pick_top(records)
    result["selection_seconds"] = round(time.time() - t0, 1)
    _write(result)
    _log(f"selection done in {result['selection_seconds']} s; selected {len(result['selected'])}")

    if result["selected"]:
        result["holdout_opened"] = True
        result["holdout"] = rs.holdout_stage(base, cands, result["selected"], log=_log)
    else:
        _log("no candidate passed selection; holdout not opened")
    _write(result)
    OUT_MD.write_text(_md(result))
    _log("wrote " + str(OUT_JSON) + " and " + str(OUT_MD))
    return 0


if __name__ == "__main__":
    sys.exit(main())
