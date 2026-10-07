# Multi-season backfill — evaluation

**Date:** 2026-10-07 · **Branch:** `feat/multi-season-backfill` · **Supersedes:** nothing; first version of this file

The plan's gate for this work: *"score the new model only on the 758 Phase A
out-of-fold games (same dates, same windows) … If the new model does not beat
Phase A on the identical games, say so and do not ship it."*

**The gate is closed.** Multi-season history beats Phase A on the identical 758
games on all five metrics.

## What was fetched

3 prior seasons plus the current one, through the ESPN endpoints already in
use (`get_scoreboard`, `get_boxscore`). 4,248 cache files, ~40 min, plus a
second pass for the playoffs (see *The correction* below).

| season | range | games | with box score |
|---|---|---|---|
| 2023 | 2023-09-25 … 2024-06-30 | 1,385 | 1,383 |
| 2024 | 2024-09-25 … 2025-06-30 | 1,396 | 1,385 |
| 2025 | 2025-09-25 … 2026-06-30 | 1,391 | 1,387 |
| 2026 | to 2026-10-07 | 17 | 12 |

**4,167 training rows**, 2023-10-05 → 2026-10-06, including **135 playoff
games**. Phase A trained on 1,368. Coverage is 99.6%, so ESPN really does hold
five seasons — no new endpoint and no new client were needed.

## The reproduction check, first

Before comparing anything, the baseline is re-run with no history. If it does
not reproduce Phase A, the comparison is meaningless and no amount of
improvement downstream would be evidence of anything.

| metric | Phase A (published) | reproduced here | Δ |
|---|---|---|---|
| pooled n | 758 | **758** | 0 |
| win log-loss | 0.6636 | **0.6636** | 0.0000 |
| win Brier | 0.2351 | **0.2351** | 0.0000 |
| win AUC | 0.6285 | 0.6288 | +0.0003 |
| margin MAE | 13.232 | 13.2319 | 0.0001 |
| total MAE | 16.512 | 16.5115 | 0.0005 |
| margin naive (train mean) | 14.018 | 14.0181 | 0.0001 |

Same 1,368-game frame, same 758 pooled out-of-fold games. **This is the exact
comparison the plan asks for**, and it is only exact because the reproduction
came first.

One column does not match: Phase A's total `naive_fixed` was 16.375, and
against this frame's own mean total (230.174) it is 16.2256. Their fixed
constant is not recorded in the doc, so I cannot say what it was. It is a
baseline, not a deciding metric, and it does not affect any verdict below.

## The comparison

Identical test games: `n_test` per window is `[250, 249, 204, 51, 4]` in *both*
runs — the mechanism's own proof that history widened training and did not move
the holdout. Pooled n=758.

| metric | Phase A | + 3 seasons | change |
|---|---|---|---|
| win log-loss | 0.6636 | **0.6542** | −0.0094 |
| win Brier | 0.2351 | **0.2309** | −0.0042 |
| win AUC | 0.6285 | **0.6448** | +0.0163 |
| margin MAE | 13.232 | **13.0638** | −0.168 |
| total MAE | 16.512 | **16.1749** | −0.337 |

Training grew from 610/860/1109/1313/1364 to 3360/3610/3859/4063/4114.

**All five improve.** The AUC move (+0.0163) is the one worth watching: Phase A
reported the win model as barely better than a coin flip at 0.6285, and 0.6448
is still modest, but it is a real improvement on a model whose log-loss already
beat the naive base rate.

## The correction, because the first version of this file said the opposite

The first run of this backfill fetched with `SEASON_END = 25 April`, and the
playoffs end in June. So the frame held **zero** playoff games, and on that frame:

- `phase_a_only` scored 0.6716, not 0.6636;
- pooled n was **808**, not 758;
- and against Phase A's published numbers, win log-loss, Brier and AUC all came
  out *worse*, so the conclusion was "history does not ship".

Every one of those numbers was wrong, and the reason shows up as soon as you
run the reproduction check instead of trusting the comparison. A baseline that
does not reproduce is not a baseline. The missing 135 playoff games were worth
0.008 of log-loss on their own — enough to account for the whole apparent
regression, and enough to flip the result once they were fetched.

CodeRabbit's *"extend the end bound if the cache must include the whole
playoffs"* is therefore not a style note. It was worth more than the entire
result it appeared alongside, and the only reason I caught it was that the
reproduction disagreed with the published doc.

## Still open

- **Season carry-over is not implemented.** `add_rolling_four_factors` does
  `shift(1).rolling(10)` per team across the whole frame, so week 1 of a season
  currently averages the *previous* season's last 10 games across a six-month
  gap. That is a real weakness in the very features the extra data feeds. It is
  deliberately built after this measurement, not before: carry-over is a
  refinement, and it should not be built until the base data is known to pay.
  On this evidence it does, so carry-over is the next thing to build, and it
  must be scored on the same 758 games before it ships too.
- Nothing ships *from this PR* — it is the fetch tool, the comparison harness
  and this document. The model change lands separately.

## Reproducing

    uv run --python 3.13 -m nba_predictor.pipeline.backfill --seasons 3
    uv run --python 3.13 python tools/compare_phase_a.py
