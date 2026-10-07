# Multi-season backfill — evaluation

**Date:** 2026-10-07 · **Branch:** `feat/multi-season-backfill`

The plan's gate for this work: *"score the new model only on the 758 Phase A
out-of-fold games (same dates, same windows) … If the new model does not beat
Phase A on the identical games, say so and do not ship it."*

**That gate is not yet answerable, and the reason is below.** What *is* measured
is a clean A/B on an identical frame, and multi-season history wins every
metric on it.

## What was fetched

3 prior seasons plus the current one, through the ESPN endpoints already in
use (`get_scoreboard`, `get_boxscore`), 4,248 cache files, ~40 min.

| season | games | with box score |
|---|---|---|
| 2023 (Oct 23 – Apr 24) | 1,332 | 1,328 |
| 2024 (Oct 24 – Apr 25) | ~1,330 | ~1,327 |
| 2025 (Oct 25 – Apr 26) | ~1,330 | ~1,327 |
| 2026 (to 2026-10-07) | 17 | 12 |

**3,983 training rows**, 2023-10-05 → 2026-10-06. Phase A trained on 1,368.

Coverage is 99.6%, so ESPN really does hold the history — no new endpoint and
no new client were needed. A failed season would be named in `failed_seasons`
rather than abandoning the run; the first attempt did exactly that, dying on a
DNS failure after a full season and losing the lot until that was fixed.

## Table 1 — the valid comparison

Same frame, same windows, **identical test games**. `n_test` per window is
`[214, 214, 209, 169, 2]` in *both* runs, which is the mechanism's proof that
history widens training and does not move the holdout. Pooled n=808.

| metric | single-season | + 3 seasons | change |
|---|---|---|---|
| win log-loss | 0.6716 | **0.6644** | −0.0072 |
| win Brier | 0.2392 | **0.2359** | −0.0033 |
| win AUC | 0.5966 | **0.6132** | +0.0166 |
| margin MAE | 13.1290 | **12.9709** | −0.158 |
| total MAE | 16.2296 | **15.9203** | −0.309 |

Training grew from 501/715/929/1138/1307 to 3126/3340/3554/3763/3932.

**History wins all five**, including the two where Phase A already beat naive.
That is a real result and it is the reason to keep going — but it is not yet
the gate.

## Table 2 — against Phase A's published numbers (NOT a valid comparison)

Shown because it is the number that would decide a ship, and because hiding it
would be the dishonest choice.

| metric | Phase A (published) | + 3 seasons | direction |
|---|---|---|---|
| win log-loss | 0.6636 | 0.6644 | worse by 0.0008 |
| win Brier | 0.2351 | 0.2359 | worse by 0.0008 |
| win AUC | 0.6285 | 0.6132 | worse by 0.0153 |
| margin MAE | 13.232 | 12.9709 | better by 0.26 |
| total MAE | 16.512 | 15.9203 | better by 0.59 |

**These two columns are not comparable and must not be read as a result.** The
frames differ:

| | Phase A | this run |
|---|---|---|
| games in the Phase A window | 1,368 | **1,331** |
| pooled out-of-fold n | 758 | **808** |

Phase A's cache was built by a fetch on 2026-10-05 and is not in the repo; this
backfill rebuilt the window from ESPN on 2026-10-07, and ESPN now answers
differently for 37 of those dates (a game whose box score was unavailable on the
5th has one now, and the window's date cuts land elsewhere). Different games, so
different windows, so a different 808.

## What is needed to close the gate

One of:

1. **Phase A's original `data/cache/training/games.json`**, if it still exists
   anywhere — restore it, score the multi-season model on its 758, and the
   comparison is exact. This is the cheap path.
2. Otherwise re-run Phase A's pipeline on *its* frame to regenerate the 758,
   which means pinning the fetch to 2026-10-05's answer for every date rather
   than today's.

Until one of those happens, the honest statement is: **multi-season history
helps a multi-season model on identical held-out games, and it has not been
shown to beat the 758-game Phase A baseline.**

## Known gaps, not glossed

- **Season carry-over is not implemented.** `add_rolling_four_factors` does
  `shift(1).rolling(10)` per team across the whole frame, so week 1 of a season
  currently averages the *previous* season's last 10 games across a six-month
  gap. That is a real weakness in the very features the extra data feeds, and it
  is the next thing to build — deliberately after this measurement, because
  carry-over is a refinement and it should not be built before knowing whether
  the extra data pays at all. On this evidence it does.
- `SEASON_START` was 20 October and was wrong — NBA seasons open in the first
  week of October, so it dropped ~3 weeks per season and the entire current
  season. Fixed to 25 September, which is why the frame above reaches
  2023-10-05 and 2026-10-06.

## Reproducing

    uv run --python 3.13 -m nba_predictor.pipeline.backfill --seasons 3
    uv run --python 3.13 python tools/compare_phase_a.py