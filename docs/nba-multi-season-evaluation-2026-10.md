# Multi-season backfill — evaluation

**Date:** 2026-10-07 · **Branch:** `feat/multi-season-backfill` · **Supersedes:** nothing; first version of this file

The plan's gate for this work: *"score the new model only on the 758 Phase A
out-of-fold games (same dates, same windows) … If the new model does not beat
Phase A on the identical games, say so and do not ship it."*

**The comparison is run on exactly those 758 games. The model does not ship:**
four of five metrics improve with intervals excluding zero, the fifth is not
distinguishable from zero, and the calibration gap widens.

An earlier version of this document said all five improved. That was true of the
point estimates and false of the evidence, and it is corrected below.

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
| win AUC | 0.6285 | 0.6448 | +0.0160 |
| margin MAE | 13.232 | **13.0638** | −0.168 |
| total MAE | 16.512 | **16.1749** | −0.337 |

Training grew from 610/860/1109/1313/1364 to 3360/3610/3859/4063/4114.

### Do the point estimates survive their own uncertainty?

A point estimate moving from 0.6636 to 0.6542 says little on its own: on 758
games, differences that size happen by chance routinely. Paired bootstrap, 2,000
resamples, seed 20261007, one draw of game indices applied to **both** models so
the game-to-game difficulty cancels. A metric counts as improved only if its 95%
interval excludes zero.

| metric | baseline | + 3 seasons | difference | 95% interval | verdict |
|---|---|---|---|---|---|
| win log-loss | 0.6636 | 0.6542 | −0.0094 | [−0.0154, −0.0032] | **improved** |
| win Brier | 0.2351 | 0.2309 | −0.0042 | [−0.0069, −0.0013] | **improved** |
| win AUC | 0.6288 | 0.6448 | +0.0160 | [−0.0023, **+0.0342**] | **not distinguishable from 0** |
| margin MAE | 13.2319 | 13.0638 | −0.1681 | [−0.2790, −0.0584] | **improved** |
| total MAE | 16.5115 | 16.1749 | −0.3366 | [−0.6170, −0.0622] | **improved** |

**Four of five clear the bar. The AUC does not.** Its interval crosses zero, so
the +0.016 is a real-looking number that 758 games cannot resolve.

### Reliability

5 equal-count quantile buckets on the same 758 games (equal-*count* because NBA
win probabilities cluster near 0.56 and equal-width would put nearly everything
in one bucket).

| bucket | n | baseline predicted | baseline observed | baseline gap | multi-season predicted | multi-season observed | multi-season gap |
|---|---|---|---|---|---|---|---|
| 0 | 152 | 0.4210 | 0.3816 | −0.0394 | 0.3989 | 0.3618 | −0.0370 |
| 1 | 151 | 0.5057 | 0.4834 | −0.0222 | 0.4974 | 0.4967 | −0.0007 |
| 2 | 152 | 0.5528 | 0.6118 | +0.0591 | 0.5530 | 0.6250 | **+0.0720** |
| 3 | 151 | 0.5999 | 0.6358 | +0.0359 | 0.6058 | 0.5828 | −0.0230 |
| 4 | 152 | 0.6928 | 0.6776 | −0.0152 | 0.7049 | 0.7237 | +0.0187 |

**max |gap|: 0.0591 → 0.0720. The calibration gap WIDENS by 0.0129**, and it
widens in the same bucket that was already the worst — bucket 2, where the model
over-predicts home wins by 6 points and now by 7.2.

Both models are over-confident in the middle of the range and that is the single
worst bucket for each. More history sharpened the middle of the probability
range without fixing the bias there, so the extra data bought accuracy and paid
for it in calibration.

## Verdict: not cleared

The bar, as stated before the measurement: all five metrics improved with 95%
intervals excluding zero, **and** the calibration gap not widened.

| requirement | result |
|---|---|
| win log-loss, interval excludes zero | met (−0.0094, [−0.0154, −0.0032]) |
| win Brier, interval excludes zero | met (−0.0042, [−0.0069, −0.0013]) |
| win AUC, interval excludes zero | **not met** (+0.0160, [−0.0023, +0.0342]) |
| margin MAE, interval excludes zero | met (−0.168, [−0.279, −0.058]) |
| total MAE, interval excludes zero | met (−0.337, [−0.617, −0.062]) |
| calibration gap not widened | **not met** (0.0591 → 0.0720) |

**Multi-season history does not ship on this evidence.** Four metrics are
genuinely better and the intervals say so; the fifth is not distinguishable
from zero, and the calibration got measurably worse.

### What an earlier version of this document claimed, and why it was wrong

The first version of the comparison table said **"all five improve"** and
singled out AUC (+0.0163) as *"the one worth watching"*. Point estimates alone
support that sentence, and it was still wrong twice over:

- **AUC's improvement is inside its own confidence interval.** Calling a
  +0.016 change "worth watching" implied more than 758 games can support.
- **Calibration was not examined at all**, and it widens. A version of this
  document that shipped a model on four intervals and an unexamined gap would
  have traded accuracy for over-confidence without noticing.

The numbers themselves did not change between the two versions. Only the
uncertainty and the reliability table were added, and they are what changed the
answer. A point estimate is a claim; an interval and a gap are the evidence for
it.

### What would clear the bar

In rough order of expected value, and none of them is "fetch more seasons":

1. **Fix the bucket-2 bias directly.** Both models over-predict home wins by
   ~7 points in the middle of the range, and it is the worst bucket for both.
   A calibration step fitted on out-of-fold predictions (Platt or isotonic) is
   the standard remedy and should be evaluated on the same 758 before anything
   else is tried. This is the honest lever, because the gap is a known,
   reproducible defect rather than something to wait out.
2. **More data will not fix it.** The gap widened *with* three extra seasons, so
   the trend is against it.
3. AUC needs a larger held-out set or a genuinely stronger ranking signal to
   resolve. It is not currently evidence of anything either way.

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
  The base data does pay on four metrics, so carry-over is worth building — and
  it must clear the same bar on the same 758, gap included.
- Nothing ships *from this PR* — it is the fetch tool, the comparison harness,
  the bootstrap and this document. The model change lands separately, and on
  this evidence it should not land at all until the calibration gap is closed.

## Method notes

**Why paired.** Both models are scored on the same 758 games, so each resample
draws game indices once and applies them to both. An unpaired bootstrap would
charge the difference for game-to-game difficulty that both models equally
experience, inflating the interval and making a real improvement look like noise.
`test_pairing_narrows_the_interval_because_game_difficulty_cancels` asserts the
paired interval is strictly narrower than an unpaired one — if pairing were
dropped, that test fails.

**Why 2,000 resamples and a fixed seed.** The seed is reported in every result so
a figure can be reproduced rather than taken on trust. A different seed will move
the interval endpoints slightly; the verdicts above are not near the boundary.

**What "improved" means.** `improved` is True only when the interval excludes
zero *in the direction that counts as better* — lower for log-loss, Brier and
MAE, higher for AUC. The sign of a point estimate is not enough, and the
difference matters: this file contains a case where a genuinely better log-loss
of −0.0070 against a bootstrap spread of 0.0066 is correctly reported as **not
improved**
(`test_a_better_looking_model_whose_gap_is_smaller_than_the_noise_is_not_an_improvement`).
Without that rule, the −0.0094 log-loss win here would have looked like a much
smaller thing than it is, and a −0.0001 change would have looked like a win.

**Degenerate resamples are dropped, not scored.** A bootstrap draw that happens
to contain one class has no defined AUC. Scoring it 0.5 would drag the interval
toward "no difference" for a reason that has nothing to do with either model, so
those draws are discarded and `n_resamples_used` is reported.

## Reproducing

    uv run --python 3.13 -m nba_predictor.pipeline.backfill --seasons 3
    uv run --python 3.13 python tools/compare_phase_a.py
