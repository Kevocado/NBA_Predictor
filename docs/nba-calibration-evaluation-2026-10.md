# Calibration step — fitted inside the walk-forward

**Date:** 2026-10-07 · **Branch:** `feat/calibration` · **Holdout:** the identical 758 Phase A out-of-fold games, 2026-01-10 → 2026-10-04

Ship rule: all five bootstrap intervals exclude zero **and** max calibration gap ≤ 0.0591 (ideally ≤ 0.05).

**Result: the calibration gap is fixed. AUC still does not resolve. Nothing ships.**

## Diagnosis first: the intercept does not drift, so there was nothing to fix there

The brief said to check whether the home-court intercept drifts before adding a calibrator. It does not — the training-window home-win rate is nearly constant while the *realised* rate wanders:

| window | test dates | n | mean predicted | realised | bias | training rate |
|---|---|---|---|---|---|---|
| 0 | 2026-01-10 … 02-11 | 250 | 0.5471 | 0.5400 | +0.0071 | 0.5488 |
| 1 | 2026-02-12 … 03-22 | 249 | 0.5374 | 0.5422 | −0.0048 | 0.5482 |
| 2 | 2026-03-23 … 04-27 | 204 | 0.5708 | 0.6127 | −0.0420 | 0.5478 |
| 3 | 2026-04-28 … 06-10 | 51 | 0.5691 | 0.5294 | +0.0397 | 0.5511 |
| 4 | 2026-06-13 … 10-04 | 4 | 0.5961 | 0.2500 | +0.3461 | 0.5508 |

The training rate sits at 0.548–0.551 in every window. The realised rate does not: 0.540, 0.542, **0.613**, 0.529. The bias swings from −0.042 to +0.040 with no trend, and the pooled bias is about −0.006 — essentially zero.

So there is no constant offset to remove. The league's home-win rate genuinely moved during this stretch, and a single intercept shift would have been fitted to noise. That is why the bucket-2 gap is a **shape** problem, not a level problem, and why a monotone recalibration is the right instrument.

## The leakage rule, and what it costs

A calibrator fitted on all 758 out-of-fold predictions and scored on those same 758 improves for a reason that has nothing to do with the model. So it is fitted **inside** the walk-forward: window *k*'s calibrator sees out-of-fold predictions from windows strictly before *k*, and nothing else.

**Windows 1 and 2 are left uncalibrated.** They have 0 and 1 earlier windows respectively, which is not enough to fit anything trustworthy. The default threshold is 2 earlier windows. Per-window record from the run:

| window | calibrated | games it was fitted on | note |
|---|---|---|---|
| 0 | no | 0 | not calibrated: 2 more earlier window(s) needed |
| 1 | no | 250 | not calibrated: 1 more earlier window(s) needed |
| 2 | yes | 499 | fitted on windows 0..1 |
| 3 | yes | 703 | fitted on windows 0..2 |
| 4 | yes | 754 | fitted on windows 0..3 |

**The cost, stated plainly: only 259 of the 758 games — 34% — are actually calibrated.** The other 499 are scored uncalibrated because there was no earlier out-of-fold data to fit on when they were predicted. Every pooled number below is therefore a mix, and `calibrated_from_window` is reported (2) precisely so no one reads the pool as uniformly calibrated.

This is a structural limit of a leak-free calibrator on a short holdout, not a tuning problem. It also caps how much any calibrator can help here.

## Reliability, 5 equal-count buckets, identical 758 games

| bucket | n | baseline gap | multi-season | **+ Platt** | **+ isotonic** |
|---|---|---|---|---|---|
| 0 | 152 | −0.0394 | −0.0370 | −0.0310 | −0.0014 |
| 1 | 151 | −0.0222 | −0.0007 | +0.0073 | −0.0039 |
| 2 | 152 | +0.0591 | +0.0720 | +0.0581 | +0.0503 |
| 3 | 151 | +0.0359 | −0.0230 | −0.0197 | +0.0029 |
| 4 | 152 | −0.0152 | +0.0187 | +0.0077 | −0.0288 |
| **max \|gap\|** | | **0.0591** | **0.0720** | **0.0581** | **0.0503** |

Both calibrators close the gap. Isotonic gets closest (0.0503, essentially at the "ideally ≤ 0.05" target). Bucket 2 remains the worst bucket for every arm — the models still over-predict home wins there, just by less.

## Paired bootstrap vs the Phase A baseline

2,000 resamples, seed 20261007, one draw of game indices applied to both models.

**multi-season + Platt** (max gap 0.0581)

| metric | baseline | this arm | diff | 95% interval | verdict |
|---|---|---|---|---|---|
| win log-loss | 0.6636 | 0.6542 | −0.0094 | [−0.0155, −0.0032] | **improved** |
| win Brier | 0.2351 | 0.2308 | −0.0043 | [−0.0072, −0.0013] | **improved** |
| win AUC | 0.6288 | 0.6447 | +0.0159 | [−0.0021, +0.0339] | not distinguishable |
| margin MAE | 13.2319 | 13.0638 | −0.1681 | [−0.2790, −0.0584] | **improved** |
| total MAE | 16.5115 | 16.1749 | −0.3366 | [−0.6170, −0.0622] | **improved** |

**multi-season + isotonic** (max gap 0.0503)

| metric | baseline | this arm | diff | 95% interval | verdict |
|---|---|---|---|---|---|
| win log-loss | 0.6636 | 0.6896 | **+0.0259** | [−0.0147, +0.1014] | not distinguishable (worse) |
| win Brier | 0.2351 | 0.2317 | −0.0033 | [−0.0069, +0.0002] | not distinguishable |
| win AUC | 0.6288 | 0.6395 | +0.0107 | [−0.0083, +0.0299] | not distinguishable |
| margin MAE | 13.2319 | 13.0638 | −0.1681 | [−0.2790, −0.0584] | **improved** |
| total MAE | 16.5115 | 16.1749 | −0.3366 | [−0.6170, −0.0622] | **improved** |

Margin and total are identical across the arms because calibration is applied to the win model only, as a probability recalibration. It does not touch a margin prediction.

## Ship rule

| arm | intervals excluding zero | max gap | verdict |
|---|---|---|---|
| multi-season | 4/5 (AUC missing) | 0.0720 | does not ship |
| **multi-season + Platt** | **4/5 (AUC missing)** | **0.0581** ✓ | **does not ship — AUC only** |
| multi-season + isotonic | 2/5 | 0.0503 ✓ | does not ship |

**Platt clears the calibration bar and keeps every accuracy metric it already had. Its one failure is AUC.**

## The two things worth knowing

**Platt is the one to keep.** It fixes the gap (0.0720 → 0.0581) at no measurable cost to log-loss or Brier, and its intervals are as tight as uncalibrated multi-season. Isonic gets a better gap (0.0503) and pays for it: log-loss goes to 0.6896, *worse than the 0.6636 baseline*, with an interval as wide as [−0.0147, +0.1014]. That is overfitting on 499 fitting games — isotonic's effective degrees of freedom grow as its steps get finer, and a test proves it collapses 400 distinct predictions to 17 values. The better-looking calibration number is bought with a worse model.

**AUC is not resolvable at n=758.** Every arm crosses zero: multi-season [−0.0023, +0.0342], +Platt [−0.0021, +0.0339], +isotonic [−0.0083, +0.0299]. The interval half-width is roughly twice the effect in each case. Two readings, and this is a decision for Kevin rather than for me:

1. **The bar is wrong for AUC.** A 0.016 AUC move may be real and simply under-powered at 758 games. Resolving it needs roughly 4× the holdout, which means waiting for the backfill cut to move or accepting a longer holdout with its own cost.
2. **The effect is not real.** AUC at 0.63–0.64 is close to nothing either way, and three independent calibrations all landing inside the same interval is consistent with "too weak to see", not "consistently present".

I would not resolve this by picking the arm with the better AUC point estimate, and I have not: isotonic has the smallest AUC change of the two and the worst log-loss, and choosing on AUC would mean choosing the worse model.

Note also that calibration cannot help AUC even in principle — a monotone recalibration preserves ranking. Platt's AUC is AUC-preserving to 1e-12; isotonic moves it only through the ties it creates, which is a tie artefact and never a better ranking.
