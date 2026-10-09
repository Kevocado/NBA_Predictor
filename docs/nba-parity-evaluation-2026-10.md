# NBA Predictor — Phase A honest evaluation

**Date:** 2026-10-05 · **Branch:** `nba-parity-review` · **Spec:** `docs/superpowers/specs/2026-10-04-nba-parity-design.md`

This is the Phase A review gate. It reports what the models actually do on
out-of-fold data against naive baselines, including where they lose. Nothing
here was tuned to look better. Two of the three models **do not beat naive**,
and the served win model was **18 points overconfident** on its most confident
picks before this work.

## Method

- **1,368 games** from `data/cache/training/games.json` (2025-10-02 → 2026-10-04),
  24 features from `features/build.py`.
- **Expanding-window walk-forward**, windows cut on **date boundaries** (the NBA
  plays 10–12 games a night; a row-wise cut splits a date and leaks). Every
  window refits from scratch on its own training slice.
- **Naive baselines are leak-free**: the home-win base rate comes from each
  window's *training* data, never its test labels.
- Pooled figures concatenate every out-of-fold prediction (n=758).

## Headline: the win model is miscalibrated, and it was the model's fault

Kevin's read that the model is badly calibrated is confirmed, and quantified.

| model | log-loss | Brier | AUC | max calibration gap |
|---|---|---|---|---|
| **logistic (race winner)** | **0.6636** | **0.2351** | **0.6285** | **0.0592** |
| xgboost (was served) | 0.7059 | 0.2516 | 0.5912 | 0.1831 |
| naive — always home | 0.6864 | 0.2466 | 0.500 | n/a |
| coin flip | 0.6931 | 0.2500 | 0.500 | n/a |

**The incumbent XGBoost lost to both naive baselines** — worse than always
picking the home team. Its calibration was the worst part: bucketed into
quintiles of predicted probability, its most confident bucket predicted **0.80**
and actually won **0.618**. Eighteen points of overconfidence on exactly the
picks a visitor would act on. That is a real defect, not a rounding concern.

The race winner cuts that gap by two thirds (0.1831 → 0.0592) and is the first
win model to beat naive on log-loss, Brier and AUC at once.

**It still misses the ±0.05-per-bucket calibration bar** at 0.0592. Under the
NFL-style gate in spec section 10 this model would be **refused**. That is
reported, not worked around.

## The race reversed the core assumption (gap G2)

The three models were fixed XGBoost by assumption. Walk-forward says otherwise.

| target | winner | winner's score | rival | naive (mean) | naive (fixed) |
|---|---|---|---|---|---|
| win | **logistic** | **0.6636** | xgboost 0.7059 | 0.6864 | 0.6931 (coin-flip) |
| margin | **ridge** | **13.232** | xgboost 13.631 | 14.018 | 13.933 (−3) |
| total | **ridge** | **16.512** | xgboost 17.110 | 16.375 | 16.458 (league avg) |

**NFL precedent held.** The NFL repo's Ridge beat XGBoost at this sample size,
and so does a plain `LogisticRegression` here. A 200-tree depth-4 booster on 24
features and 1,368 games was overfitting noise, which is what gap G2 suspected.

## Honest losses

**1. Total MAE still loses to naive, on both baselines.** Ridge 16.512 against
16.375 (per-window mean) and 16.458 (league average). The margin and total
regressors are not earning their complexity. Two of three targets beat naive;
total does not. No gate was weakened to change this.

**2. The margin model's margin over naive is thin.** 13.232 vs 13.933 (−3) is a
real 0.7-point gain; 13.232 vs 14.018 (mean) is 0.8. Worth having, not a large
edge.

**3. Elo beats every ML candidate.** A plain rating loop scores **0.6037**
log-loss — better than the race winner's 0.6636. Spec section 3.1 designates
Elo reference-only and never served, so it is **not** in the serving path and
was not promoted. But "our model loses to arithmetic by six points of log-loss"
belongs in front of Kevin, and it is a question about whether a rating book
should serve the win market at all.

**4. Walk-forward costs more than the old split, and the numbers differ.**
Scored on the 80% chronological split the win log-loss reads 0.7712 with AUC
0.5496; on the full frame it reads 0.6636 with AUC 0.6285. Same model, same
method, different data. The pooled full-frame figure is the one quoted
throughout because it uses every game.

**5. Accuracy fell from 0.6354 to 0.5844 under honest evaluation.** The old
figure came from a single 80/20 `chronological_split`. Walk-forward accuracy on
out-of-fold data is 0.5844 — about 5 points lower for the same model. This is
the single most valuable number in the report: it is why the spec demanded
re-evaluation, and it means the previously published accuracy was flattering.

## Residual σ and coherent probabilities

One Normal residual σ per output, so win/cover/over cannot disagree —
`cover_prob(m, 0, σ) == win_prob(m, σ)` exactly, because covering a pick'em line
*is* winning. `cover_prob` returns `None` with no real line rather than
fabricating a placeholder (CFB's bug, one level up).

Fitted from out-of-fold residuals: **margin σ = 16.58, total σ = 20.69**, against
the spec's naive references of σ≈12 and σ≈15–16. Larger because the model's real
error is worse than naive; using the naive σ would make every probability read
more confident than the evidence supports.

## What changed underneath

**The retrain had never succeeded.** All 17 `refresh-data.yml` runs since
2026-09-18 failed. The workflow's rolling `60 days ago` window could not cover a
season, so in October `to_training_frame` returned **one** game and
`chronological_split` raised before any manifest was written. The commit step
sat downstream, so nothing was ever written — `manifest_history.jsonl` held four
byte-identical entries. Fixed with a season-anchored window, plus a write for
`data/cache/training/games.json`, which `POST /retrain` read and **no code in
`src/` ever produced**, making that endpoint unrunnable everywhere including the
VPS.

## Props: the in-sample numbers were flattering, and matchup features help

The prop models shipped with in-sample metrics only, which the manifest admits
(`training={"in_sample_metrics": True}`). Real numbers on **3,104 player-games /
435 players / 140 games** (ESPN keyless box scores, 2026-04-07 → 2026-06-13):

| market | in-sample | holdout | gap |
|---|---|---|---|
| points | 4.059 | 4.660 | +0.601 |
| rebounds | 1.706 | 1.819 | +0.113 |
| assists | 1.167 | 1.181 | +0.014 |
| threes | 0.793 | 0.909 | +0.116 |

**Every market degrades out-of-sample.** Points degrades most — 14.8% worse. So
the prop models' published MAEs were optimistic by up to 0.6 points, and the
site has been showing in-sample numbers.

Adding the four matchup features (`opp_def_vs_pos`, `rest_days`, `usage_trend`,
`minutes_trend`) — 5 features to 9, same holdout:

| market | baseline | + matchup | change |
|---|---|---|---|
| points | 4.636 | **4.332** | **−0.304** (6.6%) |
| rebounds | 1.812 | 1.776 | −0.036 |
| assists | 1.178 | 1.123 | −0.056 |
| threes | 0.907 | 0.880 | −0.027 |

All four improve. On points the new features more than close the
generalisation gap (holdout 4.660 → 4.332).

**Caveat, stated rather than buried:** 140 games is a small sample for a
per-market MAE, and these four features have not been through the serving path.
Treat the direction as evidence and the magnitude as provisional.

**A data finding worth keeping:** ESPN already sends `fg_made_attempted`, and the
schedule already knows home/away — but `to_player_training_frame` dropped both,
so `usage_trend` was uncomputable from real data and `opp_def_vs_pos` had nothing
to key on. Fixed additively.

## The same bug, four times

Four separate instances of one defect class, each caught by a test that asserts
a *date* boundary rather than a row boundary:

1. `walk_forward_eval` windows — the NBA plays 10–12 games a night, so a row-wise
   cut splits a date. This one refused to evaluate the real data at all.
2. `prop_holdout` — `chronological_split` cuts on rows; on the real player frame
   it produced `train_max == holdout_min == 2026-05-09`, one game on both sides.
3. `prop_matchup` `opp_def_vs_pos` — several players share one matchup on one
   night, so `shift(1)` alone left same-night rows inside the window.
4. The naive baselines — derived from test labels rather than training data.

This is worth a shared helper rather than four careful implementations. Not done
here; flagged as the highest-value follow-up, because the next window anyone
writes will have the same bug.

## Gates, honestly

| gate | result |
|---|---|
| win log-loss beats naive | **PASS** (0.6636 vs 0.6864) — first time |
| calibration ±5pt/bucket at n≥100 | **FAIL** (0.0592 vs 0.05) |
| margin MAE beats naive | **PASS** (13.232 vs 13.933) |
| total MAE beats naive | **FAIL** (16.512 vs 16.375) |
| margin/total MAE improved vs §5 baseline | **MARGINAL** — total still ≥ naive |
| manifest emits log_loss/brier/auc | **PASS** — G10 labels now live |
| prop holdout exists and reports both | **PASS** — was in-sample only |
| prop matchup features improve holdout MAE | **PASS** — all 4 markets |

**Two gates fail. Both are reported as failures.** No gate was relaxed to make
it pass, and the numbers above are reproducible from the committed code.

## Open questions for Kevin

1. **Serve Elo?** It outscores the model by 6 points of log-loss. Spec says
   reference-only; changing that is a spec amendment, not an implementation
   choice.
2. **The calibration gate at ±0.05 refuses the current winner.** Options: accept
   the refusal and ship nothing, loosen to ±0.06, or hold the model until Task 7
   features are in and re-race. My read is the third — the new game-context
   features are built but **not yet in the served model**, and they are the
   plausible route to a real calibration fix.
3. **Total MAE.** If it still loses to naive after features, the honest move is
   to serve the league-average total and say so.

## Reproducing

```
python3 -m pytest tests/ -q          # 689 passed, 21 skipped
```

Walk-forward: `src/nba_predictor/models/evaluate/walk_forward_eval.py`.
Race: `src/nba_predictor/models/candidate_race.py` — a candidate that raises
**fails the race** rather than being scored `inf` and losing silently, which is
what produced a fake "XGBoost wins" result in a rejected draft.
Probabilities: `src/nba_predictor/models/probability.py`.
---

# Task 4 candidates (2026-10-09)

Audit followup `2026-10-06-nba-audit-followup.md` Task 4: one PR per candidate,
each shipped only if walk-forward beats the current winner, and a candidate that
does not win is documented here rather than shipped. This section is that record.

## 4(d) — average Ridge and XGBoost margins instead of winner-take-all — **does not ship**

The race is winner-take-all (`_pick_winner` takes the single best per target).
This asks whether the plain mean of the two margin models does better than either
alone, on the identical held-out games.

Reproduced by `tests/test_margin_average_candidate.py`. The average is built the
way a real candidate would be: both regressors fitted inside the walk-forward on
the training slice only, then averaged. 4 windows, pooled n=758.

| arm | margin MAE |
|---|---|
| Ridge (race winner) | 12.3170 |
| XGBoost | 12.4391 |
| **average** | **12.2959** |

The average is **better on the point estimate by 0.0211**. That is not the ship
rule. Paired bootstrap, 2,000 resamples, seed 20261009, one draw applied to both:

| comparison | diff | 95% interval | verdict |
|---|---|---|---|
| average vs the best single model | −0.0211 | **[−0.0954, +0.0475]** | **NOT distinguishable** |

The interval crosses zero by a wide margin — its half-width is roughly five times
the effect. **4(d) does not ship.**

The direction is worth noting because it is the *expected* one: averaging two
regressors with decorrelated errors usually helps a little, and it did here on the
point estimate. It is also unmeasurable at n=758, which is the same finding as
every AUC question on this holdout. The honest conclusion is that this holdout
cannot resolve a 0.02 MAE move, not that averaging is worthless.

**Not relaxed.** The ship rule is "beats the current winner" and this does not
demonstrably beat it, so it is documented and left out.

## Still open

| item | status |
|---|---|
| 4(a) star/minutes-weighted availability | not started — `features/availability.py` does not exist |
| 4(b) rest/back-to-back/travel interactions | rest and back-to-back are live; **travel, congestion and fatigue are dead columns** — `_OPTIONAL_COLUMNS_DEFAULT_ZERO` fills them with 0.0 and nothing computes them |
| 4(c) calibration fitted out-of-fold | done (PR #52); the **probability ceiling near 0.85** is not implemented |
| 4(d) Ridge/XGBoost margin average | **evaluated above — does not ship** |
| 4(e) total model from pace × opponent-adjusted efficiency | not started — needs the pace and adjusted blocks, which the feature-pipeline plan builds |
