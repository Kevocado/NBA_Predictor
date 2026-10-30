# NBA prop ratio-feature search: pre-registration

**Date:** 2026-10-30 · **Branch:** `nba-prop-ratio-search` · **Written before any candidate was scored.**

This file fixes the procedure, the candidate pool, the split and the pass rule
before the search runs. Any deviation found later is reported as a deviation,
not folded into the result.

## Question

Does adding one ratio feature, built only from strictly prior games, improve
the production player-prop regressors on a chronological holdout, without
making any other prop market worse?

Markets: points, rebounds, assists, threes (`fg3m`), as in
`nba_predictor.models.prop_holdout.MARKET_TARGETS`.

## Data

Only the frozen file `data/history/nba_history.json.gz` (seasons 2018-19 to
2024-25). It holds player rows with `minutes, points, rebounds, assists, fg3m,
fga` and team-game totals (`fga, fg3a, fta, tov, oreb, dreb, pts, fg3m`).

Not available, so not used (a candidate that needs them is not in the pool):
player FTA, player TOV, player 3PA, player assists-to-turnover, team assists.
The 2025-26 season is not in the frozen file, so it is not in the evaluation.

Season label = calendar year of the season's October start (`2018` = 2018-19).
Oct-2020 bubble playoff games fall into label `2020`; this is a known, small
mislabel and is left as is.

## Split

| role | seasons | use |
|---|---|---|
| warm-up | label 2018, 2019 | training only |
| selection folds | labels 2020, 2021, 2022 | walk-forward: fold f trains on every label < f |
| **HOLDOUT** | labels 2023, 2024 | opened **once**, only for the selected set |

Rows: the production frame, `regular_and_post_only`-equivalent (`minutes > 0`),
built by `features.prop_model_frame.build_prop_model_frame`. Every feature is
shift(1)-causal, so the frame is built once over all data and split by season.
Splits are by season boundary, so no date sits on both sides.

## Baseline

The current production feature set: `features.prop_model_frame.PROP_FEATURE_COLUMNS`
(14 columns), fitted with `models.player_props.train_player_stat_model` (XGBoost,
200 trees, depth 4, lr 0.05). Candidate model = baseline + exactly one ratio column,
same hyperparameters, same rows.

## Candidate pool (fixed by rule, not by inspection)

Base causal quantities (17), each a rolling value over a window of the last 10
games, using shift(1) so only prior games count:

* Player, over his last 10 games (sum, min 3 games): `p_pts, p_reb, p_ast, p_fg3m, p_fga, p_min` (6)
* Team totals in the same games he logged (sum, same window): `t_fga, t_fg3a, t_fta, t_tov, t_reb, t_pts, t_poss`, where `t_poss = FGA - OREB + TOV + 0.44 FTA` (7)
* Opponent defence, over the opponent's last 10 games (mean, min 3), matched by (opponent, game): `o_pts_allowed, o_fga_allowed, o_fg3a_allowed, o_poss_allowed` (4)

Pool = all ordered pairs `(num, den)` of distinct base quantities: 17 x 16 = **272
candidates**. Ratios with a zero or missing denominator are NaN, never inf. Columns
numerically identical to an existing column are dropped and counted. The pool
therefore covers the requested families: FGA/min, FGA share of team FGA, 3PM/FGA,
pts/FGA, pts/min, rebounds/min, assists/min, rebound share, team-pace-adjusted
rates (per possession), team 3PA and FTA rates, opponent allowed rates, and
pairwise ratios of any two rolling stats.

## Selection (uses label 2020, 2021, 2022 only)

For every (candidate c, market m):

1. Per-row paired error difference `d_i = |e_base,i| - |e_c,i|` over the test rows of the three folds (positive = candidate better).
2. Stability: the mean `d` must be **positive in each of the three fold seasons**.
3. Test: one-sided paired t on the pooled `d_i`, p-value from the normal approximation.
4. Multiplicity: Holm correction across **all candidate x market tests** (272 x 4 = 1088 tests, minus any dropped duplicates; the exact count is reported).

Selected = passes stability AND Holm-adjusted p < 0.05. Rank by Holm-adjusted p.
Keep the top **K = 3** (candidate, market) pairs. If fewer than 3 pass, keep those.
If none pass, **HOLDOUT is not opened**, and the result is a selection-stage negative.

Known limit: rows in one game or one player are correlated, so the normal-approximation
p-value is optimistic. The holdout check below is the real test. This limit is stated,
not corrected for, which is why the holdout bar is strict.

## HOLDOUT (opened once, only for the selected K pairs)

Train on labels 2018 to 2022. Test on labels 2023 and 2024 pooled. For each
selected candidate c, all four markets are refit with baseline + c.

Per selected pair, with `K` selected tests (Bonferroni-style correction):

* **Target market** (the one selected for): paired bootstrap of MAE difference (candidate minus baseline, `models.evaluate.paired_bootstrap`, 2000 resamples, default seed), at `alpha = 0.05 / K`. Pass needs the interval to exclude zero **in the improving direction**.
* **Other markets** (with c added to them too): point-estimate holdout MAE must be **not higher** than baseline. Any other market with a higher holdout MAE fails the candidate.

A candidate passes only if both hold. The result is then judged on this one pass.

## Decision rule

* At least one candidate passes: add it as a feature behind a flag, default **OFF**, and state that in the PR. The PR still requires review before merge.
* None passes: the PR is docs-only and reports the negative result. No production code path changes.

Nothing here is tuned after seeing holdout numbers. If the selection stage has
a bug, the fix is reported as a deviation and the whole search is re-run from the
pre-registered rule.

## Outputs

* `docs/nba-prop-ratio-search-2026-10.json`: every candidate tried, its per-fold numbers, Holm-adjusted p, stability flag, selected set, and holdout results for the selected set.
* `docs/nba-prop-ratio-search-2026-10.md`: the human table.
* Code: `src/nba_predictor/features/prop_ratio.py` (candidate builder), `src/nba_predictor/models/evaluate/prop_ratio_search.py` (procedure), `scripts/prop_ratio_search.py` (runner).
