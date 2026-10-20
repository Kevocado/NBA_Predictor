# Availability B3 — player minutes and starting five

**Date:** 2026-10-20 · **Branch:** `task/nba-availability-b3` · **Scope:** offline only; nothing in the serving path imports it.

**Result: it does not beat the last-game baseline.** On 3,646 completed regular-season games (the three most recent completed seasons, 2023-24 through 2025-26), the EWMA projection is worse on minutes MAE and on both starting-five measures, and every paired-bootstrap interval excludes zero. The loss is small but it is not noise.

## What was built

`src/nba_predictor/features/lineups.py`:

* `project_team(team, team_games, latest_team=..., statuses=...)` — per player `expected_minutes` and `p_start`, EWMA (span 10) over the team's last ten games. A game the player missed is a zero; games before his first appearance for the team are not.
* `starting_five(projections)` — top five by `p_start`, then minutes, then player id. Players with availability 0 are excluded.
* `status_probability(status)` — the prior table from the plan: Out 0, Doubtful 0.10, Questionable 0.50, Probable 0.85, Day-To-Day 0.60. Anything else is unknown.
* `load_injury_statuses(paths)` — latest status per player from `data/injuries/*.jsonl.gz`.
* An Out player's minutes go to fully available teammates in proportion to their own projected minutes, capped at 40. An uncertain status scales that player's own minutes by the prior but frees nothing for teammates — the report says Out, not "probably out". An unknown status changes nobody and logs a warning.

`tools/evaluate_lineups.py` runs the paired evaluation. `tests/test_features_lineups.py` pins the arithmetic, the ten-game window, the latest-team rule, the prior table, redistribution, the cap, the unknown-status warning, the starter flag, and the injury-file reader.

## Evaluation

Data: `data/cache/training/games_all.json` and the `data/cache/espn/player_boxscore_*.json` caches, read only. Regular-season date ranges only; state resets at each season boundary.

| season | games evaluated | skipped |
|---|---|---|
| 2023-24 | 1,216 | 15 |
| 2024-25 | 1,215 | 16 |
| 2025-26 | 1,215 | 16 |

Skipped games are the first game of the season and games where either team had no prior game in the season (neither arm has a baseline). 3,646 games paired by `game_id`.

The model projects before the game from prior box scores only. The baseline is the team's previous game: its five starters and each player's minutes in that game. Minutes error is the mean absolute error over the union of projected, actual and previous players, averaged over the two teams. "Starters correct" is the intersection size with the actual five (0-5 per team). "Exact five" is the share of team-games where the set matches exactly. Paired bootstrap: 2,000 resamples, seed 20261020, one draw of game indices applied to both arms.

| metric | model | last game | difference | 95% CI | excludes zero |
|---|---|---|---|---|---|
| Minutes MAE (lower is better) | 6.6089 | 6.2018 | +0.4071 | [+0.3516, +0.4669] | yes |
| Starters correct, 0-5 per team (higher is better) | 4.1278 | 4.2607 | −0.1329 | [−0.1507, −0.1152] | yes |
| Exact five, share (higher is better) | 0.4019 | 0.4798 | −0.0779 | [−0.0890, −0.0668] | yes |

**Plainly: the model loses.** Concretely, it is off by 6.61 minutes per player per team-game against 6.20 for "same as last game"; it names 4.13 of the five starters against 4.26; and it gets the exact five 40.2% of the time against 48.0%. The last game is a very strong predictor of both minutes and starters, and a ten-game EWMA is slower to react. The model's intended advantage — removing players the injury report says are out and giving their minutes to replacements — is not in this evaluation, because the checkout has no historical injury feed (see below). This is the no-injury version, which is the weakest version.

## What could not be verified

* **No historical injury statuses.** There is no `data/injuries/*.jsonl.gz` in this checkout (the only injury file is `data/cache/injuries/current_injuries.json`, one current row). The prior table and the redistribution are unit-tested with synthetic statuses, but their effect on real games is unmeasured. The evaluation above passes no statuses.
* **Starters are derived, not a column.** The cached box scores have no starter flag. ESPN lists the five starters first, and `with_starter_flags` relies on that order. Checked against the live ESPN summary endpoint on 100 random 2023-2026 games: 2,181 matched player rows, zero order/starter mismatches; two cached rows were absent from the live response and were skipped. The comparison is paired on the same derived actual starters, so a wrong order would affect both arms; the absolute accuracy numbers would move.
* **Regular season only.** No preseason, play-in or playoffs. 2018-2022 was not evaluated.
* **No serving integration.** The module is not imported by the API, the ingest, or the props model. That is deliberate for this task.

## Reproduce

```bash
uv run --extra dev python tools/evaluate_lineups.py \
  --data-dir /Users/sigey/Documents/Projects.nosync/NBA_Predictor/data/cache
```

The caches are gitignored and are not committed.
