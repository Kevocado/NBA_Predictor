# NBA TabFM benchmark (measure only)

Zero-shot TabFM (Google) vs. the current NBA model. Same frozen history
(2019-20 through 2024-25), same 28 features, same 4 expanding walk-forward
windows, no peeking at scored games. `n_estimators=1`, CPU, bfloat16.

## Context policy (selected on window 0)

| Policy | Window-0 log-loss |
|---|---|
| default (random 100) | 0.6595 |
| recent 100 | 0.6926 |
| recent 500 | 0.6665 |
| **recent 2000** | **0.6441** |
| recency-weighted 2000 | 0.6484 |

Selected policy: **most recent 2000 rows** (`recent_2000`).

## Home-win probability (pooled, n=4564)

| Model | log-loss | Brier | AUC | ECE (10-bin) |
|---|---|---|---|---|
| Current (margin->Phi) | 0.6224 | 0.2166 | 0.7027 | 0.0111 |
| TabFM (default) | 0.6353 | 0.2217 | 0.6880 | 0.0202 |
| TabFM + Platt | 0.8819 | 0.2565 | 0.6750 | 0.1597 |
| TabFM + isotonic | 2.1164 | 0.2510 | 0.6795 | 0.1361 |

TabFM loses to the current model; its Platt/isotonic layer (fit on the
training window only) makes it worse.

## Margin / total (pooled, MAE)

| Target | Current | TabFM |
|---|---|---|
| Margin | 10.800 | 10.935 |
| Total | 14.948 | 14.994 |

## Paired bootstrap (log-loss, TabFM - current, 2000 resamples)

- Pooled: +0.0129, 95% CI [+0.0077, +0.0180] (TabFM worse).
- Per season: 2021 +0.0043 [-0.0023,+0.0107]; 2022 -0.0007 [-0.0065,+0.0050];
  2023 +0.0201 [+0.0088,+0.0322]; 2024 +0.0256 [+0.0134,+0.0375].

## Runtime

- Total runtime ~5.7 h across 4 windows; peak RSS ~7.3 GB.

Raw numbers: `docs/nba_tabfm_benchmark_2019_2024.json`. Negative result: TabFM
does not beat the current NBA model on this data.