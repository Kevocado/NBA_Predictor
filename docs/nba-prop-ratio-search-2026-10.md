# NBA prop ratio-feature search: results

Pre-registered in `docs/nba-prop-ratio-preregistration-2026-10.md`. Raw numbers: `docs/nba-prop-ratio-search-2026-10.json`.

Candidates generated: **272**, dropped as duplicates: **0**, tested: **272** x 4 markets = **1088** selection tests (Holm family).

Selection tests passing stability and Holm < 0.05: **13**.

## Holdout (labels 2023 and 2024, opened once)

| candidate | target | target MAE base -> cand | diff | CI (alpha/K) | other markets not worse | passes |
|---|---|---|---|---|---|---|
| `p_pts/p_fga` | threes | 0.8881 -> 0.8859 | -0.0022 | [-0.0028, -0.0015] | False | False |
| `p_fga/p_pts` | threes | 0.8881 -> 0.8861 | -0.0020 | [-0.0027, -0.0014] | False | False |
| `t_pts/t_poss` | assists | 1.3532 -> 1.3519 | -0.0012 | [-0.0019, -0.0006] | False | False |

Any candidate passing: **False**.

## Selection, top candidates by Holm-adjusted p

| candidate | market | mean MAE gain (selection) | per-fold gains 2020/2021/2022 | p raw | p Holm | stable | selected |
|---|---|---|---|---|---|---|---|
| `p_pts/p_fga` | threes | +0.0029 | +0.0019 / +0.0031 / +0.0036 | 1.21e-32 | 1.32e-29 | True | True |
| `p_fga/p_pts` | threes | +0.0028 | +0.0020 / +0.0027 / +0.0035 | 3.72e-30 | 4.05e-27 | True | True |
| `t_pts/t_poss` | assists | +0.0023 | +0.0035 / +0.0010 / +0.0024 | 7.97e-15 | 8.66e-12 | True | True |
| `t_poss/t_pts` | assists | +0.0020 | +0.0029 / +0.0010 / +0.0021 | 2.5e-11 | 2.72e-08 | True | True |
| `p_fg3m/p_pts` | threes | +0.0011 | +0.0008 / +0.0013 / +0.0012 | 1.81e-09 | 1.97e-06 | True | True |
| `p_reb/o_fga_allowed` | threes | +0.0006 | +0.0012 / +0.0006 / +0.0002 | 1.71e-07 | 0.000185 | True | True |
| `t_pts/o_poss_allowed` | assists | +0.0014 | +0.0023 / +0.0004 / +0.0015 | 1.94e-07 | 0.00021 | True | True |
| `t_reb/t_pts` | assists | +0.0012 | +0.0028 / -0.0003 / +0.0013 | 5.45e-07 | 0.000589 | False | False |
| `p_ast/t_pts` | assists | +0.0016 | +0.0020 / +0.0001 / +0.0025 | 5.5e-07 | 0.000594 | True | True |
| `p_ast/t_fg3a` | assists | +0.0011 | +0.0019 / +0.0013 / +0.0002 | 1.39e-05 | 0.015 | True | True |
| `o_fga_allowed/t_pts` | assists | +0.0011 | +0.0013 / +0.0002 / +0.0018 | 1.46e-05 | 0.0157 | True | True |
| `p_pts/t_reb` | threes | +0.0005 | +0.0010 / +0.0003 / +0.0003 | 1.58e-05 | 0.017 | True | True |
| `o_poss_allowed/t_pts` | assists | +0.0011 | +0.0015 / +0.0005 / +0.0013 | 1.72e-05 | 0.0185 | True | True |
| `o_poss_allowed/p_pts` | threes | +0.0005 | +0.0009 / +0.0005 / +0.0002 | 2.79e-05 | 0.03 | True | True |
| `t_pts/t_fga` | assists | +0.0011 | +0.0020 / -0.0005 / +0.0018 | 6.36e-05 | 0.0683 | False | False |
| `p_fga/p_min` | threes | +0.0007 | +0.0007 / +0.0007 / +0.0006 | 7.93e-05 | 0.0851 | True | False |
| `o_poss_allowed/p_reb` | threes | +0.0005 | +0.0005 / +0.0009 / +0.0000 | 8.38e-05 | 0.0898 | True | False |
| `t_pts/o_fga_allowed` | assists | +0.0010 | +0.0012 / -0.0000 / +0.0017 | 0.000136 | 0.145 | False | False |
| `t_pts/p_ast` | assists | +0.0011 | +0.0016 / -0.0005 / +0.0023 | 0.000209 | 0.224 | False | False |
| `p_reb/o_poss_allowed` | threes | +0.0004 | +0.0005 / +0.0008 / -0.0000 | 0.000295 | 0.315 | False | False |
| `t_reb/p_pts` | threes | +0.0005 | +0.0007 / +0.0004 / +0.0002 | 0.000315 | 0.337 | True | False |
| `o_fga_allowed/p_reb` | threes | +0.0004 | +0.0009 / +0.0007 / -0.0003 | 0.000376 | 0.401 | False | False |
| `p_min/o_fga_allowed` | points | +0.0025 | +0.0047 / +0.0028 / +0.0003 | 0.000606 | 0.646 | True | False |
| `o_pts_allowed/t_pts` | assists | +0.0009 | +0.0021 / +0.0007 / +0.0000 | 0.000613 | 0.653 | True | False |
| `t_pts/p_fga` | assists | +0.0007 | +0.0004 / +0.0007 / +0.0009 | 0.000657 | 0.699 | True | False |
