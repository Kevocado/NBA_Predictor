"""Early-season props: lean on last season until this season has said enough.

A player's rolling window in October is mostly last season (George's playoff-length minutes made him a 3.7-threes
shooter in week one; his real rate is 2.6). On the 2021-2025 openers a shrinkage estimate

    w * (this season's average so far) + (1 - w) * (last season's average),   w = n / (n + K)

beat the shipped model in each player's first 20 games on all four markets (points MAE 4.33 vs 4.59 over the first 10,
paired-bootstrap interval excluding zero; 4.51 vs 4.58 over games 10-20) and lost after about game 40
(`nba-eval/props_early.py`). So it replaces the model's number for a player until `HANDOFF_GAMES` games, then steps aside.
A player with no last season (a rookie) keeps the model's number: there is nothing to shrink toward.
"""
from __future__ import annotations

import pandas as pd

K = 3.0
HANDOFF_GAMES = 20


def season_of(game_date: str) -> int:
    return int(game_date[:4]) if int(game_date[5:7]) >= 10 else int(game_date[:4]) - 1


def shrunk_projections(played: pd.DataFrame, upcoming: pd.DataFrame, stat_columns: dict[str, str]) -> dict[tuple[str, str, str], float]:
    """{(game_id, player_id, stat): value} for upcoming rows whose player is within HANDOFF_GAMES of this season's start.

    `played`: completed games he played (player_id, game_date, minutes > 0, the stat columns); `upcoming`: game_id,
    player_id, game_date. Only games strictly before the upcoming game's date are read.
    """
    if not len(played) or not len(upcoming):
        return {}
    played = played.assign(season=played["game_date"].map(season_of))
    by_player = {pid: g for pid, g in played.groupby("player_id")}
    out: dict[tuple[str, str, str], float] = {}
    for row in upcoming.itertuples():
        g = by_player.get(row.player_id)
        if g is None:
            continue
        season = season_of(row.game_date)
        before = g[g["game_date"] < row.game_date]
        this_season, last_season = before[before["season"] == season], g[g["season"] == season - 1]
        n = len(this_season)
        if n >= HANDOFF_GAMES or not len(last_season):
            continue
        w = n / (n + K)
        for stat, col in stat_columns.items():
            current = this_season[col].mean() if n else 0.0
            out[(row.game_id, row.player_id, stat)] = float(w * current + (1 - w) * last_season[col].mean())
    return out
