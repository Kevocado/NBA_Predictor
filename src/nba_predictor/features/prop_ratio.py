"""Candidate ratio features for the prop regressors (pre-registered in docs/nba-prop-ratio-preregistration-2026-10.md).

Every value is built from strictly prior games. Each base quantity is a trailing window of the
last ``WINDOW`` games, shifted by one game, so a row never sees its own game. A ratio is a
quotient of two such quantities; a zero or missing denominator gives NaN, never inf.

Three families of base quantity:

* player: his own trailing sums (points, rebounds, assists, threes, FGA, minutes);
* team: his team's totals in the same games he logged (matched on game_id and team, then
  summed over his trailing games), so they describe the same window as his own stats;
* opponent: the opponent's defence, a trailing mean over that opponent's own last games,
  matched on (opponent, game_id). That match is safe because the opponent's row for this
  game already excludes this game through shift(1).

The pool is every ordered pair of distinct base quantities. It is fixed by rule here, not
chosen after looking at scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 10
MIN_GAMES = 3

PLAYER_BASE = {"p_pts": "points", "p_reb": "rebounds", "p_ast": "assists", "p_fg3m": "fg3m", "p_fga": "fga", "p_min": "minutes"}
TEAM_BASE = ["t_fga", "t_fg3a", "t_fta", "t_tov", "t_reb", "t_pts", "t_poss"]
OPP_BASE = ["o_pts_allowed", "o_fga_allowed", "o_fg3a_allowed", "o_poss_allowed"]
BASE_QUANTITIES = list(PLAYER_BASE) + TEAM_BASE + OPP_BASE


def team_game_table(games: list[dict]) -> pd.DataFrame:
    """One row per (game, team): the team's own totals, and the opponent's totals in that game (what it allowed)."""
    rows = []
    for g in games:
        for side, other in (("home", "away"), ("away", "home")):
            team, opp = g[f"{side}_team"], g[f"{other}_team"]
            own = {k: g.get(f"{side}_{k}") for k in ("fga", "fg3a", "fta", "tov", "oreb", "dreb", "pts")}
            theirs = {k: g.get(f"{other}_{k}") for k in ("fga", "fg3a", "fta", "tov", "oreb", "dreb", "pts")}
            rows.append({
                "game_id": g["game_id"], "game_date": g["game_date"], "team": team, "opponent": opp,
                "t_fga": own["fga"], "t_fg3a": own["fg3a"], "t_fta": own["fta"], "t_tov": own["tov"],
                "t_reb": _sum2(own["oreb"], own["dreb"]), "t_pts": own["pts"],
                "t_poss": _poss(own),
                "o_pts_allowed": theirs["pts"], "o_fga_allowed": theirs["fga"], "o_fg3a_allowed": theirs["fg3a"],
                "o_poss_allowed": _poss(theirs),
            })
    return pd.DataFrame(rows)


def _sum2(a, b):
    return None if a is None or b is None else a + b


def _poss(d):
    if any(d.get(k) is None for k in ("fga", "oreb", "tov", "fta")):
        return None
    return d["fga"] - d["oreb"] + d["tov"] + 0.44 * d["fta"]


def add_base_quantities(frame: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Attach the 17 causal base quantities to a player-game frame (returns a copy, same row count)."""
    out = frame.copy()
    out["_row"] = np.arange(len(out))
    out = out.sort_values(["player_id", "game_date", "_row"], kind="mergesort").reset_index(drop=True)

    def trailing_sum(values: pd.Series) -> pd.Series:
        return values.groupby(out["player_id"]).transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).sum()
        )

    for name, col in PLAYER_BASE.items():
        out[name] = trailing_sum(pd.to_numeric(out[col], errors="coerce").astype(float))

    # Team totals in the player's own games (left merge keeps row order: team_games has one row per game and team).
    same_game = out[["game_id", "team"]].merge(
        team_games[["game_id", "team"] + TEAM_BASE], on=["game_id", "team"], how="left"
    )
    assert len(same_game) == len(out), "team_games must hold one row per (game_id, team)"
    for name in TEAM_BASE:
        out[name] = trailing_sum(pd.Series(same_game[name].to_numpy(dtype=float), index=out.index))

    # Opponent defence: trailing mean over the opponent's own team-game sequence, matched on (opponent, game).
    tg = team_games.sort_values(["team", "game_date"], kind="mergesort").copy()
    for name in OPP_BASE:
        tg[name + "_roll"] = tg.groupby("team")[name].transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean()
        )
    opp_key = out[["opponent", "game_id"]].rename(columns={"opponent": "team"})
    opp = opp_key.merge(
        tg[["team", "game_id"] + [n + "_roll" for n in OPP_BASE]], on=["team", "game_id"], how="left"
    )
    for name in OPP_BASE:
        out[name] = opp[name + "_roll"].to_numpy()

    return out.sort_values("_row").drop(columns="_row").reset_index(drop=True)


def ratio_candidates(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Every ordered pair (num, den) of distinct base quantities as num / den, NaN when den is 0 or missing."""
    cands: dict[str, pd.Series] = {}
    for num in BASE_QUANTITIES:
        for den in BASE_QUANTITIES:
            if num == den:
                continue
            d = pd.to_numeric(frame[den], errors="coerce").astype(float)
            n = pd.to_numeric(frame[num], errors="coerce").astype(float)
            val = n / d.where(d != 0)
            cands[f"{num}/{den}"] = val.replace([np.inf, -np.inf], np.nan)
    return cands
