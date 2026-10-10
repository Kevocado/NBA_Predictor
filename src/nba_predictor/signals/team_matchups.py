"""NBA team-level matchup facts: offence v defence rating, pace, rebounds, 3P%, FG%.

Plain ranked facts for the fixture's Matchup section, from the same box-score
history the four-factors duels read. They are NEUTRAL by construction:
`toward_pick` is always None. Whether a rank gap predicts anything is not
established (the duel-lift data run), so nothing here picks a side.

Every figure is a ratio of sums over a team's last `window` games strictly before
`as_of`, so a high-volume game does not outweigh a low-volume one. A defence is
what it ALLOWS, never what it scores. Ranks are out of the 30-team league; fewer
ranked teams than that and no row is produced (a rank out of 17 is not a position).

Pace is possessions per game, FGA - OREB + TOV + 0.44*FTA, averaged over both
teams in the game; overtime is not normalised away (ponytail: per-48 if it matters).
"""
from __future__ import annotations

import pandas as pd

from .duel import ranks
from .four_factors_duel import MIN_GAMES, N_TEAMS, WINDOW

_NEED = ("fgm", "fga", "fg3m", "tov", "oreb", "dreb", "fta")


def _poss(fga, oreb, tov, fta):
    return fga - oreb + tov + 0.44 * fta


def _team_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game with the team's own totals and what it allowed."""
    rows = []
    has_3pa = "home_fg3a" in games.columns
    for _, g in games.iterrows():
        for side, opp in (("home", "away"), ("away", "home")):
            if any(pd.isna(g.get(f"{s}_{f}")) for s in (side, opp) for f in _NEED):
                continue
            own_p = _poss(g[f"{side}_fga"], g[f"{side}_oreb"], g[f"{side}_tov"], g[f"{side}_fta"])
            opp_p = _poss(g[f"{opp}_fga"], g[f"{opp}_oreb"], g[f"{opp}_tov"], g[f"{opp}_fta"])
            rows.append({
                "game_date": str(g["game_date"]), "team": g[f"{side}_team"],
                "pts": g[f"{side}_pts"], "pts_allowed": g[f"{opp}_pts"],
                "poss": (own_p + opp_p) / 2,
                "reb": g[f"{side}_oreb"] + g[f"{side}_dreb"], "reb_allowed": g[f"{opp}_oreb"] + g[f"{opp}_dreb"],
                "fgm": g[f"{side}_fgm"], "fga": g[f"{side}_fga"],
                "fgm_allowed": g[f"{opp}_fgm"], "fga_allowed": g[f"{opp}_fga"],
                "fg3m": g[f"{side}_fg3m"], "fg3m_allowed": g[f"{opp}_fg3m"],
                "fg3a": g[f"{side}_fg3a"] if has_3pa else float("nan"),
                "fg3a_allowed": g[f"{opp}_fg3a"] if has_3pa else float("nan"),
            })
    return pd.DataFrame(rows)


def team_rates(games: pd.DataFrame, as_of, window: int = WINDOW, min_games: int = MIN_GAMES) -> pd.DataFrame:
    """Per-team ortg, drtg, pace, reb, reb_allowed, fg_pct, fg_pct_allowed, fg3_pct, fg3_pct_allowed."""
    if games is None or not len(games):
        return pd.DataFrame()
    long = _team_games(games)
    if not len(long):
        return pd.DataFrame()
    played = long[long["game_date"] < str(as_of)].sort_values("game_date")
    recent = played.groupby("team", sort=False).tail(window)
    n = recent.groupby("team").size()
    s = recent.groupby("team").sum(numeric_only=True, min_count=1)
    s = s[n.reindex(s.index) >= min_games]
    n = n.reindex(s.index)
    out = pd.DataFrame(index=s.index)
    out["ortg"] = 100 * s["pts"] / s["poss"]
    out["drtg"] = 100 * s["pts_allowed"] / s["poss"]
    out["pace"] = s["poss"] / n
    out["reb"] = s["reb"] / n
    out["reb_allowed"] = s["reb_allowed"] / n
    out["fg_pct"] = s["fgm"] / s["fga"]
    out["fg_pct_allowed"] = s["fgm_allowed"] / s["fga_allowed"]
    out["fg3_pct"] = s["fg3m"] / s["fg3a"]              # NaN (not 0) when 3PA is unknown
    out["fg3_pct_allowed"] = s["fg3m_allowed"] / s["fg3a_allowed"]
    return out


#: (id, stat, foil, attack column, defence column, attack higher is better, defence higher is better)
_DUELS = [
    ("rating", "offence rating", "defence rating", "ortg", "drtg", True, False),
    ("reb", "rebounds per game", "rebounds allowed per game", "reb", "reb_allowed", True, False),
    ("fg3_pct", "3P%", "3P% allowed", "fg3_pct", "fg3_pct_allowed", True, False),
    ("fg_pct", "FG%", "FG% allowed", "fg_pct", "fg_pct_allowed", True, False),
]


def team_matchups(home: str, away: str, games: pd.DataFrame, as_of, n_teams: int = N_TEAMS) -> list[dict]:
    """Facts-bundle rows for `home` v `away`; empty unless the league is fully ranked."""
    rates = team_rates(games, as_of)
    if len(rates) < n_teams or home not in rates.index or away not in rates.index:
        return []

    def row(duel_id, stat, foil, attacker, defender, a_rank, d_rank):
        return {"id": duel_id, "attacker": attacker, "defender": defender, "stat": stat, "foil": foil,
                "attacker_rank": a_rank, "defender_rank": d_rank, "n_teams": len(rates), "toward_pick": None}

    out = []
    for duel_id, stat, foil, a_col, d_col, a_high, d_high in _DUELS:
        if rates[a_col].isna().any() or rates[d_col].isna().any():
            continue
        a_ranks = ranks(rates[a_col].to_dict(), higher_is_better=a_high)
        d_ranks = ranks(rates[d_col].to_dict(), higher_is_better=d_high)
        for side, attacker, defender in (("home", home, away), ("away", away, home)):
            out.append(row(f"{duel_id}:{side}", stat, foil, attacker, defender, a_ranks[attacker], d_ranks[defender]))
    p = ranks(rates["pace"].to_dict(), higher_is_better=True)
    out.append(row("pace", "pace", "pace", home, away, p[home], p[away]))
    return out
