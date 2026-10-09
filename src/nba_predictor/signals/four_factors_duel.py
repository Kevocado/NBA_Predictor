"""NBA offence-versus-defence duels on the four factors, from box scores already known before the game.

Each duel is one side's attack (eFG%, TOV%, ORB% or FT rate) against the other side's matching defence,
expressed as league ranks. Only games strictly before `as_of` are read, and each team's last `window`
of them: ranking a team off last month's roster is a number, not a measurement.

A defence is what it ALLOWS, never what it scores: the defence rank for a factor is the rank of the
OPPONENT's value for that factor, so a defence that holds teams to a low eFG% ranks well.

`compute_four_factors` is reused rather than reimplemented -- it is the same formula the feature pipeline
already applies to the same box scores, so a duel and a feature cannot disagree about what eFG% is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.four_factors import compute_four_factors
from .duel import Duel, make_duel, ranks

#: Games per team. NBA plays 82, but a month-and-a-half of form is what a duel claims to measure.
WINDOW = 15

#: Fewer games than this and a team is not ranked at all -- 30 teams is the league, so a
#: rank out of fewer is a fabricated position.
MIN_GAMES = 5

#: A rank gap below this is noise, not an edge. 8 of 30 teams.
MIN_GAP = 8

#: The league has 30 teams. A rank of "12 of 30" means something; "12 of 17" does not.
N_TEAMS = 30

#: (duel id, factor, attack noun, defence noun). `higher_is_better` is per factor because
#: the four do not agree: a low turnover rate is good for the offence and bad for the defence.
FACTORS = [
    ("efg_pct", "efg_pct", "shooting", "shooting defence", True),
    ("tov_rate", "tov_rate", "ball security", "forcing turnovers", False),
    ("orb_pct", "orb_pct", "offensive rebounding", "defensive rebounding", True),
    ("ft_rate", "ft_rate", "getting to the line", "foul discipline", True),
]


def _long_form(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game, with the team's own four factors and its opponent's.

    Both sides of every game are unpivoted into team rows, so a team's own value and
    the value it allowed are on the same row -- which is what makes the defence rank
    the opponent's figure rather than a second, separately-computed guess.
    """
    rows = []
    for _, g in games.iterrows():
        for side, opponent in (("home", "away"), ("away", "home")):
            if pd.isna(g.get(f"{side}_fgm")) or pd.isna(g.get(f"{opponent}_fgm")):
                continue
            own = compute_four_factors(
                g[f"{side}_fgm"], g[f"{side}_fga"], g[f"{side}_fg3m"],
                g[f"{side}_tov"], g[f"{side}_oreb"], g[f"{opponent}_dreb"], g[f"{side}_fta"],
            )
            allowed = compute_four_factors(
                g[f"{opponent}_fgm"], g[f"{opponent}_fga"], g[f"{opponent}_fg3m"],
                g[f"{opponent}_tov"], g[f"{opponent}_oreb"], g[f"{side}_dreb"], g[f"{opponent}_fta"],
            )
            rows.append({
                "game_id": g["game_id"], "game_date": str(g["game_date"]),
                "team": g[f"{side}_team"], "opponent": g[f"{opponent}_team"],
                **{f: own[f] for f in ("efg_pct", "tov_rate", "orb_pct", "ft_rate")},
                **{f"{f}_allowed": allowed[f] for f in ("efg_pct", "tov_rate", "orb_pct", "ft_rate")},
            })
    if not rows:
        return pd.DataFrame(columns=[
            "game_id", "game_date", "team", "opponent",
            "efg_pct", "tov_rate", "orb_pct", "ft_rate",
            "efg_pct_allowed", "tov_rate_allowed", "orb_pct_allowed", "ft_rate_allowed",
        ])
    return pd.DataFrame(rows)


def team_game_factors(
    games: pd.DataFrame, as_of, window: int = WINDOW, min_games: int = MIN_GAMES
) -> pd.DataFrame:
    """Each team's mean four factors and mean factors-allowed over its last `window` games.

    Only games strictly before `as_of` are read. `min_games` is checked AFTER the
    window, so a team that has played enough recently is ranked and a team that has
    not is absent from the index rather than ranked off a stale sample.
    """
    if games is None or not len(games):
        return pd.DataFrame()

    long = _long_form(games)
    played = long[long["game_date"] < str(as_of)]
    if not len(played):
        return pd.DataFrame()

    played = played.sort_values("game_date")
    recent = played.groupby("team", sort=False).tail(window)
    counts = recent.groupby("team").size()
    means = recent.groupby("team").mean(numeric_only=True)
    return means[counts.reindex(means.index) >= min_games]


def four_factors_duel(
    home: str,
    away: str,
    games: pd.DataFrame,
    as_of,
    window: int = WINDOW,
    min_gap: int = MIN_GAP,
    min_games: int = MIN_GAMES,
    n_teams: int = N_TEAMS,
    limit: int = 2,
) -> list[Duel]:
    """The strongest `limit` duels for `home` v `away`, strongest first.

    Empty when either team is not yet ranked, or when nothing separates the two
    teams by `min_gap` places -- a matchup with nothing to say produces no duel,
    which is the honest answer rather than the least-uninteresting one.
    """
    means = team_game_factors(games, as_of=as_of, window=window, min_games=min_games)
    if not len(means) or home not in means.index or away not in means.index:
        return []

    out: list[Duel] = []
    for duel_id, factor, attack_noun, defence_noun, higher_is_better in FACTORS:
        attack_ranks = ranks(means[factor].to_dict(), higher_is_better=higher_is_better)
        defence_ranks = ranks(
            means[f"{factor}_allowed"].to_dict(), higher_is_better=not higher_is_better
        )
        for attacker_side in ("home", "away"):
            duel = make_duel(
                f"{duel_id}:{attacker_side}",
                home=home, away=away, attacker_side=attacker_side,
                attack_ranks=attack_ranks, defence_ranks=defence_ranks,
                history_gaps=np.array([], float), min_gap=min_gap,
                stat=attack_noun, foil=defence_noun,
            )
            if duel is not None:
                out.append(duel)
    out.sort(key=lambda d: d.strength, reverse=True)
    return out[:limit]


def to_context(duels: list[Duel], pick_side: str | None) -> list[dict]:
    """Facts-bundle form. `toward_pick` is True when the duel favours the pick."""
    rows = []
    for d in duels:
        rows.append({
            "id": d.id,
            "attacker": d.attacker,
            "defender": d.defender,
            "stat": d.stat,
            "foil": d.foil,
            "attacker_rank": d.attacker_rank,
            "defender_rank": d.defender_rank,
            "n_teams": d.n_teams,
            "toward_pick": None if pick_side is None else (d.toward == pick_side),
        })
    return rows
