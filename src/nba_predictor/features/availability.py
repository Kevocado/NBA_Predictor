"""availability.py -- lagged rotation-availability proxy (single chronological pass).

For game G and team T: of T's top-8 rotation players (ranked by cumulative minutes in
T's games BEFORE G), the share of that rotation's minutes / production (pts + 1.2 reb +
1.5 ast) held by players who did NOT play in T's PREVIOUS completed game. A player who
missed the last game is more likely to be out again; this is known before tip-off and
uses box scores only (no injury feed).

Causality: the value for G is recorded BEFORE G's own box score is folded into state, so
changing G's box score cannot move G's features.

Unplayed / upcoming games, and completed games with no player box score, are emitted from
state as it stands (T's last game that HAS a box score and the cumulative minutes so far)
and are never folded in. They are never NaN.

Default (documented): a team with no player-box history yet (its first game, or no player
boxes at all) gets 0.0 absent share on both columns, i.e. "nobody known to be missing".

Evidence: NBA#74 follow-up, held-out 2022-23..2024-25 confirmation (docs/nba-availability-evaluation).
Reference implementation: predictor-hub docs/research/2026-10-13-nba-model-gaps-scripts/feats.py.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

AVAILABILITY_COLUMNS = [
    "home_absent_min_share", "away_absent_min_share",
    "home_absent_prod_share", "away_absent_prod_share",
]
ROTATION_SIZE = 8
NO_HISTORY_DEFAULT = 0.0
_PLAYER_COLS = ["game_id", "team", "player_id", "minutes", "points", "rebounds", "assists"]


def compute_availability_features(games: pd.DataFrame, players: pd.DataFrame | None) -> pd.DataFrame:
    """One row per game_id with the four AVAILABILITY_COLUMNS.

    `games`: game_id, game_date, home_team, away_team (played or not).
    `players`: one row per (player, completed game) as `ingest.to_player_training_frame`
    builds it; None or empty means no history (every value is NO_HISTORY_DEFAULT).
    """
    order = defaultdict(list)
    for r in games.sort_values(["game_date", "game_id"], kind="stable").itertuples():
        order[r.home_team].append(r.game_id)
        order[r.away_team].append(r.game_id)

    by_team_game = {}
    if players is not None and len(players):
        p = players.sort_values(["game_date", "game_id"], kind="stable")[_PLAYER_COLS]
        for (gid, team), g in p.groupby(["game_id", "team"], sort=False):
            by_team_game[(gid, team)] = list(zip(g.player_id, g.minutes, g.points, g.rebounds, g.assists))

    feat = {}
    for team, gids in order.items():
        cum_min, cum_prod = defaultdict(float), defaultdict(float)
        prev_played = None
        for gid in gids:
            rows = by_team_game.get((gid, team))
            m_share = p_share = NO_HISTORY_DEFAULT
            if prev_played is not None and cum_min:
                top = sorted(cum_min, key=cum_min.get, reverse=True)[:ROTATION_SIZE]
                absent = [x for x in top if x not in prev_played]
                tot = sum(cum_min[x] for x in top)
                if tot:
                    m_share = sum(cum_min[x] for x in absent) / tot
                totp = sum(cum_prod[x] for x in top)
                if totp:
                    p_share = sum(cum_prod[x] for x in absent) / totp
            feat[(gid, team)] = (m_share, p_share)  # recorded BEFORE this game is folded in
            if rows is not None:
                prev_played = {pid for pid, mn, *_ in rows if mn > 0}
                for pid, mn, pts, reb, ast in rows:
                    cum_min[pid] += mn
                    cum_prod[pid] += pts + 1.2 * reb + 1.5 * ast

    d = (NO_HISTORY_DEFAULT, NO_HISTORY_DEFAULT)
    out = games[["game_id"]].copy()
    for side in ("home", "away"):
        teams = games[f"{side}_team"]
        out[f"{side}_absent_min_share"] = [feat.get((g, t), d)[0] for g, t in zip(games.game_id, teams)]
        out[f"{side}_absent_prod_share"] = [feat.get((g, t), d)[1] for g, t in zip(games.game_id, teams)]
    return out[["game_id", *AVAILABILITY_COLUMNS]]
