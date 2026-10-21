"""The Players page's extra data: each player's next game, and their last ten.

The per-game averages, the ratings and the position are already in the hub
players cache. What is not there is what the model expects next, and what
actually happened -- and both live in the tracking database, not in a cache
the ingest writes.

Two shapes, for two different needs:

* `next_game_projections` is for the table: every player, the team's next
  game, and the model's own projection for that player in that game. One
  query per team, so the whole league costs thirty.
* `player_detail` is for the drawer: one player, their last ten games with
  the actual beside the projection, their next game, and their status.

A projection that is not there is `None` and never zero: a player with no
stored projection has no projection, and zero would read as "the model
expects nothing from him".
"""
from __future__ import annotations

from typing import Any

from ..tracking import store

#: The stats the page talks about.
STATS = ("points", "rebounds", "assists", "threes")

#: How many games the drawer shows.
DRAWER_GAMES = 10


def _latest_per_player(rows) -> dict[tuple[str, str], dict]:
    """The newest stored row per (player, stat).

    The store keeps every snapshot -- a rerun of the models is history, not a
    replacement -- so the newest row is the one the site would show.
    """
    latest: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row["player_id"]), str(row["stat"]))
        current = latest.get(key)
        if current is None or row["created_at"] > current["created_at"]:
            latest[key] = dict(row)
    return latest


def _team_next_games(schedule: list[dict], today: str) -> dict[str, dict]:
    """Each team's first game still to come."""
    out: dict[str, dict] = {}
    for game in sorted(schedule, key=lambda g: (str(g.get("game_date", "")), str(g.get("game_id", "")))):
        if game.get("completed") or str(game.get("game_date", "")) < today:
            continue
        for side in ("home_team", "away_team"):
            team = game.get(side)
            if team and team not in out:
                out[team] = game
    return out


def next_game_projections(
    schedule: list[dict],
    db_path,
    team_of: dict[str, str],
    today: str,
) -> dict[str, dict]:
    """Every player's next game, with the model's projection for it."""
    next_games = _team_next_games(schedule, today)
    projections_by_game: dict[str, dict[tuple[str, str], dict]] = {}
    for game in next_games.values():
        game_id = str(game.get("game_id"))
        if game_id not in projections_by_game:
            projections_by_game[game_id] = _latest_per_player(
                store.get_player_predictions_for_game(db_path, game_id)
            )

    out: dict[str, dict] = {}
    for player_id, team in team_of.items():
        game = next_games.get(team)
        if not game:
            continue
        rows = projections_by_game.get(str(game.get("game_id")), {})
        home = game.get("home_team") == team
        entry: dict[str, Any] = {
            "game_id": game.get("game_id"),
            "game_date": game.get("game_date"),
            "tip_off": game.get("tip_off"),
            "opponent": game.get("away_team") if home else game.get("home_team"),
            "home": home,
            "projections": {},
        }
        for stat in STATS:
            row = rows.get((str(player_id), stat))
            entry["projections"][stat] = float(row["predicted_value"]) if row is not None else None
        out[str(player_id)] = entry
    return out


def player_detail(
    schedule: list[dict],
    db_path,
    player_id: str,
    team: str | None,
    injuries: list[dict] | None = None,
    today: str = "",
) -> dict:
    """One player: their last ten games, their next game, their status."""
    games_by_id = {str(g.get("game_id")): g for g in schedule}
    outcomes = store.get_player_outcomes_for_player(db_path, str(player_id))

    actuals: dict[str, dict[str, float]] = {}
    for row in outcomes:
        actuals.setdefault(str(row["game_id"]), {})[str(row["stat"])] = float(row["actual_value"])

    played = sorted(
        (game_id for game_id in actuals if game_id in games_by_id),
        key=lambda game_id: str(games_by_id[game_id].get("game_date", "")),
    )[-DRAWER_GAMES:]

    games = []
    for game_id in played:
        game = games_by_id[game_id]
        rows = _latest_per_player(store.get_player_predictions_for_game(db_path, game_id))
        home = game.get("home_team") == team
        entry: dict[str, Any] = {
            "game_id": game_id,
            "game_date": game.get("game_date"),
            "opponent": game.get("away_team") if home else game.get("home_team"),
            "home": home,
            "score": f"{game.get('home_pts')}–{game.get('away_pts')}",
            "actual": {stat: actuals[game_id].get(stat) for stat in STATS},
            "projection": {
                stat: (float(rows[(str(player_id), stat)]["predicted_value"])
                        if (str(player_id), stat) in rows else None)
                for stat in STATS
            },
        }
        games.append(entry)

    status = None
    for injury in injuries or ():
        if str(injury.get("player_id")) == str(player_id):
            status = {
                "status": injury.get("status") or "",
                "source": injury.get("source") or "ESPN's injury report",
                "dated": injury.get("dated") or "",
            }
            break

    return {
        "player_id": str(player_id),
        "team": team,
        "games": games,
        "status": status,
    }
