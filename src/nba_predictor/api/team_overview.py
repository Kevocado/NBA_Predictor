"""The Teams page's data: the league's ratings, every team's form, and its next game.

One endpoint, one shape, so the page does not have to join four feeds and
hope they agree. The ratings come from the same box-score window the game
detail reads (`signals/team_matchups.team_rates`), so a team's rating on the
Teams page and its rating on a game page are the same number.

The form and the next game come from the schedule: the schedule is the
authority on who played whom and when, and the tracking database is the
authority on what the model picked.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from ..tracking import store
from ..tracking.timing import latest_pre_tip

#: How many results the form column shows.
FORM_GAMES = 10

#: How many games the row's detail shows.
DETAIL_GAMES = 5


def _team_games(schedule: list[dict], team: str, before: str | None = None) -> list[dict]:
    """Every game `team` is in, oldest first."""
    games = [
        g
        for g in schedule
        if g.get("home_team") == team or g.get("away_team") == team
    ]
    if before is not None:
        games = [g for g in games if str(g.get("game_date", "")) < before]
    return sorted(games, key=lambda g: (str(g.get("game_date", "")), str(g.get("game_id", ""))))


def _result(game: dict, team: str) -> str:
    """"W" or "L" for this team, or "" when the game has no score yet."""
    home = game.get("home_team") == team
    home_pts = game.get("home_pts")
    away_pts = game.get("away_pts")
    if home_pts is None or away_pts is None:
        return ""
    won = (home_pts > away_pts) if home else (away_pts > home_pts)
    return "W" if won else "L"


def form_for_team(schedule: list[dict], team: str, before: str | None = None) -> list[str]:
    """The team's last `FORM_GAMES` results, oldest first.

    A game with no score yet contributes nothing: an unplayed game is not a
    result, and counting it as one would put a "L" in a form column for a game
    that has not happened.
    """
    played = [g for g in _team_games(schedule, team, before) if g.get("completed")]
    results = [_result(g, team) for g in played]
    return [r for r in results if r][-FORM_GAMES:]


def last_games_for_team(schedule: list[dict], team: str, before: str | None = None) -> list[dict]:
    """The team's last `DETAIL_GAMES` played games, with the score and the result."""
    played = [g for g in _team_games(schedule, team, before) if g.get("completed")]
    out = []
    for game in played[-DETAIL_GAMES:]:
        home = game.get("home_team") == team
        out.append({
            "game_id": game.get("game_id"),
            "game_date": game.get("game_date"),
            "opponent": game.get("away_team") if home else game.get("home_team"),
            "home": home,
            # His own score first, so it agrees with the W/L beside it (home-away read as a loss for an away win).
            "score": f"{game.get('home_pts')}–{game.get('away_pts')}" if home else f"{game.get('away_pts')}–{game.get('home_pts')}",
            "result": _result(game, team),
        })
    return out


def next_game_for_team(schedule: list[dict], team: str, db_path, today: str) -> dict | None:
    """The team's next game, with what the model picked for it.

    The pick is the newest row made before tip-off, the same rule the game
    detail uses. A game with no pre-tip pick has no pick: the model has not
    said anything about it yet, and saying so is not the same as guessing.
    """
    upcoming = [
        g
        for g in _team_games(schedule, team)
        if not g.get("completed") and str(g.get("game_date", "")) >= today
    ]
    if not upcoming:
        return None
    game = upcoming[0]
    rows = list(store.get_predictions_for_game(db_path, str(game.get("game_id"))))
    pick = latest_pre_tip(rows, game) if rows else None
    home = game.get("home_team") == team
    home_prob = float(pick["home_win_prob"]) if pick else None
    if home_prob is None:
        pick_label = None
        pick_prob = None
    else:
        favoured = game["home_team"] if home_prob >= 0.5 else game["away_team"]
        pick_label = favoured
        pick_prob = home_prob if favoured == game["home_team"] else 1 - home_prob
    return {
        "game_id": game.get("game_id"),
        "game_date": game.get("game_date"),
        "tip_off": game.get("tip_off"),
        "opponent": game.get("away_team") if home else game.get("home_team"),
        "home": home,
        "pick": pick_label,
        "pick_probability": pick_prob,
    }


def overview(
    schedule: list[dict],
    db_path,
    ratings: dict | None,
    teams: list[str],
    today: str,
) -> dict[str, Any]:
    """The Teams page's payload: the league's ratings and every team's form."""
    payload: dict[str, Any] = {
        "as_of": today,
        "n_teams": ratings.get("n_teams") if ratings else 0,
        "caption": ratings.get("caption") if ratings else "",
        "metrics": ratings.get("metrics") if ratings else [],
        "ratings": ratings.get("teams") if ratings else {},
        "form": {team: form_for_team(schedule, team) for team in teams},
        "last_games": {team: last_games_for_team(schedule, team) for team in teams},
        "next_game": {team: next_game_for_team(schedule, team, db_path, today) for team in teams},
    }
    return payload


def today_iso() -> str:
    return date.today().isoformat()
