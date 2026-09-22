import json
from datetime import date, timedelta
from pathlib import Path


def load_schedule(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def get_games_for_date(schedule: list[dict], game_date: str) -> list[dict]:
    return [game for game in schedule if game["game_date"] == game_date]


def get_game(schedule: list[dict], game_id: str) -> dict | None:
    return next((game for game in schedule if game["game_id"] == game_id), None)


def get_games_for_week(schedule: list[dict], week_start: str) -> list[dict]:
    """Games in the 7-day window [week_start, week_start + 6 days], sorted by date."""
    start = date.fromisoformat(week_start)
    week_dates = {(start + timedelta(days=i)).isoformat() for i in range(7)}
    games = [game for game in schedule if game["game_date"] in week_dates]
    return sorted(games, key=lambda g: (g["game_date"], g["game_id"]))


def monday_of(iso_date: str) -> str:
    """The Monday of the week containing iso_date (YYYY-MM-DD)."""
    d = date.fromisoformat(iso_date)
    return (d - timedelta(days=d.weekday())).isoformat()


def default_week_start(schedule: list[dict], today: str) -> str | None:
    """Monday of the week containing the next not-yet-completed game, so
    the page always lands on upcoming predictions rather than a season's
    opening week (which is typically long since completed by the time
    anyone loads the page). Falls back to the Monday of the most recent
    (latest) game when nothing is upcoming — e.g. the season has already
    finished — rather than the earliest game ever played. None only if
    the schedule is completely empty."""
    if not schedule:
        return None

    # Real ESPN data can carry stale entries — a game whose completed flag
    # never got set to true (e.g. postponed/orphaned), dated well before
    # today, from a season that has already finished. A genuinely current
    # not-completed game is never more than a few weeks stale relative to
    # today (the season it belongs to is still being played); a 30-day
    # cutoff is a deliberately generous, simple way to exclude leftover
    # artifacts from an already-finished season without excluding a real
    # game the schedule just hasn't marked completed yet.
    stale_cutoff = (date.fromisoformat(today) - timedelta(days=30)).isoformat()
    upcoming_dates = sorted(
        g["game_date"] for g in schedule if not g.get("completed") and g["game_date"] >= stale_cutoff
    )
    if upcoming_dates:
        return monday_of(max(today, upcoming_dates[0]))

    latest = max(game["game_date"] for game in schedule)
    return monday_of(latest)


def get_head_to_head(schedule: list[dict], team_a: str, team_b: str, before_date: str, limit: int = 5) -> list[dict]:
    """Prior completed meetings between team_a and team_b, strictly before
    before_date, most recent first."""
    matches = [
        game
        for game in schedule
        if game.get("completed")
        and game["game_date"] < before_date
        and {game["home_team"], game["away_team"]} == {team_a, team_b}
    ]
    matches.sort(key=lambda g: g["game_date"], reverse=True)
    return matches[:limit]


def get_recent_form(schedule: list[dict], team: str, before_date: str, limit: int = 5) -> list[str]:
    """Team's last `limit` completed results strictly before before_date, as
    "W"/"L", most recent first."""
    games = [
        game
        for game in schedule
        if game.get("completed")
        and game["game_date"] < before_date
        and team in (game["home_team"], game["away_team"])
    ]
    games.sort(key=lambda g: g["game_date"], reverse=True)

    results = []
    for game in games[:limit]:
        is_home = game["home_team"] == team
        team_pts = game["home_pts"] if is_home else game["away_pts"]
        opp_pts = game["away_pts"] if is_home else game["home_pts"]
        results.append("W" if team_pts > opp_pts else "L")
    return results
