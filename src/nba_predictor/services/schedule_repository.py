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
