import json
from pathlib import Path


def load_schedule(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def get_games_for_date(schedule: list[dict], game_date: str) -> list[dict]:
    return [game for game in schedule if game["game_date"] == game_date]


def get_game(schedule: list[dict], game_id: str) -> dict | None:
    return next((game for game in schedule if game["game_id"] == game_id), None)
