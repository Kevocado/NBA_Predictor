"""Scores and status for the schedule cache, every few minutes on game days.

The full refresh runs once a day, so a game that ended at 9pm showed "awaiting result" until the next afternoon.
This reads only the last/next day's ESPN scoreboards (two calls) and updates `completed`, the scores and `tip_off` of
games already in the schedule cache; it adds nothing, trains nothing, and writes the file only when something changed.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from nba_predictor import config
from nba_predictor.data import espn

FIELDS = ("completed", "home_pts", "away_pts", "tip_off")


def refresh_scores(schedule_path: Path, today: date | None = None, scoreboard=espn.get_scoreboard) -> int:
    """Number of schedule rows updated."""
    today = today or date.today()
    rows = json.loads(schedule_path.read_text())
    by_id = {r["game_id"]: r for r in rows}
    changed = 0
    for offset in (-1, 0, 1):
        for game in scoreboard((today + timedelta(days=offset)).isoformat()):
            row = by_id.get(game["game_id"])
            if row is None:
                continue
            new = {f: game.get(f) for f in FIELDS}
            if any(row.get(f) != v for f, v in new.items()):
                row.update(new)
                changed += 1
    if changed:
        tmp = schedule_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows))
        tmp.replace(schedule_path)
    return changed


if __name__ == "__main__":
    print("updated", refresh_scores(config.DATA_DIR / "cache" / "schedule" / "games.json"))
