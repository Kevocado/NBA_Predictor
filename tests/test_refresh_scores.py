import json
from datetime import date

from nba_predictor.pipeline.refresh_scores import refresh_scores


def _sched(tmp_path):
    path = tmp_path / "games.json"
    path.write_text(json.dumps([
        {"game_id": "a", "game_date": "2026-10-20", "home_team": "DET", "away_team": "BOS", "completed": False, "home_pts": None, "away_pts": None, "tip_off": "t"},
        {"game_id": "b", "game_date": "2026-10-20", "home_team": "NYK", "away_team": "PHI", "completed": True, "home_pts": 100, "away_pts": 90, "tip_off": "t"},
    ]))
    return path


def test_a_finished_game_is_updated_and_unchanged_rows_are_left_alone(tmp_path):
    path = _sched(tmp_path)
    board = lambda day: [
        {"game_id": "a", "completed": True, "home_pts": 121, "away_pts": 119, "tip_off": "t"},
        {"game_id": "b", "completed": True, "home_pts": 100, "away_pts": 90, "tip_off": "t"},
        {"game_id": "zz", "completed": True, "home_pts": 1, "away_pts": 0, "tip_off": "t"},   # not in the schedule: never added
    ] if day == "2026-10-20" else []
    assert refresh_scores(path, date(2026, 10, 20), board) == 1
    rows = {r["game_id"]: r for r in json.loads(path.read_text())}
    assert rows["a"]["completed"] and rows["a"]["home_pts"] == 121 and len(rows) == 2


def test_nothing_changed_writes_nothing(tmp_path):
    path = _sched(tmp_path)
    before = path.stat().st_mtime_ns
    assert refresh_scores(path, date(2026, 10, 20), lambda day: []) == 0
    assert path.stat().st_mtime_ns == before
