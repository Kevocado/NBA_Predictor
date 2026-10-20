"""Frozen earlier seasons for the retrain (data/history/nba_history.json.gz)."""
import gzip
import json

import pandas as pd

from nba_predictor.pipeline import history

PLAYER_COLS = history.PLAYER_COLS


def _file(tmp_path, games, players, before="2025-10-01"):
    path = tmp_path / "h.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"before": before, "games": games, "players": players}, f)
    return path


def _game(gid, day):
    return {"game_id": gid, "game_date": day, "home_team": "BOS", "away_team": "NYK", "completed": True, "home_pts": 100, "away_pts": 90}


def _player(gid, day):
    return {"game_id": gid, "game_date": day, "team": "BOS", "player_id": "p1", "minutes": 30.0, "points": 10.0, "rebounds": 3.0, "assists": 2.0}


def test_load_history_keeps_only_games_before_the_window(tmp_path):
    path = _file(tmp_path, [_game("a", "2025-05-01"), _game("b", "2025-10-05")], [_player("a", "2025-05-01"), _player("b", "2025-10-05")])
    games, players = history.load_history("2025-10-01", path)
    assert [g["game_id"] for g in games] == ["a"]
    assert list(players["game_id"]) == ["a"]


def test_a_missing_file_means_no_history_not_an_error(tmp_path):
    games, players = history.load_history("2025-10-01", tmp_path / "nope.json.gz")
    assert games == [] and list(players.columns) == PLAYER_COLS and len(players) == 0


def test_merge_history_prepends_and_never_duplicates_a_game(tmp_path):
    path = _file(tmp_path, [_game("a", "2025-05-01"), _game("w", "2025-09-30")], [_player("a", "2025-05-01")])
    window = [_game("w", "2025-09-30"), _game("c", "2025-11-01")]            # "w" is in both: the live copy wins
    window_players = pd.DataFrame([_player("c", "2025-11-01")])
    games, players = history.merge_history(window, window_players, "2025-10-02", path)
    assert [g["game_id"] for g in games] == ["a", "w", "c"]
    assert sorted(players["game_id"]) == ["a", "c"]


def test_the_history_path_follows_project_root(tmp_path, monkeypatch):
    from nba_predictor import config

    target = tmp_path / "data" / "history"
    target.mkdir(parents=True)
    _file(target.parent / "history", [_game("a", "2025-05-01")], [_player("a", "2025-05-01")])
    (target / "h.json.gz").rename(target / "nba_history.json.gz")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    games, _ = history.load_history("2025-10-01")
    assert [g["game_id"] for g in games] == ["a"]
