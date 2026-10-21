"""'Key out' on an upcoming game: reported-Out players who normally play real minutes, from the cached rosters."""
import json

import pytest
from fastapi.testclient import TestClient

from nba_predictor.api import deps
from nba_predictor.api.app import app
from nba_predictor.api.key_out import key_out_for_game
from nba_predictor.tracking import store

ROSTERS = {
    "BOS": [
        {"player_id": "tatum", "player_name": "Jayson Tatum", "position": "SF", "injury_status": "Out"},
        {"player_id": "bench", "player_name": "Bench Guy", "position": "G", "injury_status": "Out"},
        {"player_id": "white", "player_name": "Derrick White", "position": "G", "injury_status": "Day-To-Day"},
    ],
    "DET": [{"player_id": "cade", "player_name": "Cade Cunningham", "position": "PG", "injury_status": "Out"}],
}
MINUTES = {"tatum": 36.0, "bench": 6.0, "white": 32.0, "cade": 34.0}


def test_only_out_players_with_real_minutes_count():
    game = {"home_team": "DET", "away_team": "BOS", "completed": False}
    got = key_out_for_game(game, ROSTERS, MINUTES)
    assert [(k["team"], k["player_name"]) for k in got] == [("DET", "Cade Cunningham"), ("BOS", "Jayson Tatum")]


def test_a_finished_game_or_unknown_rosters_say_nothing():
    assert key_out_for_game({"home_team": "DET", "away_team": "BOS", "completed": True}, ROSTERS, MINUTES) == []
    assert key_out_for_game({"home_team": "DET", "away_team": "BOS", "completed": False}, None, MINUTES) == []


def test_at_most_two_per_team_most_minutes_first():
    rosters = {"BOS": [{"player_id": f"p{i}", "player_name": f"P{i}", "position": "G", "injury_status": "Out"} for i in range(4)], "DET": []}
    minutes = {"p0": 21.0, "p1": 30.0, "p2": 25.0, "p3": 28.0}
    got = key_out_for_game({"home_team": "DET", "away_team": "BOS", "completed": False}, rosters, minutes)
    assert [k["player_name"] for k in got] == ["P1", "P3"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    db = tmp_path / "t.db"; store.init_db(db)
    sched = tmp_path / "games.json"
    sched.write_text(json.dumps([{"game_id": "g", "game_date": "2026-10-21", "home_team": "DET", "away_team": "BOS", "tip_off": "2026-10-21T23:00Z", "completed": False}]))
    hub = tmp_path / "data" / "cache" / "hub"; hub.mkdir(parents=True)
    (hub / "players.json").write_text(json.dumps([{"player_id": k, "player_name": k, "minutes_per_game": v} for k, v in MINUTES.items()]))
    from nba_predictor import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    app.dependency_overrides[deps.get_db_path] = lambda: db
    app.dependency_overrides[deps.get_schedule_path] = lambda: sched
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_the_week_endpoint_carries_key_out(client):
    app.dependency_overrides[deps.get_roster_cache] = lambda: ROSTERS
    game = client.get("/games/week", params={"start": "2026-10-19"}).json()[0]
    assert [(k["team"], k["player_name"]) for k in game["key_out"]] == [("DET", "Cade Cunningham"), ("BOS", "Jayson Tatum")]


def test_without_a_roster_cache_the_field_is_empty(client):
    app.dependency_overrides[deps.get_roster_cache] = lambda: None
    assert client.get("/games/week", params={"start": "2026-10-19"}).json()[0]["key_out"] == []
