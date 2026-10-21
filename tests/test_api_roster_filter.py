"""An upcoming game shows only players on the two teams' CURRENT rosters; a finished game keeps whoever played."""
import json

import pytest
from fastapi.testclient import TestClient

from nba_predictor.api import deps
from nba_predictor.api.app import app
from nba_predictor.tracking import store

SCHEDULE = [
    {"game_id": "up", "game_date": "2026-10-21", "home_team": "DET", "away_team": "BOS", "tip_off": "2026-10-21T23:00Z", "completed": False},
    {"game_id": "done", "game_date": "2026-04-10", "home_team": "DET", "away_team": "BOS", "tip_off": "2026-04-10T23:00Z", "completed": True,
     "home_pts": 100, "away_pts": 99},
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    for gid, when in (("up", "2026-10-20T12:00:00+00:00"), ("done", "2026-04-10T12:00:00+00:00")):
        for pid in ("tatum", "brown"):                      # Brown was scored for BOS by an old run; he is on PHI now
            store.insert_player_prediction(db_path, game_id=gid, player_id=pid, stat="points", predicted_value=20.0, created_at=when, position="G")
    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(json.dumps(SCHEDULE))
    hub = tmp_path / "data" / "cache" / "hub"
    hub.mkdir(parents=True)
    (hub / "players.json").write_text(json.dumps([{"player_id": "tatum", "player_name": "Jayson Tatum"}, {"player_id": "brown", "player_name": "Jaylen Brown"}]))
    from nba_predictor import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    app.dependency_overrides[deps.get_injury_report] = lambda: []
    yield TestClient(app)
    app.dependency_overrides.clear()


def _record_outcome(client, game_id, player_id):
    from nba_predictor import config  # noqa: F401  (fixture already pointed DATA_DIR at tmp)
    db_path = app.dependency_overrides[deps.get_db_path]()
    store.insert_player_outcome(db_path, game_id=game_id, player_id=player_id, stat="points", actual_value=22.0, recorded_at="2026-04-11T03:00:00+00:00")


def _names(client, game_id):
    return sorted({row["player_name"] for row in client.get(f"/games/{game_id}/players").json()})


def test_a_traded_players_stale_rows_are_hidden_on_an_upcoming_game(client):
    app.dependency_overrides[deps.get_roster_index] = lambda: {"BOS": {"tatum"}, "DET": set(), "PHI": {"brown"}}
    assert _names(client, "up") == ["Jayson Tatum"]


def test_a_finished_game_keeps_whoever_played_in_it(client):
    app.dependency_overrides[deps.get_roster_index] = lambda: {"BOS": {"tatum"}, "DET": set(), "PHI": {"brown"}}
    assert _names(client, "done") == ["Jaylen Brown", "Jayson Tatum"]


def test_unknown_rosters_filter_nothing(client):
    app.dependency_overrides[deps.get_roster_index] = lambda: None
    assert _names(client, "up") == ["Jaylen Brown", "Jayson Tatum"]


def test_a_finished_game_with_results_shows_only_the_players_who_played(client):
    app.dependency_overrides[deps.get_roster_index] = lambda: None
    _record_outcome(client, "done", "tatum")                      # Brown never appeared
    assert _names(client, "done") == ["Jayson Tatum"]


def test_an_upcoming_projection_carries_the_team_he_is_on_now(client):
    app.dependency_overrides[deps.get_roster_index] = lambda: {"BOS": {"tatum"}, "DET": {"brown"}, "PHI": set()}
    rows = client.get("/games/up/players").json()
    assert {row["player_name"]: row["team"] for row in rows} == {"Jayson Tatum": "BOS", "Jaylen Brown": "DET"}


def test_a_finished_game_projection_has_no_team_claim(client):
    app.dependency_overrides[deps.get_roster_index] = lambda: {"BOS": {"tatum"}, "DET": set()}
    assert {row["team"] for row in client.get("/games/done/players").json()} == {None}
