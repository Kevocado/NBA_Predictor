import json

def _client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app
    from nba_predictor.api import deps
    from nba_predictor import config
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    app.dependency_overrides[deps.get_db_path] = lambda: db_path

    return TestClient(app)


def test_hub_teams_empty_when_no_cache(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.get("/hub/teams")
    assert response.status_code == 200
    assert response.json() == []


def test_hub_players_empty_when_no_cache(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.get("/hub/players")
    assert response.status_code == 200
    assert response.json() == []


def test_hub_rankings_empty_when_no_cache(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    assert client.get("/hub/rankings").json() == []


def test_hub_standings_empty_when_no_cache(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    assert client.get("/hub/standings").json() == []


def test_hub_standings_previous_season_excludes_preseason_and_current_season(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    sched_dir = tmp_path / "cache" / "schedule"
    sched_dir.mkdir(parents=True)
    (sched_dir / "games.json").write_text(json.dumps([
        # Previous season (season start year 2025): regular games count.
        {"game_id": "p1", "game_date": "2025-10-25", "home_team": "BOS", "away_team": "MIA",
         "completed": True, "home_pts": 110, "away_pts": 100, "season_type": 2},
        {"game_id": "p2", "game_date": "2025-11-01", "home_team": "LAL", "away_team": "GSW",
         "completed": True, "home_pts": 95, "away_pts": 105, "season_type": 2},
        # Preseason of the previous season: excluded via is_preseason.
        {"game_id": "pr1", "game_date": "2025-10-02", "home_team": "BOS", "away_team": "PHI",
         "completed": True, "home_pts": 99, "away_pts": 84, "season_type": 1},
        # Current season (season start year 2026): not part of the previous season.
        {"game_id": "c1", "game_date": "2026-10-25", "home_team": "BOS", "away_team": "MIA",
         "completed": True, "home_pts": 110, "away_pts": 100, "season_type": 2},
    ]))
    body = client.get("/hub/standings?season=previous").json()
    by_team = {row["abbreviation"]: row for row in body}
    assert by_team["BOS"]["wins"] == 1 and by_team["BOS"]["losses"] == 0
    assert by_team["MIA"]["wins"] == 0 and by_team["MIA"]["losses"] == 1
    assert by_team["LAL"]["wins"] == 0 and by_team["LAL"]["losses"] == 1
    assert by_team["GSW"]["wins"] == 1 and by_team["GSW"]["losses"] == 0
    # Preseason and current-season teams do not appear.
    assert "PHI" not in by_team


def test_hub_standings_previous_season_empty_when_no_previous_season(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    sched_dir = tmp_path / "cache" / "schedule"
    sched_dir.mkdir(parents=True)
    (sched_dir / "games.json").write_text(json.dumps([
        # Only current season (season start year 2026): no previous season.
        {"game_id": "c1", "game_date": "2026-10-25", "home_team": "BOS", "away_team": "MIA",
         "completed": True, "home_pts": 110, "away_pts": 100, "season_type": 2},
    ]))
    # No previous season in the cache: the control stays hidden.
    assert client.get("/hub/standings?season=previous").json() == []


def test_hub_track_record_empty_when_no_predictions(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.get("/hub/track-record")
    assert response.status_code == 200
    assert response.json() == []


def test_hub_vs_market_empty_comparison_still_explains_itself(tmp_path, monkeypatch):
    """Nothing compared is a state, not an error: the block comes back with
    zero counts and the method sentences, so the page can say what it would
    show instead of showing a hole."""
    client = _client(tmp_path, monkeypatch)
    response = client.get("/hub/vs-market")
    assert response.status_code == 200
    body = response.json()
    assert body["market"] == "h2h"
    assert body["n"] == 0
    assert body["weekly"] == []
    assert body["disagreement_hit_rate"] is None
    assert "not_a_profit_claim" in body["method"]


def test_calibration_empty_when_no_predictions(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.get("/calibration")
    assert response.status_code == 200
    assert response.json() == []


def test_player_props_manifest_404_without_retrain(tmp_path, monkeypatch):
    from nba_predictor.api import deps
    from nba_predictor.api.app import app

    client = _client(tmp_path, monkeypatch)
    # get_models_dir points at the repo's real models/ (which has a manifest);
    # point it at an empty dir so "without retrain" is actually exercised.
    app.dependency_overrides[deps.get_models_dir] = lambda: tmp_path / "models"
    try:
        assert client.get("/player-props-manifest").status_code == 404
    finally:
        del app.dependency_overrides[deps.get_models_dir]
