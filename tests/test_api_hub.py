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


def test_hub_track_record_empty_when_no_predictions(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.get("/hub/track-record")
    assert response.status_code == 200
    assert response.json() == []


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
