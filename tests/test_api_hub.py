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
