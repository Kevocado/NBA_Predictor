import json


def _client(tmp_path, monkeypatch, public_mode: bool):
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app
    from nba_predictor.api import deps
    from nba_predictor import config

    monkeypatch.setattr(config, "PUBLIC_MODE", public_mode)
    app.dependency_overrides[deps.get_models_dir] = lambda: tmp_path / "models"
    app.dependency_overrides[deps.get_training_games_path] = lambda: tmp_path / "training_games.json"

    return TestClient(app)


def test_retrain_404_when_public_mode(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, public_mode=True)
    response = client.post("/retrain")
    assert response.status_code == 404


def test_refresh_odds_404_when_public_mode(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, public_mode=True)
    response = client.post("/refresh-odds")
    assert response.status_code == 404


def test_retrain_400_when_no_training_cache(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, public_mode=False)
    response = client.post("/retrain")
    assert response.status_code == 400


def test_retrain_succeeds_when_training_cache_present(tmp_path, monkeypatch):
    import pandas as pd

    client = _client(tmp_path, monkeypatch, public_mode=False)

    teams = ["BOS", "MIA", "LAL", "GSW"]
    dates = pd.date_range("2026-10-21", periods=60).astype(str)
    rows = []
    for i, game_date in enumerate(dates):
        home, away = teams[i % 4], teams[(i + 1) % 4]
        rows.append(
            {
                "game_id": f"g{i}", "game_date": str(game_date), "home_team": home, "away_team": away,
                "home_pts": 110, "away_pts": 105,
                "home_fgm": 40, "home_fga": 88, "home_fg3m": 12, "home_tov": 11, "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
                "away_fgm": 38, "away_fga": 90, "away_fg3m": 10, "away_tov": 13, "away_oreb": 10, "away_dreb": 30, "away_fta": 18,
                "home_win": i % 2,
            }
        )
    (tmp_path / "training_games.json").write_text(json.dumps(rows))

    response = client.post("/retrain")
    assert response.status_code == 200
    assert response.json()["models"] == ["win_probability", "margin", "total"]


def test_refresh_odds_returns_202_with_zero_stored_when_no_upcoming_games(tmp_path, monkeypatch):
    from nba_predictor.api import deps
    from nba_predictor.api.app import app
    from nba_predictor.tracking import store

    client = _client(tmp_path, monkeypatch, public_mode=False)

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    schedule_path = tmp_path / "games.json"
    schedule_path.write_text("[]")
    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path

    response = client.post("/refresh-odds")

    assert response.status_code == 202
    assert response.json() == {"status": "ok", "market_predictions_stored": 0}


def test_get_manifest_404_when_absent(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, public_mode=False)
    response = client.get("/manifest")
    assert response.status_code == 404


def test_get_manifest_returns_file_contents(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, public_mode=False)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "manifest.json").write_text(json.dumps({"model_version": "v1", "trained_at": "t", "models": [], "metrics": {}}))

    response = client.get("/manifest")
    assert response.status_code == 200
    assert response.json()["model_version"] == "v1"
