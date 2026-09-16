def _client_with_overrides(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app
    from nba_predictor.api import deps
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(
        '[{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}]'
    )

    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path

    client = TestClient(app)
    return client, db_path


def test_list_games_requires_date_query_param(tmp_path, monkeypatch):
    client, _ = _client_with_overrides(tmp_path, monkeypatch)
    response = client.get("/games")
    assert response.status_code == 422


def test_list_games_returns_games_for_date_with_null_prediction(tmp_path, monkeypatch):
    client, _ = _client_with_overrides(tmp_path, monkeypatch)
    response = client.get("/games", params={"date": "2026-11-01"})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["game_id"] == "g1"
    assert body[0]["prediction"] is None


def test_list_games_includes_latest_prediction_when_present(tmp_path, monkeypatch):
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-11-01T12:00:00", model_version="v1",
        home_win_prob=0.62, predicted_margin=3.5, predicted_total=224.5,
    )

    response = client.get("/games", params={"date": "2026-11-01"})
    body = response.json()
    assert body[0]["prediction"]["home_win_probability"] == 0.62


def test_get_game_detail_404_for_unknown_game(tmp_path, monkeypatch):
    client, _ = _client_with_overrides(tmp_path, monkeypatch)
    response = client.get("/games/does-not-exist")
    assert response.status_code == 404


def test_get_game_detail_includes_markets(tmp_path, monkeypatch):
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="home", model_probability=0.62,
        market_probability=0.55, edge=0.07, bookmaker="DraftKings", american_odds=-130,
        created_at="2026-11-01T12:00:00",
    )

    response = client.get("/games/g1")
    assert response.status_code == 200
    body = response.json()
    assert body["home_team"] == "BOS"
    assert len(body["markets"]) == 1
    assert body["markets"][0]["market"] == "h2h"


def test_get_game_players_404_for_unknown_game(tmp_path, monkeypatch):
    client, _ = _client_with_overrides(tmp_path, monkeypatch)
    response = client.get("/games/does-not-exist/players")
    assert response.status_code == 404


def test_get_game_players_returns_tracked_props(tmp_path, monkeypatch):
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="203999", stat="points",
        predicted_value=27.5, created_at="2026-11-01T12:00:00",
    )

    response = client.get("/games/g1/players")
    assert response.status_code == 200
    body = response.json()
    assert body[0]["stat"] == "points"
    assert body[0]["predicted_value"] == 27.5
