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


def test_get_game_players_uses_real_player_name_from_hub_cache(tmp_path, monkeypatch):
    import json

    from nba_predictor.tracking import store
    from nba_predictor import config

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="203999", stat="points",
        predicted_value=27.5, created_at="2026-11-01T12:00:00",
    )

    hub_dir = tmp_path / "data" / "cache" / "hub"
    hub_dir.mkdir(parents=True)
    (hub_dir / "players.json").write_text(json.dumps([{"player_id": "203999", "player_name": "Nikola Jokic"}]))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")

    response = client.get("/games/g1/players")
    assert response.status_code == 200
    assert response.json()[0]["player_name"] == "Nikola Jokic"


def test_get_game_players_falls_back_to_id_when_name_unknown(tmp_path, monkeypatch):
    from nba_predictor.tracking import store
    from nba_predictor import config

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="203999", stat="points",
        predicted_value=27.5, created_at="2026-11-01T12:00:00",
    )
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "no-such-data-dir")

    response = client.get("/games/g1/players")
    assert response.json()[0]["player_name"] == "203999"


def _client_with_week_schedule(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import json

    from nba_predictor.api.app import app
    from nba_predictor.api import deps
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(
        json.dumps(
            [
                {"game_id": "g1", "game_date": "2026-02-16", "home_team": "BOS", "away_team": "MIA"},
                {"game_id": "g2", "game_date": "2026-02-19", "home_team": "LAL", "away_team": "GSW"},
                {"game_id": "g3", "game_date": "2026-02-23", "home_team": "DEN", "away_team": "PHX"},
            ]
        )
    )

    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path

    return TestClient(app)


def test_list_games_for_week_requires_start_param(tmp_path, monkeypatch):
    client = _client_with_week_schedule(tmp_path, monkeypatch)
    response = client.get("/games/week")
    assert response.status_code == 422


def test_list_games_for_week_returns_only_games_in_window(tmp_path, monkeypatch):
    client = _client_with_week_schedule(tmp_path, monkeypatch)
    response = client.get("/games/week", params={"start": "2026-02-16"})

    assert response.status_code == 200
    body = response.json()
    assert [g["game_id"] for g in body] == ["g1", "g2"]


def test_season_first_week_uses_the_injected_today(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import json

    from nba_predictor.api.app import app
    from nba_predictor.api import deps

    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(json.dumps([
        {"game_id": "g0", "game_date": "2025-10-21", "home_team": "BOS", "away_team": "MIA", "completed": True, "home_pts": 110, "away_pts": 100},
        {"game_id": "g1", "game_date": "2026-10-21", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None},
    ]))
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    app.dependency_overrides[deps.get_today] = lambda: "2026-10-25"

    client = TestClient(app)
    response = client.get("/season/first-week")

    assert response.status_code == 200
    from nba_predictor.services.schedule_repository import monday_of
    assert response.json() == {"first_week_start": monday_of("2026-10-25")}


def test_get_game_players_includes_actual_value_when_settled(tmp_path, monkeypatch):
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="203999", stat="points",
        predicted_value=27.5, created_at="2026-11-01T12:00:00",
    )
    store.insert_player_outcome(
        db_path, game_id="g1", player_id="203999", stat="points",
        actual_value=24.0, recorded_at="2026-11-01T22:00:00",
    )

    response = client.get("/games/g1/players")
    assert response.status_code == 200
    assert response.json()[0]["actual_value"] == 24.0


def test_get_game_players_actual_value_is_null_when_not_settled(tmp_path, monkeypatch):
    from nba_predictor.tracking import store

    client, db_path = _client_with_overrides(tmp_path, monkeypatch)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="203999", stat="points",
        predicted_value=27.5, created_at="2026-11-01T12:00:00",
    )

    response = client.get("/games/g1/players")
    assert response.json()[0]["actual_value"] is None


def _client_with_final(tmp_path):
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import app
    from nba_predictor.api import deps
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(
        '[{"game_id": "g1", "game_date": "2026-03-01", "tip_off": "2026-03-02T00:30Z", "home_team": "BOS",'
        ' "away_team": "MIA", "completed": true, "home_pts": 110, "away_pts": 100}]'
    )
    app.dependency_overrides[deps.get_db_path] = lambda: db_path
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    return TestClient(app), db_path


def test_final_game_shows_the_pre_tip_pick_not_a_later_rebuild(tmp_path):
    from nba_predictor.tracking import store

    client, db_path = _client_with_final(tmp_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-03-01T10:00:00+00:00", model_version="v1",
        home_win_prob=0.7, predicted_margin=5.0, predicted_total=220.0,
    )
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-09-20T08:00:00+00:00", model_version="v2",
        home_win_prob=0.3, predicted_margin=-2.0, predicted_total=220.0,
    )

    game = client.get("/games", params={"date": "2026-03-01"}).json()[0]

    assert game["prediction"]["home_win_probability"] == 0.7
    assert game["rebuilt"] is False
    assert game["tip_off"] == "2026-03-02T00:30Z"


def test_final_game_with_only_a_backtest_pick_is_marked_rebuilt(tmp_path):
    from nba_predictor.tracking import store

    client, db_path = _client_with_final(tmp_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-09-20T08:00:00+00:00", model_version="v2",
        home_win_prob=0.3, predicted_margin=-2.0, predicted_total=220.0,
    )

    game = client.get("/games/week", params={"start": "2026-02-23"}).json()[0]
    detail = client.get("/games/g1").json()

    assert game["rebuilt"] is True
    assert game["prediction"]["home_win_probability"] == 0.3
    assert detail["rebuilt"] is True
