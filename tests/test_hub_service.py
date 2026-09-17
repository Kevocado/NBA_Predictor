def test_load_hub_cache_returns_empty_list_when_missing(tmp_path):
    from nba_predictor.services.hub_service import load_hub_cache

    assert load_hub_cache(tmp_path / "missing.json") == []


def test_load_hub_cache_reads_json_array(tmp_path):
    import json

    from nba_predictor.services.hub_service import load_hub_cache

    path = tmp_path / "teams.json"
    path.write_text(json.dumps([{"abbreviation": "BOS", "points_per_game": 118.2}]))

    assert load_hub_cache(path) == [{"abbreviation": "BOS", "points_per_game": 118.2}]


def test_load_player_name_map_builds_id_to_name_dict(tmp_path):
    import json

    from nba_predictor.services.hub_service import load_player_name_map

    path = tmp_path / "players.json"
    path.write_text(json.dumps([
        {"player_id": "4251", "player_name": "Paul George"},
        {"player_id": "203999", "player_name": "Nikola Jokic"},
    ]))

    assert load_player_name_map(path) == {"4251": "Paul George", "203999": "Nikola Jokic"}


def test_load_player_name_map_missing_file_returns_empty_dict(tmp_path):
    from nba_predictor.services.hub_service import load_player_name_map

    assert load_player_name_map(tmp_path / "does-not-exist.json") == {}


def test_compute_track_record_groups_by_market(tmp_path):
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="home", model_probability=0.6,
        market_probability=0.5, edge=0.1, bookmaker="DraftKings", american_odds=-130,
        created_at="2026-11-01T12:00:00",
    )
    store.insert_market_prediction(
        db_path, game_id="g2", market="h2h", selection="away", model_probability=0.55,
        market_probability=0.5, edge=0.05, bookmaker="DraftKings", american_odds=120,
        created_at="2026-11-02T12:00:00",
    )
    store.insert_market_prediction(
        db_path, game_id="g1", market="spread", selection="home", model_probability=0.52,
        market_probability=0.5, edge=0.02, bookmaker="DraftKings", american_odds=-110,
        created_at="2026-11-01T12:00:00",
    )

    records = compute_track_record(db_path)
    by_market = {r.market: r for r in records}

    assert by_market["h2h"].total_predictions == 2
    assert by_market["spread"].total_predictions == 1


def test_compute_track_record_empty_db_returns_empty_list(tmp_path):
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    assert compute_track_record(db_path) == []


def _completed_game(game_id, home, away, home_pts, away_pts):
    return {
        "game_id": game_id, "game_date": "2026-03-01", "home_team": home, "away_team": away,
        "completed": True, "home_pts": home_pts, "away_pts": away_pts,
    }


def test_compute_track_record_settles_game_outcome_against_real_results(tmp_path):
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    # Correct call: predicted BOS (home) to win, BOS did win.
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-03-01T00:00:00", model_version="v1",
        home_win_prob=0.7, predicted_margin=5.0, predicted_total=220.0,
    )
    # Incorrect call: predicted LAL (home) to win, LAL lost.
    store.insert_prediction(
        db_path, game_id="g2", created_at="2026-03-01T00:00:00", model_version="v1",
        home_win_prob=0.6, predicted_margin=3.0, predicted_total=215.0,
    )

    schedule = [
        _completed_game("g1", "BOS", "MIA", 110, 100),
        _completed_game("g2", "LAL", "GSW", 95, 105),
    ]

    records = compute_track_record(db_path, schedule)
    game_outcome = next(r for r in records if r.market == "game_outcome")

    assert game_outcome.total_predictions == 2
    assert game_outcome.correct_predictions == 1
    assert game_outcome.hit_rate == 0.5


def test_compute_track_record_ignores_predictions_for_incomplete_games(tmp_path):
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-03-01T00:00:00", model_version="v1",
        home_win_prob=0.6, predicted_margin=3.0, predicted_total=220.0,
    )
    schedule = [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None}]

    records = compute_track_record(db_path, schedule)

    assert not any(r.market == "game_outcome" for r in records)


def test_compute_track_record_settles_h2h_market_predictions(tmp_path):
    from nba_predictor.services.hub_service import compute_track_record
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="BOS", model_probability=0.65,
        market_probability=0.55, edge=0.1, bookmaker="DraftKings", american_odds=-140,
        created_at="2026-03-01T00:00:00",
    )
    schedule = [_completed_game("g1", "BOS", "MIA", 110, 100)]

    records = compute_track_record(db_path, schedule)
    h2h = next(r for r in records if r.market == "h2h")

    assert h2h.total_predictions == 1
    assert h2h.correct_predictions == 1
    assert h2h.hit_rate == 1.0
