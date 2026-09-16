def test_load_hub_cache_returns_empty_list_when_missing(tmp_path):
    from nba_predictor.services.hub_service import load_hub_cache

    assert load_hub_cache(tmp_path / "missing.json") == []


def test_load_hub_cache_reads_json_array(tmp_path):
    import json

    from nba_predictor.services.hub_service import load_hub_cache

    path = tmp_path / "teams.json"
    path.write_text(json.dumps([{"abbreviation": "BOS", "points_per_game": 118.2}]))

    assert load_hub_cache(path) == [{"abbreviation": "BOS", "points_per_game": 118.2}]


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
