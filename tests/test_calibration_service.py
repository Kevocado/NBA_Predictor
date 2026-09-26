def _completed_game(game_id, home, away, home_pts, away_pts):
    return {
        "game_id": game_id, "game_date": "2026-03-01", "home_team": home, "away_team": away,
        "completed": True, "home_pts": home_pts, "away_pts": away_pts,
    }


def test_compute_model_calibration_returns_bins_from_real_predictions(tmp_path):
    from nba_predictor.services.calibration_service import compute_model_calibration
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    for i in range(20):
        store.insert_prediction(
            db_path, game_id=f"g{i}", created_at="2026-03-01T00:00:00", model_version="v1",
            home_win_prob=0.9 if i < 10 else 0.1, predicted_margin=5.0, predicted_total=220.0,
        )

    schedule = [_completed_game(f"g{i}", "BOS", "MIA", 110 if i < 9 else 100, 100 if i < 9 else 110) for i in range(20)]

    bins = compute_model_calibration(db_path, schedule, n_bins=10)

    assert len(bins) > 0
    high_bin = next(b for b in bins if b["bin_start"] >= 0.8)
    assert high_bin["count"] == 10


def test_compute_model_calibration_empty_db_returns_empty_list(tmp_path):
    from nba_predictor.services.calibration_service import compute_model_calibration
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    assert compute_model_calibration(db_path, []) == []


def test_compute_model_calibration_ignores_incomplete_games(tmp_path):
    from nba_predictor.services.calibration_service import compute_model_calibration
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-03-01T00:00:00", model_version="v1",
        home_win_prob=0.6, predicted_margin=3.0, predicted_total=220.0,
    )
    schedule = [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None}]

    assert compute_model_calibration(db_path, schedule) == []


def test_compute_model_calibration_uses_only_pre_tip_picks(tmp_path):
    from nba_predictor.services.calibration_service import compute_model_calibration
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-03-01T10:00:00+00:00", model_version="v1",
        home_win_prob=0.9, predicted_margin=5.0, predicted_total=220.0,
    )
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-09-20T08:00:00+00:00", model_version="v2",
        home_win_prob=0.1, predicted_margin=-5.0, predicted_total=220.0,
    )
    store.insert_prediction(
        db_path, game_id="g2", created_at="2026-09-20T08:00:00+00:00", model_version="v2",
        home_win_prob=0.1, predicted_margin=-5.0, predicted_total=220.0,
    )
    schedule = [_completed_game("g1", "BOS", "MIA", 110, 100), _completed_game("g2", "BOS", "MIA", 110, 100)]

    bins = compute_model_calibration(db_path, schedule, n_bins=10)

    assert sum(b["count"] for b in bins) == 1
    assert next(b for b in bins if b["count"])["bin_start"] >= 0.8
