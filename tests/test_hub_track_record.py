# tests/test_hub_track_record.py
from pathlib import Path

from nba_predictor.tracking import store
from nba_predictor.services import hub_service

TS = "2026-01-01T00:00:00+00:00"


def _db(tmp_path) -> Path:
    db_path = tmp_path / "t.db"
    store.init_db(db_path)
    return db_path


def _schedule():
    return [
        {"game_id": "g1", "completed": True, "home_team": "LAL", "away_team": "BOS",
         "home_pts": 110, "away_pts": 100, "game_date": "2026-01-05"},  # LAL wins
        {"game_id": "g2", "completed": True, "home_team": "LAL", "away_team": "BOS",
         "home_pts": 95, "away_pts": 105, "game_date": "2026-01-06"},   # BOS wins
        {"game_id": "g3", "completed": True, "home_team": "LAL", "away_team": "BOS",
         "home_pts": 120, "away_pts": 100, "game_date": "2026-01-07"},  # LAL wins
    ]


def test_h2h_confidence_buckets(tmp_path):
    db_path = _db(tmp_path)
    for gid, prob, pick in [("g1", 0.55, "LAL"), ("g2", 0.65, "LAL"), ("g3", 0.75, "LAL")]:
        store.insert_market_prediction(
            db_path, game_id=gid, market="h2h", selection=pick,
            model_probability=prob, market_probability=0.5, edge=0.05,
            bookmaker="b", american_odds=-110, created_at=TS,
        )
    rows = {r.market: r for r in hub_service.compute_track_record(db_path, _schedule())}
    buckets = {b.bucket: b for b in rows["h2h"].confidence_buckets}
    # g1 (0.55, LAL won) -> hit; g2 (0.65, BOS won) -> miss; g3 (0.75, LAL won) -> hit
    assert buckets["50-60%"].hit_rate == 1.0
    assert buckets["60-70%"].hit_rate == 0.0
    assert buckets["70%+"].hit_rate == 1.0


def test_player_prop_signed_error_and_position(tmp_path):
    db_path = _db(tmp_path)
    # p1 (G) consistently over-predicted by 4; p2 (C) consistently under by 2
    preds = [("g1", "p1", "points", 24.0, "G"), ("g1", "p2", "points", 8.0, "C")]
    for gid, pid, stat, pv, pos in preds:
        store.insert_player_prediction(
            db_path, game_id=gid, player_id=pid, stat=stat,
            predicted_value=pv, created_at=TS, position=pos,
        )
    store.insert_player_outcome(db_path, game_id="g1", player_id="p1", stat="points", actual_value=20.0, recorded_at=TS)
    store.insert_player_outcome(db_path, game_id="g1", player_id="p2", stat="points", actual_value=10.0, recorded_at=TS)
    rows = {r.market: r for r in hub_service.compute_track_record(db_path, _schedule())}
    prop = rows["player_props"]
    by_stat = {s.stat: s for s in prop.per_stat}
    assert by_stat["points"].n == 2
    assert by_stat["points"].mae == 3.0            # (|4| + |-2|) / 2
    assert by_stat["points"].mean_signed_error == 1.0  # (4 + -2) / 2, positive = over-prediction
    assert prop.per_position_mae == {"C": 2.0, "G": 4.0}


def test_null_position_grouped_as_unknown(tmp_path):
    db_path = _db(tmp_path)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="p9", stat="points",
        predicted_value=15.0, created_at=TS, position=None,
    )
    store.insert_player_outcome(db_path, game_id="g1", player_id="p9", stat="points", actual_value=10.0, recorded_at=TS)
    rows = {r.market: r for r in hub_service.compute_track_record(db_path, _schedule())}
    assert rows["player_props"].per_position_mae == {"Unknown": 5.0}


def test_no_prop_data_no_prop_row(tmp_path):
    db_path = _db(tmp_path)
    rows = {r.market: r for r in hub_service.compute_track_record(db_path, _schedule())}
    assert "player_props" not in rows
