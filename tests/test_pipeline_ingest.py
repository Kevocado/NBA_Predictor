from unittest.mock import patch

import pytest


def _sample_completed_games():
    return [
        {
            "game_id": "1", "game_date": "2026-03-01", "home_team": "BOS", "away_team": "MIA",
            "completed": True, "home_pts": 110, "away_pts": 100,
            "home_fgm": 40, "home_fga": 88, "home_fg3m": 12, "home_tov": 11, "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
            "away_fgm": 36, "away_fga": 90, "away_fg3m": 8, "away_tov": 14, "away_oreb": 10, "away_dreb": 28, "away_fta": 18,
        },
        {
            "game_id": "2", "game_date": "2026-03-03", "home_team": "MIA", "away_team": "BOS",
            "completed": True, "home_pts": 95, "away_pts": 105,
            "home_fgm": 34, "home_fga": 89, "home_fg3m": 9, "home_tov": 13, "home_oreb": 8, "home_dreb": 27, "home_fta": 15,
            "away_fgm": 39, "away_fga": 87, "away_fg3m": 11, "away_tov": 10, "away_oreb": 11, "away_dreb": 30, "away_fta": 19,
        },
    ]


def test_daterange_inclusive_of_both_ends():
    from nba_predictor.pipeline.ingest import _daterange

    days = _daterange("2026-03-01", "2026-03-03")
    assert days == ["2026-03-01", "2026-03-02", "2026-03-03"]


@patch("nba_predictor.pipeline.ingest.espn.get_scoreboard")
def test_fetch_schedule_range_calls_scoreboard_per_day(mock_scoreboard):
    from nba_predictor.pipeline.ingest import fetch_schedule_range

    mock_scoreboard.side_effect = lambda d: [{"game_id": d, "game_date": d}]

    games = fetch_schedule_range("2026-03-01", "2026-03-02")

    assert mock_scoreboard.call_count == 2
    assert [g["game_id"] for g in games] == ["2026-03-01", "2026-03-02"]


@patch("nba_predictor.pipeline.ingest.espn.get_boxscore")
def test_enrich_with_boxscores_adds_flat_fields_for_completed_games(mock_boxscore):
    from nba_predictor.pipeline.ingest import enrich_with_boxscores

    mock_boxscore.return_value = {
        "BOS": {"fgm": 40, "fga": 88, "fg3m": 12, "tov": 11, "oreb": 9, "dreb": 32, "fta": 20},
        "MIA": {"fgm": 36, "fga": 90, "fg3m": 8, "tov": 14, "oreb": 10, "dreb": 28, "fta": 18},
    }
    games = [{"game_id": "1", "game_date": "2026-03-01", "home_team": "BOS", "away_team": "MIA", "completed": True, "home_pts": 110, "away_pts": 100}]

    enriched = enrich_with_boxscores(games)

    assert enriched[0]["home_fgm"] == 40
    assert enriched[0]["away_fga"] == 90


def test_enrich_with_boxscores_skips_upcoming_games():
    from nba_predictor.pipeline.ingest import enrich_with_boxscores

    games = [{"game_id": "1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None}]

    enriched = enrich_with_boxscores(games)

    assert "home_fgm" not in enriched[0]


def test_to_schedule_cache_returns_minimal_shape():
    from nba_predictor.pipeline.ingest import to_schedule_cache

    games = [{"game_id": "1", "game_date": "2026-03-01", "home_team": "BOS", "away_team": "MIA", "completed": True, "home_pts": 110, "away_pts": 100, "home_fgm": 40}]

    cache = to_schedule_cache(games)

    assert cache == [{"game_id": "1", "game_date": "2026-03-01", "home_team": "BOS", "away_team": "MIA"}]


def test_to_training_frame_only_includes_completed_games_with_box_scores():
    from nba_predictor.pipeline.ingest import to_training_frame

    games = _sample_completed_games() + [
        {"game_id": "3", "game_date": "2026-03-05", "home_team": "LAL", "away_team": "GSW", "completed": False, "home_pts": None, "away_pts": None},
    ]

    df = to_training_frame(games)

    assert len(df) == 2
    assert set(df["game_id"]) == {"1", "2"}
    assert df.loc[df["game_id"] == "1", "home_win"].iloc[0] == 1


def test_compute_team_hub_aggregates_real_numbers():
    from nba_predictor.pipeline.ingest import compute_team_hub

    rows = compute_team_hub(_sample_completed_games())
    by_team = {r["abbreviation"]: r for r in rows}

    assert by_team["BOS"]["wins"] == 2
    assert by_team["BOS"]["losses"] == 0
    assert by_team["BOS"]["points_per_game"] == pytest.approx((110 + 105) / 2, abs=0.1)
    assert by_team["MIA"]["wins"] == 0
    assert by_team["MIA"]["losses"] == 2


def test_compute_power_rankings_ranks_winner_higher():
    from nba_predictor.pipeline.ingest import compute_power_rankings

    rows = compute_power_rankings(_sample_completed_games())
    by_team = {r["abbreviation"]: r for r in rows}

    assert by_team["BOS"]["power_rating"] > by_team["MIA"]["power_rating"]
    assert by_team["BOS"]["rank"] == 1


def test_score_and_store_predictions_stores_real_model_output(tmp_path):
    import numpy as np
    import pandas as pd

    from nba_predictor.pipeline.ingest import score_and_store_predictions
    from nba_predictor.pipeline.retrain import run_retrain_pipeline
    from nba_predictor.tracking import store

    rng = np.random.default_rng(3)
    teams = ["BOS", "MIA", "LAL", "GSW"]
    dates = pd.date_range("2026-02-01", periods=50).astype(str)
    rows = []
    for i, game_date in enumerate(dates):
        home, away = teams[i % 4], teams[(i + 1) % 4]
        rows.append(
            {
                "game_id": f"g{i}", "game_date": game_date, "home_team": home, "away_team": away,
                "home_pts": 110, "away_pts": 105,
                "home_fgm": 40, "home_fga": 88, "home_fg3m": 12, "home_tov": 11, "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
                "away_fgm": 38, "away_fga": 90, "away_fg3m": 10, "away_tov": 13, "away_oreb": 10, "away_dreb": 30, "away_fta": 18,
                "home_win": int(rng.random() > 0.4),
            }
        )
    games_df = pd.DataFrame(rows)

    models_dir = tmp_path / "models"
    run_retrain_pipeline(games_df, models_dir, model_version="v-test", trained_at="2026-03-01T00:00:00")

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    stored_count = score_and_store_predictions(games_df, models_dir, db_path, model_version="v-test")

    assert stored_count > 0
    row = store.get_predictions_for_game(db_path, "g10")
    if row:
        assert 0.0 <= row[0]["home_win_prob"] <= 1.0
        assert row[0]["model_version"] == "v-test"


def test_compute_standings_assigns_seeds_by_win_pct():
    from nba_predictor.pipeline.ingest import compute_standings

    rows = compute_standings(_sample_completed_games())
    east = [r for r in rows if r["conference"] == "East"]

    assert east[0]["abbreviation"] == "BOS"
    assert east[0]["seed"] == 1
    assert east[0]["win_pct"] == 1.0
    assert east[1]["abbreviation"] == "MIA"
    assert east[1]["games_back"] == 2.0
