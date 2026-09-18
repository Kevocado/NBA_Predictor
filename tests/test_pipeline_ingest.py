from unittest.mock import patch

import pandas as pd
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


def test_to_schedule_cache_includes_scores_and_completion():
    from nba_predictor.pipeline.ingest import to_schedule_cache

    games = [{"game_id": "1", "game_date": "2026-03-01", "home_team": "BOS", "away_team": "MIA", "completed": True, "home_pts": 110, "away_pts": 100, "home_fgm": 40}]

    cache = to_schedule_cache(games)

    assert cache == [
        {
            "game_id": "1", "game_date": "2026-03-01", "home_team": "BOS", "away_team": "MIA",
            "completed": True, "home_pts": 110, "away_pts": 100,
        }
    ]


def test_to_schedule_cache_handles_upcoming_games_without_scores():
    from nba_predictor.pipeline.ingest import to_schedule_cache

    games = [{"game_id": "2", "game_date": "2026-11-01", "home_team": "LAL", "away_team": "GSW", "completed": False}]

    cache = to_schedule_cache(games)

    assert cache[0]["completed"] is False
    assert cache[0]["home_pts"] is None


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


@patch("nba_predictor.pipeline.ingest.espn.get_player_boxscore")
def test_fetch_player_boxscores_only_fetches_completed_games(mock_boxscore):
    from nba_predictor.pipeline.ingest import fetch_player_boxscores

    mock_boxscore.return_value = [{"player_id": "1"}]
    games = [
        {"game_id": "1", "completed": True},
        {"game_id": "2", "completed": False},
    ]

    result = fetch_player_boxscores(games)

    assert list(result.keys()) == ["1"]
    mock_boxscore.assert_called_once_with("1")


def _player_box_row(player_id, name, team, minutes, points, rebounds, assists, fg="4-8", fg3="1-3", ft="2-2"):
    return {
        "player_id": player_id, "player_name": name, "team": team, "position": "G",
        "minutes": minutes, "points": points, "rebounds": rebounds, "assists": assists,
        "fg_made_attempted": fg, "three_made_attempted": fg3, "ft_made_attempted": ft,
    }


def test_compute_player_hub_aggregates_real_per_game_rows():
    from nba_predictor.pipeline.ingest import compute_player_hub

    games = [
        {"game_id": "g1", "game_date": "2026-03-01", "home_team": "BOS", "away_team": "MIA", "home_fga": 90, "away_fga": 85},
        {"game_id": "g2", "game_date": "2026-03-03", "home_team": "MIA", "away_team": "BOS", "home_fga": 88, "away_fga": 92},
    ]
    player_boxscores = {
        "g1": [_player_box_row("p1", "Jayson Tatum", "BOS", 34.0, 30.0, 8.0, 5.0)],
        "g2": [_player_box_row("p1", "Jayson Tatum", "BOS", 36.0, 26.0, 6.0, 7.0)],
    }

    rows = compute_player_hub(games, player_boxscores)

    assert len(rows) == 1
    row = rows[0]
    assert row["player_name"] == "Jayson Tatum"
    assert row["points_per_game"] == pytest.approx(28.0)
    assert row["rebounds_per_game"] == pytest.approx(7.0)
    assert row["assists_per_game"] == pytest.approx(6.0)
    assert row["fg_pct"] == pytest.approx(8 / 16, abs=0.01)
    assert row["minutes_per_game"] == pytest.approx(35.0)
    assert row["rating"] > 0


def test_compute_player_hub_live_form_uses_last_five_games_only():
    from nba_predictor.pipeline.ingest import compute_player_hub

    games = [
        {"game_id": f"g{i}", "game_date": f"2026-03-{i:02d}", "home_team": "BOS", "away_team": "MIA", "home_fga": 90, "away_fga": 85}
        for i in range(1, 8)
    ]
    # First two games: low output. Last five: high output.
    player_boxscores = {}
    for i in range(1, 8):
        points = 10.0 if i <= 2 else 30.0
        player_boxscores[f"g{i}"] = [_player_box_row("p1", "Jayson Tatum", "BOS", 30.0, points, 5.0, 5.0)]

    rows = compute_player_hub(games, player_boxscores)
    row = rows[0]

    # Season average is pulled down by the two low games; form rating (last 5) should be higher.
    assert row["live_form_rating"] > row["rating"]


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


def test_to_scoring_frame_includes_upcoming_games_with_null_fields():
    from nba_predictor.pipeline.ingest import to_scoring_frame

    games = _sample_completed_games() + [
        {"game_id": "3", "game_date": "2026-03-05", "home_team": "LAL", "away_team": "GSW", "completed": False, "home_pts": None, "away_pts": None},
    ]

    df = to_scoring_frame(games)

    assert len(df) == 3
    upcoming_row = df[df["game_id"] == "3"].iloc[0]
    assert pd.isna(upcoming_row["home_win"])
    assert pd.isna(upcoming_row["home_fgm"])
    completed_row = df[df["game_id"] == "1"].iloc[0]
    assert completed_row["home_win"] == 1


def test_score_upcoming_games_stores_predictions_for_not_yet_played_games(tmp_path):
    import numpy as np
    import pandas as pd

    from nba_predictor.pipeline.ingest import score_upcoming_games
    from nba_predictor.pipeline.retrain import run_retrain_pipeline
    from nba_predictor.tracking import store

    rng = np.random.default_rng(3)
    teams = ["BOS", "MIA", "LAL", "GSW"]
    dates = pd.date_range("2026-02-01", periods=50).astype(str)
    completed_games = []
    for i, game_date in enumerate(dates):
        home, away = teams[i % 4], teams[(i + 1) % 4]
        completed_games.append(
            {
                "game_id": f"g{i}", "game_date": game_date, "home_team": home, "away_team": away,
                "completed": True, "home_pts": 110, "away_pts": 105,
                "home_fgm": 40, "home_fga": 88, "home_fg3m": 12, "home_tov": 11, "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
                "away_fgm": 38, "away_fga": 90, "away_fg3m": 10, "away_tov": 13, "away_oreb": 10, "away_dreb": 30, "away_fta": 18,
            }
        )
    upcoming_game = {
        "game_id": "g-upcoming", "game_date": "2026-04-01", "home_team": "BOS", "away_team": "MIA",
        "completed": False, "home_pts": None, "away_pts": None,
    }
    games = completed_games + [upcoming_game]

    from nba_predictor.pipeline.ingest import to_training_frame
    train_df = to_training_frame(completed_games)
    for i, row in train_df.iterrows():
        train_df.at[i, "home_win"] = int(rng.random() > 0.4)

    models_dir = tmp_path / "models"
    run_retrain_pipeline(train_df, models_dir, model_version="v-test", trained_at="2026-03-01T00:00:00")

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    stored = score_upcoming_games(games, models_dir, db_path, model_version="v-test")

    assert stored == 1
    rows = store.get_predictions_for_game(db_path, "g-upcoming")
    assert len(rows) == 1
    assert 0.0 <= rows[0]["home_win_prob"] <= 1.0


def test_score_upcoming_games_returns_zero_when_nothing_upcoming(tmp_path):
    from nba_predictor.pipeline.ingest import score_upcoming_games
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    stored = score_upcoming_games(_sample_completed_games(), tmp_path / "models", db_path, model_version="v-test")

    assert stored == 0


def test_to_player_scoring_frame_adds_a_row_per_estimated_roster_player():
    from nba_predictor.pipeline.ingest import to_player_scoring_frame

    games = [
        {"game_id": "g1", "game_date": "2026-10-21", "home_team": "BOS", "away_team": "MIA", "completed": True, "home_pts": 110, "away_pts": 100},
        {"game_id": "g2", "game_date": "2026-10-25", "home_team": "BOS", "away_team": "LAL", "completed": False, "home_pts": None, "away_pts": None},
    ]
    player_boxscores = {
        "g1": [{"player_id": "p1", "player_name": "Jayson Tatum", "team": "BOS", "position": "F", "minutes": 34.0, "points": 28.0, "rebounds": 7.0, "assists": 5.0, "fg_made_attempted": "10-19", "three_made_attempted": "3-7", "ft_made_attempted": "5-6"}]
    }

    df = to_player_scoring_frame(games, player_boxscores)

    upcoming_rows = df[df["game_id"] == "g2"]
    assert len(upcoming_rows) == 1
    assert upcoming_rows.iloc[0]["player_id"] == "p1"
    assert pd.isna(upcoming_rows.iloc[0]["points"])


def test_to_player_scoring_frame_only_includes_players_from_the_named_team():
    from nba_predictor.pipeline.ingest import to_player_scoring_frame

    games = [
        {"game_id": "g1", "game_date": "2026-10-21", "home_team": "BOS", "away_team": "MIA", "completed": True, "home_pts": 110, "away_pts": 100},
        {"game_id": "g2", "game_date": "2026-10-25", "home_team": "BOS", "away_team": "LAL", "completed": False, "home_pts": None, "away_pts": None},
    ]
    player_boxscores = {
        "g1": [
            {"player_id": "p1", "player_name": "BOS Player", "team": "BOS", "position": "F", "minutes": 34.0, "points": 28.0, "rebounds": 7.0, "assists": 5.0, "fg_made_attempted": "10-19", "three_made_attempted": "3-7", "ft_made_attempted": "5-6"},
            {"player_id": "p2", "player_name": "MIA Player", "team": "MIA", "position": "G", "minutes": 30.0, "points": 18.0, "rebounds": 3.0, "assists": 6.0, "fg_made_attempted": "7-15", "three_made_attempted": "2-5", "ft_made_attempted": "2-2"},
        ]
    }

    df = to_player_scoring_frame(games, player_boxscores)

    upcoming_players = set(df[df["game_id"] == "g2"]["player_id"])
    assert "p2" not in upcoming_players


def test_store_player_outcomes_stores_real_actual_values_for_all_four_stats(tmp_path):
    import pandas as pd

    from nba_predictor.pipeline.ingest import store_player_outcomes
    from nba_predictor.tracking import store

    training_df = pd.DataFrame([
        {
            "player_id": "p1", "player_name": "Jayson Tatum", "team": "BOS",
            "game_id": "g1", "game_date": "2026-03-01",
            "points": 28.0, "rebounds": 7.0, "assists": 5.0, "fg3m": 3.0, "minutes": 34.0,
        }
    ])

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    stored = store_player_outcomes(training_df, db_path)

    assert stored == 4
    outcomes = store.get_player_outcomes_for_game(db_path, "g1")
    by_stat = {row["stat"]: row["actual_value"] for row in outcomes}
    assert by_stat == {"points": 28.0, "rebounds": 7.0, "assists": 5.0, "threes": 3.0}


def test_score_upcoming_player_props_stores_predictions(tmp_path):
    import numpy as np
    import pandas as pd

    from nba_predictor.pipeline.ingest import score_upcoming_player_props, train_player_prop_models
    from nba_predictor.tracking import store

    rng = np.random.default_rng(13)
    dates = pd.date_range("2026-10-01", periods=20).astype(str)
    games = []
    player_boxscores = {}
    for i, game_date in enumerate(dates):
        game_id = f"g{i}"
        games.append({"game_id": game_id, "game_date": game_date, "home_team": "BOS", "away_team": "MIA", "completed": True, "home_pts": 110, "away_pts": 105})
        player_boxscores[game_id] = [
            {
                "player_id": "p1", "player_name": "Jayson Tatum", "team": "BOS", "position": "F",
                "minutes": float(rng.integers(28, 38)), "points": float(rng.integers(15, 35)),
                "rebounds": float(rng.integers(3, 10)), "assists": float(rng.integers(2, 8)),
                "fg_made_attempted": "10-19", "three_made_attempted": "3-7", "ft_made_attempted": "5-6",
            }
        ]
    upcoming_game = {"game_id": "g-upcoming", "game_date": "2026-10-25", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None}
    games.append(upcoming_game)

    from nba_predictor.pipeline.ingest import to_player_training_frame
    training_df = to_player_training_frame(games, player_boxscores)

    models_dir = tmp_path / "models"
    train_player_prop_models(training_df, models_dir, model_version="v-test", trained_at="2026-10-20T00:00:00")

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    stored = score_upcoming_player_props(games, player_boxscores, models_dir, db_path, model_version="v-test")

    assert stored == 4
    rows = store.get_player_predictions_for_game(db_path, "g-upcoming")
    assert len(rows) == 4


def test_score_and_store_player_predictions_stores_real_model_output(tmp_path):
    import numpy as np
    import pandas as pd

    from nba_predictor.pipeline.ingest import score_and_store_player_predictions, train_player_prop_models
    from nba_predictor.tracking import store

    rng = np.random.default_rng(11)
    dates = pd.date_range("2026-02-01", periods=20).astype(str)
    rows = []
    for i, game_date in enumerate(dates):
        rows.append(
            {
                "player_id": "p1", "player_name": "Jayson Tatum", "team": "BOS",
                "game_id": f"g{i}", "game_date": game_date,
                "points": float(rng.integers(15, 35)), "rebounds": float(rng.integers(3, 10)),
                "assists": float(rng.integers(2, 8)), "fg3m": float(rng.integers(0, 6)),
                "minutes": float(rng.integers(28, 38)),
            }
        )
    training_df = pd.DataFrame(rows)

    models_dir = tmp_path / "models"
    train_player_prop_models(training_df, models_dir, model_version="v-test", trained_at="2026-03-01T00:00:00")

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    stored = score_and_store_player_predictions(training_df, models_dir, db_path, model_version="v-test")

    # 19 rows survive feature assembly (first game has no rolling history) x 4 stats
    assert stored == 19 * 4
    rows = store.get_player_predictions_for_game(db_path, "g5")
    assert len(rows) == 4
    stats = {r["stat"] for r in rows}
    assert stats == {"points", "rebounds", "assists", "threes"}


def test_score_and_store_player_predictions_returns_zero_for_empty_frame(tmp_path):
    import pandas as pd

    from nba_predictor.pipeline.ingest import score_and_store_player_predictions
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    empty_df = pd.DataFrame(columns=["player_id", "player_name", "team", "game_id", "game_date", "points", "rebounds", "assists", "fg3m", "minutes"])

    stored = score_and_store_player_predictions(empty_df, tmp_path / "models", db_path, model_version="v-test")

    assert stored == 0


def test_train_player_prop_models_writes_four_model_files_and_manifest(tmp_path):
    import numpy as np
    import pandas as pd

    from nba_predictor.pipeline.ingest import train_player_prop_models

    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-02-01", periods=20).astype(str)
    rows = []
    for i, game_date in enumerate(dates):
        rows.append(
            {
                "player_id": "p1", "player_name": "Jayson Tatum", "team": "BOS",
                "game_id": f"g{i}", "game_date": game_date,
                "points": float(rng.integers(15, 35)), "rebounds": float(rng.integers(3, 10)),
                "assists": float(rng.integers(2, 8)), "fg3m": float(rng.integers(0, 6)),
                "minutes": float(rng.integers(28, 38)),
            }
        )
    training_df = pd.DataFrame(rows)

    models_dir = tmp_path / "models"
    manifest = train_player_prop_models(training_df, models_dir, model_version="v-test", trained_at="2026-03-01T00:00:00")

    assert (models_dir / "player_points_model.pkl").exists()
    assert (models_dir / "player_rebounds_model.pkl").exists()
    assert (models_dir / "player_assists_model.pkl").exists()
    assert (models_dir / "player_threes_model.pkl").exists()
    assert (models_dir / "player_props_manifest.json").exists()
    assert manifest["model_version"] == "v-test"
    assert set(manifest["models"]) == {"points", "rebounds", "assists", "threes"}
    assert manifest["metrics"]["points"]["mae"] is not None


def test_train_player_prop_models_handles_empty_training_frame(tmp_path):
    import pandas as pd

    from nba_predictor.pipeline.ingest import train_player_prop_models

    models_dir = tmp_path / "models"
    empty_df = pd.DataFrame(columns=["player_id", "player_name", "team", "game_id", "game_date", "points", "rebounds", "assists", "fg3m", "minutes"])

    manifest = train_player_prop_models(empty_df, models_dir, model_version="v-test", trained_at="2026-03-01T00:00:00")

    assert manifest["metrics"]["points"]["mae"] is None
    assert not (models_dir / "player_points_model.pkl").exists()


def test_to_player_training_frame_builds_one_row_per_player_game():
    from nba_predictor.pipeline.ingest import to_player_training_frame

    games = [
        {"game_id": "g1", "game_date": "2026-03-01", "home_team": "BOS", "away_team": "MIA"},
    ]
    player_boxscores = {
        "g1": [
            {
                "player_id": "p1", "player_name": "Jayson Tatum", "team": "BOS", "position": "F",
                "minutes": 34.0, "points": 28.0, "rebounds": 7.0, "assists": 5.0,
                "fg_made_attempted": "10-19", "three_made_attempted": "3-7", "ft_made_attempted": "5-6",
            }
        ]
    }

    df = to_player_training_frame(games, player_boxscores)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["player_id"] == "p1"
    assert row["game_id"] == "g1"
    assert row["game_date"] == "2026-03-01"
    assert row["points"] == 28.0
    assert row["fg3m"] == 3.0
    assert row["minutes"] == 34.0


def test_to_player_training_frame_skips_games_missing_from_schedule():
    from nba_predictor.pipeline.ingest import to_player_training_frame

    player_boxscores = {"g-unknown": [{"player_id": "p1", "player_name": "X", "team": "BOS", "position": "F", "minutes": 30.0, "points": 10.0, "rebounds": 5.0, "assists": 2.0, "fg_made_attempted": "4-8", "three_made_attempted": "1-2", "ft_made_attempted": "1-1"}]}

    df = to_player_training_frame([], player_boxscores)

    assert len(df) == 0
