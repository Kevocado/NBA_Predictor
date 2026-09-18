import pandas as pd


def _sample_player_games() -> pd.DataFrame:
    dates = ["2026-10-21", "2026-10-23", "2026-10-25", "2026-10-27"]
    rows = []
    for i, game_date in enumerate(dates):
        rows.append(
            {
                "player_id": "p1", "player_name": "Jayson Tatum", "team": "BOS",
                "game_id": f"g{i}", "game_date": game_date,
                "points": 20.0 + i, "rebounds": 6.0, "assists": 4.0, "fg3m": 3.0, "minutes": 34.0,
            }
        )
    return pd.DataFrame(rows)


def test_build_player_feature_frame_returns_feature_columns():
    from nba_predictor.features.player_stats import PLAYER_FEATURE_COLUMNS, build_player_feature_frame

    player_games = _sample_player_games()
    result_df, feature_cols = build_player_feature_frame(player_games)

    assert feature_cols == PLAYER_FEATURE_COLUMNS
    for col in feature_cols:
        assert col in result_df.columns


def test_build_player_feature_frame_drops_first_game_with_no_rolling_history():
    from nba_predictor.features.player_stats import build_player_feature_frame

    player_games = _sample_player_games()
    result_df, _ = build_player_feature_frame(player_games)

    assert len(result_df) == 3  # first game has no prior history, dropped
    assert "g0" not in set(result_df["game_id"])


def test_build_player_feature_frame_rolling_value_is_leak_free():
    from nba_predictor.features.player_stats import build_player_feature_frame

    player_games = _sample_player_games()
    result_df, _ = build_player_feature_frame(player_games)

    # g1's points_roll should be exactly g0's real points (20.0), not
    # influenced by g1's own points (21.0).
    g1_row = result_df[result_df["game_id"] == "g1"].iloc[0]
    assert g1_row["points_roll"] == 20.0


def test_build_player_feature_frame_computes_features_for_upcoming_player_game():
    from nba_predictor.features.player_stats import build_player_feature_frame

    player_games = _sample_player_games()
    upcoming = pd.DataFrame([
        {
            "player_id": "p1", "player_name": "Jayson Tatum", "team": "BOS",
            "game_id": "g4", "game_date": "2026-10-29",
            "points": None, "rebounds": None, "assists": None, "fg3m": None, "minutes": None,
        }
    ])
    combined = pd.concat([player_games, upcoming], ignore_index=True)

    result_df, feature_cols = build_player_feature_frame(combined)

    upcoming_row = result_df[result_df["game_id"] == "g4"]
    assert len(upcoming_row) == 1
    assert upcoming_row[feature_cols].isna().sum().sum() == 0
