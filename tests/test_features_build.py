import pandas as pd
import pytest


def _sample_games() -> pd.DataFrame:
    rows = []
    teams = ["BOS", "MIA"]
    dates = ["2026-10-21", "2026-10-23", "2026-10-25", "2026-10-27"]
    for i, game_date in enumerate(dates):
        home, away = (teams[0], teams[1]) if i % 2 == 0 else (teams[1], teams[0])
        rows.append(
            {
                "game_id": f"g{i}",
                "game_date": game_date,
                "home_team": home,
                "away_team": away,
                "home_pts": 110 + i,
                "away_pts": 105 + i,
                "home_fgm": 40, "home_fga": 88, "home_fg3m": 12, "home_tov": 11,
                "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
                "away_fgm": 38, "away_fga": 90, "away_fg3m": 10, "away_tov": 13,
                "away_oreb": 10, "away_dreb": 30, "away_fta": 18,
                "home_win": 1,
            }
        )
    return pd.DataFrame(rows)


def test_build_training_frame_returns_feature_columns():
    from nba_predictor.features.build import FEATURE_COLUMNS, build_training_frame

    games = _sample_games()
    result_df, feature_cols = build_training_frame(games)

    assert feature_cols == FEATURE_COLUMNS
    for col in feature_cols:
        assert col in result_df.columns


def test_build_training_frame_drops_rows_with_no_rolling_history():
    from nba_predictor.features.build import build_training_frame

    games = _sample_games()
    result_df, feature_cols = build_training_frame(games)

    assert len(result_df) < len(games)
    assert result_df[feature_cols].isna().sum().sum() == 0


def test_build_training_frame_fills_missing_optional_columns_with_zero():
    from nba_predictor.features.build import build_training_frame

    games = _sample_games()
    result_df, _ = build_training_frame(games)

    assert (result_df["home_power_rating"] == 0.0).all()
    assert (result_df["home_fatigue_index"] == 0.0).all()
    assert (result_df["home_missing_value"] == 0.0).all()


def test_build_training_frame_respects_precomputed_optional_columns():
    from nba_predictor.features.build import build_training_frame

    games = _sample_games()
    games["home_power_rating"] = 1550.0
    result_df, _ = build_training_frame(games)

    assert (result_df["home_power_rating"] == 1550.0).all()
