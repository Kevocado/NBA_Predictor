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
    """Columns the caller cannot supply are zero-filled — and ONLY those.

    `home_fatigue_index` used to be in this list while also being in
    `FEATURE_COLUMNS`, which is how it stayed a constant zero for so long: the
    zero-fill made the column exist and nothing checked that it moved. It is
    computed now, so it is not zero-filled and this test must not claim it is.
    """
    from nba_predictor.features.build import build_training_frame

    games = _sample_games()
    result_df, _ = build_training_frame(games)

    assert (result_df["home_power_rating"] == 0.0).all()
    assert (result_df["home_missing_value"] == 0.0).all()

    # The fatigue index is computed, not filled, so it must actually move.
    assert not (result_df["home_fatigue_index"] == 0.0).all(), (
        "home_fatigue_index is still a constant zero; it is being zero-filled "
        "rather than computed"
    )


def test_fatigue_index_is_computed_rather_than_zero_filled():
    """The defect this replaces: a column can be in FEATURE_COLUMNS, be absent from
    the incoming frame, and be filled with 0.0 — which makes it exist without
    making it a measurement."""
    from nba_predictor.features.build import build_training_frame, _OPTIONAL_COLUMNS_DEFAULT_ZERO

    assert "home_fatigue_index" not in _OPTIONAL_COLUMNS_DEFAULT_ZERO, (
        "home_fatigue_index is zero-filled; it should be computed"
    )
    games = _sample_games()
    assert "home_fatigue_index" not in games.columns, "fixture must not pre-supply it"
    result_df, feature_cols = build_training_frame(games, include_travel_fatigue=True)
    assert "home_fatigue_index" in feature_cols
    assert len(set(result_df["home_fatigue_index"])) > 1, (
        "the computed fatigue index has only one value; it is not reading anything"
    )


def test_build_training_frame_respects_precomputed_optional_columns():
    from nba_predictor.features.build import build_training_frame

    games = _sample_games()
    games["home_power_rating"] = 1550.0
    result_df, _ = build_training_frame(games)

    assert (result_df["home_power_rating"] == 1550.0).all()


def test_build_feature_frame_computes_rolling_features_for_upcoming_game():
    from nba_predictor.features.build import build_feature_frame

    games = _sample_games()
    upcoming = pd.DataFrame([
        {
            "game_id": "g4", "game_date": "2026-10-29", "home_team": "BOS", "away_team": "MIA",
            "home_win": None,
            "home_fgm": None, "home_fga": None, "home_fg3m": None, "home_tov": None,
            "home_oreb": None, "home_dreb": None, "home_fta": None,
            "away_fgm": None, "away_fga": None, "away_fg3m": None, "away_tov": None,
            "away_oreb": None, "away_dreb": None, "away_fta": None,
        }
    ])
    combined = pd.concat([games, upcoming], ignore_index=True)

    result_df, feature_cols = build_feature_frame(combined)

    upcoming_row = result_df[result_df["game_id"] == "g4"]
    assert len(upcoming_row) == 1
    assert upcoming_row[feature_cols].isna().sum().sum() == 0
    assert upcoming_row["home_efg_pct_roll"].iloc[0] > 0


def test_build_feature_frame_does_not_record_a_result_for_upcoming_games():
    from nba_predictor.features.build import build_feature_frame

    games = _sample_games()
    upcoming = pd.DataFrame([
        {
            "game_id": "g4", "game_date": "2026-10-29", "home_team": "BOS", "away_team": "MIA",
            "home_win": None,
            "home_fgm": None, "home_fga": None, "home_fg3m": None, "home_tov": None,
            "home_oreb": None, "home_dreb": None, "home_fta": None,
            "away_fgm": None, "away_fga": None, "away_fg3m": None, "away_tov": None,
            "away_oreb": None, "away_dreb": None, "away_fta": None,
        },
        {
            "game_id": "g5", "game_date": "2026-10-31", "home_team": "BOS", "away_team": "MIA",
            "home_win": None,
            "home_fgm": None, "home_fga": None, "home_fg3m": None, "home_tov": None,
            "home_oreb": None, "home_dreb": None, "home_fta": None,
            "away_fgm": None, "away_fga": None, "away_fg3m": None, "away_tov": None,
            "away_oreb": None, "away_dreb": None, "away_fta": None,
        },
    ])
    combined = pd.concat([games, upcoming], ignore_index=True)

    result_df, _ = build_feature_frame(combined)

    g4_streak = result_df.loc[result_df["game_id"] == "g4", "home_streak"].iloc[0]
    g5_streak = result_df.loc[result_df["game_id"] == "g5", "home_streak"].iloc[0]
    assert g4_streak == g5_streak


def test_build_training_frame_still_works_as_a_thin_wrapper():
    from nba_predictor.features.build import FEATURE_COLUMNS, build_training_frame

    games = _sample_games()
    result_df, feature_cols = build_training_frame(games)

    assert feature_cols == FEATURE_COLUMNS
    assert len(result_df) < len(games)
