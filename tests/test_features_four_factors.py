import math

import pandas as pd
import pytest


def test_compute_four_factors_known_values():
    from nba_predictor.features.four_factors import compute_four_factors

    result = compute_four_factors(
        fgm=40, fga=90, fg3m=10, tov=12, oreb=10, opp_dreb=30, fta=20
    )

    assert result["efg_pct"] == pytest.approx((40 + 0.5 * 10) / 90)
    assert result["tov_rate"] == pytest.approx(12 / (90 + 0.44 * 20 + 12))
    assert result["orb_pct"] == pytest.approx(10 / (10 + 30))
    assert result["ft_rate"] == pytest.approx(20 / 90)


def test_compute_four_factors_zero_fga_does_not_raise():
    from nba_predictor.features.four_factors import compute_four_factors

    result = compute_four_factors(
        fgm=0, fga=0, fg3m=0, tov=0, oreb=0, opp_dreb=0, fta=0
    )
    assert result["efg_pct"] == 0.0
    assert result["ft_rate"] == 0.0
    assert result["orb_pct"] == 0.0


def test_add_rolling_four_factors_first_game_is_nan():
    from nba_predictor.features.four_factors import add_rolling_four_factors

    games = pd.DataFrame(
        {
            "team": ["BOS", "BOS"],
            "game_date": ["2026-10-01", "2026-10-03"],
            "fgm": [40, 42],
            "fga": [90, 88],
            "fg3m": [10, 12],
            "tov": [12, 10],
            "oreb": [10, 11],
            "opp_dreb": [30, 28],
            "fta": [20, 18],
        }
    )

    result = add_rolling_four_factors(games, window=10)

    assert math.isnan(result.loc[0, "efg_pct_roll"])
    assert not math.isnan(result.loc[1, "efg_pct_roll"])


def test_add_rolling_four_factors_uses_only_prior_games():
    from nba_predictor.features.four_factors import (
        add_rolling_four_factors,
        compute_four_factors,
    )

    games = pd.DataFrame(
        {
            "team": ["BOS", "BOS", "BOS"],
            "game_date": ["2026-10-01", "2026-10-03", "2026-10-05"],
            "fgm": [40, 42, 44],
            "fga": [90, 88, 86],
            "fg3m": [10, 12, 14],
            "tov": [12, 10, 8],
            "oreb": [10, 11, 12],
            "opp_dreb": [30, 28, 26],
            "fta": [20, 18, 16],
        }
    )

    result = add_rolling_four_factors(games, window=10)

    game1_factors = compute_four_factors(40, 90, 10, 12, 10, 30, 20)
    game2_factors = compute_four_factors(42, 88, 12, 10, 11, 28, 18)
    expected_third_row = (game1_factors["efg_pct"] + game2_factors["efg_pct"]) / 2

    assert result.loc[2, "efg_pct_roll"] == pytest.approx(expected_third_row)


def test_add_rolling_four_factors_keeps_teams_independent():
    from nba_predictor.features.four_factors import add_rolling_four_factors

    games = pd.DataFrame(
        {
            "team": ["BOS", "MIA"],
            "game_date": ["2026-10-01", "2026-10-01"],
            "fgm": [40, 35],
            "fga": [90, 85],
            "fg3m": [10, 8],
            "tov": [12, 15],
            "oreb": [10, 9],
            "opp_dreb": [30, 32],
            "fta": [20, 19],
        }
    )

    result = add_rolling_four_factors(games, window=10)
    assert math.isnan(result.loc[0, "efg_pct_roll"])
    assert math.isnan(result.loc[1, "efg_pct_roll"])
