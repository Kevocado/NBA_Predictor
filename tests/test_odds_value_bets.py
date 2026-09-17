import pytest


def test_implied_probability_positive_american_odds():
    from nba_predictor.odds.value_bets import implied_probability

    assert implied_probability(150) == pytest.approx(100 / 250)


def test_implied_probability_negative_american_odds():
    from nba_predictor.odds.value_bets import implied_probability

    assert implied_probability(-110) == pytest.approx(110 / 210)


def test_shin_devig_symmetric_odds_split_evenly():
    from nba_predictor.odds.value_bets import implied_probability, shin_devig

    raw = [implied_probability(-110), implied_probability(-110)]
    devigged = shin_devig(raw)

    assert devigged[0] == pytest.approx(0.5, abs=1e-4)
    assert devigged[1] == pytest.approx(0.5, abs=1e-4)
    assert sum(devigged) == pytest.approx(1.0, abs=1e-6)


def test_shin_devig_preserves_favorite_ordering():
    from nba_predictor.odds.value_bets import implied_probability, shin_devig

    raw = [implied_probability(-200), implied_probability(170)]
    devigged = shin_devig(raw)

    assert devigged[0] > devigged[1]
    assert sum(devigged) == pytest.approx(1.0, abs=1e-6)


def test_shin_devig_no_overround_just_normalizes():
    from nba_predictor.odds.value_bets import shin_devig

    devigged = shin_devig([0.5, 0.5])
    assert devigged == pytest.approx([0.5, 0.5])


def test_compute_edge_positive_when_model_favors_selection():
    from nba_predictor.odds.value_bets import compute_edge

    assert compute_edge(0.60, 0.52) == pytest.approx(0.08)


def test_compute_edge_negative_when_market_favors_selection():
    from nba_predictor.odds.value_bets import compute_edge

    assert compute_edge(0.45, 0.52) == pytest.approx(-0.07)


import math


def test_std_from_mae_converts_via_half_normal_relation():
    from nba_predictor.odds.value_bets import std_from_mae

    mae = 10.0
    std = std_from_mae(mae)

    assert std == pytest.approx(mae * math.sqrt(math.pi / 2))


def test_normal_cover_probability_is_half_when_mean_equals_line():
    from nba_predictor.odds.value_bets import normal_cover_probability

    assert normal_cover_probability(mean=5.0, line=5.0, std=10.0) == pytest.approx(0.5)


def test_normal_cover_probability_increases_with_higher_mean():
    from nba_predictor.odds.value_bets import normal_cover_probability

    low = normal_cover_probability(mean=1.0, line=5.0, std=10.0)
    high = normal_cover_probability(mean=9.0, line=5.0, std=10.0)

    assert high > low
    assert 0.0 < low < 1.0
    assert 0.0 < high < 1.0


def test_normal_cover_probability_zero_std_is_a_step_function():
    from nba_predictor.odds.value_bets import normal_cover_probability

    assert normal_cover_probability(mean=6.0, line=5.0, std=0.0) == 1.0
    assert normal_cover_probability(mean=4.0, line=5.0, std=0.0) == 0.0
