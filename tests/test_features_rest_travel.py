import pytest


def test_haversine_miles_same_point_is_zero():
    from nba_predictor.features.rest_travel import haversine_miles

    assert haversine_miles(40.0, -75.0, 40.0, -75.0) == pytest.approx(0.0, abs=1e-6)


def test_haversine_miles_boston_to_lakers_is_roughly_2600_miles():
    from nba_predictor.features.rest_travel import haversine_miles

    distance = haversine_miles(42.3662, -71.0621, 34.0430, -118.2673)
    assert 2550 < distance < 2650


def test_compute_rest_days_counts_full_days_between_games():
    from nba_predictor.features.rest_travel import compute_rest_days

    assert compute_rest_days("2026-11-05", "2026-11-03") == 1


def test_compute_rest_days_zero_for_back_to_back():
    from nba_predictor.features.rest_travel import compute_rest_days

    assert compute_rest_days("2026-11-05", "2026-11-04") == 0


def test_compute_rest_days_large_for_season_opener():
    from nba_predictor.features.rest_travel import compute_rest_days

    assert compute_rest_days("2026-10-21", None) == 99


def test_is_back_to_back_true_for_zero_rest_days():
    from nba_predictor.features.rest_travel import is_back_to_back

    assert is_back_to_back(0) is True
    assert is_back_to_back(1) is False


def test_congestion_flags_three_in_four():
    from nba_predictor.features.rest_travel import congestion_flags

    flags = congestion_flags(
        prior_game_dates=["2026-11-01", "2026-11-03"], current_date="2026-11-04"
    )
    assert flags["three_in_four"] is True
    assert flags["four_in_six"] is False


def test_congestion_flags_four_in_six():
    from nba_predictor.features.rest_travel import congestion_flags

    flags = congestion_flags(
        prior_game_dates=["2026-11-01", "2026-11-02", "2026-11-04"],
        current_date="2026-11-06",
    )
    assert flags["four_in_six"] is True


def test_congestion_flags_false_when_well_rested():
    from nba_predictor.features.rest_travel import congestion_flags

    flags = congestion_flags(prior_game_dates=["2026-10-28"], current_date="2026-11-04")
    assert flags["three_in_four"] is False
    assert flags["four_in_six"] is False


def test_rolling_travel_miles_sums_consecutive_hops():
    from nba_predictor.features.rest_travel import rolling_travel_miles

    total = rolling_travel_miles(["BOS", "MIA", "LAL"])
    assert total > 0


def test_rolling_travel_miles_zero_for_single_location():
    from nba_predictor.features.rest_travel import rolling_travel_miles

    assert rolling_travel_miles(["BOS"]) == 0.0


def test_timezone_change_count_counts_differences():
    from nba_predictor.features.rest_travel import timezone_change_count

    assert timezone_change_count(["BOS", "MIA", "LAL"]) == 1  # NY->NY (0), NY->LA (1)
    assert timezone_change_count(["BOS", "LAL", "GSW"]) == 1  # NY->LA (1), LA->LA (0)


def test_fatigue_index_increases_with_travel_and_fewer_rest_days():
    from nba_predictor.features.rest_travel import fatigue_index

    low_fatigue = fatigue_index(travel_miles=0, timezone_changes=0, rest_days=3)
    high_fatigue = fatigue_index(travel_miles=2500, timezone_changes=3, rest_days=0)
    assert high_fatigue > low_fatigue
