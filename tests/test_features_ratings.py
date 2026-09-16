import pytest


def test_compute_possessions_known_value():
    from nba_predictor.features.ratings import compute_possessions

    result = compute_possessions(fga=90, fta=20, oreb=10, tov=12)
    assert result == pytest.approx(90 - 10 + 12 + 0.44 * 20)


def test_compute_offensive_rating_known_value():
    from nba_predictor.features.ratings import compute_offensive_rating

    assert compute_offensive_rating(points=110, possessions=100) == pytest.approx(110.0)


def test_compute_offensive_rating_zero_possessions_does_not_raise():
    from nba_predictor.features.ratings import compute_offensive_rating

    assert compute_offensive_rating(points=0, possessions=0) == 0.0


def test_compute_pace_averages_both_teams():
    from nba_predictor.features.ratings import compute_pace

    result = compute_pace(team_possessions=98, opp_possessions=102, minutes=48.0)
    assert result == pytest.approx(100.0)


def test_elo_expected_equal_ratings_is_half():
    from nba_predictor.features.ratings import elo_expected

    assert elo_expected(1500, 1500) == pytest.approx(0.5)


def test_elo_expected_higher_rating_favored():
    from nba_predictor.features.ratings import elo_expected

    assert elo_expected(1600, 1500) > 0.5


def test_elo_expected_home_adjustment_increases_home_probability():
    from nba_predictor.features.ratings import elo_expected

    no_adjustment = elo_expected(1500, 1500, home_adjustment=0)
    with_adjustment = elo_expected(1500, 1500, home_adjustment=100)
    assert with_adjustment > no_adjustment


def test_elo_update_win_increases_rating():
    from nba_predictor.features.ratings import elo_update

    new_rating = elo_update(rating=1500, expected=0.5, actual=1.0, k=20)
    assert new_rating == pytest.approx(1510.0)


def test_elo_update_loss_decreases_rating():
    from nba_predictor.features.ratings import elo_update

    new_rating = elo_update(rating=1500, expected=0.5, actual=0.0, k=20)
    assert new_rating == pytest.approx(1490.0)


def test_init_elo_ratings_sets_base_for_all_teams():
    from nba_predictor.features.ratings import init_elo_ratings

    ratings = init_elo_ratings(["BOS", "MIA", "LAL"], base_rating=1500.0)
    assert ratings == {"BOS": 1500.0, "MIA": 1500.0, "LAL": 1500.0}
