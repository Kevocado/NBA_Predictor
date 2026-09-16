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
