def test_head_to_head_record_counts_wins():
    from nba_predictor.features.context import head_to_head_record

    result = head_to_head_record(["team_a", "team_b", "team_a"])
    assert result == {"team_a_wins": 2, "team_b_wins": 1}


def test_head_to_head_record_empty_history():
    from nba_predictor.features.context import head_to_head_record

    assert head_to_head_record([]) == {"team_a_wins": 0, "team_b_wins": 0}


def test_current_streak_positive_for_win_streak():
    from nba_predictor.features.context import current_streak

    assert current_streak(["L", "W", "W", "W"]) == 3


def test_current_streak_negative_for_loss_streak():
    from nba_predictor.features.context import current_streak

    assert current_streak(["W", "L", "L"]) == -2


def test_current_streak_empty_is_zero():
    from nba_predictor.features.context import current_streak

    assert current_streak([]) == 0


def test_is_high_altitude_true_for_denver():
    from nba_predictor.features.context import is_high_altitude

    assert is_high_altitude("DEN") is True


def test_is_high_altitude_false_for_boston():
    from nba_predictor.features.context import is_high_altitude

    assert is_high_altitude("BOS") is False


def test_game_flags_conference_and_division_game():
    from nba_predictor.features.context import game_flags

    flags = game_flags("BOS", "NYK")  # both Atlantic, both East
    assert flags == {"conference_game": True, "division_game": True}


def test_game_flags_conference_only():
    from nba_predictor.features.context import game_flags

    flags = game_flags("BOS", "MIA")  # both East, different division
    assert flags == {"conference_game": True, "division_game": False}


def test_game_flags_non_conference_game():
    from nba_predictor.features.context import game_flags

    flags = game_flags("BOS", "LAL")  # East vs West
    assert flags == {"conference_game": False, "division_game": False}
