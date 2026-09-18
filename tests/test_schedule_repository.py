import json


def test_load_schedule_returns_empty_list_when_file_missing(tmp_path):
    from nba_predictor.services.schedule_repository import load_schedule

    assert load_schedule(tmp_path / "does_not_exist.json") == []


def test_load_schedule_reads_json_array(tmp_path):
    from nba_predictor.services.schedule_repository import load_schedule

    path = tmp_path / "games.json"
    path.write_text(json.dumps([{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}]))

    schedule = load_schedule(path)
    assert schedule == [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}]


def test_get_games_for_date_filters_correctly():
    from nba_predictor.services.schedule_repository import get_games_for_date

    schedule = [
        {"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"},
        {"game_id": "g2", "game_date": "2026-11-02", "home_team": "LAL", "away_team": "GSW"},
    ]
    result = get_games_for_date(schedule, "2026-11-01")
    assert result == [schedule[0]]


def test_get_game_returns_none_when_not_found():
    from nba_predictor.services.schedule_repository import get_game

    assert get_game([], "missing") is None


def test_get_game_finds_by_id():
    from nba_predictor.services.schedule_repository import get_game

    schedule = [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}]
    assert get_game(schedule, "g1") == schedule[0]


def test_get_games_for_week_includes_full_seven_day_window():
    from nba_predictor.services.schedule_repository import get_games_for_week

    schedule = [
        {"game_id": "g0", "game_date": "2026-02-15", "home_team": "BOS", "away_team": "MIA"},  # Sunday before
        {"game_id": "g1", "game_date": "2026-02-16", "home_team": "BOS", "away_team": "MIA"},  # Monday (start)
        {"game_id": "g2", "game_date": "2026-02-19", "home_team": "LAL", "away_team": "GSW"},  # Thursday
        {"game_id": "g3", "game_date": "2026-02-22", "home_team": "DEN", "away_team": "PHX"},  # Sunday (end)
        {"game_id": "g4", "game_date": "2026-02-23", "home_team": "DEN", "away_team": "PHX"},  # Monday after
    ]

    result = get_games_for_week(schedule, "2026-02-16")

    assert [g["game_id"] for g in result] == ["g1", "g2", "g3"]


def test_get_games_for_week_sorts_by_date_then_id():
    from nba_predictor.services.schedule_repository import get_games_for_week

    schedule = [
        {"game_id": "g2", "game_date": "2026-02-18", "home_team": "LAL", "away_team": "GSW"},
        {"game_id": "g1", "game_date": "2026-02-16", "home_team": "BOS", "away_team": "MIA"},
    ]

    result = get_games_for_week(schedule, "2026-02-16")

    assert [g["game_id"] for g in result] == ["g1", "g2"]


def test_monday_of_returns_same_date_when_already_monday():
    from nba_predictor.services.schedule_repository import monday_of

    assert monday_of("2026-02-16") == "2026-02-16"


def test_monday_of_returns_preceding_monday():
    from nba_predictor.services.schedule_repository import monday_of

    assert monday_of("2026-02-19") == "2026-02-16"  # Thursday -> that week's Monday
    assert monday_of("2026-02-22") == "2026-02-16"  # Sunday -> that week's Monday


def _completed_game(game_id, date, home, away, home_pts, away_pts):
    return {
        "game_id": game_id, "game_date": date, "home_team": home, "away_team": away,
        "completed": True, "home_pts": home_pts, "away_pts": away_pts,
    }


def test_get_head_to_head_returns_prior_meetings_sorted_most_recent_first():
    from nba_predictor.services.schedule_repository import get_head_to_head

    schedule = [
        _completed_game("g1", "2026-01-01", "BOS", "MIA", 110, 100),
        _completed_game("g2", "2026-02-01", "MIA", "BOS", 95, 105),
        _completed_game("g3", "2026-03-01", "BOS", "LAL", 120, 100),
        _completed_game("g4", "2026-04-01", "BOS", "MIA", 90, 92),
    ]

    result = get_head_to_head(schedule, "BOS", "MIA", before_date="2026-04-01", limit=5)

    assert [g["game_id"] for g in result] == ["g2", "g1"]


def test_get_head_to_head_only_counts_completed_games():
    from nba_predictor.services.schedule_repository import get_head_to_head

    schedule = [
        {"game_id": "g1", "game_date": "2026-01-01", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None},
    ]

    assert get_head_to_head(schedule, "BOS", "MIA", before_date="2026-02-01") == []


def test_get_head_to_head_respects_limit():
    from nba_predictor.services.schedule_repository import get_head_to_head

    schedule = [_completed_game(f"g{i}", f"2026-0{i}-01", "BOS", "MIA", 100 + i, 90 + i) for i in range(1, 6)]

    result = get_head_to_head(schedule, "BOS", "MIA", before_date="2026-06-01", limit=2)
    assert len(result) == 2


def test_get_recent_form_returns_win_loss_letters_most_recent_first():
    from nba_predictor.services.schedule_repository import get_recent_form

    schedule = [
        _completed_game("g1", "2026-01-01", "BOS", "MIA", 110, 100),
        _completed_game("g2", "2026-01-05", "LAL", "BOS", 100, 90),
        _completed_game("g3", "2026-01-10", "BOS", "DEN", 88, 95),
    ]

    result = get_recent_form(schedule, "BOS", before_date="2026-01-15", limit=5)
    assert result == ["L", "L", "W"]


def test_get_recent_form_excludes_games_on_or_after_before_date():
    from nba_predictor.services.schedule_repository import get_recent_form

    schedule = [
        _completed_game("g1", "2026-01-01", "BOS", "MIA", 110, 100),
        _completed_game("g2", "2026-01-10", "BOS", "MIA", 90, 100),
    ]

    assert get_recent_form(schedule, "BOS", before_date="2026-01-10", limit=5) == ["W"]


def test_default_week_start_empty_schedule_returns_none():
    from nba_predictor.services.schedule_repository import default_week_start

    assert default_week_start([], today="2026-09-17") is None


def test_default_week_start_uses_earliest_game_when_far_from_next_upcoming():
    from nba_predictor.services.schedule_repository import default_week_start, monday_of

    schedule = [
        _completed_game("g0", "2025-10-21", "BOS", "MIA", 110, 100),
        {"game_id": "g1", "game_date": "2026-10-21", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None},
    ]

    # More than 7 days before 2026-10-21 (the only upcoming game).
    result = default_week_start(schedule, today="2026-09-17")

    assert result == monday_of("2025-10-21")


def test_default_week_start_switches_to_upcoming_week_exactly_seven_days_before():
    from nba_predictor.services.schedule_repository import default_week_start, monday_of

    schedule = [
        _completed_game("g0", "2025-10-21", "BOS", "MIA", 110, 100),
        {"game_id": "g1", "game_date": "2026-10-21", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None},
    ]

    result = default_week_start(schedule, today="2026-10-14")  # exactly 7 days before

    assert result == monday_of("2026-10-21")


def test_default_week_start_stays_on_old_season_one_day_before_the_switch():
    from nba_predictor.services.schedule_repository import default_week_start, monday_of

    schedule = [
        _completed_game("g0", "2025-10-21", "BOS", "MIA", 110, 100),
        {"game_id": "g1", "game_date": "2026-10-21", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None},
    ]

    result = default_week_start(schedule, today="2026-10-13")  # 8 days before

    assert result == monday_of("2025-10-21")


def test_default_week_start_uses_today_once_season_is_underway():
    from nba_predictor.services.schedule_repository import default_week_start, monday_of

    schedule = [
        _completed_game("g0", "2025-10-21", "BOS", "MIA", 110, 100),
        {"game_id": "g1", "game_date": "2026-10-21", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None},
    ]

    result = default_week_start(schedule, today="2026-10-25")  # after the next game's date

    assert result == monday_of("2026-10-25")


def test_default_week_start_falls_back_to_earliest_when_nothing_upcoming():
    from nba_predictor.services.schedule_repository import default_week_start, monday_of

    schedule = [_completed_game("g0", "2025-10-21", "BOS", "MIA", 110, 100)]

    result = default_week_start(schedule, today="2026-09-17")

    assert result == monday_of("2025-10-21")


def test_default_week_start_ignores_stale_not_completed_games_from_the_past():
    """Real ESPN data can leave an already-finished season's postponed/
    orphaned game stuck with completed=False forever. That must not be
    mistaken for the next real upcoming game."""
    from nba_predictor.services.schedule_repository import default_week_start, monday_of

    schedule = [
        _completed_game("g0", "2025-10-21", "BOS", "MIA", 110, 100),
        # Stale: dated in the (real-world) past relative to `today` below,
        # but never resolved to completed=True.
        {"game_id": "g1", "game_date": "2026-01-08", "home_team": "CHI", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None},
        # The real next upcoming game.
        {"game_id": "g2", "game_date": "2026-10-21", "home_team": "BOS", "away_team": "MIA", "completed": False, "home_pts": None, "away_pts": None},
    ]

    result = default_week_start(schedule, today="2026-09-18")

    assert result == monday_of("2025-10-21")
