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
