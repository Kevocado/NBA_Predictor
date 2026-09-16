from nba_predictor.data.team_reference import get_team


def head_to_head_record(results: list[str], perspective: str = "team_a") -> dict:
    return {
        "team_a_wins": results.count("team_a"),
        "team_b_wins": results.count("team_b"),
    }


def current_streak(results: list[str]) -> int:
    if not results:
        return 0
    streak_char = results[-1]
    streak = 0
    for result in reversed(results):
        if result != streak_char:
            break
        streak += 1
    return streak if streak_char == "W" else -streak


def is_high_altitude(home_team_abbr: str) -> bool:
    return get_team(home_team_abbr).altitude_ft > 1000


def game_flags(home_team_abbr: str, away_team_abbr: str) -> dict:
    home, away = get_team(home_team_abbr), get_team(away_team_abbr)
    conference_game = home.conference == away.conference
    division_game = conference_game and home.division == away.division
    return {"conference_game": conference_game, "division_game": division_game}
