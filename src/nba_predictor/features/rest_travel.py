import math
from datetime import date

from nba_predictor.data.team_reference import get_team


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_miles = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_miles * c


def compute_rest_days(current_date: str, previous_game_date: str | None) -> int:
    if previous_game_date is None:
        return 99
    current = date.fromisoformat(current_date)
    previous = date.fromisoformat(previous_game_date)
    return (current - previous).days - 1


def is_back_to_back(rest_days: int) -> bool:
    return rest_days == 0


def congestion_flags(prior_game_dates: list[str], current_date: str) -> dict:
    current = date.fromisoformat(current_date)
    all_dates = sorted(date.fromisoformat(d) for d in prior_game_dates) + [current]

    def games_within(window_size: int) -> int:
        cutoff = current.toordinal() - (window_size - 1)
        return sum(1 for d in all_dates if d.toordinal() >= cutoff)

    return {
        "three_in_four": games_within(4) >= 3,
        "four_in_six": games_within(6) >= 4,
    }


def rolling_travel_miles(game_locations: list[str], window_days: int = 7) -> float:
    if len(game_locations) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(game_locations, game_locations[1:]):
        team_a, team_b = get_team(a), get_team(b)
        total += haversine_miles(team_a.arena_lat, team_a.arena_lon, team_b.arena_lat, team_b.arena_lon)
    return total


def timezone_change_count(game_locations: list[str]) -> int:
    if len(game_locations) < 2:
        return 0
    changes = 0
    for a, b in zip(game_locations, game_locations[1:]):
        if get_team(a).timezone != get_team(b).timezone:
            changes += 1
    return changes


def fatigue_index(travel_miles: float, timezone_changes: int, rest_days: int) -> float:
    return travel_miles / 500 + timezone_changes * 1.5 + max(0, 2 - rest_days)
