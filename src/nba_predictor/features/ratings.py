def compute_possessions(fga: float, fta: float, oreb: float, tov: float) -> float:
    return fga - oreb + tov + 0.44 * fta


def compute_offensive_rating(points: float, possessions: float) -> float:
    return (points / possessions) * 100 if possessions else 0.0


def compute_pace(team_possessions: float, opp_possessions: float, minutes: float = 48.0) -> float:
    return (team_possessions + opp_possessions) / 2 * (48.0 / minutes) if minutes else 0.0


def elo_expected(rating_a: float, rating_b: float, home_adjustment: float = 0.0) -> float:
    adjusted_a = rating_a + home_adjustment
    return 1.0 / (1.0 + 10 ** ((rating_b - adjusted_a) / 400))


def elo_update(rating: float, expected: float, actual: float, k: float = 20.0) -> float:
    return rating + k * (actual - expected)


def init_elo_ratings(team_abbreviations: list[str], base_rating: float = 1500.0) -> dict[str, float]:
    return {team: base_rating for team in team_abbreviations}
