import math

from scipy.optimize import brentq


def implied_probability(american_odds: int) -> float:
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)


def _shin_probabilities(z: float, raw_probabilities: list[float], total: float) -> list[float]:
    return [
        (math.sqrt(z**2 + 4 * (1 - z) * p**2 / total) - z) / (2 * (1 - z))
        for p in raw_probabilities
    ]


def shin_devig(raw_probabilities: list[float]) -> list[float]:
    total = sum(raw_probabilities)
    if total <= 1.0:
        return [p / total for p in raw_probabilities]

    def sum_minus_one(z: float) -> float:
        return sum(_shin_probabilities(z, raw_probabilities, total)) - 1

    z = brentq(sum_minus_one, 1e-9, 0.999)
    return _shin_probabilities(z, raw_probabilities, total)


def compute_edge(model_probability: float, market_probability: float) -> float:
    return model_probability - market_probability


def std_from_mae(mae: float) -> float:
    return mae * math.sqrt(math.pi / 2)


def normal_cover_probability(mean: float, line: float, std: float) -> float:
    if std <= 0:
        return 1.0 if mean > line else 0.0
    z = (mean - line) / (std * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))
