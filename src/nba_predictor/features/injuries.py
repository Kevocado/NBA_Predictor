"""Missing-player production value, from values the caller supplies.

There is no server-side source for a player's production value. data/injuries.py
used to appear to be one -- its _fetch_player_stats returned {} and its
get_missing_player_value turned that into 0.0 per missing player, a number
indistinguishable from a real measurement of zero. That stub is neutralised now
(see data/injuries.py), so this module has no default data path at all: a
caller with real values passes a value_fn, and a caller without one is told so
instead of receiving an invented total.
"""

from nba_predictor.data import injuries as injuries_data


def missing_players_value(injured_player_ids: list[int], season: int, value_fn=None) -> float:
    if not injured_player_ids:
        # A true empty sum: no missing players, so no value missing. This is
        # arithmetic, not a measurement standing in for one.
        return 0.0
    if value_fn is None:
        raise NotImplementedError(
            "missing_players_value needs a value_fn: this repo has no source for a "
            "player's production value. data/injuries.py, which used to return a "
            "fabricated 0.0 per player, is neutralised. For availability, use "
            "nba_predictor.data.espn.get_injuries()."
        )
    return sum(value_fn(player_id, season) for player_id in injured_player_ids)


__all__ = ["missing_players_value", "injuries_data"]
