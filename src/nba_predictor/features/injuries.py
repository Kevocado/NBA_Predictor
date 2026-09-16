from nba_predictor.data import injuries as injuries_data


def missing_players_value(injured_player_ids: list[int], season: int, value_fn=None) -> float:
    fn = value_fn or injuries_data.get_missing_player_value
    return sum(fn(player_id, season) for player_id in injured_player_ids)
