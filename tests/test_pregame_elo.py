"""Leak test for pre-game Elo: a game's rating never reads that game's result.

Also checks the converse, so the test cannot pass by accident: flipping the result
of a game does move the ratings of games played after it.
"""

import pandas as pd

from nba_predictor.features.pregame_elo import pregame_elo_ratings


def _games(home_win_of_g: float, home_win_of_later: float) -> pd.DataFrame:
    # Two games on 2025-11-01 share a team in the same day; one later game sees both.
    return pd.DataFrame(
        {
            "game_id": ["early_a", "G", "later"],
            "game_date": ["2025-11-01", "2025-11-01", "2025-11-03"],
            "home_team": ["BOS", "NYK", "BOS"],
            "away_team": ["MIA", "BOS", "NYK"],
            "home_win": [1.0, home_win_of_g, home_win_of_later],
        }
    )


def test_rating_for_a_game_never_uses_that_games_result():
    base = pregame_elo_ratings(_games(1.0, 1.0), k=20.0, hca=50.0).set_index("game_id")

    # Flip the result of G (same date) and of the later game: G's own pre-game ratings must not move.
    flipped_g = pregame_elo_ratings(_games(0.0, 1.0), k=20.0, hca=50.0).set_index("game_id")
    assert flipped_g.loc["G", ["home_pregame_elo", "away_pregame_elo"]].tolist() == \
        base.loc["G", ["home_pregame_elo", "away_pregame_elo"]].tolist()

    flipped_later = pregame_elo_ratings(_games(1.0, 0.0), k=20.0, hca=50.0).set_index("game_id")
    assert flipped_later.loc["G", ["home_pregame_elo", "away_pregame_elo"]].tolist() == \
        base.loc["G", ["home_pregame_elo", "away_pregame_elo"]].tolist()

    # Same-date results are not read either: flipping the other game of G's date leaves G unchanged.
    flipped_same_day = pregame_elo_ratings(
        _games(1.0, 1.0).assign(home_win=[0.0, 1.0, 1.0]), k=20.0, hca=50.0
    ).set_index("game_id")
    assert flipped_same_day.loc["G", ["home_pregame_elo", "away_pregame_elo"]].tolist() == \
        base.loc["G", ["home_pregame_elo", "away_pregame_elo"]].tolist()

    # Converse: a game's result does reach games played after it.
    assert flipped_g.loc["later", "home_pregame_elo"] != base.loc["later", "home_pregame_elo"]
