"""Pre-game Elo ratings: each game carries its two teams' ratings AS OF TIP-OFF.

The rating written on a game row is read before anything that happens in that
game or on that date is folded in. Games on the same date are all read from the
ratings left by earlier dates, then that date's results are applied together.
So a row's ratings never contain its own result, nor a result from the same day.

Season boundaries regress each team a third of the way back to the league mean
(the usual Elo offseason correction). K and home-court advantage are not
hard-coded: `select_elo_params` chooses them from a small grid, scored only on the
last season of the training data it is given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_RATING = 1500.0
ELO_SCALE = 400.0
#: Fraction of the distance to the mean that is given back at each season boundary.
SEASON_REGRESSION = 1.0 / 3.0

K_GRID = (10.0, 20.0, 30.0)
HCA_GRID = (0.0, 50.0, 100.0)

PREGAME_ELO_COLUMNS = ["home_pregame_elo", "away_pregame_elo"]
PREGAME_ELO_FEATURE = "pregame_elo_diff_s"


def season_year(game_date: str) -> int:
    """NBA season starting year: Oct-Dec belongs to the season that starts that year."""
    year, month = int(str(game_date)[:4]), int(str(game_date)[5:7])
    return year if month >= 10 else year - 1


def _expected(rating_home: float, rating_away: float, hca: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(rating_home + hca - rating_away) / ELO_SCALE))


def pregame_elo_ratings(games: pd.DataFrame, k: float, hca: float) -> pd.DataFrame:
    """One row per game: `game_id`, `home_pregame_elo`, `away_pregame_elo`.

    `games` needs game_id, game_date, home_team, away_team and home_win (NaN for
    games not yet played; those are read but never folded in).
    """
    ratings: dict[str, float] = {}
    season_of: dict[str, int] = {}
    rows = []

    ordered = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    for game_date, day in ordered.groupby("game_date", sort=True):
        sy = season_year(game_date)
        for team in pd.concat([day["home_team"], day["away_team"]]).unique():
            if team in season_of and season_of[team] != sy:
                ratings[team] = BASE_RATING + (1 - SEASON_REGRESSION) * (ratings[team] - BASE_RATING)
            season_of[team] = sy

        # Read every game on this date before any of them is folded in.
        for row in day.itertuples():
            rows.append((row.game_id,
                         ratings.get(row.home_team, BASE_RATING),
                         ratings.get(row.away_team, BASE_RATING)))

        for row in day.itertuples():
            if pd.isna(row.home_win):
                continue
            r_h = ratings.get(row.home_team, BASE_RATING)
            r_a = ratings.get(row.away_team, BASE_RATING)
            delta = k * (float(row.home_win) - _expected(r_h, r_a, hca))
            ratings[row.home_team] = r_h + delta
            ratings[row.away_team] = r_a - delta

    return pd.DataFrame(rows, columns=["game_id", *PREGAME_ELO_COLUMNS])


def select_elo_params(games: pd.DataFrame, k_grid=K_GRID, hca_grid=HCA_GRID) -> tuple[float, float]:
    """Pick (K, home-court advantage) by log-loss of the pre-game win probability.

    Scored only on the last season present in `games` (the training data handed
    in), and every rating it reads was built from earlier games. Nothing after
    the training window is ever seen.
    """
    played = games[games["home_win"].notna()]
    seasons = played["game_date"].map(season_year)
    val = played[seasons == seasons.max()]
    y = val["home_win"].astype(float).to_numpy()

    best, best_loss = None, np.inf
    for k in k_grid:
        for hca in hca_grid:
            ratings = pregame_elo_ratings(games, k=k, hca=hca).set_index("game_id")
            r = ratings.loc[val["game_id"]]
            p = np.array([_expected(h, a, hca) for h, a in
                          zip(r["home_pregame_elo"], r["away_pregame_elo"])])
            p = np.clip(p, 1e-12, 1 - 1e-12)
            loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
            if loss < best_loss:
                best, best_loss = (float(k), float(hca)), loss
    return best
