import pandas as pd


def compute_four_factors(
    fgm: float,
    fga: float,
    fg3m: float,
    tov: float,
    oreb: float,
    opp_dreb: float,
    fta: float,
) -> dict:
    if fgm is None or fga is None or fg3m is None or tov is None or oreb is None or opp_dreb is None or fta is None:
        return {"efg_pct": float("nan"), "tov_rate": float("nan"), "orb_pct": float("nan"), "ft_rate": float("nan")}
    efg_pct = (fgm + 0.5 * fg3m) / fga if fga else 0.0
    tov_rate = tov / (fga + 0.44 * fta + tov) if (fga + 0.44 * fta + tov) else 0.0
    orb_pct = oreb / (oreb + opp_dreb) if (oreb + opp_dreb) else 0.0
    ft_rate = fta / fga if fga else 0.0
    return {
        "efg_pct": efg_pct,
        "tov_rate": tov_rate,
        "orb_pct": orb_pct,
        "ft_rate": ft_rate,
    }


def add_rolling_four_factors(games: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    games = games.sort_values(["team", "game_date"]).reset_index(drop=True)

    factor_rows = games.apply(
        lambda row: compute_four_factors(
            row["fgm"], row["fga"], row["fg3m"], row["tov"], row["oreb"], row["opp_dreb"], row["fta"]
        ),
        axis=1,
        result_type="expand",
    )
    games = pd.concat([games, factor_rows], axis=1)

    for factor in ["efg_pct", "tov_rate", "orb_pct", "ft_rate"]:
        games[f"{factor}_roll"] = games.groupby("team")[factor].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )

    return games
