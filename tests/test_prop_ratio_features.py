"""Causality and shape checks for features/prop_ratio.py (the pre-registered ratio candidates)."""
import copy

import numpy as np
import pandas as pd

from nba_predictor.features.prop_ratio import (
    BASE_QUANTITIES,
    add_base_quantities,
    ratio_candidates,
    team_game_table,
)


def _synthetic(seed=0):
    """Three teams, eight players, 40 games, a shuffled-but-deterministic schedule."""
    rng = np.random.default_rng(seed)
    teams = ["AAA", "BBB", "CCC"]
    games, players = [], []
    dates = pd.date_range("2024-01-01", periods=40, freq="D").strftime("%Y-%m-%d")
    for i, day in enumerate(dates):
        home, away = teams[i % 3], teams[(i + 1) % 3]
        gid = f"g{i:03d}"
        g = {"game_id": gid, "game_date": day, "home_team": home, "away_team": away}
        for side in ("home", "away"):
            g.update({
                f"{side}_pts": float(rng.integers(90, 130)), f"{side}_fga": float(rng.integers(75, 95)),
                f"{side}_fg3a": float(rng.integers(20, 45)), f"{side}_fta": float(rng.integers(12, 30)),
                f"{side}_tov": float(rng.integers(8, 20)), f"{side}_oreb": float(rng.integers(5, 15)),
                f"{side}_dreb": float(rng.integers(25, 35)), f"{side}_fg3m": float(rng.integers(8, 16)),
                f"{side}_fgm": float(rng.integers(30, 45)),
            })
        games.append(g)
        for team, opp in ((home, away), (away, home)):
            for k in range(2):
                players.append({
                    "game_id": gid, "game_date": day, "team": team, "player_id": f"{team}{k}",
                    "player_name": f"{team}{k}", "position": "G" if k == 0 else "F",
                    "minutes": float(rng.integers(10, 38)), "points": float(rng.integers(0, 35)),
                    "rebounds": float(rng.integers(0, 12)), "assists": float(rng.integers(0, 10)),
                    "fg3m": float(rng.integers(0, 6)), "fga": float(rng.integers(3, 25)),
                    "home_team": home, "away_team": away, "opponent": opp,
                })
    return games, pd.DataFrame(players)


def _features(games, players):
    frame = players.copy()
    return add_base_quantities(frame, team_game_table(games))


def test_pool_is_all_ordered_pairs_of_distinct_base_quantities():
    games, players = _synthetic()
    cands = ratio_candidates(_features(games, players))
    assert len(BASE_QUANTITIES) == 17
    assert len(cands) == 17 * 16 == 272
    assert len(set(cands)) == 272


def test_features_at_a_row_do_not_change_when_later_games_are_removed():
    games, players = _synthetic()
    full = _features(games, players)
    cutoff = "2024-02-01"
    g_cut = [g for g in games if g["game_date"] < cutoff]
    p_cut = players[players["game_date"] < cutoff]
    cut = _features(g_cut, p_cut)

    early_full = full[full["game_date"] < cutoff].sort_values(["player_id", "game_date"]).reset_index(drop=True)
    early_cut = cut.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    for col in BASE_QUANTITIES:
        np.testing.assert_allclose(early_full[col].to_numpy(float), early_cut[col].to_numpy(float), equal_nan=True, err_msg=col)


def test_a_game_cannot_see_its_own_stats():
    games, players = _synthetic()
    base = _features(games, players)

    # Perturb the stats of one game (both teams, and the player's own line in it) and check its row is unchanged.
    target_gid = "g020"
    g2 = copy.deepcopy(games)
    for g in g2:
        if g["game_id"] == target_gid:
            g["home_fga"] += 40.0
            g["away_pts"] += 50.0
            g["home_oreb"] += 9.0
    p2 = players.copy()
    mask = p2["game_id"] == target_gid
    p2.loc[mask, "points"] += 30.0
    p2.loc[mask, "minutes"] += 5.0
    pert = _features(g2, p2)

    key = ["player_id", "game_id"]
    a = base[base["game_id"] == target_gid].set_index(key)[BASE_QUANTITIES].sort_index()
    b = pert[pert["game_id"] == target_gid].set_index(key)[BASE_QUANTITIES].sort_index()
    np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)


def test_zero_denominator_is_nan_not_inf():
    frame = pd.DataFrame({q: [1.0, 1.0] for q in BASE_QUANTITIES})
    frame["p_min"] = [0.0, 10.0]
    frame["p_pts"] = [5.0, 20.0]
    val = ratio_candidates(frame)["p_pts/p_min"]
    assert np.isnan(val.iloc[0])
    assert val.iloc[1] == 2.0
