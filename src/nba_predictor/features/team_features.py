"""team_features.py — single-pass chronological team features (elo, adjusted off/def, pace).

Computes pregame features for every game from completed games only (shift-1 by construction).
No lookahead: the value for game G is recorded BEFORE G's own result is folded into state.
This mirrors predictor-hub's feats.py team_blocks with the same causal discipline.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from nba_predictor.features.ratings import compute_possessions, elo_expected, elo_update


WIN = 10
ELO_K, ELO_SCALE = 20.0, 400.0
HOME_COURT = 50.0  # points-ish prior for elo_mov


def _poss(fga: float, oreb: float, tov: float, fta: float) -> float:
    return fga - oreb + tov + 0.44 * fta


def _shrunk_fg3_pct(team_fg3m: deque, team_fg3a: deque, lg3: float) -> float:
    m = sum(team_fg3m)
    att = sum(team_fg3a)
    return (m + 50 * lg3) / (att + 50) if (att + 50) > 0 else lg3


def compute_team_blocks(games: pd.DataFrame) -> pd.DataFrame:
    """One row per game_id with home_/away_ columns for:
    elo_diff/100, home/away adj_off/5, adj_def/5, pace_roll (x-100)/5.

    Returns a DataFrame indexed by game_id with scaled H features ready for FEATURE_COLUMNS.
    """
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    # State updated AFTER each game's row is emitted
    elo = defaultdict(lambda: 1500.0)
    elo_mov = defaultdict(lambda: 1500.0)
    hist = defaultdict(lambda: defaultdict(lambda: deque(maxlen=WIN)))  # team -> stat -> last WIN values
    cum = defaultdict(lambda: defaultdict(float))  # team -> running sums (season-to-date) for opp-adjustment
    ngames = defaultdict(int)
    last_date = {}
    dates_played = defaultdict(list)
    lg_n, lg_ortg_sum = 0, 0.0
    lg3_made = lg3_att = 0.0

    out = []

    def roll(team: str, stat: str) -> float:
        d = hist[team][stat]
        return float(np.mean(d)) if len(d) else np.nan

    for r in games.itertuples():
        h, a = r.home_team, r.away_team
        lg_ortg = lg_ortg_sum / lg_n if lg_n else np.nan
        lg3 = lg3_made / lg3_att if lg3_att else 0.35
        row = {"game_id": r.game_id}

        # Safely get box score columns (may not exist for upcoming games)
        def g(name):
            return getattr(r, name, np.nan)

        # Check if this game has box scores (completed game)
        has_box = (
            pd.notna(g("home_fga")) and pd.notna(g("home_oreb")) and pd.notna(g("home_tov")) and pd.notna(g("home_fta")) and
            pd.notna(g("away_fga")) and pd.notna(g("away_oreb")) and pd.notna(g("away_tov")) and pd.notna(g("away_fta")) and
            pd.notna(g("home_pts")) and pd.notna(g("away_pts"))
        )

        for side, t, o in (("home", h, a), ("away", a, h)):
            row[f"{side}_elo"] = elo[t]
            row[f"{side}_elo_mov"] = elo_mov[t]
            for s in ("pace", "ortg", "drtg", "adj_off", "adj_def"):
                row[f"{side}_{s}_roll"] = roll(t, s)
            # shrunk 3P% (not used in H but computed for completeness)
            row[f"{side}_fg3_pct_shrunk"] = _shrunk_fg3_pct(hist[t]["fg3m"], hist[t]["fg3a"], lg3)
            ld = last_date.get(t)
            gap = (pd.Timestamp(r.game_date) - pd.Timestamp(ld)).days - 1 if ld else 7
            row[f"{side}_rest_team"] = min(gap, 7)
            row[f"{side}_g_last7"] = sum(1 for d in dates_played[t] if 0 < (pd.Timestamp(r.game_date) - pd.Timestamp(d)).days <= 7)
        out.append(row)

        # ---- fold this game's result into state AFTER the row is recorded
        if has_box:
            ph = _poss(g("home_fga"), g("home_oreb"), g("home_tov"), g("home_fta"))
            pa = _poss(g("away_fga"), g("away_oreb"), g("away_tov"), g("away_fta"))
            poss = (ph + pa) / 2

            res = {}
            for side, t, o, pts, opp_pts, fga, fg3m, fg3a in (
                ("home", h, a, g("home_pts"), g("away_pts"), g("home_fga"), g("home_fg3m"), g("home_fg3a")),
                ("away", a, h, g("away_pts"), g("home_pts"), g("away_fga"), g("away_fg3m"), g("away_fg3a")),
            ):
                ortg, drtg = pts / poss * 100, opp_pts / poss * 100
                o_alld = cum[o]["drtg"] / ngames[o] if ngames[o] else (lg_ortg if lg_n else ortg)
                o_scor = cum[o]["ortg"] / ngames[o] if ngames[o] else (lg_ortg if lg_n else drtg)
                res[t] = dict(
                    pace=poss,
                    ortg=ortg,
                    drtg=drtg,
                    adj_off=ortg - o_alld,
                    adj_def=drtg - o_scor,
                    fg3a_rate=(fg3a / fga) if fga and not np.isnan(fg3a) else np.nan,
                    fg3m=fg3m,
                    fg3a=fg3a,
                )

            for t, d in res.items():
                for k, v in d.items():
                    if not (isinstance(v, float) and np.isnan(v)):
                        hist[t][k].append(v)
                cum[t]["ortg"] += d["ortg"]
                cum[t]["drtg"] += d["drtg"]
                ngames[t] += 1
                lg_n += 1
                lg_ortg_sum += d["ortg"]
                if not np.isnan(d["fg3a"]):
                    lg3_made += d["fg3m"]
                    lg3_att += d["fg3a"]
                last_date[t] = r.game_date
                dates_played[t].append(r.game_date)

            # Elo (repo constants) and a margin-of-victory variant
            exp = 1.0 / (1.0 + 10 ** (-(elo[h] - elo[a]) / ELO_SCALE))
            win = 1.0 if g("home_pts") > g("away_pts") else 0.0
            elo[h] += ELO_K * (win - exp)
            elo[a] -= ELO_K * (win - exp)
            expm = 1.0 / (1.0 + 10 ** (-(elo_mov[h] - elo_mov[a] + HOME_COURT) / ELO_SCALE))
            mult = np.log(abs(g("home_pts") - g("away_pts")) + 1.0) * 0.7 + 0.5
            elo_mov[h] += ELO_K * mult * (win - expm)
            elo_mov[a] -= ELO_K * mult * (win - expm)

    df = pd.DataFrame(out).set_index("game_id")

    # league-to-date fill for the first games, so no row is dropped for lack of history
    for c in df.columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].expanding().mean().shift(1)).fillna(df[c].mean())

    # Compute scaled H features (matching step4_team.py scaling)
    df["elo_diff_s"] = (df.home_elo - df.away_elo) / 100
    df["elo_mov_diff_s"] = (df.home_elo_mov - df.away_elo_mov) / 100
    df["home_adj_off_s"] = df.home_adj_off_roll / 5
    df["away_adj_off_s"] = df.away_adj_off_roll / 5
    df["home_adj_def_s"] = df.home_adj_def_roll / 5
    df["away_adj_def_s"] = df.away_adj_def_roll / 5
    df["home_pace_roll_s"] = (df.home_pace_roll - 100) / 5
    df["away_pace_roll_s"] = (df.away_pace_roll - 100) / 5

    # Return only the H feature columns
    h_cols = [
        "elo_diff_s",
        "home_adj_off_s", "away_adj_off_s",
        "home_adj_def_s", "away_adj_def_s",
        "home_pace_roll_s", "away_pace_roll_s",
    ]
    return df[h_cols].reset_index()