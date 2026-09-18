import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_predictor import config
from nba_predictor.data import espn
from nba_predictor.data.team_reference import get_team
from nba_predictor.features.build import build_feature_frame, build_training_frame
from nba_predictor.features.player_stats import build_player_feature_frame
from nba_predictor.features.context import current_streak
from nba_predictor.features.ratings import compute_possessions
from nba_predictor.models.game_outcome import predict_win_probability
from nba_predictor.models.manifest import build_manifest, write_manifest
from nba_predictor.models.player_props import predict_player_stat, train_player_stat_model
from nba_predictor.pipeline.retrain import run_retrain_pipeline
from nba_predictor.tracking import store

BOX_FIELDS = ["fgm", "fga", "fg3m", "tov", "oreb", "dreb", "fta"]


def _daterange(start_date: str, end_date: str) -> list[str]:
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    days = (end - start).days
    return [(start + timedelta(days=i)).isoformat() for i in range(days + 1)]


def _parse_made_attempted(value: str) -> tuple[float, float]:
    try:
        made, attempted = value.split("-")
        return float(made), float(attempted)
    except (ValueError, AttributeError):
        return 0.0, 0.0


def store_player_outcomes(training_df: pd.DataFrame, db_path: Path) -> int:
    """Stores the real actual stat value for every completed player-game
    row, for each of the four tracked stat targets — settlement data."""
    recorded_at = datetime.now(timezone.utc).isoformat()
    stored = 0
    for _, row in training_df.iterrows():
        for stat, column in PLAYER_STAT_TARGET_COLUMNS.items():
            store.insert_player_outcome(
                db_path,
                game_id=row["game_id"],
                player_id=row["player_id"],
                stat=stat,
                actual_value=float(row[column]),
                recorded_at=recorded_at,
            )
            stored += 1
    return stored


PLAYER_STAT_TARGET_COLUMNS = {"points": "points", "rebounds": "rebounds", "assists": "assists", "threes": "fg3m"}


def to_player_scoring_frame(games: list[dict], player_boxscores: dict[str, list[dict]], roster_window: int = 5) -> pd.DataFrame:
    """Estimate upcoming roster from recent completed games; rows for
    upcoming games have None stats (causal — never touches a future result)."""
    # Gather recent completed games per team
    team_recent: dict[str, list[dict]] = {}
    for g in games:
        if g.get("completed"):
            for side in ["home_team", "away_team"]:
                team = g.get(side)
                if team:
                    team_recent.setdefault(team, []).append({"game_id": g["game_id"], "game_date": g["game_date"]})
    # For upcoming games, estimate roster from recent completed boxscores
    upcoming = [g for g in games if not g.get("completed")]
    rows = []
    for g in upcoming:
        for side in ["home_team", "away_team"]:
            team = g.get(side)
            if team:
                # Find the latest completed game for this team from the games list
                team_completed_ids = [g_["game_id"] for g_ in games if g_.get("completed") and (g_.get("home_team") == team or g_.get("away_team") == team)]
                if team_completed_ids:
                    last_id = team_completed_ids[-1]
                    if last_id in player_boxscores:
                        for box_row in player_boxscores[last_id]:
                            if box_row.get("team") == team:
                                rows.append({
                                    "player_id": box_row["player_id"],
                                    "player_name": box_row["player_name"],
                                    "team": team,
                                    "game_id": g["game_id"],
                                    "game_date": g["game_date"],
                                    "points": None,
                                    "rebounds": None,
                                    "assists": None,
                                    "fg3m": None,
                                    "minutes": None,
                                })
    # Also include completed player's real rows so feature frame can compute
    completed_rows = to_player_training_frame(games, player_boxscores)
    # Combine, drop duplicates for upcoming (keep first per player/game)
    combined = pd.concat([completed_rows, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset=["player_id", "game_id"], keep="first").reset_index(drop=True)
    return combined


def score_upcoming_player_props(games: list[dict], player_boxscores: dict[str, list[dict]], models_dir: Path, db_path: Path, model_version: str) -> int:
    """Score upcoming player props using roster estimate + causal rolling features."""
    scoring_df = to_player_scoring_frame(games, player_boxscores)
    frame, feature_cols = build_player_feature_frame(scoring_df)
    upcoming_frame = frame[frame["points"].isna()].reset_index(drop=True)
    if len(upcoming_frame) == 0:
        return 0

    models = {stat: joblib.load(models_dir / f"player_{stat}_model.pkl") for stat in PLAYER_STAT_TARGET_COLUMNS}
    predictions = {stat: predict_player_stat(model, upcoming_frame[feature_cols]) for stat, model in models.items()}

    created_at = datetime.now(timezone.utc).isoformat()
    stored = 0
    for i, row in upcoming_frame.iterrows():
        for stat in PLAYER_STAT_TARGET_COLUMNS:
            store.insert_player_prediction(
                db_path,
                game_id=row["game_id"],
                player_id=row["player_id"],
                stat=stat,
                predicted_value=float(predictions[stat][i]),
                created_at=created_at,
            )
            stored += 1
    return stored


def train_player_prop_models(training_df: pd.DataFrame, models_dir: Path, model_version: str, trained_at: str) -> dict:
    """Trains one XGBRegressor per stat target on real per-player rolling
    features. Metrics are in-sample (fit then scored on the same rows) —
    same honesty tradeoff score_and_store_predictions already accepts for
    the team backtest, not a proper chronological holdout, stated here
    rather than hidden."""
    models_dir.mkdir(parents=True, exist_ok=True)
    frame, feature_cols = build_player_feature_frame(training_df)

    metrics: dict[str, dict] = {}
    for stat, target_col in PLAYER_STAT_TARGET_COLUMNS.items():
        if len(frame) == 0:
            metrics[stat] = {"mae": None}
            continue
        model = train_player_stat_model(frame[feature_cols], frame[target_col])
        joblib.dump(model, models_dir / f"player_{stat}_model.pkl")
        preds = predict_player_stat(model, frame[feature_cols])
        metrics[stat] = {"mae": float(np.mean(np.abs(preds - frame[target_col])))}

    manifest = build_manifest(
        model_names=list(PLAYER_STAT_TARGET_COLUMNS.keys()),
        metrics=metrics,
        model_version=model_version,
        trained_at=trained_at,
    )
    write_manifest(manifest, models_dir / "player_props_manifest.json")
    return manifest


def score_and_store_player_predictions(training_df: pd.DataFrame, models_dir: Path, db_path: Path, model_version: str) -> int:
    """Backtest: scores every completed player-game row that survived
    feature assembly with the trained per-stat models, stores each as a
    tracked player prediction."""
    frame, feature_cols = build_player_feature_frame(training_df)
    if len(frame) == 0:
        return 0

    models = {stat: joblib.load(models_dir / f"player_{stat}_model.pkl") for stat in PLAYER_STAT_TARGET_COLUMNS}
    predictions = {stat: predict_player_stat(model, frame[feature_cols]) for stat, model in models.items()}

    created_at = datetime.now(timezone.utc).isoformat()
    stored = 0
    for i, row in frame.iterrows():
        for stat in PLAYER_STAT_TARGET_COLUMNS:
            store.insert_player_prediction(
                db_path,
                game_id=row["game_id"],
                player_id=row["player_id"],
                stat=stat,
                predicted_value=float(predictions[stat][i]),
                created_at=created_at,
            )
            stored += 1
    return stored


def fetch_schedule_range(start_date: str, end_date: str) -> list[dict]:
    """All games (completed and upcoming) for each date in the range, via ESPN."""
    games = []
    for day in _daterange(start_date, end_date):
        games.extend(espn.get_scoreboard(day))
    return games


def enrich_with_boxscores(games: list[dict]) -> list[dict]:
    """Adds flat home_*/away_* box-score fields to completed games with a full box score."""
    enriched = []
    for game in games:
        game = dict(game)
        if game["completed"]:
            box = espn.get_boxscore(game["game_id"])
            home_box, away_box = box.get(game["home_team"]), box.get(game["away_team"])
            if home_box and away_box and all(k in home_box for k in BOX_FIELDS) and all(k in away_box for k in BOX_FIELDS):
                for field in BOX_FIELDS:
                    game[f"home_{field}"] = home_box[field]
                    game[f"away_{field}"] = away_box[field]
        enriched.append(game)
    return enriched


def to_schedule_cache(games: list[dict]) -> list[dict]:
    """Schedule repository shape, including final scores/completion status
    so settlement (track record, calibration) can determine actual outcomes
    without a separate DB table."""
    return [
        {
            "game_id": g["game_id"],
            "game_date": g["game_date"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "completed": g.get("completed", False),
            "home_pts": g.get("home_pts"),
            "away_pts": g.get("away_pts"),
        }
        for g in games
    ]


def to_training_frame(games: list[dict]) -> pd.DataFrame:
    """Completed games with full box scores, shaped for features.build.build_training_frame."""
    rows = []
    for g in games:
        if not g.get("completed") or f"home_{BOX_FIELDS[0]}" not in g:
            continue
        row = {
            "game_id": g["game_id"],
            "game_date": g["game_date"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "home_pts": g["home_pts"],
            "away_pts": g["away_pts"],
            "home_win": int(g["home_pts"] > g["away_pts"]),
        }
        for field in BOX_FIELDS:
            row[f"home_{field}"] = g[f"home_{field}"]
            row[f"away_{field}"] = g[f"away_{field}"]
        rows.append(row)
    return pd.DataFrame(rows)


def to_scoring_frame(games: list[dict]) -> pd.DataFrame:
    """All games (completed + upcoming), shaped for
    features.build.build_feature_frame. Upcoming games (or completed games
    missing a full box score) get None for home_win and every box-score
    field — build_feature_frame computes their rolling features from real
    prior games only, never touching a nonexistent result."""
    rows = []
    for g in games:
        row = {
            "game_id": g["game_id"],
            "game_date": g["game_date"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
        }
        if g.get("completed") and f"home_{BOX_FIELDS[0]}" in g:
            row["home_win"] = int(g["home_pts"] > g["away_pts"])
            for field in BOX_FIELDS:
                row[f"home_{field}"] = g[f"home_{field}"]
                row[f"away_{field}"] = g[f"away_{field}"]
        else:
            row["home_win"] = None
            for field in BOX_FIELDS:
                row[f"home_{field}"] = None
                row[f"away_{field}"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def score_upcoming_games(games: list[dict], models_dir: Path, db_path: Path, model_version: str) -> int:
    """Scores every not-yet-completed game using real prior-game rolling
    features (via to_scoring_frame + build_feature_frame) and stores the
    result in the same predictions table score_and_store_predictions
    writes to. Only rows for games that are not completed are scored —
    a completed game already gets its backtest prediction from
    score_and_store_predictions."""
    scoring_df = to_scoring_frame(games)
    frame, feature_cols = build_feature_frame(scoring_df)
    upcoming_frame = frame[frame["home_win"].isna()].reset_index(drop=True)
    if len(upcoming_frame) == 0:
        return 0

    # Trusted artifacts: these .pkl files are written by run_retrain_pipeline
    # (via joblib.dump) in this same pipeline run — not from an external or
    # user-uploaded source (same trust boundary as score_and_store_predictions).
    win_model = joblib.load(models_dir / "win_probability_model.pkl")
    margin_model = joblib.load(models_dir / "margin_model.pkl")
    total_model = joblib.load(models_dir / "total_model.pkl")

    win_probs = predict_win_probability(win_model, upcoming_frame[feature_cols])
    margins = margin_model.predict(upcoming_frame[feature_cols])
    totals = total_model.predict(upcoming_frame[feature_cols])

    created_at = datetime.now(timezone.utc).isoformat()
    for i, row in upcoming_frame.iterrows():
        store.insert_prediction(
            db_path,
            game_id=row["game_id"],
            created_at=created_at,
            model_version=model_version,
            home_win_prob=float(win_probs[i]),
            predicted_margin=float(margins[i]),
            predicted_total=float(totals[i]),
        )
    return len(upcoming_frame)


def _completed_with_box(games: list[dict]) -> list[dict]:
    return [g for g in games if g.get("completed") and f"home_{BOX_FIELDS[0]}" in g]


def compute_team_hub(games: list[dict]) -> list[dict]:
    completed = sorted(_completed_with_box(games), key=lambda g: g["game_date"])
    stats: dict[str, dict] = {}

    for g in completed:
        for side, opp_side in [("home", "away"), ("away", "home")]:
            abbr = g[f"{side}_team"]
            pts, opp_pts = g[f"{side}_pts"], g[f"{opp_side}_pts"]
            s = stats.setdefault(abbr, {"pts": 0, "opp_pts": 0, "poss": 0.0, "games": 0, "results": []})
            s["games"] += 1
            s["pts"] += pts
            s["opp_pts"] += opp_pts
            s["poss"] += compute_possessions(g[f"{side}_fga"], g[f"{side}_fta"], g[f"{side}_oreb"], g[f"{side}_tov"])
            s["results"].append("W" if pts > opp_pts else "L")

    rows = []
    for abbr, s in stats.items():
        team = get_team(abbr)
        n = s["games"]
        rows.append(
            {
                "abbreviation": abbr,
                "conference": team.conference,
                "division": team.division,
                "wins": s["results"].count("W"),
                "losses": s["results"].count("L"),
                "points_per_game": round(s["pts"] / n, 1),
                "opp_points_per_game": round(s["opp_pts"] / n, 1),
                "net_rating": round((s["pts"] - s["opp_pts"]) / n, 1),
                "pace": round(s["poss"] / n, 1),
                "streak": current_streak(s["results"]),
            }
        )
    return rows


def compute_power_rankings(games: list[dict]) -> list[dict]:
    from nba_predictor.features.ratings import elo_expected, elo_update, init_elo_ratings

    completed = sorted(_completed_with_box(games), key=lambda g: g["game_date"])
    teams = sorted({g["home_team"] for g in completed} | {g["away_team"] for g in completed})
    ratings = init_elo_ratings(teams)
    history: dict[str, list[float]] = {t: [ratings[t]] for t in teams}

    for g in completed:
        home, away = g["home_team"], g["away_team"]
        expected_home = elo_expected(ratings[home], ratings[away], home_adjustment=50)
        actual_home = 1.0 if g["home_pts"] > g["away_pts"] else 0.0
        ratings[home] = elo_update(ratings[home], expected_home, actual_home)
        ratings[away] = elo_update(ratings[away], 1 - expected_home, 1 - actual_home)
        history[home].append(ratings[home])
        history[away].append(ratings[away])

    ranked = sorted(ratings.items(), key=lambda kv: -kv[1])
    rows = []
    for rank, (team, rating) in enumerate(ranked, start=1):
        h = history[team]
        trend = "steady"
        if len(h) >= 2:
            trend = "up" if h[-1] > h[-2] else "down" if h[-1] < h[-2] else "steady"
        rows.append({"rank": rank, "abbreviation": team, "power_rating": round(rating, 1), "trend": trend})
    return rows


def compute_standings(games: list[dict]) -> list[dict]:
    completed = _completed_with_box(games)
    records: dict[str, list[int]] = {}

    for g in completed:
        winner = g["home_team"] if g["home_pts"] > g["away_pts"] else g["away_team"]
        loser = g["away_team"] if winner == g["home_team"] else g["home_team"]
        records.setdefault(winner, [0, 0])[0] += 1
        records.setdefault(loser, [0, 0])[1] += 1

    standings = []
    for conference in ["East", "West"]:
        conf_teams = [t for t in records if get_team(t).conference == conference]
        conf_teams.sort(key=lambda t: -(records[t][0] / (records[t][0] + records[t][1])))
        if not conf_teams:
            continue
        leader_wins, leader_losses = records[conf_teams[0]]
        for seed, team in enumerate(conf_teams, start=1):
            wins, losses = records[team]
            win_pct = wins / (wins + losses) if (wins + losses) else 0.0
            games_back = ((leader_wins - wins) + (losses - leader_losses)) / 2
            status = "clinched" if seed <= 6 else "play-in" if seed <= 10 else "eliminated"
            standings.append(
                {
                    "conference": conference,
                    "seed": seed,
                    "abbreviation": team,
                    "wins": wins,
                    "losses": losses,
                    "win_pct": round(win_pct, 3),
                    "games_back": round(games_back, 1),
                    "playoff_status": status,
                }
            )
    return standings


def fetch_player_boxscores(games: list[dict]) -> dict[str, list[dict]]:
    """game_id -> per-player box score rows, for completed games only.

    A second ESPN request per game (get_player_boxscore is cached
    separately from get_boxscore) — callers should bound `games` to a
    recent window rather than a full season to avoid doubling the whole
    backfill's request count.
    """
    result = {}
    for game in games:
        if game.get("completed"):
            result[game["game_id"]] = espn.get_player_boxscore(game["game_id"])
    return result


def to_player_training_frame(games: list[dict], player_boxscores: dict[str, list[dict]]) -> pd.DataFrame:
    """One row per (player, completed game) with real box-score stats,
    shaped for features.player_stats.build_player_feature_frame."""
    games_by_id = {g["game_id"]: g for g in games}
    rows = []
    for game_id, boxscore_rows in player_boxscores.items():
        game = games_by_id.get(game_id)
        if game is None:
            continue
        for row in boxscore_rows:
            fg3m, _ = _parse_made_attempted(row["three_made_attempted"])
            rows.append(
                {
                    "player_id": row["player_id"],
                    "player_name": row["player_name"],
                    "team": row["team"],
                    "game_id": game_id,
                    "game_date": game["game_date"],
                    "points": row["points"],
                    "rebounds": row["rebounds"],
                    "assists": row["assists"],
                    "fg3m": fg3m,
                    "minutes": row["minutes"],
                }
            )
    return pd.DataFrame(rows)


def compute_player_hub(games: list[dict], player_boxscores: dict[str, list[dict]]) -> list[dict]:
    """Real per-player aggregates from real box scores.

    `rating`/`live_form_rating` are a simple game-score-style composite
    (PTS + 0.4*REB + 0.7*AST) over the full window vs. the last 5 games —
    a simplified, clearly-labeled formula, not the league's own advanced
    metric. `usage_rate` is each game's (player FGA + 0.44*player FTA) /
    team FGA, averaged — a field-goal-attempt-share proxy, not the full
    NBA usage% formula (which also needs team possessions/minutes-on-court
    context this data doesn't carry per-player).
    """
    games_by_id = {g["game_id"]: g for g in games}
    per_player_games: dict[str, list[dict]] = {}

    ordered_game_ids = sorted(player_boxscores, key=lambda gid: games_by_id.get(gid, {}).get("game_date", ""))
    for game_id in ordered_game_ids:
        game = games_by_id.get(game_id)
        if game is None:
            continue
        for row in player_boxscores[game_id]:
            team_fga = game["home_fga"] if row["team"] == game["home_team"] else game.get("away_fga")
            fgm, fga = _parse_made_attempted(row["fg_made_attempted"])
            fg3m, fg3a = _parse_made_attempted(row["three_made_attempted"])
            ftm, fta = _parse_made_attempted(row["ft_made_attempted"])
            per_player_games.setdefault(row["player_id"], []).append(
                {
                    "player_name": row["player_name"],
                    "team": row["team"],
                    "position": row["position"],
                    "minutes": row["minutes"],
                    "points": row["points"],
                    "rebounds": row["rebounds"],
                    "assists": row["assists"],
                    "fgm": fgm, "fga": fga, "fg3m": fg3m, "fg3a": fg3a, "ftm": ftm, "fta": fta,
                    "usage_share": (fga + 0.44 * fta) / team_fga if team_fga else 0.0,
                }
            )

    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _game_score(pts: float, reb: float, ast: float) -> float:
        return pts + 0.4 * reb + 0.7 * ast

    rows = []
    for player_id, entries in per_player_games.items():
        total_fgm = sum(e["fgm"] for e in entries)
        total_fga = sum(e["fga"] for e in entries)
        total_fg3m = sum(e["fg3m"] for e in entries)
        total_fg3a = sum(e["fg3a"] for e in entries)
        total_ftm = sum(e["ftm"] for e in entries)
        total_fta = sum(e["fta"] for e in entries)
        last5 = entries[-5:]

        rows.append(
            {
                "player_id": player_id,
                "player_name": entries[-1]["player_name"],
                "team": entries[-1]["team"],
                "position": entries[-1]["position"],
                "rating": round(
                    _game_score(
                        _avg([e["points"] for e in entries]),
                        _avg([e["rebounds"] for e in entries]),
                        _avg([e["assists"] for e in entries]),
                    ),
                    1,
                ),
                "live_form_rating": round(
                    _game_score(
                        _avg([e["points"] for e in last5]),
                        _avg([e["rebounds"] for e in last5]),
                        _avg([e["assists"] for e in last5]),
                    ),
                    1,
                ),
                "points_per_game": round(_avg([e["points"] for e in entries]), 1),
                "rebounds_per_game": round(_avg([e["rebounds"] for e in entries]), 1),
                "assists_per_game": round(_avg([e["assists"] for e in entries]), 1),
                "fg_pct": round(total_fgm / total_fga, 3) if total_fga else 0.0,
                "three_pt_pct": round(total_fg3m / total_fg3a, 3) if total_fg3a else 0.0,
                "ft_pct": round(total_ftm / total_fta, 3) if total_fta else 0.0,
                "usage_rate": round(_avg([e["usage_share"] for e in entries]), 3),
                "minutes_per_game": round(_avg([e["minutes"] for e in entries]), 1),
            }
        )
    return rows


def score_and_store_predictions(games_df: pd.DataFrame, models_dir: Path, db_path: Path, model_version: str) -> int:
    """Scores every game that survives feature assembly with the trained model
    and stores the result as a tracked prediction. Returns the number stored.

    Games in `games_df` are expected to already be completed (their real
    pre-game rolling features are leak-free since build_training_frame only
    ever looks at *prior* games via shift(1)) - this is how a backtest lets
    us show real model output for real historical games in the UI.
    """
    # Trusted artifacts: these .pkl files are written by run_retrain_pipeline
    # (via joblib.dump) in this same pipeline run — not from an external or
    # user-uploaded source.
    win_model = joblib.load(models_dir / "win_probability_model.pkl")
    margin_model = joblib.load(models_dir / "margin_model.pkl")
    total_model = joblib.load(models_dir / "total_model.pkl")

    frame, feature_cols = build_training_frame(games_df)
    if len(frame) == 0:
        return 0

    win_probs = predict_win_probability(win_model, frame[feature_cols])
    margins = margin_model.predict(frame[feature_cols])
    totals = total_model.predict(frame[feature_cols])

    created_at = datetime.now(timezone.utc).isoformat()
    for i, row in frame.iterrows():
        store.insert_prediction(
            db_path,
            game_id=row["game_id"],
            created_at=created_at,
            model_version=model_version,
            home_win_prob=float(win_probs[i]),
            predicted_margin=float(margins[i]),
            predicted_total=float(totals[i]),
        )
    return len(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest real NBA schedule/box-score data and refresh caches.")
    parser.add_argument("--start", default="2026-02-16")
    parser.add_argument("--end", default="2026-03-22")
    parser.add_argument(
        "--player-hub-days", type=int, default=21,
        help="How many days (most recent, within --start/--end) to fetch per-player box scores for. "
        "0 skips Player Hub entirely. Bounded by default since it's a second ESPN request per game.",
    )
    parser.add_argument(
        "--skip-predictions", action="store_true",
        help="Skip scoring/storing predictions into the tracking DB. For CI contexts (e.g. a scheduled "
        "workflow with no persistent tracking.db) that only need fresh schedule/hub caches and a "
        "retrained model committed to git — predictions belong on the deployed server's live DB, "
        "not a stateless CI runner's throwaway one.",
    )
    args = parser.parse_args()

    print(f"Fetching schedule {args.start} to {args.end} from ESPN...")
    games = fetch_schedule_range(args.start, args.end)
    print(f"  {len(games)} games found")

    print("Fetching box scores for completed games...")
    games = enrich_with_boxscores(games)

    schedule_path = config.DATA_DIR / "cache" / "schedule" / "games.json"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(json.dumps(to_schedule_cache(games)))
    print(f"  wrote {schedule_path}")

    hub_dir = config.DATA_DIR / "cache" / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    (hub_dir / "teams.json").write_text(json.dumps(compute_team_hub(games)))
    (hub_dir / "rankings.json").write_text(json.dumps(compute_power_rankings(games)))
    (hub_dir / "standings.json").write_text(json.dumps(compute_standings(games)))

    if args.player_hub_days > 0:
        cutoff = (date.fromisoformat(args.end) - timedelta(days=args.player_hub_days)).isoformat()
        recent_games = [g for g in games if g["game_date"] >= cutoff]
        print(f"Fetching player box scores for {len(recent_games)} games since {cutoff}...")
        player_boxscores = fetch_player_boxscores(recent_games)
        (hub_dir / "players.json").write_text(json.dumps(compute_player_hub(recent_games, player_boxscores)))
    else:
        (hub_dir / "players.json").write_text(json.dumps([]))
    print(f"  wrote hub caches to {hub_dir}")

    training_df = to_training_frame(games)
    print(f"Training on {len(training_df)} completed games with full box scores...")

    models_dir = config.PROJECT_ROOT / "models"
    model_version = datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
    manifest = run_retrain_pipeline(
        training_df, models_dir, model_version=model_version, trained_at=datetime.now(timezone.utc).isoformat()
    )
    print(f"  trained {model_version}: {manifest['metrics']}")

    if args.skip_predictions:
        print("  --skip-predictions set: not scoring/storing predictions")
    else:
        store.init_db(config.TRACKING_DB_PATH)
        stored = score_and_store_predictions(training_df, models_dir, config.TRACKING_DB_PATH, model_version)
        print(f"  stored {stored} real predictions for browsing in the UI")


if __name__ == "__main__":
    main()
