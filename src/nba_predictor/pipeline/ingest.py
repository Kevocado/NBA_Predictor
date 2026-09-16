import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd

from nba_predictor import config
from nba_predictor.data import espn
from nba_predictor.data.team_reference import get_team
from nba_predictor.features.build import build_training_frame
from nba_predictor.features.context import current_streak
from nba_predictor.features.ratings import compute_possessions
from nba_predictor.models.game_outcome import predict_win_probability
from nba_predictor.pipeline.retrain import run_retrain_pipeline
from nba_predictor.tracking import store

BOX_FIELDS = ["fgm", "fga", "fg3m", "tov", "oreb", "dreb", "fta"]


def _daterange(start_date: str, end_date: str) -> list[str]:
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    days = (end - start).days
    return [(start + timedelta(days=i)).isoformat() for i in range(days + 1)]


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
    """Minimal {game_id, game_date, home_team, away_team} shape for the schedule repository."""
    return [
        {"game_id": g["game_id"], "game_date": g["game_date"], "home_team": g["home_team"], "away_team": g["away_team"]}
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

    store.init_db(config.TRACKING_DB_PATH)
    stored = score_and_store_predictions(training_df, models_dir, config.TRACKING_DB_PATH, model_version)
    print(f"  stored {stored} real predictions for browsing in the UI")


if __name__ == "__main__":
    main()
