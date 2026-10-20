"""Evaluate availability B3 on completed 2023-24, 2024-25 and 2025-26 games.

Two projections, both made before the game from regular-season box scores only:

* the model: EWMA minutes and starter probability over each team's last ten
  games, with the latest-team roster rule from ``features/lineups.py``;
* the baseline: the team's previous game -- its five starters and each
  player's minutes in that game.

The comparison is paired by game. The script reports the minutes MAE and the
number of correctly projected starters (0-5 per team), each with a 95% paired
bootstrap interval. Negative MAE difference is better; positive starter
difference is better.

The ESPN caches have no starter column. ESPN's box score lists the five
starters first, and ``with_starter_flags`` uses that order. It was checked
against the live ESPN summary endpoint on 100 random 2023-2026 games: 2,181
matched player rows, zero mismatches, and two cached rows absent from the live
response. That check is not part of this offline run.

Run from a checkout:
    uv run --extra dev python tools/evaluate_lineups.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_predictor.features.lineups import (  # noqa: E402
    WINDOW,
    project_team,
    starting_five,
    with_starter_flags,
)

SEASONS = {
    "2023-24": ("2023-10-24", "2024-04-14"),
    "2024-25": ("2024-10-22", "2025-04-13"),
    "2025-26": ("2025-10-21", "2026-04-12"),
}
N_RESAMPLES = 2000
SEED = 20261020


def _load_boxscores(data_dir: Path) -> dict[str, list[dict]]:
    """game_id -> rows with starter flags, for every cached player box score."""
    out = {}
    for path in (data_dir / "espn").glob("player_boxscore_*.json"):
        game_id = path.stem.split("_")[-1]
        rows = json.loads(path.read_text())["rows"]
        out[game_id] = with_starter_flags(rows)
    return out


def _by_team(rows: list[dict]) -> dict[str, list[dict]]:
    teams: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        teams[str(row["team"])].append(row)
    return teams


def _team_errors(projections, actual_rows, previous_rows):
    """Minutes errors on the union of projected, actual and previous players."""
    projected = {p.player_id: p.expected_minutes for p in projections}
    actual = {str(r["player_id"]): float(r.get("minutes") or 0.0) for r in actual_rows}
    previous = {str(r["player_id"]): float(r.get("minutes") or 0.0) for r in previous_rows}
    players = set(projected) | set(actual) | set(previous)

    model_error = np.mean([abs(projected.get(pid, 0.0) - actual.get(pid, 0.0)) for pid in players])
    baseline_error = np.mean([abs(previous.get(pid, 0.0) - actual.get(pid, 0.0)) for pid in players])

    model_five = {p.player_id for p in starting_five(projections)}
    actual_five = {str(r["player_id"]) for r in actual_rows if r.get("started")}
    baseline_five = {str(r["player_id"]) for r in previous_rows if r.get("started")}

    return {
        "model_error": float(model_error),
        "baseline_error": float(baseline_error),
        "model_correct": len(model_five & actual_five),
        "baseline_correct": len(baseline_five & actual_five),
        "model_exact": float(model_five == actual_five and len(actual_five) == 5),
        "baseline_exact": float(baseline_five == actual_five and len(actual_five) == 5),
    }


def _bootstrap_difference(model: list[float], baseline: list[float]) -> dict:
    model_arr = np.asarray(model, dtype=float)
    baseline_arr = np.asarray(baseline, dtype=float)
    difference = model_arr - baseline_arr
    rng = np.random.default_rng(SEED)
    draws = np.asarray(
        [
            difference[rng.integers(0, len(difference), len(difference))].mean()
            for _ in range(N_RESAMPLES)
        ]
    )
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "model": float(model_arr.mean()),
        "baseline": float(baseline_arr.mean()),
        "difference": float(difference.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "excludes_zero": bool(low > 0 or high < 0),
    }


def evaluate(data_dir: Path) -> dict:
    games = json.loads((data_dir / "training" / "games_all.json").read_text())
    boxscores = _load_boxscores(data_dir)

    seasons = {}
    for season, (start, end) in SEASONS.items():
        season_games = sorted(
            (g for g in games if start <= g["game_date"] <= end),
            key=lambda g: (g["game_date"], g["game_id"]),
        )
        team_games: dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW))
        latest_team: dict[str, str] = {}
        rows_out = []
        skipped = 0

        for game in season_games:
            game_id = str(game["game_id"])
            rows = boxscores.get(game_id)
            if not rows:
                skipped += 1
                continue
            teams = _by_team(rows)
            home, away = game["home_team"], game["away_team"]
            home_rows, away_rows = teams.get(home), teams.get(away)
            if not home_rows or not away_rows:
                skipped += 1
                continue
            if len([r for r in home_rows if r["started"]]) != 5 or len(
                [r for r in away_rows if r["started"]]
            ) != 5:
                skipped += 1
                continue

            if team_games[home] and team_games[away]:
                home_proj = project_team(home, list(team_games[home]), latest_team=latest_team)
                away_proj = project_team(away, list(team_games[away]), latest_team=latest_team)
                home_metrics = _team_errors(home_proj, home_rows, team_games[home][-1]["players"])
                away_metrics = _team_errors(away_proj, away_rows, team_games[away][-1]["players"])
                rows_out.append(
                    {
                        "game_id": game_id,
                        "model_error": (home_metrics["model_error"] + away_metrics["model_error"]) / 2,
                        "baseline_error": (home_metrics["baseline_error"] + away_metrics["baseline_error"]) / 2,
                        "model_correct": (home_metrics["model_correct"] + away_metrics["model_correct"]) / 2,
                        "baseline_correct": (home_metrics["baseline_correct"] + away_metrics["baseline_correct"]) / 2,
                        "model_exact": (home_metrics["model_exact"] + away_metrics["model_exact"]) / 2,
                        "baseline_exact": (home_metrics["baseline_exact"] + away_metrics["baseline_exact"]) / 2,
                    }
                )
            else:
                skipped += 1

            for team, team_rows in ((home, home_rows), (away, away_rows)):
                team_games[team].append({"game_id": game_id, "players": team_rows})
                for row in team_rows:
                    latest_team[str(row["player_id"])] = team

        seasons[season] = {"rows": rows_out, "skipped": skipped, "games": len(season_games)}

    all_rows = [row for season in seasons.values() for row in season["rows"]]
    return {
        "seasons": seasons,
        "overall": {
            "games": len(all_rows),
            "minutes_mae": _bootstrap_difference(
                [r["model_error"] for r in all_rows],
                [r["baseline_error"] for r in all_rows],
            ),
            "starters_correct": _bootstrap_difference(
                [r["model_correct"] for r in all_rows],
                [r["baseline_correct"] for r in all_rows],
            ),
            "exact_five": _bootstrap_difference(
                [r["model_exact"] for r in all_rows],
                [r["baseline_exact"] for r in all_rows],
            ),
        },
    }


def _print_result(name: str, result: dict, better: str) -> None:
    direction = "negative is better" if better == "lower" else "positive is better"
    print(
        f"| {name} | {result['model']:.4f} | {result['baseline']:.4f} | "
        f"{result['difference']:+.4f} | [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] | "
        f"{'yes' if result['excludes_zero'] else 'no'} ({direction}) |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cache"))
    args = parser.parse_args()

    result = evaluate(args.data_dir)
    print("| season | games | skipped |")
    print("|---|---|---|")
    for season, data in result["seasons"].items():
        print(f"| {season} | {len(data['rows'])} | {data['skipped']} |")

    overall = result["overall"]
    print(f"\nEvaluated games: {overall['games']} (paired by game_id)")
    print("| metric | model | last game | difference | 95% CI | excludes zero |")
    print("|---|---|---|---|---|---|")
    _print_result("Minutes MAE", overall["minutes_mae"], "lower")
    _print_result("Starters correct (0-5 per team)", overall["starters_correct"], "higher")
    _print_result("Exact five (share)", overall["exact_five"], "higher")


if __name__ == "__main__":
    main()
