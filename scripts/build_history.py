"""Freeze completed seasons into data/history/nba_history.json.gz so the retrain can use several seasons.

Why: the ingest window starts at the previous season's Oct 1 (one season of box scores). Walk-forward on 2019-2025
openers (nba-eval/race3.py) shows training on all earlier seasons beats the one-season window by 0.0045 log-loss
(95% interval excludes zero; first 15 games 0.0054) and 0.09 margin MAE. Fetching 8 seasons from ESPN on every CI run
or container refresh is not realistic, so completed seasons are frozen here from a machine that has the ESPN cache.

    python scripts/build_history.py --before 2025-10-01 [--cache data/cache]

Games are stored as ingest game dicts (team box score fields included); players as the slim rows the availability
features need. Re-run once a year, moving --before to the new window start, and commit the file.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

from nba_predictor.pipeline import ingest

OUT = Path("data/history/nba_history.json.gz")
PLAYER_COLS = ["game_id", "game_date", "team", "player_id", "minutes", "points", "rebounds", "assists"]


def build(cache: Path, before: str) -> dict:
    games = json.loads((cache / "training" / "games_all.json").read_text())
    games = [dict(g, completed=True) for g in games if g["game_date"] < before]
    boxes = {Path(f).stem.split("_")[-1]: json.loads(Path(f).read_text())["rows"] for f in glob.glob(str(cache / "espn" / "player_boxscore_*.json"))}
    players = ingest.to_player_training_frame(games, {k: v for k, v in boxes.items() if k in {g["game_id"] for g in games}})
    return {
        "before": before,
        "games": games,
        "players": players[PLAYER_COLS].to_dict(orient="records"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="first date NOT in the history, i.e. the ingest window start")
    ap.add_argument("--cache", default="data/cache")
    args = ap.parse_args()
    data = build(Path(args.cache), args.before)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"{len(data['games'])} games, {len(data['players'])} player rows -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
