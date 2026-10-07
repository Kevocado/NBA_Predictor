"""Multi-season backfill: warm the ESPN cache for prior seasons.

The biggest model lever available is simply having more games. Phase A trained
on about one season (1,118 games), so every rolling feature -- form, rest,
pace, ratings -- had barely a season of history behind it.

**What this does and does not do.** It walks the dates of N prior seasons,
fetches each day's scoreboard and each finished game's box score, and writes the
combined training cache. It does NOT retrain, does NOT touch the tracking
database, and does NOT score predictions. That separation is the point: a
five-season backfill through `run_ingest` would write ~6,000 predictions into
the tracking DB and train on them, which is a backtest dressed up as a backfill
and would corrupt the record the site publishes.

**Resumability is free.** Every ESPN read is disk-cached per date and per event
(`data/espn.py`), so a re-run re-reads what it has and fetches only what is
missing. Interrupt it and run it again.

**No network in tests.** Every function here takes its fetchers as arguments, so
a test passes fakes and asserts on what was asked for and what was written. The
one test that touches `requests` at all asserts that it does not.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path

from nba_predictor import config
from nba_predictor.pipeline.ingest import (
    enrich_with_boxscores,
    fetch_schedule_range,
    to_training_frame,
)

#: An NBA season runs from late October to the following April. These bounds are
#: the outermost dates a season can contain; the loop asks ESPN for each date and
#: takes what comes back, so the bounds only have to be generous, not exact.
SEASON_START = (10, 20)
SEASON_END = (4, 15)

logger = logging.getLogger(__name__)


def season_bounds(season: int) -> tuple[str, str]:
    """The date range covering one season, where `season` is its starting year.

    `season=2024` is Oct 2024 - Apr 2025. `season=2023` is Oct 2023 - Apr 2024.
    """
    return (
        f"{season:04d}-{SEASON_START[0]:02d}-{SEASON_START[1]:02d}",
        f"{season + 1:04d}-{SEASON_END[0]:02d}-{SEASON_END[1]:02d}",
    )


def prior_seasons(n: int, *, before: date | None = None) -> list[int]:
    """The starting years of the `n` seasons before the one containing `before`.

    The season containing a date is the calendar year for Oct-Dec and the year
    before for Jan-Sep -- the same rule `pipeline/retrain.py` uses, so "this
    season" means the same thing in both places.
    """
    today = before or date.today()
    current = today.year if today.month >= 10 else today.year - 1
    return [current - offset for offset in range(n, 0, -1)]


def backfill_seasons(
    seasons: list[int],
    *,
    training_path: Path | None = None,
    fetch=fetch_schedule_range,
    enrich=enrich_with_boxscores,
    log=print,
) -> dict:
    """Warm the cache for `seasons`, then write the combined training cache.

    `fetch` and `enrich` are the two real network stages, injected so tests pass
    fakes. `fetch(start, end)` returns the games in that range; `enrich(games)`
    returns them with box-score fields where one exists.
    """
    training_path = training_path or (config.DATA_DIR / "cache" / "training" / "games.json")

    games: list[dict] = []
    per_season: dict[str, dict] = {}
    for season in seasons:
        start, end = season_bounds(season)
        log(f"Season {season}: {start} .. {end}")
        # One season failing must not abandon the others. This is a multi-hour
        # fetch and the network will drop: the first run of this took a whole
        # season and then died on a DNS failure inside the second, throwing away
        # everything it had fetched because it had not written its file yet.
        # Failing per season keeps that fetch, names the season that failed, and
        # the re-run resumes from the cache.
        try:
            found = fetch(start, end)
            enriched = enrich(found)
        except Exception as exc:  # noqa: BLE001 - one season's failure is reported, not fatal
            logger.error("Season %s failed, continuing: %s: %s", season, type(exc).__name__, exc)
            per_season[str(season)] = {
                "start": start, "end": end, "failed": f"{type(exc).__name__}: {exc}",
            }
            log(f"  FAILED: {type(exc).__name__}: {exc}")
            continue
        completed = [g for g in enriched if g.get("completed") and "home_fgm" in g]
        per_season[str(season)] = {
            "start": start,
            "end": end,
            "games_found": len(found),
            "games_with_box": len(completed),
        }
        log(f"  {len(found)} games, {len(completed)} with a box score")
        games.extend(enriched)

    # A backfill must not invent rows: only games that are finished AND carry a
    # box score can become training rows, and `to_training_frame` is already the
    # one place that decides that.
    training_df = to_training_frame(games)
    training_path.parent.mkdir(parents=True, exist_ok=True)
    payload = training_df.to_dict(orient="records")
    training_path.write_text(json.dumps(payload))
    log(f"Wrote {len(payload)} training rows to {training_path}")

    return {
        "seasons": per_season,
        "n_training_rows": len(payload),
        "training_path": str(training_path),
        "earliest_game": str(training_df["game_date"].min()) if len(training_df) else None,
        "latest_game": str(training_df["game_date"].max()) if len(training_df) else None,
        # A season that failed is named here so a partial run cannot be mistaken
        # for a complete one. Absent means every season fetched.
        "failed_seasons": [s for s, stats in per_season.items() if "failed" in stats],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seasons", type=int, default=3,
        help="how many prior seasons to backfill (default 3)",
    )
    parser.add_argument(
        "--training-path", type=Path, default=None,
        help="where to write the training cache (default data/cache/training/games.json)",
    )
    args = parser.parse_args(argv)

    seasons = prior_seasons(args.seasons)
    print(f"Backfilling seasons: {seasons}")
    summary = backfill_seasons(seasons, training_path=args.training_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())