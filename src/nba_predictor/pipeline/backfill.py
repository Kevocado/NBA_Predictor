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
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from nba_predictor import config
from nba_predictor.pipeline.ingest import (
    enrich_with_boxscores,
    fetch_schedule_range,
    to_training_frame,
)

#: An NBA season runs from early October to the following April.
#:
#: The start is deliberately EARLY. This was 20 October and it was wrong: the
#: first games of a season are in the first week of October, and Phase A's own
#: training frame opens on 2025-10-02. A bound of Oct 20 silently dropped about
#: three weeks per season -- and the whole current season, whose games run
#: 2026-10-01 to 2026-10-07 and were therefore absent from a backfill run on
#: 2026-10-07. The end is the other side of the same coin: late enough to hold
#: the whole playoffs run.
#:
#: The end is late June, not April. The playoffs run into June and their games
#: are real training rows like any other; an April bound dropped them, which is
#: how a season ends up "complete" while missing the games that decided it.
#:
#: Over-wide costs a few empty scoreboard queries per season and is harmless,
#: because the loop asks ESPN for each date and takes what comes back.
SEASON_START = (9, 25)
SEASON_END = (6, 30)

logger = logging.getLogger(__name__)


def season_bounds(season: int, *, today: date | None = None) -> tuple[str, str]:
    """The date range covering one season, where `season` is its starting year.

    `season=2024` is Oct 2024 - Apr 2025. `season=2023` is Oct 2023 - Apr 2024.

    **A season in progress is clamped to today.** Asking for the current season
    used to return Oct 20 - Apr 15, which silently dropped the opening weeks of
    the season that is actually being played -- and those are the games the
    Phase A comparison is scored on, so the frame would have been quietly
    missing exactly the rows it is measured against.
    """
    start = f"{season:04d}-{SEASON_START[0]:02d}-{SEASON_START[1]:02d}"
    end = f"{season + 1:04d}-{SEASON_END[0]:02d}-{SEASON_END[1]:02d}"
    now = today or date.today()
    if start <= now.isoformat() < end:
        end = now.isoformat()
    return start, end


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
    fetch=None,
    enrich=None,
    log=print,
    today: date | None = None,
) -> dict:
    """Warm the cache for `seasons`, then write the combined training cache.

    `fetch` and `enrich` are the two real network stages, injected so tests pass
    fakes. `fetch(start, end)` returns the games in that range; `enrich(games)`
    returns them with box-score fields where one exists.

    They default to None rather than to the functions themselves so the CLI --
    which passes nothing -- still picks them up at call time, and so a test can
    patch the module attribute and have the CLI honour it. A default bound at
    def time captures the original and silently ignores the patch.
    """
    fetch = fetch or fetch_schedule_range
    enrich = enrich or enrich_with_boxscores
    training_path = training_path or (config.DATA_DIR / "cache" / "training" / "games.json")

    games: list[dict] = []
    per_season: dict[str, dict] = {}
    for season in seasons:
        start, end = season_bounds(season, today=today)
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

    # **Merge, never overwrite.** A season that failed is absent from `games`,
    # and this file is written wholesale -- so overwriting would delete every row
    # that season had contributed from earlier runs. Rescuing the run from a DNS
    # failure by dropping that season's data is the same loss, quieter. The
    # existing file is the union of previous runs; a game fetched again keeps the
    # new row (identical, since it is the same endpoint) and is deduped on
    # game_id.
    training_path.parent.mkdir(parents=True, exist_ok=True)
    previous = _load_existing(training_path)

    # Refusing rather than writing is the point of this guard: with no seasons
    # asked for and nothing to carry over, the run would write an EMPTY file
    # that reads as a successful, empty backfill. Checked before the sort,
    # because an empty frame carries no columns to sort by.
    if training_df.empty and previous.empty:
        raise ValueError("the backfill produced no training rows; refusing to write an empty cache")

    merged = pd.concat([previous, training_df], ignore_index=True) if len(previous) else training_df
    merged = merged.drop_duplicates(subset=["game_id"], keep="last").sort_values("game_date")
    payload = merged.to_dict(orient="records")

    training_path.write_text(json.dumps(payload))
    log(f"Wrote {len(payload)} training rows to {training_path} "
        f"({len(training_df)} new this run, {len(previous)} carried over)")

    return {
        "seasons": per_season,
        "n_training_rows": len(payload),
        "n_new_rows": len(training_df),
        "n_carried_over": len(previous),
        "training_path": str(training_path),
        "earliest_game": str(merged["game_date"].min()) if len(merged) else None,
        "latest_game": str(merged["game_date"].max()) if len(merged) else None,
        # A season that failed is named here so a partial run cannot be mistaken
        # for a complete one. Absent means every season fetched.
        "failed_seasons": [s for s, stats in per_season.items() if "failed" in stats],
    }


def _load_existing(training_path: Path) -> pd.DataFrame:
    """Whatever is already in the training cache, as a frame. Empty when absent
    or unreadable -- an unreadable cache is not a reason to refuse to write."""
    if not training_path.exists():
        return pd.DataFrame()
    try:
        return pd.DataFrame(json.loads(training_path.read_text()))
    except (json.JSONDecodeError, OSError, ValueError):
        return pd.DataFrame()


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

    # Refuse before fetching. `--seasons 0` asks for nothing, and writing an
    # empty training cache over a good one is silent, permanent data loss.
    if args.seasons <= 0:
        parser.error("--seasons must be at least 1")

    seasons = prior_seasons(args.seasons)
    print(f"Backfilling seasons: {seasons}")
    summary = backfill_seasons(seasons, training_path=args.training_path)
    print(json.dumps(summary, indent=2))

    # A partial backfill is not a successful one. Exiting 0 on a run that lost a
    # season is how a caller concludes the fetch is done when it is not.
    if summary["failed_seasons"]:
        print(
            f"INCOMPLETE: these seasons failed and were not fetched this run: "
            f"{', '.join(summary['failed_seasons'])}. Re-run to resume from the cache.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())