"""Keep upcoming player projections on the CURRENT roster.

Bug (live, 2026-10-10): the fixture detail showed one team's predictions and a
trickle of the other's. `/games/401909837` (CHI v TOR) had 28 predicted players,
of whom 2 played for CHI; BOS and WAS were the same. The CHI/BOS rows were
exactly the roster of the team's last pre-preseason game, and none of the
players from the preseason game after it had a projection.

Root cause: an upcoming game's player projections are a roster ESTIMATE, taken
from the team's latest completed game when `score_upcoming_player_props` last
ran -- and that only runs inside `run_ingest`, which in production is the manual
`POST /admin/refresh-full` (the daily CI uses `--skip-predictions`). Nothing
re-scored when a team played another game, so the estimate stayed frozen at the
roster of whatever game preceded the last manual refresh.

This re-scores exactly the upcoming games whose estimate predates a completed
game of either team, then goes quiet until the next one.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

from nba_predictor.tracking import store

logger = logging.getLogger(__name__)

#: Upcoming games further out than this are not worth a projection yet.
HORIZON_DAYS = 14
#: Box-score history behind the rolling features. The full-season ingest uses a
#: whole season; this reaches back over the summer to the last regular-season
#: games so a player's rolling average is not three preseason minutes.
HISTORY_DAYS = 240
#: A game is "complete" this long after tip-off (stored `tip_off` is the start).
GAME_LENGTH = timedelta(hours=3)
TICK_SECONDS = 1800


def _completed_at(game: dict) -> pd.Timestamp:
    tip = game.get("tip_off")
    start = pd.Timestamp(tip) if tip else pd.Timestamp(f"{game['game_date']}T23:59:59Z")
    return start + GAME_LENGTH


def stale_upcoming_games(schedule: list[dict], db_path: Path, now: pd.Timestamp) -> list[dict]:
    """Upcoming games whose newest projection predates a completed game of either team."""
    team_last: dict[str, pd.Timestamp] = {}
    for g in schedule:
        if g.get("completed"):
            done = _completed_at(g)
            for side in ("home_team", "away_team"):
                if g.get(side) and done > team_last.get(g[side], done - timedelta(days=1)):
                    team_last[g[side]] = done
    with store.get_connection(db_path) as conn:
        newest = {
            r[0]: pd.Timestamp(r[1])
            for r in conn.execute(
                "SELECT game_id, MAX(created_at) FROM player_prediction_snapshots GROUP BY game_id"
            )
        }
    horizon = now + timedelta(days=HORIZON_DAYS)
    out = []
    for g in schedule:
        if g.get("completed") or pd.Timestamp(f"{g['game_date']}T00:00Z") > horizon:
            continue
        latest_game = max((team_last[t] for t in (g["home_team"], g["away_team"]) if t in team_last), default=None)
        if latest_game is not None and latest_game > newest.get(g["game_id"], pd.Timestamp.min.tz_localize("UTC")):
            out.append(g)
    return out


def refresh_stale_player_props(
    schedule: list[dict], db_path: Path, models_dir: Path, *, fetch=None, score=None, now=None
) -> int:
    """Re-score the stale upcoming games. Returns rows stored (0 when nothing was stale)."""
    from nba_predictor.pipeline.ingest import fetch_player_boxscores, score_upcoming_player_props

    fetch = fetch or fetch_player_boxscores
    score = score or score_upcoming_player_props
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    stale = stale_upcoming_games(schedule, db_path, now)
    if not stale:
        return 0
    since = (now - timedelta(days=HISTORY_DAYS)).date().isoformat()
    completed = [g for g in schedule if g.get("completed") and g["game_date"] >= since]
    return score(completed + stale, fetch(completed), models_dir, db_path, model_version="roster-refresh")


def start_player_props_refresher(db_path: Path, schedule_path: Path, models_dir: Path, *, tick=TICK_SECONDS) -> threading.Thread:
    from nba_predictor.services.schedule_repository import load_schedule

    def _run() -> None:
        while True:
            try:
                n = refresh_stale_player_props(load_schedule(schedule_path), db_path, models_dir)
                if n:
                    logger.info("re-scored %d player projections on the current roster", n)
            except Exception:  # noqa: BLE001 - a failed pass keeps the previous projections
                logger.exception("player-props roster refresh failed")
            time.sleep(tick)

    thread = threading.Thread(target=_run, name="player-props-refresher", daemon=True)
    thread.start()
    return thread
