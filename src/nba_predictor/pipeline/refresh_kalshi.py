"""Kalshi moneylines as the market side of "against the odds".

Free, public and keyless, unlike the RapidAPI sportsbook feed (which returned 429 on the daily refresh and costs a
request per game). Each upcoming game with an open Kalshi moneyline gets one `h2h` market row per side, bookmaker
"Kalshi": the market probability is the two mid prices normalised to sum to 1 (no Shin step needed for a two-outcome
exchange price), the edge is the model's probability minus that. A side with no bid, or a spread over 8 points, is
skipped: a stale or empty quote is not a market. A Kalshi fault stores nothing and raises nothing.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import requests

from nba_predictor.data import kalshi_lines
from nba_predictor.odds.value_bets import compute_edge
from nba_predictor.tracking import store

logger = logging.getLogger(__name__)

MAX_SPREAD = 0.08
MATCH_HOURS = 30.0


def _american(p: float) -> int:
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def _mid(row: dict) -> float | None:
    bid, ask = row.get("yes_bid"), row.get("yes_ask")
    if not bid or not ask or ask < bid or ask - bid > MAX_SPREAD:
        return None
    return (bid + ask) / 2


def _hours_apart(a: str | None, b: str | None) -> float:
    try:
        fa = datetime.fromisoformat(a.replace("Z", "+00:00")); fb = datetime.fromisoformat(b.replace("Z", "+00:00"))
        return abs((fa - fb).total_seconds()) / 3600
    except (AttributeError, ValueError):
        return float("inf")


def refresh_kalshi_markets(schedule: list[dict], db_path: Path, session=None, now: datetime | None = None) -> int:
    upcoming = [g for g in schedule if not g.get("completed")]
    if not upcoming:
        return 0
    now = now or datetime.now(timezone.utc)
    try:
        raw = kalshi_lines.fetch_series(session or requests.Session(), "KXNBAGAME", "open")
    except Exception as exc:  # noqa: BLE001 - a market-feed fault must not take the refresh down
        logger.warning("kalshi moneyline fetch failed (%s); no market rows stored", exc)
        return 0
    ts = now.isoformat(timespec="seconds")
    by_event: dict[str, dict[str, dict]] = {}
    for m in raw:
        row = kalshi_lines.normalize(m, "KXNBAGAME", ts)
        event, ticker = row["event_ticker"] or "", row["ticker"] or ""
        if event and ticker.startswith(event + "-"):
            by_event.setdefault(event, {})[ticker[len(event) + 1:]] = row

    stored = 0
    created_at = now.isoformat()
    for game in upcoming:
        sides = {game["home_team"], game["away_team"]}
        match = next((s for s in by_event.values() if sides <= set(s) and _hours_apart(next(iter(s.values())).get("occurrence"), game.get("tip_off")) <= MATCH_HOURS), None)
        pick = store.get_latest_prediction_for_game(db_path, game["game_id"])
        if match is None or pick is None:
            continue
        mids = {team: _mid(match[team]) for team in sides}
        if any(v is None for v in mids.values()):
            continue
        total = sum(mids.values())
        for team, mid in mids.items():
            market_p = mid / total
            model_p = pick["home_win_prob"] if team == game["home_team"] else 1 - pick["home_win_prob"]
            store.insert_market_prediction(
                db_path, game_id=game["game_id"], market="h2h", selection=team, model_probability=model_p, market_probability=market_p,
                edge=compute_edge(model_p, market_p), bookmaker="Kalshi", american_odds=_american(market_p), created_at=created_at,
            )
            stored += 1
    return stored
