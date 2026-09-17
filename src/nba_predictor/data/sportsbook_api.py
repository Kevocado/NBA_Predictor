"""NBA odds via RapidAPI's Sportsbook API
(https://rapidapi.com/sportsbook-api-sportsbook-api-default/api/sportsbook-api2).

Endpoints, headers, and response shape below are confirmed against the live
API (competition list, NBA/WNBA events, real priced markets on a live WNBA
game) — the module this replaced called a fabricated api.the-odds-api.com/
sportsbook/... path that 404s.

Confirmed live:
- Base URL: https://sportsbook-api2.p.rapidapi.com/v0
- Auth: X-RapidAPI-Key / X-RapidAPI-Host headers (not the the-odds-api.com
  `Authorization` header style).
- NBA competition key: d791-wddv-30fU (from GET /v0/competitions,
  shortName "NBA").
- No bulk odds endpoint — GET /v0/competitions/{key}/events lists
  fixtures with market *keys* but no prices; GET /v0/events?eventKeys=
  returns one event with each market's `outcomes` populated (keyed by
  bookmaker) once a book has posted a price — empty `outcomes` for a
  market with no live prices yet is normal, not an error.
- Each outcome is `{modifier, payout, type, source, participantKey,
  participant}`: `payout` is a **decimal** price (e.g. 1.91), `modifier`
  is the spread/total line, `type` is "WIN" for moneyline/spread or
  "OVER"/"UNDER" for totals, `participant` is the team the outcome is
  for (null for totals). Team-level markets (MONEYLINE, POINT_SPREAD,
  POINT_TOTAL) have the *market's own* `participantKey: null`; a
  player-level market would have that populated instead — no such
  market was observed live during development (NBA/WNBA off-season at
  implementation time), so get_player_props() below is written against
  that schema signal, not a guessed market-type name.
"""

import json
import time
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from nba_predictor import config
from nba_predictor.data.team_reference import TEAMS

SPORTSBOOK_API_HOST = "sportsbook-api2.p.rapidapi.com"
SPORTSBOOK_API_BASE_URL = f"https://{SPORTSBOOK_API_HOST}/v0"
SPORTSBOOK_NBA_COMPETITION_KEY = "d791-wddv-30fU"

CACHE_DIR = config.CACHE_DIR / "sportsbook"
CACHE_TTL_SECONDS = 6 * 60 * 60

MARKET_TYPE_TO_NAME = {
    "MONEYLINE": "h2h",
    "POINT_SPREAD": "spread",
    "POINT_TOTAL": "total",
}

_NAME_TO_ABBREVIATION = {team.name: team.abbreviation for team in TEAMS}


class SportsbookAPIKeyMissing(RuntimeError):
    pass


def _get_api_key() -> str:
    if not config.SPORTSBOOK_API_KEY:
        raise SportsbookAPIKeyMissing("SPORTSBOOK_API_KEY is not set")
    return config.SPORTSBOOK_API_KEY


def _headers() -> dict:
    return {"X-RapidAPI-Key": _get_api_key(), "X-RapidAPI-Host": SPORTSBOOK_API_HOST}


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.json"


def _is_fresh(path: Path, ttl_seconds: int) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl_seconds


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=60), reraise=True)
def _get(url: str, params: dict | None = None) -> dict:
    response = requests.get(url, headers=_headers(), params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def decimal_to_american(decimal_odds: float) -> int:
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))


def _selection_for(outcome: dict, market_type: str) -> str:
    if market_type == "POINT_TOTAL":
        return outcome.get("type", "").lower()
    participant = outcome.get("participant")
    if participant is None:
        return outcome.get("type", "")
    return _NAME_TO_ABBREVIATION.get(participant.get("name"), participant.get("shortName", ""))


def fetch_nba_events_raw(force_refresh: bool = False) -> list[dict]:
    """Upcoming NBA events with market keys attached (no prices yet)."""
    cache_path = _cache_path("nba_events")
    if not force_refresh and _is_fresh(cache_path, CACHE_TTL_SECONDS):
        return json.loads(cache_path.read_text())

    data = _get(f"{SPORTSBOOK_API_BASE_URL}/competitions/{SPORTSBOOK_NBA_COMPETITION_KEY}/events")
    events = data.get("events", [])
    cache_path.write_text(json.dumps(events))
    return events


def fetch_event_odds_raw(event_key: str, force_refresh: bool = False) -> dict | None:
    """Real prices for one event. One request per call, cached per event."""
    cache_path = _cache_path(f"event_{event_key}")
    if not force_refresh and _is_fresh(cache_path, CACHE_TTL_SECONDS):
        return json.loads(cache_path.read_text())

    data = _get(f"{SPORTSBOOK_API_BASE_URL}/events", params={"eventKeys": event_key})
    groups = data.get("events", [])
    event = groups[0][0] if groups and groups[0] else None
    if event is not None:
        cache_path.write_text(json.dumps(event))
    return event


def get_odds(event_key: str) -> list[dict]:
    """Real priced team-market rows for one event.

    Each row: market ("h2h"/"spread"/"total"), selection (team
    abbreviation for h2h/spread, "over"/"under" for total), bookmaker,
    american_odds (converted from the API's decimal payout), point (the
    spread/total line, None for h2h).
    """
    event = fetch_event_odds_raw(event_key)
    if event is None:
        return []

    rows = []
    for market in event.get("markets", []):
        market_name = MARKET_TYPE_TO_NAME.get(market.get("type"))
        if market_name is None or market.get("participantKey") is not None:
            continue
        for bookmaker, outcomes in market.get("outcomes", {}).items():
            for outcome in outcomes:
                rows.append(
                    {
                        "market": market_name,
                        "selection": _selection_for(outcome, market.get("type")),
                        "bookmaker": bookmaker,
                        "american_odds": decimal_to_american(outcome["payout"]),
                        "point": outcome.get("modifier") if market_name != "h2h" else None,
                    }
                )
    return rows


def get_player_props(event_key: str) -> list[dict]:
    """Player-level market rows (market's own participantKey populated), if any are posted."""
    event = fetch_event_odds_raw(event_key)
    if event is None:
        return []

    rows = []
    for market in event.get("markets", []):
        if market.get("participantKey") is None:
            continue
        participant = market.get("participant") or {}
        for bookmaker, outcomes in market.get("outcomes", {}).items():
            for outcome in outcomes:
                rows.append(
                    {
                        "player": participant.get("name"),
                        "market_type": market.get("type"),
                        "bookmaker": bookmaker,
                        "outcome_type": outcome.get("type"),
                        "american_odds": decimal_to_american(outcome["payout"]) if outcome.get("payout") else None,
                        "point": outcome.get("modifier"),
                    }
                )
    return rows
