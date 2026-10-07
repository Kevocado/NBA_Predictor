"""ESPN data module for NBA schedule, box scores, and injuries.

Uses ESPN's public (keyless) site API. Endpoints verified against the
live API — not the stats.nba.com-style paths nba_api uses, which are
blocked from some network environments.
"""

import json
import os
import re
import threading
import time
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from nba_predictor import config
from nba_predictor.data.team_reference import TEAMS

_NAME_TO_ABBREVIATION = {team.name: team.abbreviation for team in TEAMS}
_KNOWN_ABBREVIATIONS = {team.abbreviation for team in TEAMS}

ESPN_SITE_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

ESPN_CACHE_DIR = config.CACHE_DIR / "espn"

# The availability gate reads the injury report on EVERY request. A cache with no age
# limit froze it: a player ruled Out later stayed ranked, and one who had returned
# stayed removed. Fifteen minutes is short against ESPN's update cadence and cheap
# against the request volume; a refetch that fails raises (the route answers 503)
# rather than serving a report older than this.
INJURY_CACHE_TTL_SECONDS = int(os.getenv("INJURY_CACHE_TTL_SECONDS", "900"))

# ESPN uses a handful of abbreviations that differ from team_reference's.
ESPN_ABBREVIATION_MAP = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
}


def normalize_abbreviation(espn_abbr: str) -> str:
    return ESPN_ABBREVIATION_MAP.get(espn_abbr, espn_abbr)


def _cache_path(endpoint: str, id_value: str) -> Path:
    safe_id = id_value.replace("/", "_").replace(":", "_")
    return ESPN_CACHE_DIR / f"{endpoint}_{safe_id}.json"


def _load_cache(endpoint: str, id_value: str) -> dict | None:
    cache_file = _cache_path(endpoint, id_value)
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _cache_age_seconds(endpoint: str, id_value: str) -> float:
    """Seconds since the cache file was last written; infinite when unreadable, so
    a cache whose age cannot be established is never trusted."""
    try:
        return max(0.0, time.time() - _cache_path(endpoint, id_value).stat().st_mtime)
    except OSError:
        return float("inf")


def _save_cache(endpoint: str, id_value: str, data: dict) -> None:
    cache_file = _cache_path(endpoint, id_value)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=32), reraise=True)
def _fetch_json(url: str, params: dict | None = None) -> dict:
    _throttle()
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


# Minimum seconds between two ESPN requests, and the time of the last one.
#
# The daily ingest makes a few dozen calls and is not affected either way. A
# multi-season backfill makes thousands: five seasons is ~410 scoreboard and
# ~6,000 box-score requests, and firing those at full speed is how a fetch gets
# throttled into 429s, which `retry` then turns into five slow failures per game.
# One floor for every caller is simpler than a per-call-site budget and cannot be
# bypassed by a new call site.
#
# Set ESPN_MIN_REQUEST_INTERVAL_SECONDS=0 to turn it off for a local fixture run
# that must not sleep. `sleep` is injected so the throttle is testable without
# real waiting, and the clock with it, so a test can assert the pacing rather
# than merely that a function was called.
ESPN_MIN_REQUEST_INTERVAL_SECONDS = float(os.getenv("ESPN_MIN_REQUEST_INTERVAL_SECONDS", "0.2"))
_last_request_at: float | None = None
_throttle_lock = threading.Lock()


def _throttle(sleep=time.sleep, clock=time.monotonic) -> None:
    """Block until at least `ESPN_MIN_REQUEST_INTERVAL_SECONDS` has passed since
    the previous request. A no-op when the interval is zero or negative."""
    global _last_request_at
    if ESPN_MIN_REQUEST_INTERVAL_SECONDS <= 0:
        return
    with _throttle_lock:
        now = clock()
        if _last_request_at is not None:
            wait = ESPN_MIN_REQUEST_INTERVAL_SECONDS - (now - _last_request_at)
            if wait > 0:
                sleep(wait)
                now = clock()
        _last_request_at = now


def get_scoreboard(date: str) -> list[dict]:
    """Games for a date (YYYY-MM-DD), with final scores when completed.

    Returns a list of dicts: game_id, game_date, home_team, away_team
    (normalized abbreviations), completed, home_pts, away_pts (None if
    not yet played).
    """
    cache_key = date
    cached = _load_cache("scoreboard", cache_key)
    if cached is not None:
        return cached["games"]

    espn_date = date.replace("-", "")
    data = _fetch_json(f"{ESPN_SITE_BASE}/scoreboard", params={"dates": espn_date})

    games = []
    for event in data.get("events", []):
        competitors = event["competitions"][0]["competitors"]
        home = next((c for c in competitors if c["homeAway"] == "home"), None)
        away = next((c for c in competitors if c["homeAway"] == "away"), None)
        if home is None or away is None or "abbreviation" not in home.get("team", {}) or "abbreviation" not in away.get("team", {}):
            # Non-standard entries (e.g. international preseason exhibitions
            # played under a special one-off team branding) don't have a
            # normal team structure — skip rather than crash, since these
            # aren't real regular-franchise games our model tracks anyway.
            continue

        home_abbr = normalize_abbreviation(home["team"]["abbreviation"])
        away_abbr = normalize_abbreviation(away["team"]["abbreviation"])
        if home_abbr not in _KNOWN_ABBREVIATIONS or away_abbr not in _KNOWN_ABBREVIATIONS:
            continue

        status = event["competitions"][0].get("status", {}).get("type", {})
        completed = bool(status.get("completed", False))

        games.append(
            {
                "game_id": event["id"],
                "game_date": date,
                # UTC start time; tracking uses it to tell pre-tip picks apart.
                "tip_off": event.get("date"),
                "home_team": home_abbr,
                "away_team": away_abbr,
                "completed": completed,
                "home_pts": int(home["score"]) if completed and "score" in home else None,
                "away_pts": int(away["score"]) if completed and "score" in away else None,
            }
        )

    _save_cache("scoreboard", cache_key, {"games": games})
    return games


_STAT_KEYS = {
    "fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("fg3m", "fg3a"),
    "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta"),
    "offensiveRebounds": ("oreb", None),
    "defensiveRebounds": ("dreb", None),
    "turnovers": ("tov", None),
}


def _parse_team_box(team_block: dict) -> dict:
    parsed: dict = {}
    for stat in team_block.get("statistics", []):
        mapping = _STAT_KEYS.get(stat.get("name"))
        if mapping is None:
            continue
        made_key, attempted_key = mapping
        value = stat.get("displayValue", "")
        if attempted_key is not None and "-" in value:
            made, attempted = value.split("-")
            parsed[made_key] = float(made)
            parsed[attempted_key] = float(attempted)
        else:
            try:
                parsed[made_key] = float(value)
            except ValueError:
                parsed[made_key] = 0.0
    return parsed


def get_boxscore(event_id: str) -> dict:
    """Team box score stats for a completed game, keyed by normalized abbreviation.

    Returns {"BOS": {"fgm":..., "fga":..., "fg3m":..., "ftm":..., "fta":...,
    "oreb":..., "dreb":..., "tov":...}, "MIA": {...}}. Missing/unparseable
    stats default to 0.0 via _parse_team_box's mapping absence.
    """
    cached = _load_cache("boxscore", event_id)
    if cached is not None:
        return cached

    data = _fetch_json(f"{ESPN_SITE_BASE}/summary", params={"event": event_id})
    teams = data.get("boxscore", {}).get("teams", [])

    result = {}
    for team_block in teams:
        abbr = normalize_abbreviation(team_block["team"]["abbreviation"])
        result[abbr] = _parse_team_box(team_block)

    _save_cache("boxscore", event_id, result)
    return result


def _to_float(value: str | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def get_player_boxscore(event_id: str) -> list[dict]:
    """Per-player box score rows for a completed game, one row per athlete
    who actually played (didNotPlay entries have no stats to parse).

    ESPN's summary response pairs a per-team `labels` array (e.g.
    ["MIN","PTS","FG","3PT","FT","REB","AST","TO","STL","BLK","OREB",
    "DREB","PF","+/-"]) with each athlete's `stats`, a parallel array of
    string values in the same order — confirmed live.
    """
    cached = _load_cache("player_boxscore", event_id)
    if cached is not None:
        return cached["rows"]

    data = _fetch_json(f"{ESPN_SITE_BASE}/summary", params={"event": event_id})
    player_blocks = data.get("boxscore", {}).get("players", [])

    rows = []
    for team_block in player_blocks:
        abbr = normalize_abbreviation(team_block["team"]["abbreviation"])
        for stat_group in team_block.get("statistics", []):
            labels = stat_group.get("labels", [])
            for entry in stat_group.get("athletes", []):
                if entry.get("didNotPlay") or not entry.get("stats"):
                    continue
                stats = dict(zip(labels, entry["stats"]))
                athlete = entry.get("athlete", {})
                rows.append(
                    {
                        "player_id": athlete.get("id", ""),
                        "player_name": athlete.get("displayName", ""),
                        "team": abbr,
                        "position": (athlete.get("position") or {}).get("abbreviation", ""),
                        "minutes": _to_float(stats.get("MIN")),
                        "points": _to_float(stats.get("PTS")),
                        "rebounds": _to_float(stats.get("REB")),
                        "assists": _to_float(stats.get("AST")),
                        "fg_made_attempted": stats.get("FG", "0-0"),
                        "three_made_attempted": stats.get("3PT", "0-0"),
                        "ft_made_attempted": stats.get("FT", "0-0"),
                    }
                )

    _save_cache("player_boxscore", event_id, {"rows": rows})
    return rows


# ESPN's injuries endpoint does NOT send athlete.id (measured 2026-10-01: 0 of
# 65 live entries had one), but every athlete carries links whose href embeds
# the same ESPN athlete id get_player_boxscore already stores as player_id --
# https://www.espn.com/nba/player/_/id/5105571/henri-veesaar, or
# sportscenter://...?uid=s:40~l:2~a:5105571. Reading it back is an exact key
# match against player_id, so no player name is ever compared to another. That
# matters: the injuries feed keys by name and props key by id, and a name join
# can resolve to the wrong person, which would remove the wrong player from a
# ranking. Verified against the live payload -- of the 65 entries, the 52 whose
# displayName was also in data/cache/hub/players.json had ids matching that
# player's player_id in 52 of 52 cases.
_PLAYERCARD_ID = re.compile(r"/nba/player/(?:_/)?id/(\d+)")
# ESPN's playercard link is the first link on the athlete and carries the id
# plainly. The uid form is a fallback, and only where a: follows one of the
# delimiters the uid actually uses -- the live href reads
# "...?uid=s:40~l:46~a:5105571&section=stats" -- so a word ending in "a:"
# inside some other parameter cannot be read as an id.
_SPORTSCENTER_UID = re.compile(r"[?&~]a:(\d+)(?![0-9])")


def athlete_id_from_links(links: list[dict] | None) -> str:
    """The ESPN athlete id from an athlete's own links, or "" when absent.

    "" means "could not resolve", never "unknown player": the caller treats it
    as resolving to nobody, which is the safe direction to fail.
    """
    for pattern in (_PLAYERCARD_ID, _SPORTSCENTER_UID):
        for link in links or ():
            found = pattern.search(link.get("href") or "")
            if found:
                return found.group(1)
    return ""


def _cached_injuries_have_ids(rows) -> bool:
    """A report cached before ids existed must not be reused: it would carry
    no resolvable id for every row and silently disable the availability gate,
    which is the one thing decision 6 forbids. Refetch instead."""
    return all(isinstance(row, dict) and "player_id" in row for row in rows)


def get_injuries() -> list[dict]:
    """League-wide current injury report.

    Rows carry the ESPN athlete id (see athlete_id_from_links) so the
    availability gate can exclude an out player by exact key. Names are kept
    for display only.
    """
    cached = _load_cache("injuries", "current")
    if (
        cached is not None
        and _cached_injuries_have_ids(cached.get("injuries"))
        and _cache_age_seconds("injuries", "current") < INJURY_CACHE_TTL_SECONDS
    ):
        return cached["injuries"]

    data = _fetch_json(f"{ESPN_SITE_BASE}/injuries")

    injuries = []
    for team_block in data.get("injuries", []):
        team_abbr = _NAME_TO_ABBREVIATION.get(team_block.get("displayName", ""), "")
        for entry in team_block.get("injuries", []):
            athlete = entry.get("athlete", {})
            injuries.append(
                {
                    "team": team_abbr,
                    "player_id": athlete_id_from_links(athlete.get("links")),
                    "player_name": athlete.get("displayName", ""),
                    "status": entry.get("status", ""),
                    # ESPN stamps each entry with when the report was updated.
                    # Reported verbatim; a blank date stays blank rather than
                    # being backfilled with "today", which would be a fiction.
                    "dated": entry.get("date", ""),
                }
            )

    _save_cache("injuries", "current", {"injuries": injuries})
    return injuries
