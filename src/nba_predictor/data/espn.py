"""ESPN data module for NBA schedule, box scores, and injuries.

Uses ESPN's public (keyless) site API. Endpoints verified against the
live API — not the stats.nba.com-style paths nba_api uses, which are
blocked from some network environments.
"""

import json
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from nba_predictor import config
from nba_predictor.data.team_reference import TEAMS

_NAME_TO_ABBREVIATION = {team.name: team.abbreviation for team in TEAMS}
_KNOWN_ABBREVIATIONS = {team.abbreviation for team in TEAMS}

ESPN_SITE_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

ESPN_CACHE_DIR = config.CACHE_DIR / "espn"

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


def _save_cache(endpoint: str, id_value: str, data: dict) -> None:
    cache_file = _cache_path(endpoint, id_value)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=32), reraise=True)
def _fetch_json(url: str, params: dict | None = None) -> dict:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


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


def get_injuries() -> list[dict]:
    """League-wide current injury report."""
    cached = _load_cache("injuries", "current")
    if cached is not None:
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
                    "player_name": athlete.get("displayName", ""),
                    "status": entry.get("status", ""),
                }
            )

    _save_cache("injuries", "current", {"injuries": injuries})
    return injuries
