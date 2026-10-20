"""Current NBA rosters from ESPN (free, no key): who is on which team NOW, and each player's reported injury status.

Why: projections were assigned to teams from box scores, so after a summer of trades a player showed up on his old
team, his new team, or both (Jaylen Brown and Paul George, opening week 2026). The roster endpoint carries the real,
current assignment keyed by the same ESPN athlete id the box scores use. A feed failure returns the last good copy,
or None when there is none: "unknown" must never be read as "an empty roster".
"""
from __future__ import annotations

import logging

from nba_predictor.data import espn

logger = logging.getLogger(__name__)

ROSTER_TTL_SECONDS = 3 * 3600
_CACHE = ("rosters", "current")


def _latest_status(injuries) -> str:
    entries = [e for e in (injuries or []) if isinstance(e, dict)]
    if not entries:
        return ""
    return max(entries, key=lambda e: e.get("date") or "").get("status", "") or ""


def fetch_rosters() -> dict[str, list[dict]]:
    """{team abbreviation: [{player_id, player_name, position, injury_status}]}, one team per player."""
    teams = espn._fetch_json(f"{espn.ESPN_SITE_BASE}/teams")["sports"][0]["leagues"][0]["teams"]
    out: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for entry in teams:
        team = entry["team"]
        abbr = espn._NAME_TO_ABBREVIATION.get(team.get("displayName", "")) or espn.normalize_abbreviation(team.get("abbreviation", ""))
        payload = espn._fetch_json(f"{espn.ESPN_SITE_BASE}/teams/{team['id']}/roster")
        players = []
        for a in payload.get("athletes", []) or []:
            pid = str(a.get("id", ""))
            if not pid or pid in seen:                       # a player is on one team; the first listing wins
                continue
            seen.add(pid)
            players.append({
                "player_id": pid,
                "player_name": a.get("displayName", ""),
                "position": (a.get("position") or {}).get("abbreviation", ""),
                "injury_status": _latest_status(a.get("injuries")),
            })
        out[abbr] = players
    return out


def get_rosters(max_age_seconds: float = ROSTER_TTL_SECONDS) -> dict[str, list[dict]] | None:
    """Cached rosters, refetched when older than `max_age_seconds`; the stale copy on a failed refetch; None if neither."""
    cached = espn._load_cache(*_CACHE)
    if cached is not None and espn._cache_age_seconds(*_CACHE) < max_age_seconds:
        return cached["teams"]
    try:
        fresh = fetch_rosters()
    except Exception as exc:  # noqa: BLE001 - a roster feed fault must not take scoring down
        logger.warning("roster feed failed (%s); %s", exc, "using the last good copy" if cached else "no roster available")
        return cached["teams"] if cached is not None else None
    if sum(len(v) for v in fresh.values()) < 300:            # 30 teams x ~15: anything much smaller is a broken response
        logger.warning("roster feed returned only %d players; ignoring it", sum(len(v) for v in fresh.values()))
        return cached["teams"] if cached is not None else None
    espn._save_cache(*_CACHE, {"teams": fresh})
    return fresh


def cached_rosters() -> dict[str, list[dict]] | None:
    """The cached rosters whatever their age, never touching the network: for request paths (the ingest refreshes them)."""
    cached = espn._load_cache(*_CACHE)
    return cached["teams"] if cached is not None else None

