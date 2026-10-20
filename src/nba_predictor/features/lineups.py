"""Player minutes and starting-five projections from regular-season box scores.

This is availability B3 of the NBA model plan, offline only: nothing in the
serving path imports it yet. The evaluation lives in
``tools/evaluate_lineups.py`` and the numbers in
``docs/nba-availability-b3-2026-10.md``.

The two quantities are deliberately separate:

* ``expected_minutes`` is an EWMA of minutes over the team's last ten games.
  A game the player did not appear in is a zero, because a player who keeps
  missing games should not keep his old minutes. Games before his first
  appearance for the team are not zeros: a traded player is not diluted by
  games he could not have played for this team.
* ``p_start`` is the same EWMA over the starter flag. ESPN's box score lists
  the five starters first (validated against the live summary endpoint on 100
  random 2023-2026 games: 2,181 matched rows, zero mismatches), so
  ``with_starter_flags`` turns row order into that flag. The caches carry no
  starter column.

Injury statuses are prior probabilities, not measurements. An unknown status
changes nobody and logs a warning: a guess here silently deletes or inflates a
real player's minutes.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

WINDOW = 10
#: A player cannot absorb more than a full game's minutes. The cap is applied
#: after redistribution, so an Out teammate's minutes never turn a bench
#: player into a 48-minute player.
MAX_MINUTES = 40.0
#: EWMA span 10, the same "last ~10 team games" the plan asks for.
ALPHA = 2 / (WINDOW + 1)

#: The prior table from the plan. Only these exact words are probabilities.
STATUS_PROBABILITY: dict[str, float] = {
    "out": 0.0,
    "doubtful": 0.10,
    "questionable": 0.50,
    "probable": 0.85,
    "day-to-day": 0.60,
    "day to day": 0.60,
}


@dataclass(frozen=True)
class PlayerProjection:
    player_id: str
    player_name: str
    team: str
    position: str
    expected_minutes: float
    p_start: float
    availability: float = 1.0


def _key(status: object) -> str | None:
    if not isinstance(status, str):
        return None
    return " ".join(status.strip().lower().split())


def status_probability(status: object) -> float | None:
    """The prior probability for an injury status, or None when unknown."""
    key = _key(status)
    return STATUS_PROBABILITY.get(key) if key is not None else None


def with_starter_flags(rows: Iterable[Mapping]) -> list[dict]:
    """Copy box-score rows, flagging the first five rows of each team as starters.

    ESPN's summary endpoint lists starters first; the cache preserves that
    order. The flag is per team, not per file, because a file holds both teams.
    """
    seen: dict[str, int] = {}
    flagged = []
    for row in rows:
        team = str(row.get("team") or "")
        index = seen.get(team, 0)
        seen[team] = index + 1
        flagged.append({**row, "started": index < 5})
    return flagged


def load_injury_statuses(paths: Iterable[str | Path]) -> dict[str, str]:
    """Latest status per player from gzipped JSONL injury files.

    Each line is ``{"player_id": ..., "status": ..., "date": ...}``. A later
    date wins; a row with no date loses to a dated row but wins over nothing.
    An empty path list is not an error: the caller decides whether no statuses
    is acceptable for its game.
    """
    latest: dict[str, tuple[str, str]] = {}
    for path in sorted(Path(p) for p in paths):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                player_id = str(row.get("player_id") or "")
                status = row.get("status")
                if not player_id or not isinstance(status, str):
                    continue
                date = str(row.get("date") or "")
                if player_id not in latest or date >= latest[player_id][0]:
                    latest[player_id] = (date, status)
    return {player_id: status for player_id, (_, status) in latest.items()}


def _ewma(values: Sequence[float], alpha: float = ALPHA) -> float:
    if not values:
        return 0.0
    out = float(values[0])
    for value in values[1:]:
        out = alpha * float(value) + (1 - alpha) * out
    return out


def _redistribute(
    projections: list[PlayerProjection], removed: float
) -> list[PlayerProjection]:
    """Give an Out player's minutes to fully available teammates.

    Role is the player's own projected minutes: the sixth man already plays
    more than the twelfth man, so he absorbs more of an absent starter's
    minutes. Only ``availability == 1`` players receive; a Questionable player
    does not get his own minutes handed back to him, and a partial status is
    scaled but not redistributed -- the report says Out, not "probably out".
    """
    recipients = [p for p in projections if p.availability >= 1.0]
    if removed <= 0 or not recipients:
        return projections

    base = {p.player_id: p.expected_minutes for p in recipients}
    extra = {p.player_id: 0.0 for p in recipients}
    remaining = removed
    for _ in range(10):
        pool = [pid for pid in base if base[pid] + extra[pid] < MAX_MINUTES - 1e-9]
        weight = sum(base[pid] for pid in pool)
        if not pool or weight <= 0:
            break
        added = 0.0
        for pid in pool:
            share = remaining * base[pid] / weight
            room = MAX_MINUTES - (base[pid] + extra[pid])
            take = min(share, room)
            extra[pid] += take
            added += take
        remaining -= added
        if added <= 1e-9:
            break

    return [
        replace(p, expected_minutes=p.expected_minutes + extra[p.player_id])
        if p.player_id in extra
        else p
        for p in projections
    ]


def project_team(
    team: str,
    team_games: Sequence[Mapping],
    *,
    latest_team: Mapping[str, str] | None = None,
    statuses: Mapping[str, str] | None = None,
    window: int = WINDOW,
) -> list[PlayerProjection]:
    """Project every player whose latest team is ``team``.

    ``team_games`` is chronological, one entry per completed game:
    ``{"game_id": ..., "players": [row, ...]}``. Rows need ``player_id``,
    ``player_name``, ``team``, ``minutes`` and ``started``; ``position`` is
    carried through. Only the last ``window`` games are used.

    ``latest_team`` is the player -> team map as of before the target game.
    When it is omitted every player appearing in ``team_games`` is assumed to
    still be on the team; the evaluation always passes the real map, because a
    traded player must not be projected for his old team.
    """
    games = list(team_games)[-window:]
    observations: dict[str, dict[str, tuple[float, bool]]] = {}
    first_game: dict[str, int] = {}
    names: dict[str, str] = {}
    positions: dict[str, str] = {}

    for index, game in enumerate(games):
        game_id = str(game["game_id"])
        for row in game.get("players", []):
            if str(row.get("team") or "") != team:
                continue
            player_id = str(row["player_id"])
            observations.setdefault(player_id, {})[game_id] = (
                float(row.get("minutes") or 0.0),
                bool(row.get("started")),
            )
            first_game.setdefault(player_id, index)
            names[player_id] = str(row.get("player_name") or player_id)
            positions[player_id] = str(row.get("position") or "")

    projections = []
    removed = 0.0
    for player_id, by_game in observations.items():
        if latest_team is not None and latest_team.get(player_id) != team:
            continue

        minutes = []
        starts = []
        for index, game in enumerate(games):
            if index < first_game[player_id]:
                continue
            game_minutes, started = by_game.get(str(game["game_id"]), (0.0, False))
            minutes.append(game_minutes)
            starts.append(1.0 if started else 0.0)

        base_minutes = _ewma(minutes)
        base_start = _ewma(starts)
        probability = 1.0
        status = (statuses or {}).get(player_id)
        if status is not None:
            prior = status_probability(status)
            if prior is None:
                logging.warning(
                    "unknown injury status %r for player %s; leaving projection unchanged",
                    status,
                    player_id,
                )
            else:
                probability = prior
        # Only a plain Out frees minutes for someone else. A partial status
        # scales this player's own projection; handing the rest to a teammate
        # as well would count the same absence twice.
        removed += base_minutes if probability == 0.0 else 0.0
        projections.append(
            PlayerProjection(
                player_id=player_id,
                player_name=names[player_id],
                team=team,
                position=positions[player_id],
                expected_minutes=base_minutes * probability,
                p_start=base_start * probability,
                availability=probability,
            )
        )

    return _redistribute(projections, removed)


def starting_five(
    projections: Iterable[PlayerProjection], size: int = 5
) -> list[PlayerProjection]:
    """The projected starters: highest P(start), then minutes, then player id."""
    available = [p for p in projections if p.availability > 0]
    return sorted(
        available,
        key=lambda p: (-p.p_start, -p.expected_minutes, p.player_id),
    )[:size]
