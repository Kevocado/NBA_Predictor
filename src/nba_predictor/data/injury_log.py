"""Append-only snapshots of ESPN's league-wide NBA injury report (free, no key).

History is the point: a model that discounts Out/Doubtful/Questionable players has to learn what each
status really meant for availability, and ESPN keeps no history, so every report is stored with the time we
saw it. Stdlib + requests only, so the workflow installs nothing else. One gzip file per UTC day:
data/injuries/nba_injuries_YYYY-MM-DD.jsonl.gz. Joins use the ESPN athlete id (read from the athlete's links);
names are display only.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
OUT_DIR = Path("data/injuries")

# The playercard link carries the athlete id plainly; the uid form is the fallback (see data/espn.py, which
# measured that the injuries payload has no athlete.id of its own).
_PLAYERCARD_ID = re.compile(r"/nba/player/(?:_/)?id/(\d+)")
_UID = re.compile(r"[?&~]a:(\d+)(?![0-9])")


def athlete_id(links) -> str:
    for pattern in (_PLAYERCARD_ID, _UID):
        for link in links or ():
            found = pattern.search((link or {}).get("href") or "")
            if found:
                return found.group(1)
    return ""


def normalise(payload: dict, seen_at: str) -> list[dict]:
    """One row per reported player. Missing pieces stay blank: nothing is backfilled."""
    rows = []
    for block in payload.get("injuries", []) or []:
        for entry in block.get("injuries", []) or []:
            athlete = entry.get("athlete") or {}
            details = entry.get("details") or {}
            rows.append({
                "seen_at": seen_at,
                "team_name": block.get("displayName", ""),
                "player_id": athlete_id(athlete.get("links")),
                "player_name": athlete.get("displayName", ""),
                "position": (athlete.get("position") or {}).get("abbreviation", ""),
                "status": entry.get("status", ""),
                "reported_at": entry.get("date", ""),
                "injury": details.get("type", ""),
                "return_date": details.get("returnDate", ""),
                "comment": (entry.get("shortComment") or "")[:400],
            })
    return rows


def append(rows: list[dict], seen_at: str, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"nba_injuries_{seen_at[:10]}.jsonl.gz"
    with gzip.open(path, "at", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def main() -> int:
    seen_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()
    rows = normalise(resp.json(), seen_at)
    # An empty report is possible on a quiet day but is far likelier a feed fault: store nothing, say so.
    if not rows:
        print("no injury rows in the feed; nothing stored")
        return 0
    print(f"stored {len(rows)} rows -> {append(rows, seen_at)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
