"""When a pick has to be made to count.

Only picks made before tip-off go on the record. The backtest in
pipeline/ingest.py writes completed games into the same predictions table
with created_at set to when it ran, so every read that judges the model
filters through made_before_tip.
"""
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

_EASTERN = ZoneInfo("America/New_York")
# ESPN's scoreboard dates are Eastern dates, and no NBA game tips before noon
# Eastern, so noon on game_date is a safe cutoff when the tip time is unknown
# (scoreboards cached before tip_off was recorded).
_FALLBACK_TIP = time(12, 0)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def pick_cutoff(game: dict) -> datetime:
    tip_off = game.get("tip_off")
    if tip_off:
        return _parse_utc(tip_off)
    day = datetime.strptime(game["game_date"], "%Y-%m-%d").date()
    return datetime.combine(day, _FALLBACK_TIP, tzinfo=_EASTERN).astimezone(timezone.utc)


def made_before_tip(created_at: str, game: dict) -> bool:
    """True only when created_at is readable and strictly before the cutoff."""
    try:
        return _parse_utc(created_at) < pick_cutoff(game)
    except (TypeError, ValueError, KeyError):
        return False


def latest_pre_tip(rows: list, game: dict):
    """The newest row made before tip-off, or None."""
    eligible = [r for r in rows if made_before_tip(r["created_at"], game)]
    return max(eligible, key=lambda r: _parse_utc(r["created_at"]), default=None)


def latest_by_instant(rows: list):
    """The newest row by parsed time, not by string order (zoneless and
    '+00:00' timestamps don't sort together as text). None when empty."""
    def key(r):
        try:
            return _parse_utc(r["created_at"])
        except (TypeError, ValueError):
            return datetime.min.replace(tzinfo=timezone.utc)
    return max(rows, key=key, default=None)
