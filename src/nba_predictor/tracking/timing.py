"""Which pick a row is, and when it was made.

Two questions, kept in one module because they are the two halves of one
rule and must never be answered by different code:

* **which row counts** -- `earliest_recorded`, one per key, the EARLIEST
  recorded;
* **when it was made** -- `made_before_tip`, derived on every read from that
  row's own `created_at` against the game's tip-off.

The backtest in pipeline/ingest.py writes completed games into the same tables
the live pollers write to, with `created_at` set to when the backtest ran. That
is what used to end the record: every read that judged the model filtered
through `made_before_tip` and threw the rest away, so re-running the models
stopped the record tracking anything. Kevin's decision (2026-10-01,
predictor-hub #66): recorded stays recorded, the headline counts every
recorded pick, and honesty moves from exclusion to disclosure.

So the filter is gone from the read path. What replaces it is a LABEL, and the
label fails closed: an unreadable timestamp, or a game with no `game_date` to
cut against, is never presented as a pre-tip pick, because nothing here can
prove that it was.
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
    if parsed.tzinfo is None:
        # A stamp written with no offset is UTC, which is how every writer in
        # this repo writes it -- not naive local time.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _instant_or_none(value) -> datetime | None:
    """The UTC instant of a stamp, or None when it cannot be read."""
    try:
        return _parse_utc(value)
    except (TypeError, ValueError, AttributeError):
        return None


def pick_cutoff(game: dict) -> datetime:
    tip_off = game.get("tip_off")
    if tip_off:
        return _parse_utc(tip_off)
    day = datetime.strptime(game["game_date"], "%Y-%m-%d").date()
    return datetime.combine(day, _FALLBACK_TIP, tzinfo=_EASTERN).astimezone(timezone.utc)


def made_before_tip(created_at: str, game: dict) -> bool:
    """Whether this row's OWN stamp proves the pick was made before tip-off.

    DERIVED, on every read, from the two timestamps and nothing else. There is
    no `made_before_tip` column in any of the three prediction tables and there
    must not be one: a stored flag can drift out of step with the timestamps it
    claims to describe and nothing would notice, and there would be a
    pre-existing backfill to get wrong.

    Both sides are compared as UTC INSTANTS, never as strings and never as
    naive wall clocks. That distinction is the whole risk here, because a
    `tip_off` handed over by the schedule cache can carry any offset, so
    `2026-09-12T23:30:00+09:00` (14:30Z) is half an hour BEFORE a
    `2026-09-12T15:00:00+00:00` tip even though the wall clocks read "23:30"
    against "15:00" and sort the other way. A comparison that dropped the
    offset, or read the host's local zone, would label that pick post-tip.
    `test_made_before_tip_is_derived_from_utc_instants_not_wall_clock_strings`
    is that pair plus its mirror, each asserting the string order as a
    precondition so it cannot silently stop straddling;
    `test_made_before_tip_ignores_the_machine_timezone` runs the same pair under
    four host zones.

    Fails CLOSED, and that is the rule's one remaining prohibition. A stamp
    that does not parse, and a game the schedule cannot date at all, both give
    False. Such a row still COUNTS as a recorded pick; it is simply never
    presented as a pre-game one, because nothing here can prove that it was.
    """
    try:
        return _parse_utc(created_at) < pick_cutoff(game)
    except (TypeError, ValueError, KeyError):
        return False


def _stamp_key(row, column: str) -> tuple:
    """Earliest instant first; an unreadable stamp last.

    `na_position="last"` is the point: a row whose stamp cannot be parsed must
    never win a "which was recorded first" comparison, so it sorts behind
    everything provable instead of jumping the queue on a string comparison.
    """
    instant = _instant_or_none(row[column])
    return (instant is None, instant or datetime.max.replace(tzinfo=timezone.utc))


def _sort_key(row) -> tuple:
    return _stamp_key(row, "created_at")


def _counting_key(row) -> tuple:
    """The key a counted pick is unique on, for the two tables that need one.

    `(game, player, stat)` for a player prop: a points pick is one pick per
    PLAYER per game, and collapsing a game's twenty rotation players into one
    pick would delete the player record rather than deduplicate it. The
    `predictions` table has no player and no market -- the model publishes one
    probability per game there, and `game_outcome` is that market -- so its key
    reduces to the game.

    The same triple is the key `store._UNIQUE_INDEXES` enforces on
    `game_player_outcomes`, so this function and the schema cannot drift apart
    without one of them being wrong. It is the reader half of the same key.
    """
    keys = row.keys()
    return (row["game_id"],
            row["player_id"] if "player_id" in keys else None,
            row["stat"] if "stat" in keys else None)


def earliest_recorded(rows: list) -> list:
    """One row per (game, player, stat), the EARLIEST recorded.

    Rule 2 of the 2026-10-01 spec: a later rerun of the model on the same game
    and market is kept as history -- it is still in the table, and the
    per-pick list still publishes it -- but it neither replaces the counted
    pick nor counts a second time. Without that, re-running until the model was
    right would be free, and every model change would silently restate the
    record.

    **This dedupe is load-bearing, not belt to braces, and it is load-bearing
    precisely because the schema does NOT enforce the key it dedupes.**
    `predictions` and `player_prediction_snapshots` carry no uniqueness
    constraint at all -- their only PRIMARY KEY is the surrogate `id`, which
    makes each row distinct and constrains no business key -- so a rerun
    genuinely lands twice and every aggregate path calls this function to stop
    it being graded twice: `hub_service.counted_picks` for the hit rate,
    `player_props.resolved_player_props` for the MAE. The scheduled refresh
    re-presents a rolling window of finished games every day, so those
    duplicates are the product working as specified, not an accident to be
    constrained away. See `store._UNIQUE_INDEXES` for why
    `game_player_outcomes` -- a fact table, where a duplicate WOULD move the
    MAE -- is the one table that now has a unique index, and why the reader
    there (`earliest_recorded_outcome`) stays anyway.

    For the odds tables the unit is the whole run rather than the row, because
    `refresh_odds` writes every side of every bookmaker at one instant -- that
    is `earliest_run`.

    "Earliest" is by instant, so which row wins does not depend on how its
    timestamp happens to be spelled. A stamp that cannot be parsed cannot be
    proven earliest, so it sorts last and never displaces a row carrying a real
    instant; where a key has only unparseable stamps the first row stands and
    `made_before_tip` fails closed on it rather than the pick being lost --
    rule 1 has no exception for a timestamp nobody can read.

    Reversible in one place: keep the LAST row per key instead if that is ever
    the better answer (`.head(1)` -> `.tail(1)`).
    """
    kept: list = []
    seen: set = set()
    for row in sorted(rows, key=_sort_key):
        key = _counting_key(row)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def earliest_recorded_outcome(rows: list) -> list:
    """One row per (game, player, stat), the EARLIEST recorded. For OUTCOMES.

    The same rule `earliest_recorded` applies to predictions, on the same
    `(game, player, stat)` key, reading `recorded_at` instead of `created_at`
    because that is the column an outcome is stamped with.

    Defence in depth behind `uq_game_player_outcomes_key`, and it stays because
    a unique index is only as complete as the writers: this is reached for any
    database file that predates the index, and it is the reason adding the
    index cannot change what serving reads on a clean file (there is one row
    per key, so this returns the input unchanged).

    What it fixes is that the outcome read had NO dedupe at all and no ORDER BY
    -- it folded rows into a dict keyed by (player, stat), so on a file with a
    duplicated key the MAE was graded against whichever row SQLite happened to
    hand back last. That is the silent version of this defect: a number moves
    and nothing records why. Here the winner is the earliest recorded one, the
    same answer every other counted pick in this module gets.
    """
    kept: list = []
    seen: set = set()
    for row in sorted(rows, key=lambda r: _stamp_key(r, "recorded_at")):
        key = _counting_key(row)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def earliest_run(rows: list) -> list:
    """Every row sharing the EARLIEST instant in `rows` -- one odds refresh.

    The odds tables store both selections for every bookmaker on every run, so
    a single run is several rows with one `created_at`. The counted unit is the
    run, not the row: the pick it contains is the side the model priced
    highest, and picking the run by its first row instead would grade a
    bookmaker's row rather than the model's call.
    """
    if not rows:
        return []
    ordered = sorted(rows, key=_sort_key)
    first = _sort_key(ordered[0])
    return [row for row in ordered if _sort_key(row) == first]


def latest_pre_tip(rows: list, game: dict):
    """The newest row made before tip-off, or None.

    **Not the record's pick rule.** This is the SERVING rule: what the game
    detail, the facts bundle and the live ranking show. The track record counts
    the earliest recorded pick per key (`earliest_recorded`); this still
    prefers the freshest pre-tip read, because a projection shown beside a
    game's box score has to be the newest one the model actually made, not the
    first thing it ever said about it.

    The two disagree on purpose and the disagreement is checked, not tolerated:
    `tests/test_track_record_counted_picks.py` pins that the record grades the
    first pick while this still serves the last.
    """
    eligible = [r for r in rows if made_before_tip(r["created_at"], game)]
    return max(eligible, key=lambda r: _parse_utc(r["created_at"]), default=None)


def latest_by_instant(rows: list):
    """The newest row by parsed time, not by string order (zoneless and
    '+00:00' timestamps don't sort together as text). None when empty."""
    def key(r):
        return _instant_or_none(r["created_at"]) or datetime.min.replace(tzinfo=timezone.utc)
    return max(rows, key=key, default=None)



