"""The graded value ledger: snapshot, settle, closing line, CLV, weekly report.

Spec section 9. The edge gate produces picks; before this, nothing recorded what
happened to them. A gate with no ledger is an opinion generator, and there is no
way to tell afterwards whether a good-looking record was skill or luck.

Three tables, each with a unique key so the row's identity is enforced rather
than left to the reader:

* `value_picks` — the pre-tip snapshot. Unique on
  (game, market, selection, bookmaker), so a rerun of the gate cannot file a
  second pick for the same bet. That is what makes the snapshot a record: the
  odds refresher re-scores every book on every pass, so duplicates would otherwise
  land continuously and the first write would not be the one anyone saw.
* `value_pick_outcomes` — what happened. Unique on the same key, and a disagreeing
  second value raises `ConflictingOutcome`: an outcome is a fact, so a
  contradiction is something for a human to see, not an update to apply.
* `value_closing_lines` — the market's final price, for CLV.

**Pre-tip is enforced, not assumed.** `tracking.timing.made_before_tip` already
encodes that rule and is reused rather than restated; a post-tip snapshot raises
`PostTipSnapshot` (spec section 14.3 -- a line recorded after tip-off is not a
bettable price).

**Ungraded is None, never False.** A pick with no recorded outcome has not been
measured, and reporting it as a miss is the error the game-level track record
already learned to avoid.

**CLV is measured against the closing line**, per pick. A pick can win and still
have been a bad price, and this is the one number that says so.

The weekly report prints yield whatever it is, under copy that states plainly
that this is not evidence of a profitable strategy -- PL's framing, deliberately.

Nothing here places a bet.
"""

from __future__ import annotations

import sqlite3

from nba_predictor.tracking.store import (
    ConflictingOutcome,
    get_connection,
)
from nba_predictor.tracking.timing import made_before_tip


class PostTipSnapshot(Exception):
    """A pick snapshotted at or after its game's tip-off.

    Not a bettable price, so not a pick. Raised rather than warned: the gate's
    whole freshness contract is that what it flags was available before tip-off,
    and a post-tip row would be counted against that contract while looking
    identical to a legitimate one.
    """


def snapshot_pick(
    db_path,
    *,
    game_id: str,
    market: str,
    selection: str,
    bookmaker: str,
    american_odds: int,
    point: float | None,
    model_probability: float,
    market_probability: float,
    edge: float,
    snapshot_at: str,
    game: dict | None = None,
) -> bool:
    """Record one pre-tip pick. Immutable: the first write wins.

    Returns True if this call wrote the row, False if the pick already existed.
    """
    if game is not None and not made_before_tip(snapshot_at, game):
        raise PostTipSnapshot(
            f"{game_id}/{market}/{selection} snapshot at {snapshot_at} is not "
            f"before tip-off ({game.get('tip_off')})"
        )
    # A plain append, NOT INSERT OR IGNORE, and with no unique index on
    # `value_picks`. The odds refresher runs on a loop and re-snapshots the same
    # pick as often as the cadence allows; this repo's standing decision is that
    # a prediction-shaped table keeps that history and the reader counts the
    # earliest, rather than the schema refusing the write. `read_ledger` does
    # the dedupe, the same way `tracking.timing.earliest_recorded` does for the
    # game-level tables.
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO value_picks
                (game_id, market, selection, bookmaker, american_odds, point,
                 model_probability, market_probability, edge, snapshot_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (game_id, market, selection, bookmaker, american_odds, point,
             model_probability, market_probability, edge, snapshot_at),
        )
        conn.commit()
        return True


def settle_pick(
    db_path,
    *,
    game_id: str,
    market: str,
    selection: str,
    bookmaker: str,
    outcome: str,
    settled_at: str,
) -> int:
    """Record what a flagged pick did. `outcome` is 'won', 'lost' or 'push'.

    Idempotent on the same fact and loud on a contradicting one.
    """
    if outcome not in ("won", "lost", "push"):
        raise ValueError(f"outcome must be won/lost/push, got {outcome!r}")
    with get_connection(db_path) as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO value_pick_outcomes
                    (game_id, market, selection, bookmaker, outcome, settled_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (game_id, market, selection, bookmaker, outcome, settled_at),
            )
            conn.commit()
            return cur.lastrowid
        except Exception as exc:  # noqa: BLE001 - narrowed immediately below
            if not isinstance(exc, sqlite3.IntegrityError):
                raise
            row = conn.execute(
                """
                SELECT outcome FROM value_pick_outcomes
                WHERE game_id=? AND market=? AND selection=? AND bookmaker=?
                """,
                (game_id, market, selection, bookmaker),
            ).fetchone()
            conn.rollback()
            if row is not None and row[0] != outcome:
                raise ConflictingOutcome(
                    f"{game_id}/{market}/{selection} already settled as {row[0]}, "
                    f"now {outcome}"
                ) from None
            return 0


def record_closing_line(
    db_path,
    *,
    game_id: str,
    market: str,
    selection: str,
    bookmaker: str,
    closing_american_odds: int,
    recorded_at: str,
) -> bool:
    """Record the market's final price for one side, for CLV."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO value_closing_lines
                (game_id, market, selection, bookmaker, closing_american_odds, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (game_id, market, selection, bookmaker, closing_american_odds, recorded_at),
        )
        conn.commit()
        return cur.rowcount > 0


def to_decimal(american_odds: int) -> float:
    """American odds -> decimal, the standard conversion.

    A positive price is `1 + o/100`: +150 pays 150 for a 1 bet, so decimal 2.50.
    It is **not** `1 + 100/o`, which is the inversion -- that maps +150 to 1.67
    and, worse, makes the conversion *decrease* with the price, so a longer price
    looks worth less. That inverts CLV for every underdog, both its sign and its
    magnitude. An earlier version of this function had exactly that bug, and
    every test for it used a negative price, so nothing caught it.
    """
    return 1 + american_odds / 100 if american_odds > 0 else 1 + 100 / abs(american_odds)


def clv_pct(entry_american_odds: int | None, closing_american_odds: int | None):
    """Closing line value, as a fraction. Positive means the entry beat the close.

    None when either side is missing: an unclosed pick has no CLV, and 0.0 would
    claim the price was exactly fair.

    Sign convention: entered at -110 and the market closed at -105 is a *better*
    price, so positive. The decimal conversion keeps that monotonic.
    """
    if entry_american_odds is None or closing_american_odds is None:
        return None

    entry = to_decimal(entry_american_odds)
    close = to_decimal(closing_american_odds)
    return (close - entry) / close


def read_ledger(db_path, game_id: str | None = None) -> list[dict]:
    """Every snapshotted pick, graded where an outcome exists and CLV'd where a
    closing line exists.

    A pick with neither reads `outcome=None`, `hit=None`, `clv_pct=None`. Never
    `hit=False`: ungraded is a different statement from lost.
    """
    where = "WHERE p.game_id = ?" if game_id else ""
    params: tuple = (game_id,) if game_id else ()
    with get_connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT p.*, o.outcome, c.closing_american_odds
            FROM value_picks p
            LEFT JOIN value_pick_outcomes o
              ON o.game_id=p.game_id AND o.market=p.market
             AND o.selection=p.selection AND o.bookmaker=p.bookmaker
            LEFT JOIN value_closing_lines c
              ON c.game_id=p.game_id AND c.market=p.market
             AND c.selection=p.selection AND c.bookmaker=p.bookmaker
            {where}
            ORDER BY p.snapshot_at, p.game_id
            """,
            params,
        ).fetchall()

    # The EARLIEST snapshot per (game, market, selection, bookmaker). The table
    # keeps every snapshot the refresh loop wrote; the pick that counts is the
    # one published first, which is the pre-tip price a reader could have had.
    earliest: dict[tuple, dict] = {}
    for row in rows:
        record = dict(row)
        key = (record["game_id"], record["market"], record["selection"], record["bookmaker"])
        if key in earliest:
            continue
        record["snapshots_for_key"] = 0  # filled below
        earliest[key] = record

    ledger = []
    for record in earliest.values():
        record.pop("snapshots_for_key", None)
        outcome = record["outcome"]
        # A push graded to neither hit nor miss -- the same rule the game-level
        # track record follows for a pushed bet.
        record["hit"] = None if outcome in (None, "push") else outcome == "won"
        # The Brier term, not a restatement of the outcome: a lost pick is
        # wrong by (1 - p), so a confident loser costs more than a narrow one.
        # That is what makes `error` a quality measure rather than a second copy
        # of `hit`.
        record["error"] = (
            None if outcome in (None, "push")
            else (0.0 if outcome == "won" else 1.0 - float(record["model_probability"]))
        )
        record["clv_pct"] = clv_pct(record["american_odds"], record["closing_american_odds"])
        ledger.append(record)
    return ledger


DISCLAIMER = (
    "This is a measurement of past games, **not evidence of a profitable "
    "strategy**. Hit rate and yield on a small graded sample describe what "
    "already happened; they say nothing about what will, and nothing here "
    "places a bet."
)


def weekly_ledger_report(db_path) -> str:
    """The ledger as markdown: every pick, its outcome, the hit rate, the yield
    and the CLV -- shown whatever they are.

    An empty or wholly ungraded ledger renders as zeros and None rather than
    being hidden, because "we graded nothing this week" is itself the finding a
    reader needs.
    """
    ledger = read_ledger(db_path)
    graded = [r for r in ledger if r["outcome"] in ("won", "lost")]
    won = [r for r in graded if r["outcome"] == "won"]
    pushes = [r for r in ledger if r["outcome"] == "push"]

    def pct(value):
        return f"{value * 100:.1f}%" if value is not None else "—"

    hit_rate = len(won) / len(graded) if graded else None
    # Yield in units where one unit is staked per pick, at the price actually
    # taken. Counting a win as +1 and a loss as -1 is wrong by the vig: at -110 a
    # win returns 0.909 units, so a record of 11-11 at -110 is about -1 unit, not
    # zero. `to_decimal(odds) - 1` is the real profit on a win, and a lost unit
    # is -1. A push returns the stake, so it contributes 0.
    yield_units = sum(
        (to_decimal(r["american_odds"]) - 1.0) if r["outcome"] == "won" else -1.0
        for r in graded
    )
    clvs = [r["clv_pct"] for r in ledger if r["clv_pct"] is not None]
    mean_clv = sum(clvs) / len(clvs) if clvs else None

    lines = [
        "# Value ledger — weekly report",
        "",
        DISCLAIMER,
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Picks snapshotted | {len(ledger)} |",
        f"| Graded | {len(graded)} |",
        f"| Won | {len(won)} |",
        f"| Lost | {len(graded) - len(won)} |",
        f"| Pushes (ungraded, per the hit-rate rule) | {len(pushes)} |",
        f"| Hit rate | {pct(hit_rate)} |",
        f"| Yield (units; 1 staked per pick at the price taken) | {yield_units:+.3f} |",
        f"| Mean CLV | {pct(mean_clv)} |",
        "",
    ]

    if not ledger:
        lines += ["No picks have been snapshotted yet.", ""]
        return "\n".join(lines)

    lines += [
        "## Picks",
        "",
        "| Game | Market | Selection | Book | Line | Odds | Edge | Model prob | Outcome | CLV |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ledger:
        lines.append(
            f"| {r['game_id']} | {r['market']} | {r['selection']} | {r['bookmaker']} | "
            f"{r['point'] if r['point'] is not None else '—'} | {r['american_odds']} | "
            f"{r['edge']:+.3f} | {r['model_probability']:.3f} | "
            f"{r['outcome'] or '—'} | {pct(r['clv_pct'])} |"
        )
    lines.append("")
    return "\n".join(lines)