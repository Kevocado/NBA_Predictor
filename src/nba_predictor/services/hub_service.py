import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from nba_predictor.api.schemas import TrackRecordOut, TrackRecordWeekOut
from nba_predictor.tracking import store
from nba_predictor.tracking.store import get_connection
from nba_predictor.tracking.timing import latest_pre_tip


def load_hub_cache(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def load_player_name_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {row["player_id"]: row["player_name"] for row in rows}


def pre_tip_picks(db_path: Path, schedule: list[dict]) -> tuple[list[tuple[dict, dict]], int]:
    """(game, latest pick made before tip-off) for every completed game, plus
    how many completed games have only picks rebuilt after tip-off. The
    backtest writes completed games into the same table long after they were
    played; those rows are never judged."""
    schedule_by_id = {g["game_id"]: g for g in schedule}
    by_game: dict[str, list] = {}
    with get_connection(db_path) as conn:
        for row in conn.execute("SELECT * FROM predictions"):
            by_game.setdefault(row["game_id"], []).append(row)

    picks: list[tuple[dict, dict]] = []
    rebuilt = 0
    for game_id, rows in by_game.items():
        game = schedule_by_id.get(game_id)
        if game is None or not game.get("completed") or game.get("home_pts") is None:
            continue
        pick = latest_pre_tip(rows, game)
        if pick is None:
            rebuilt += 1
        else:
            picks.append((game, pick))
    return picks, rebuilt


def _settle_game_outcome(db_path: Path, schedule: list[dict]) -> tuple[TrackRecordOut | None, list[tuple[date, bool]]]:
    """Settles the model's own win/loss call (predictions.home_win_prob >= 0.5)
    against each game's actual result, using only the pick made before
    tip-off. Every result is a real completed game, joined via the schedule
    cache's home_pts/away_pts (populated by pipeline/ingest.py).

    Returns the row and its graded picks as (game_date, correct) pairs, which
    is what the weekly breakdown is built from -- one list, one rule, so a
    week can never be graded more loosely than the headline above it."""
    picks, rebuilt = pre_tip_picks(db_path, schedule)
    if not picks and not rebuilt:
        return None, []

    graded: list[tuple[date, bool]] = []
    for game, pick in picks:
        day = _game_date(game)
        if day is None:
            # Untimed and undated: `made_before_tip` cannot have judged it
            # pre-tip either, but if it somehow graded, dropping it here would
            # break the weekly-to-headline identity the panel reconciles with.
            continue
        graded.append((day, (pick["home_win_prob"] >= 0.5) == (game["home_pts"] > game["away_pts"])))

    correct = sum(1 for _, ok in graded if ok)
    total = len(graded)
    return TrackRecordOut(
        market="game_outcome", total_predictions=total, correct_predictions=correct,
        hit_rate=round(correct / total, 3) if total else None, n_rebuilt=rebuilt,
        settled=True,
    ), graded


def _grade_market_row(row, game: dict) -> bool | None:
    """True = the pick cleared its line, False = it did not, None = no grade.

    A grade is the pick against the line it was priced at, never against a
    line re-read from anywhere else:

    - **h2h**: `selection` is the side the model backed; it wins the game.
    - **spread**: `point` is the selection's own line in book convention
      (BOS -4.5 stores -4.5 for BOS), so the side covers when its margin
      beats `-point`. The same threshold `refresh_odds._model_probability`
      prices with (`line=-point`); `test_spread_grading_uses_the_same_
      threshold_as_the_odds_writer` pins the two together, because a grader
      reading the sign the other way produces plausible rates that are all
      mirrored.
    - **total**: `point` is the line; `selection` is "over" or "under".

    None is a push (the result landed exactly on the line) or a row with no
    line to judge: nobody won that one, so it is counted in `n_push` and
    stays out of the rate rather than becoming a miss.
    """
    market = row["market"]
    home_pts, away_pts = game["home_pts"], game["away_pts"]

    if market == "h2h":
        winner = game["home_team"] if home_pts > away_pts else game["away_team"]
        return row["selection"] == winner

    point = row["point"]
    if point is None:
        return None

    if market == "spread":
        if row["selection"] == game["home_team"]:
            side_margin = home_pts - away_pts
        elif row["selection"] == game["away_team"]:
            side_margin = away_pts - home_pts
        else:
            return None
        # Covers when side_margin > -point; == is the push.
        margin_vs_line = side_margin + point
        return None if margin_vs_line == 0 else margin_vs_line > 0

    if market == "total":
        margin_vs_line = (home_pts + away_pts) - point
        if margin_vs_line == 0:
            return None
        return (margin_vs_line > 0) == (str(row["selection"]).lower() == "over")

    return None


def _settle_market_predictions(
    db_path: Path, schedule: list[dict], market: str
) -> tuple[TrackRecordOut | None, list[tuple[date, bool]]]:
    """Settles one priced market against actual results, once per game.

    Odds refresh stores both selections for every bookmaker on every run, so
    judging every row would pin the rate near 50%: per game, take the latest
    run made before tip-off and its selection with the higher model
    probability, then grade that one pick. Returns (None, []) when the
    schedule holds no finished game for this market -- the caller then
    reports the stored rows as unsettled rather than as 0%.
    """
    schedule_by_id = {g["game_id"]: g for g in schedule}
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM game_market_predictions WHERE market = ?", (market,)
        ).fetchall()

    by_game: dict[str, list] = {}
    for row in rows:
        by_game.setdefault(row["game_id"], []).append(row)

    completed_games = 0
    graded: list[tuple[date, bool]] = []
    n_push = 0
    for game_id, game_rows in by_game.items():
        game = schedule_by_id.get(game_id)
        if game is None or not game.get("completed") or game.get("home_pts") is None:
            continue
        completed_games += 1
        latest = latest_pre_tip(game_rows, game)
        if latest is None:
            continue
        day = _game_date(game)
        if day is None:
            continue
        run = [r for r in game_rows if r["created_at"] == latest["created_at"]]
        pick = max(run, key=lambda r: r["model_probability"])
        verdict = _grade_market_row(pick, game)
        if verdict is None:
            n_push += 1
            continue
        graded.append((day, verdict))

    if completed_games == 0:
        return None, []
    correct = sum(1 for _, ok in graded if ok)
    total = len(graded)
    return TrackRecordOut(
        market=market, total_predictions=total, correct_predictions=correct,
        hit_rate=round(correct / total, 3) if total else None,
        n_push=n_push, settled=True,
    ), graded


# The markets this repo has a grading rule for. Anything else in
# game_market_predictions is reported with its stored count and `settled=false`.
_SETTLED_MARKETS = ("h2h", "spread", "total")


def _game_date(game: dict | None) -> date | None:
    """The day a game was played, for week grouping.

    `game_date` first (the schedule's own key); a `tip_off` date only, for a
    row written before `game_date` existed. None when neither parses -- and
    such a game can never have been graded anyway, because `timing.pick_cutoff`
    would have raised before `made_before_tip` could return True.
    """
    if game is None:
        return None
    raw = game.get("game_date")
    if raw:
        try:
            return date.fromisoformat(str(raw)[:10])
        except ValueError:
            pass
    tip = game.get("tip_off")
    if tip:
        try:
            return datetime.fromisoformat(str(tip).replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def _week_start(day: date) -> date:
    """Monday on or before `day` -- the same Monday-based week the site's
    `frontend/src/lib/weeks.ts` groups its schedule by, so a week the visitor
    already knows by name is the week this table reports."""
    return day - timedelta(days=day.weekday())


def _tracking_window(db_path: Path, schedule: list[dict], today: date) -> list[date]:
    """Every week from the first game the tracker wrote a pick for, through
    the current week.

    Enumerating weeks from the data (rather than emitting one row per group)
    is the whole point: a week the tracker skipped must read as "not tracked",
    not vanish, because a reader cannot tell a gap from an absence. The start
    is games with ROWS rather than with grades, so a week whose picks were all
    rebuilt or all unsettled still belongs to the window -- it just appears
    untracked.

    `/hub/vs-market` derives its window from this same function, so the two
    week tables on the page line up row for row.

    Empty when the tracking DB has nothing the schedule can date.
    """
    schedule_by_id = {g["game_id"]: g for g in schedule}
    with get_connection(db_path) as conn:
        ids = {row[0] for row in conn.execute("SELECT DISTINCT game_id FROM predictions")}
        ids |= {row[0] for row in conn.execute("SELECT DISTINCT game_id FROM game_market_predictions")}

    weeks = [_week_start(day) for day in
             (d for d in (_game_date(schedule_by_id.get(game_id)) for game_id in ids) if d is not None)]
    if not weeks:
        return []
    first = min(weeks)
    # Never before tracking began; always up to this week, so an in-progress
    # week with nothing graded yet shows as a row instead of ending the table.
    last = max(_week_start(today), first)
    return [first + timedelta(weeks=i) for i in range((last - first).days // 7 + 1)]


def _weekly_rows(graded: list[tuple[date, bool]], window: list[date]) -> list[TrackRecordWeekOut]:
    """One row per week in `window` for one market.

    `hit_rate` is None (not 0.0) when a week graded nothing: 0% is a claim
    that every pick missed, which is not what "never measured" means. The
    headline's rule is reused verbatim -- this is the only other place the
    market's record is computed, deliberately.
    """
    by_week: dict[date, list[bool]] = defaultdict(list)
    for day, correct in graded:
        by_week[_week_start(day)].append(correct)

    rows = []
    for week in window:
        results = by_week.get(week, [])
        n = len(results)
        correct = sum(1 for ok in results if ok)
        rows.append(TrackRecordWeekOut(
            week_start=week.isoformat(), n=n, correct=correct,
            hit_rate=round(correct / n, 3) if n else None, tracked=n > 0,
        ))
    return rows


def compute_track_record(
    db_path: Path, schedule: list[dict] | None = None, today: date | None = None
) -> list[TrackRecordOut]:
    schedule = schedule or []
    today = today or date.today()
    rows: list[TrackRecordOut] = []
    settled_rows: dict[str, TrackRecordOut] = {}
    graded_by_market: dict[str, list[tuple[date, bool]]] = {}

    with get_connection(db_path) as conn:
        market_counts = conn.execute(
            "SELECT market, COUNT(*) as total FROM game_market_predictions GROUP BY market"
        ).fetchall()

    for market in _SETTLED_MARKETS:
        row, graded = _settle_market_predictions(db_path, schedule, market)
        if row is not None:
            settled_rows[market] = row
            graded_by_market[market] = graded

    for row in market_counts:
        settled_row = settled_rows.get(row["market"])
        if settled_row is not None:
            rows.append(settled_row)
        else:
            # Stored rows we cannot judge against results: the count is real,
            # the rate is not measured. hit_rate stays None so no surface can
            # print a 0% for it.
            rows.append(TrackRecordOut(
                market=row["market"], total_predictions=row["total"],
                correct_predictions=0, hit_rate=None, settled=False,
            ))

    game_outcome, outcome_graded = _settle_game_outcome(db_path, schedule)
    if game_outcome is not None:
        settled_rows["game_outcome"] = game_outcome
        graded_by_market["game_outcome"] = outcome_graded
        rows.append(game_outcome)

    # Every settled market gets the SAME window, so the weekly tables align
    # row for row on the page without the frontend inventing a join.
    window = _tracking_window(db_path, schedule, today)
    if window:
        for row in rows:
            if row.settled:
                row.weekly = _weekly_rows(graded_by_market.get(row.market, []), window)

    return rows
