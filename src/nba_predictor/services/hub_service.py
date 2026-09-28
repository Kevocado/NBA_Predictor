import json
from pathlib import Path

from nba_predictor.api.schemas import TrackRecordOut
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


def _settle_game_outcome(db_path: Path, schedule: list[dict]) -> TrackRecordOut | None:
    """Settles the model's own win/loss call (predictions.home_win_prob >= 0.5)
    against each game's actual result, using only the pick made before
    tip-off. Every result is a real completed game, joined via the schedule
    cache's home_pts/away_pts (populated by pipeline/ingest.py)."""
    picks, rebuilt = pre_tip_picks(db_path, schedule)
    if not picks and not rebuilt:
        return None

    correct = sum(
        1 for game, pick in picks
        if (pick["home_win_prob"] >= 0.5) == (game["home_pts"] > game["away_pts"])
    )
    total = len(picks)
    return TrackRecordOut(
        market="game_outcome", total_predictions=total, correct_predictions=correct,
        hit_rate=round(correct / total, 3) if total else None, n_rebuilt=rebuilt,
        settled=True,
    )


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


def _settle_market_predictions(db_path: Path, schedule: list[dict], market: str) -> TrackRecordOut | None:
    """Settles one priced market against actual results, once per game.

    Odds refresh stores both selections for every bookmaker on every run, so
    judging every row would pin the rate near 50%: per game, take the latest
    run made before tip-off and its selection with the higher model
    probability, then grade that one pick. Returns None when the schedule
    holds no finished game for this market -- the caller then reports the
    stored rows as unsettled rather than as 0%.
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
    correct = 0
    n_push = 0
    total = 0
    for game_id, game_rows in by_game.items():
        game = schedule_by_id.get(game_id)
        if game is None or not game.get("completed") or game.get("home_pts") is None:
            continue
        completed_games += 1
        latest = latest_pre_tip(game_rows, game)
        if latest is None:
            continue
        run = [r for r in game_rows if r["created_at"] == latest["created_at"]]
        pick = max(run, key=lambda r: r["model_probability"])
        verdict = _grade_market_row(pick, game)
        if verdict is None:
            n_push += 1
            continue
        total += 1
        if verdict:
            correct += 1

    if completed_games == 0:
        return None
    return TrackRecordOut(
        market=market, total_predictions=total, correct_predictions=correct,
        hit_rate=round(correct / total, 3) if total else None,
        n_push=n_push, settled=True,
    )


# The markets this repo has a grading rule for. Anything else in
# game_market_predictions is reported with its stored count and `settled=false`.
_SETTLED_MARKETS = ("h2h", "spread", "total")


def compute_track_record(db_path: Path, schedule: list[dict] | None = None) -> list[TrackRecordOut]:
    schedule = schedule or []
    rows: list[TrackRecordOut] = []

    with get_connection(db_path) as conn:
        market_counts = conn.execute(
            "SELECT market, COUNT(*) as total FROM game_market_predictions GROUP BY market"
        ).fetchall()

    settled = {market: _settle_market_predictions(db_path, schedule, market) for market in _SETTLED_MARKETS}
    for row in market_counts:
        graded = settled.get(row["market"])
        if graded is not None:
            rows.append(graded)
        else:
            # Stored rows we cannot judge against results: the count is real,
            # the rate is not measured. hit_rate stays None so no surface can
            # print a 0% for it.
            rows.append(TrackRecordOut(
                market=row["market"], total_predictions=row["total"],
                correct_predictions=0, hit_rate=None, settled=False,
            ))

    game_outcome = _settle_game_outcome(db_path, schedule)
    if game_outcome is not None:
        rows.append(game_outcome)

    return rows
