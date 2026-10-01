"""Measure the before -> after of the track-record reversal, on real games.

The shipped `data/tracking.db` in this checkout holds ZERO rows in all four
tables and the shipped `data/public_snapshot.json` carries `track_record: []`,
so on this checkout the real before/after is 0 -> 0. That is a fact about the
checkout, not a finding: NBA has no populated record here for a headline to
hide, which is the opposite of PL's shape and is reported as such in the PR.

To put a number on the change rather than assert one, this seeds throwaway
databases from the REAL games and REAL final scores in
`data/cache/schedule/games.json` (1,387 completed games) and evaluates both
rules on them. The schedule and the results are the repo's own; the picks are
synthetic and are labelled as such wherever a number comes from them. Nothing
is written into the repo.

Two shapes, because they are the two things that happened in practice:

  * WITH a pre-tip snapshot: every game has a read made before tip-off and
    then a re-run after it. The old rule and the new one agree on n, because
    the earliest recorded pick for each game is the pre-tip one -- which is the
    property that makes rule 2 honest rather than flattering.
  * WITHOUT one: every game's only pick is a re-run. This is the "stops
    tracking" failure. The old rule published an EMPTY headline and counted
    every pick as withheld; the new one counts them all and says how many were
    made after tip-off.

Run: python3 tools/measure_track_record_swap.py
"""
import json
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_predictor.services.hub_service import compute_track_record  # noqa: E402
from nba_predictor.tracking import store  # noqa: E402
from nba_predictor.tracking.timing import latest_pre_tip  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RERUN_AT = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc).isoformat()


def seed(tmp: Path, schedule: list[dict], *, with_pre_tip: bool) -> Path:
    """One completed game -> a pre-tip read and/or a post-tip re-run.

    The re-run carries the answer baked in (0.9 on the side that in fact won
    most of the time), which is what a backtest over a finished game does, and
    it is why the two rules disagree on the VERDICT and not only on the n.
    """
    db = tmp / f"tracking-{'with' if with_pre_tip else 'without'}-pretip.db"
    store.init_db(db)
    for i, game in enumerate(schedule):
        if not game.get("completed") or game.get("home_pts") is None:
            continue
        if with_pre_tip:
            # 14:00Z is 09:00/10:00 Eastern, before the noon-Eastern cutoff
            # `timing.pick_cutoff` falls back to when a cached scoreboard has
            # no tip_off. The model is right most of the time, as it is.
            store.insert_prediction(
                db, game_id=game["game_id"], created_at=f"{game['game_date']}T14:00:00+00:00",
                model_version="v1", home_win_prob=0.62 if i % 5 else 0.38,
                predicted_margin=4.0, predicted_total=224.0,
            )
        store.insert_prediction(
            db, game_id=game["game_id"], created_at=RERUN_AT, model_version="v2",
            home_win_prob=0.9, predicted_margin=9.0, predicted_total=230.0,
        )
    return db


def headline_before(db: Path, schedule: list[dict]) -> tuple[int, int, int]:
    """The rule as it stood on `main`: the latest pre-tip row per game, with
    every game whose only rows came from a re-run counted as withheld.

    Reimplemented here from the definition rather than measured from `main`, so
    the comparison is two rules on one dataset and not two checkouts.
    """
    by_id = {g["game_id"]: g for g in schedule}
    with store.get_connection(db) as conn:
        rows = conn.execute("SELECT * FROM predictions").fetchall()
    by_game: dict[str, list] = {}
    for row in rows:
        by_game.setdefault(row["game_id"], []).append(row)

    total = correct = withheld = 0
    for game_id, game_rows in by_game.items():
        game = by_id.get(game_id)
        if game is None or not game.get("completed") or game.get("home_pts") is None:
            continue
        pick = latest_pre_tip(game_rows, game)
        if pick is None:
            withheld += 1
            continue
        total += 1
        correct += (pick["home_win_prob"] >= 0.5) == (game["home_pts"] > game["away_pts"])
    return total, correct, withheld


def report(db: Path, schedule: list[dict], label: str) -> None:
    before_n, before_ok, before_withheld = headline_before(db, schedule)
    rows = {r.market: r for r in compute_track_record(db, schedule, today=date(2026, 11, 1))
            if r.settled}
    games = rows["game_outcome"]
    pre = games.pre_tip

    print(f"\n{label}")
    print(f"  before (pre-tip only, `main`'s rule): n={before_n}, "
          f"{before_ok / before_n:.3f}" if before_n else "  before: n=0, no rate at all")
    if not before_n:
        print(f"    ...and {before_withheld} picks withheld from the headline entirely.")
    print(f"  after  (headline, every counted pick): n={games.total_predictions}, "
          f"{games.hit_rate:.3f}")
    print(f"  secondary (pre-tip subset):            n={pre.total_predictions}, "
          f"{pre.hit_rate:.3f}" if pre.hit_rate is not None else
          f"  secondary (pre-tip subset):            n={pre.total_predictions}, no rate")
    print(f"  of the headline, made at/after tip-off: n={games.n_rebuilt}")
    assert games.total_predictions == pre.total_predictions + games.n_rebuilt, (
        "the reconciliation does not hold: "
        f"{games.total_predictions} != {pre.total_predictions} + {games.n_rebuilt}"
    )
    assert sum(w.n for w in games.weekly) == games.total_predictions, "the week table drifted"
    assert all(p.created_at for p in games.per_pick), "a pick row lost its own timestamp"


def main() -> int:
    schedule = json.loads((REPO / "data" / "cache" / "schedule" / "games.json").read_text())
    completed = sum(1 for g in schedule if g.get("completed") and g.get("home_pts") is not None)
    print(f"REAL schedule: {len(schedule)} games, {completed} completed with a final score.")
    print("Seeded picks are SYNTHETIC; the games and results are the repo's own.")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        report(seed(tmp, schedule, with_pre_tip=True), schedule,
               "A. a pre-tip snapshot was captured for every game, then the model was re-run")
        report(seed(tmp, schedule, with_pre_tip=False), schedule,
               "B. no pre-tip snapshot: every game's only pick is a re-run  <- the 'stops tracking' case")
    print("\nBoth rows: total == pre_tip + n_rebuilt, and the week table sums to the headline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
