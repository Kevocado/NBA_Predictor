"""Kalshi moneylines as the market side of 'against the odds': free, public, no key, no rate-limited paid feed."""
import pytest

from nba_predictor.data import kalshi_lines
from nba_predictor.pipeline import refresh_kalshi
from nba_predictor.tracking import store


def market(team, yes_bid, yes_ask, event="KXNBAGAME-26OCT21BOSDET", volume=500):
    return {
        "ticker": f"{event}-{team}", "event_ticker": event, "title": f"{team} wins", "status": "active",
        "yes_bid_dollars": str(yes_bid), "yes_ask_dollars": str(yes_ask), "no_bid_dollars": str(round(1 - yes_ask, 2)), "no_ask_dollars": str(round(1 - yes_bid, 2)),
        "volume_fp": str(volume), "occurrence_datetime": "2026-10-21T23:00:00Z", "close_time": "2026-10-24T00:00:00Z",
    }


SCHEDULE = [
    {"game_id": "g1", "game_date": "2026-10-21", "home_team": "DET", "away_team": "BOS", "tip_off": "2026-10-21T23:00Z", "completed": False},
    {"game_id": "g2", "game_date": "2026-10-21", "home_team": "MIA", "away_team": "NYK", "tip_off": "2026-10-21T23:30Z", "completed": False},
]


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.db"
    store.init_db(path)
    store.insert_prediction(path, game_id="g1", created_at="2026-10-20T00:00:00+00:00", model_version="v", home_win_prob=0.60, predicted_margin=3.0, predicted_total=225.0)
    return path


def test_each_matched_game_gets_both_sides_with_normalised_market_probabilities(db, monkeypatch):
    monkeypatch.setattr(kalshi_lines, "fetch_series", lambda session, series, status, min_close_ts=None: [market("DET", 0.44, 0.46), market("BOS", 0.53, 0.55)])
    assert refresh_kalshi.refresh_kalshi_markets(SCHEDULE, db) == 2
    rows = {r["selection"]: r for r in store.get_market_predictions_for_game(db, "g1")}
    assert set(rows) == {"DET", "BOS"}
    det, bos = rows["DET"], rows["BOS"]
    assert det["bookmaker"] == "Kalshi" and det["market"] == "h2h" and det["point"] is None
    assert det["market_probability"] + bos["market_probability"] == pytest.approx(1.0)
    assert det["model_probability"] == pytest.approx(0.60) and bos["model_probability"] == pytest.approx(0.40)
    assert det["edge"] == pytest.approx(det["model_probability"] - det["market_probability"])
    assert det["american_odds"] is not None and bos["american_odds"] is not None


def test_an_unquoted_or_wide_market_is_skipped_not_guessed(db, monkeypatch):
    monkeypatch.setattr(kalshi_lines, "fetch_series", lambda *a, **k: [market("DET", 0.0, 0.46), market("BOS", 0.30, 0.55)])
    assert refresh_kalshi.refresh_kalshi_markets(SCHEDULE, db) == 0       # no bid on one side; 25-point spread on the other


def test_a_game_without_a_stored_pick_is_skipped(db, monkeypatch):
    monkeypatch.setattr(kalshi_lines, "fetch_series", lambda *a, **k: [market("MIA", 0.40, 0.42, "KXNBAGAME-26OCT21NYKMIA"), market("NYK", 0.57, 0.59, "KXNBAGAME-26OCT21NYKMIA")])
    assert refresh_kalshi.refresh_kalshi_markets(SCHEDULE, db) == 0


def test_a_kalshi_outage_stores_nothing_and_raises_nothing(db, monkeypatch):
    def down(*a, **k):
        raise RuntimeError("kalshi down")

    monkeypatch.setattr(kalshi_lines, "fetch_series", down)
    assert refresh_kalshi.refresh_kalshi_markets(SCHEDULE, db) == 0
