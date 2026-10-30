import gzip
import json

from nba_predictor.data import injury_log

PAYLOAD = {
    "injuries": [{
        "displayName": "Boston Celtics",
        "injuries": [
            {
                "status": "Out", "date": "2026-10-20T14:00Z", "shortComment": "Knee soreness.",
                "details": {"type": "Knee", "returnDate": "2026-10-28"},
                "athlete": {
                    "displayName": "Jaylen Brown", "position": {"abbreviation": "SG"},
                    "links": [{"href": "https://www.espn.com/nba/player/_/id/3917376/jaylen-brown"}],
                },
            },
            {"status": "Questionable", "athlete": {"displayName": "No Links", "links": []}},
        ],
    }]
}


def test_rows_are_keyed_by_athlete_id_and_never_backfilled():
    rows = injury_log.normalise(PAYLOAD, "2026-10-20T15:00:00+00:00")
    assert [r["player_id"] for r in rows] == ["3917376", ""]
    brown = rows[0]
    assert (brown["status"], brown["injury"], brown["return_date"], brown["position"]) == ("Out", "Knee", "2026-10-28", "SG")
    assert brown["team_name"] == "Boston Celtics" and brown["seen_at"].startswith("2026-10-20")
    # Unknown stays blank: no invented id, no invented date.
    assert rows[1]["reported_at"] == "" and rows[1]["return_date"] == ""


def test_the_uid_form_is_a_fallback_but_cannot_match_other_parameters():
    assert injury_log.athlete_id([{"href": "sportscenter://x?uid=s:40~l:46~a:5105571&section=stats"}]) == "5105571"
    assert injury_log.athlete_id([{"href": "https://x/?data:123"}]) == ""


def test_snapshots_append_never_overwrite(tmp_path):
    rows = injury_log.normalise(PAYLOAD, "2026-10-20T15:00:00+00:00")
    path = injury_log.append(rows, "2026-10-20T15:00:00+00:00", tmp_path)
    injury_log.append(rows, "2026-10-20T21:00:00+00:00", tmp_path)
    assert path.name == "nba_injuries_2026-10-20.jsonl.gz"
    lines = gzip.open(path, "rt").read().splitlines()
    assert len(lines) == 4 and json.loads(lines[0])["player_name"] == "Jaylen Brown"


def test_main_makes_one_stubbed_fetch_and_appends_a_timestamped_snapshot(tmp_path, monkeypatch):
    # OUT_DIR is relative, so run from a temp dir; requests.get is stubbed, so no network.
    monkeypatch.chdir(tmp_path)
    calls = []

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return PAYLOAD

    monkeypatch.setattr(injury_log.requests, "get", lambda url, timeout: calls.append(url) or Resp())
    assert injury_log.main() == 0
    assert calls == [injury_log.URL]
    (path,) = (tmp_path / "data" / "injuries").glob("nba_injuries_*.jsonl.gz")
    rows = [json.loads(line) for line in gzip.open(path, "rt")]
    assert len(rows) == 2 and rows[0]["player_name"] == "Jaylen Brown"
    assert rows[0]["seen_at"].endswith("+00:00") and rows[0]["seen_at"][:10] == path.name[len("nba_injuries_"):-len(".jsonl.gz")]
