from datetime import datetime, timezone
from pathlib import Path

from nba_predictor.data import sportsbook_api
from nba_predictor.data.team_reference import TEAMS
from nba_predictor.odds.value_bets import compute_edge, implied_probability, shin_devig
from nba_predictor.tracking import store

_NAME_TO_ABBREVIATION = {team.name: team.abbreviation for team in TEAMS}


def match_schedule_to_sportsbook_events(schedule: list[dict], events: list[dict]) -> dict[str, str]:
    """Maps our game_id -> sportsbook event key, matched by (home, away, date)."""
    by_matchup: dict[tuple[str, str, str], str] = {}
    for event in events:
        participants = event.get("participants", [])
        home_key = event.get("homeParticipantKey")
        if len(participants) != 2 or home_key is None:
            continue
        home = next((p for p in participants if p["key"] == home_key), None)
        away = next((p for p in participants if p["key"] != home_key), None)
        if home is None or away is None:
            continue
        home_abbr = _NAME_TO_ABBREVIATION.get(home.get("name"))
        away_abbr = _NAME_TO_ABBREVIATION.get(away.get("name"))
        if home_abbr is None or away_abbr is None:
            continue
        event_date = event.get("startTime", "")[:10]
        by_matchup[(home_abbr, away_abbr, event_date)] = event["key"]

    mapping = {}
    for game in schedule:
        key = (game["home_team"], game["away_team"], game["game_date"])
        if key in by_matchup:
            mapping[game["game_id"]] = by_matchup[key]
    return mapping


def refresh_market_predictions(schedule: list[dict], db_path: Path) -> int:
    """Fetches live odds for upcoming (not completed) scheduled games, de-vigs
    each bookmaker's own h2h market against the model's stored win
    probability (Shin's method), and stores value-bet rows. Returns the
    number of market-prediction rows written.

    Only h2h is wired here — spread/total would need the same devig
    treatment plus a stored line value the current schema doesn't carry
    (see hub_service._settle_h2h_market_predictions for the matching
    settlement-side gap); a natural follow-up once that's added.
    """
    upcoming = [g for g in schedule if not g.get("completed")]
    if not upcoming:
        return 0

    events = sportsbook_api.fetch_nba_events_raw()
    event_by_game = match_schedule_to_sportsbook_events(upcoming, events)
    if not event_by_game:
        return 0

    created_at = datetime.now(timezone.utc).isoformat()
    stored = 0

    for game in upcoming:
        event_key = event_by_game.get(game["game_id"])
        if event_key is None:
            continue

        prediction = store.get_latest_prediction_for_game(db_path, game["game_id"])
        if prediction is None:
            continue

        odds_rows = sportsbook_api.get_odds(event_key)
        h2h_by_bookmaker: dict[str, list[dict]] = {}
        for row in odds_rows:
            if row["market"] == "h2h":
                h2h_by_bookmaker.setdefault(row["bookmaker"], []).append(row)

        for bookmaker, rows in h2h_by_bookmaker.items():
            if len(rows) != 2:
                continue
            raw_probs = [implied_probability(r["american_odds"]) for r in rows]
            fair_probs = shin_devig(raw_probs)

            for row, fair_prob in zip(rows, fair_probs):
                model_prob = (
                    prediction["home_win_prob"] if row["selection"] == game["home_team"] else 1 - prediction["home_win_prob"]
                )
                store.insert_market_prediction(
                    db_path,
                    game_id=game["game_id"],
                    market="h2h",
                    selection=row["selection"],
                    model_probability=model_prob,
                    market_probability=fair_prob,
                    edge=compute_edge(model_prob, fair_prob),
                    bookmaker=bookmaker,
                    american_odds=row["american_odds"],
                    created_at=created_at,
                )
                stored += 1

    return stored
