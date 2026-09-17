from datetime import datetime, timezone
from pathlib import Path

from nba_predictor.data import sportsbook_api
from nba_predictor.data.team_reference import TEAMS
from nba_predictor.odds.value_bets import compute_edge, implied_probability, normal_cover_probability, shin_devig, std_from_mae
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


def _model_probability(
    market: str,
    selection: str,
    point: float | None,
    game: dict,
    prediction,
    margin_std: float,
    total_std: float,
) -> float | None:
    """Model's probability that `selection` wins its side of `market`."""
    if market == "h2h":
        return prediction["home_win_prob"] if selection == game["home_team"] else 1 - prediction["home_win_prob"]

    if market == "spread":
        if point is None:
            return None
        cover_mean = prediction["predicted_margin"] if selection == game["home_team"] else -prediction["predicted_margin"]
        return normal_cover_probability(mean=cover_mean, line=-point, std=margin_std)

    if market == "total":
        if point is None:
            return None
        over_probability = normal_cover_probability(mean=prediction["predicted_total"], line=point, std=total_std)
        return over_probability if selection == "over" else 1 - over_probability

    return None


def refresh_market_predictions(
    schedule: list[dict],
    db_path: Path,
    margin_std: float = 12.0,
    total_std: float = 15.0,
) -> int:
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
        rows_by_market_bookmaker: dict[tuple[str, str], list[dict]] = {}
        for row in odds_rows:
            rows_by_market_bookmaker.setdefault((row["market"], row["bookmaker"]), []).append(row)

        for (market, bookmaker), rows in rows_by_market_bookmaker.items():
            if len(rows) != 2:
                continue

            model_probs = [
                _model_probability(market, row["selection"], row.get("point"), game, prediction, margin_std, total_std)
                for row in rows
            ]
            if any(p is None for p in model_probs):
                continue

            raw_probs = [implied_probability(row["american_odds"]) for row in rows]
            fair_probs = shin_devig(raw_probs)

            for row, model_prob, fair_prob in zip(rows, model_probs, fair_probs):
                store.insert_market_prediction(
                    db_path,
                    game_id=game["game_id"],
                    market=market,
                    selection=row["selection"],
                    model_probability=model_prob,
                    market_probability=fair_prob,
                    edge=compute_edge(model_prob, fair_prob),
                    bookmaker=bookmaker,
                    american_odds=row["american_odds"],
                    point=row.get("point"),
                    created_at=created_at,
                )
                stored += 1

    return stored
