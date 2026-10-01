from pydantic import BaseModel


class TeamOut(BaseModel):
    abbreviation: str
    name: str
    conference: str
    division: str


class PredictionOut(BaseModel):
    home_win_probability: float
    predicted_margin: float
    predicted_total: float


class MarketPredictionOut(BaseModel):
    market: str
    selection: str
    model_probability: float
    market_probability: float | None = None
    edge: float | None = None
    bookmaker: str | None = None
    american_odds: int | None = None
    point: float | None = None
    # Priced after tip-off (e.g. a refresh during the game): shown, never judged.
    rebuilt: bool = False


class GameOut(BaseModel):
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    tip_off: str | None = None
    prediction: PredictionOut | None = None
    # True when the pick shown was made at or after tip-off (a backtest). It
    # is labelled on the site and never counted.
    rebuilt: bool = False
    completed: bool = False
    home_pts: int | None = None
    away_pts: int | None = None


class HeadToHeadMeetingOut(BaseModel):
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    home_pts: int | None = None
    away_pts: int | None = None


class GameDetailOut(GameOut):
    markets: list[MarketPredictionOut] = []
    head_to_head: list[HeadToHeadMeetingOut] = []
    home_recent_form: list[str] = []
    away_recent_form: list[str] = []


class PlayerPropOut(BaseModel):
    player_id: str
    player_name: str
    stat: str
    # A projection, in the stat's own unit -- points, rebounds, assists, made
    # threes. Deliberately not a probability: models/player_props.py is an
    # XGBRegressor returning a raw point total, and this repo has no calibrated
    # probability for it to serve. predict_double_double_probability exists and
    # has zero callers; shipping it is not this change. Do not add a
    # probability field here without calibration evidence.
    predicted_value: float
    actual_value: float | None = None
    # Built after tip-off (the retrain backtest): shown, never judged.
    rebuilt: bool = False
    # In-sample mean absolute error for this stat over resolved rows only (see
    # models.player_props.in_sample_mae_by_stat). None when nothing has been
    # resolved for the stat yet -- never 0.0, which would claim the model never
    # missed by a tenth of a point. The site renders this as "+/- N" next to
    # the projection, or says it has no error estimate yet.
    mae: float | None = None


class OutPlayerOut(BaseModel):
    """A player the availability gate removed from the ranking.

    Removed, not flagged in place: no list, no bar, no rank position. Shown
    once, below the lists, attributed and dated.
    """

    player_id: str
    player_name: str
    team: str
    status: str
    source: str
    dated: str


class DoubtfulPlayerOut(BaseModel):
    """A player the availability gate flagged as doubtful but kept ranked.

    Flagged in place, NOT removed: the player is still a legitimate call and
    still appears in /games/{id}/players. This is the note that makes the
    ranking honest -- before it, a day-to-day player reached no feed at all,
    which reads as "checked and clear".

    Deliberately the same six fields as OutPlayerOut and no more. ``status`` is
    the feed's own word ("Day-To-Day"), reported verbatim rather than reworded
    into a judgement; there is NO probability and NO downgrade coefficient
    field, because a status is not a number and this repo has no calibrated
    quantity to pair it with. Nothing here is a recommendation against the
    player: it states availability, nothing more.
    """

    player_id: str
    player_name: str
    team: str
    status: str
    source: str
    dated: str


class TrackRecordOut(BaseModel):
    market: str
    total_predictions: int
    correct_predictions: int
    # None when nothing was graded: 0.0 would claim every graded pick missed,
    # which is a different statement from "never measured".
    hit_rate: float | None = None
    # Final games whose only picks were made after tip-off: left out above.
    n_rebuilt: int = 0
    # Picks left out of the rate because there was nothing to grade them
    # against: the margin landed exactly on the line (a push), or the row
    # carried no line. Counted, never scored as a miss.
    n_push: int = 0
    # False when this repo has no rule for judging the market (or no results
    # to judge it against). The site shows the stored count and says so, and
    # never a fabricated 0%.
    settled: bool = True
    # Every week from the first tracked week through this week, gaps filled in
    # with tracked=false so a week with no picks reads as "not tracked"
    # instead of vanishing. Empty when the tracking DB has nothing to date.
    weekly: list["TrackRecordWeekOut"] = []


class TrackRecordWeekOut(BaseModel):
    week_start: str  # ISO date of the Monday the week starts on
    n: int
    correct: int
    # None for a week with no graded picks -- see TrackRecordOut.hit_rate.
    hit_rate: float | None = None
    tracked: bool


class VsMarketWeekOut(BaseModel):
    week_start: str  # ISO Monday date
    tracked: bool  # n > 0: games actually compared with a price
    n: int
    # Percentage points; None when nothing was compared that week.
    mean_edge_points: float | None = None
    disagreement_n: int = 0
    disagreement_hit_rate: float | None = None


class VsMarketScopeOut(BaseModel):
    population: str
    weekly_from: str | None = None
    weekly_through: str | None = None
    n_games_total: int
    n_games_in_weekly: int
    n_games_outside_weekly: int


class VsMarketOut(BaseModel):
    """The model's moneyline pick beside the price it was measured against.

    NBA stores the market's own de-vigged probability per side (see
    pipeline/refresh_odds), so unlike the NFL/CFB version this needs no
    implied-probability conversion of a line: the comparison reads the same
    row the pick was priced with.
    """

    market: str = "h2h"
    n: int = 0
    mean_model_probability: float | None = None
    mean_market_probability: float | None = None
    # Percentage points, signed toward the model; None when n == 0.
    mean_edge_points: float | None = None
    # Games where the model backed the side the price did not favour.
    disagreement_n: int = 0
    disagreement_hit_rate: float | None = None
    disagreement_game_ids: list[str] = []
    weekly: list[VsMarketWeekOut] = []
    scope: VsMarketScopeOut
    # Sentences the page prints verbatim; see hub_service._VS_MARKET_METHOD.
    method: dict[str, str | float] = {}


class SeasonBoundsOut(BaseModel):
    first_week_start: str | None = None
