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
    predicted_value: float
    actual_value: float | None = None
    # Built after tip-off (the retrain backtest): shown, never judged.
    rebuilt: bool = False


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


class SeasonBoundsOut(BaseModel):
    first_week_start: str | None = None
