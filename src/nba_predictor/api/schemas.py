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


class TrackRecordOut(BaseModel):
    market: str
    total_predictions: int
    correct_predictions: int
    hit_rate: float
    # Final games whose only picks were made after tip-off: left out above.
    n_rebuilt: int = 0


class SeasonBoundsOut(BaseModel):
    first_week_start: str | None = None
