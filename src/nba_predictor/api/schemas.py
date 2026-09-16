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


class GameOut(BaseModel):
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    prediction: PredictionOut | None = None


class GameDetailOut(GameOut):
    markets: list[MarketPredictionOut] = []


class PlayerPropOut(BaseModel):
    player_id: str
    player_name: str
    stat: str
    predicted_value: float


class TrackRecordOut(BaseModel):
    market: str
    total_predictions: int
    correct_predictions: int
    hit_rate: float
