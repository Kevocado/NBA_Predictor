const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export interface Team {
  abbreviation: string;
  name: string;
  conference: "East" | "West";
  division: string;
}

export interface Prediction {
  home_win_probability: number;
  predicted_margin: number;
  predicted_total: number;
}

export interface Game {
  game_id: string;
  game_date: string;
  home_team: string;
  away_team: string;
  prediction: Prediction | null;
}

export interface MarketPrediction {
  market: string;
  selection: string;
  model_probability: number;
  market_probability: number | null;
  edge: number | null;
  bookmaker: string | null;
  american_odds: number | null;
}

export interface GameDetail extends Game {
  markets: MarketPrediction[];
}

export interface PlayerProp {
  player_id: string;
  player_name: string;
  stat: string;
  predicted_value: number;
}

export interface TrackRecord {
  market: string;
  total_predictions: number;
  correct_predictions: number;
  hit_rate: number;
}

export interface TeamHubRow {
  abbreviation: string;
  conference: "East" | "West";
  division: string;
  wins: number;
  losses: number;
  points_per_game: number;
  opp_points_per_game: number;
  net_rating: number;
  pace: number;
  streak: number;
}

export interface PlayerHubRow {
  player_id: string;
  player_name: string;
  team: string;
  position: string;
  rating: number;
  live_form_rating: number;
  points_per_game: number;
  rebounds_per_game: number;
  assists_per_game: number;
  fg_pct: number;
  three_pt_pct: number;
  ft_pct: number;
  usage_rate: number;
  minutes_per_game: number;
}

export interface PowerRankingRow {
  rank: number;
  abbreviation: string;
  power_rating: number;
  trend: "up" | "down" | "steady";
}

export interface StandingsRow {
  conference: "East" | "West";
  seed: number;
  abbreviation: string;
  wins: number;
  losses: number;
  win_pct: number;
  games_back: number;
  playoff_status: "clinched" | "play-in" | "eliminated" | "in-hunt";
}

export interface Manifest {
  model_version: string;
  trained_at: string;
  models: string[];
  metrics: Record<string, Record<string, number | null>>;
}

export interface CalibrationBin {
  bin_start: number;
  bin_end: number;
  predicted_rate: number;
  actual_rate: number;
  count: number;
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`Request to ${path} failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  getTeams: () => fetchJson<Team[]>("/teams"),
  getTeam: (abbreviation: string) => fetchJson<Team>(`/teams/${abbreviation}`),
  getGames: (date: string) => fetchJson<Game[]>(`/games?date=${date}`),
  getGamesWeek: (start: string) => fetchJson<Game[]>(`/games/week?start=${start}`),
  getGameDetail: (gameId: string) => fetchJson<GameDetail>(`/games/${gameId}`),
  getGamePlayers: (gameId: string) => fetchJson<PlayerProp[]>(`/games/${gameId}/players`),
  getHubTeams: () => fetchJson<TeamHubRow[]>("/hub/teams"),
  getHubPlayers: () => fetchJson<PlayerHubRow[]>("/hub/players"),
  getHubRankings: () => fetchJson<PowerRankingRow[]>("/hub/rankings"),
  getHubStandings: () => fetchJson<StandingsRow[]>("/hub/standings"),
  getTrackRecord: () => fetchJson<TrackRecord[]>("/hub/track-record"),
  getManifest: () => fetchJson<Manifest>("/manifest"),
  getCalibration: () => fetchJson<CalibrationBin[]>("/calibration"),
};