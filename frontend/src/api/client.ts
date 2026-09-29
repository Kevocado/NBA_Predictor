
import type { Explanation } from "../predictor-ui";

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

export interface HeadToHeadMeeting {
  game_id: string;
  game_date: string;
  home_team: string;
  away_team: string;
  home_pts: number | null;
  away_pts: number | null;
}

export interface Game {
  game_id: string;
  game_date: string;
  /** UTC start time; missing for games cached before it was recorded. */
  tip_off?: string | null;
  home_team: string;
  away_team: string;
  prediction: Prediction | null;
  /** The pick shown was made after tip-off: labelled, never counted. */
  rebuilt?: boolean;
  completed: boolean;
  home_pts: number | null;
  away_pts: number | null;
}

export interface MarketPrediction {
  market: string;
  selection: string;
  model_probability: number;
  market_probability: number | null;
  edge: number | null;
  bookmaker: string | null;
  american_odds: number | null;
  point: number | null;
  /** Priced after tip-off: shown, never judged. */
  rebuilt?: boolean;
}

export interface GameDetail extends Game {
  markets: MarketPrediction[];
  head_to_head: HeadToHeadMeeting[];
  home_recent_form: string[];
  away_recent_form: string[];
}

export interface SeasonBounds {
  first_week_start: string | null;
}

export interface PlayerProp {
  player_id: string;
  player_name: string;
  stat: string;
  predicted_value: number;
  actual_value: number | null;
  /** Built after tip-off (a retrain backtest): shown, never judged. */
  rebuilt?: boolean;
}

export interface TrackRecordWeek {
  /** ISO date of the Monday the week starts on. */
  week_start: string;
  n: number;
  correct: number;
  /** null when the week graded nothing: an em dash, never a fabricated 0%. */
  hit_rate: number | null;
  tracked: boolean;
}

export interface TrackRecord {
  market: string;
  total_predictions: number;
  correct_predictions: number;
  /** null when nothing was graded — a dash, never a 0% claim. */
  hit_rate: number | null;
  /** Finals whose only picks were made after tip-off (left out of the counts). */
  n_rebuilt?: number;
  /** Picks left out of the rate: the result landed on the line, or there was no line. */
  n_push?: number;
  /** False when the backend has no rule for judging this market: never render a rate. */
  settled?: boolean;
  /** Every week since tracking began, through this week; gaps carried as tracked=false. */
  weekly?: TrackRecordWeek[];
}

export interface VsMarketWeek {
  week_start: string;
  /** Games actually compared with a price that week. */
  tracked: boolean;
  n: number;
  /** Model minus price, in percentage points; null when nothing was compared. */
  mean_edge_points: number | null;
  disagreement_n: number;
  disagreement_hit_rate: number | null;
}

export interface VsMarketScope {
  population: string;
  weekly_from: string | null;
  weekly_through: string | null;
  n_games_total: number;
  n_games_in_weekly: number;
  n_games_outside_weekly: number;
}

export interface VsMarket {
  market: string;
  n: number;
  mean_model_probability: number | null;
  mean_market_probability: number | null;
  mean_edge_points: number | null;
  disagreement_n: number;
  disagreement_hit_rate: number | null;
  disagreement_game_ids: string[];
  weekly: VsMarketWeek[];
  scope: VsMarketScope;
  /** Sentences the panel prints verbatim — see hub_service._VS_MARKET_METHOD. */
  method: Record<string, string | number>;
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

// A backend that accepts the connection but never answers must still end in
// the page's error state (with Try again), never an endless "Loading…".
export const REQUEST_TIMEOUT_MS = 15_000;

function fetchWithTimeout(url: string, init: RequestInit = {}): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  return fetch(url, { ...init, signal: controller.signal }).finally(() => clearTimeout(timer));
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetchWithTimeout(`${API_BASE}${path}`);
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
  /** The shared plain-English summary, via this API's explainer proxy. The id
   *  is the game's own id — the only value the route needs, and the only one
   *  the site has. */
  explainGame: (gameId: string) => fetchJson<Explanation>(`/api/explain/nba/${gameId}`),
  getSeasonFirstWeek: () => fetchJson<SeasonBounds>("/season/first-week"),
  getHubTeams: () => fetchJson<TeamHubRow[]>("/hub/teams"),
  getHubPlayers: () => fetchJson<PlayerHubRow[]>("/hub/players"),
  getHubRankings: () => fetchJson<PowerRankingRow[]>("/hub/rankings"),
  getHubStandings: () => fetchJson<StandingsRow[]>("/hub/standings"),
  getTrackRecord: () => fetchJson<TrackRecord[]>("/hub/track-record"),
  getVsMarket: () => fetchJson<VsMarket>("/hub/vs-market"),
  getManifest: () => fetchJson<Manifest>("/manifest"),
  getCalibration: () => fetchJson<CalibrationBin[]>("/calibration"),
};