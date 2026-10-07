
import type { Explanation, Signal } from "../predictor-ui";

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
  // Spread cover probability (None when no line available)
  cover_prob_spread?: number | null;
  // Total over probability (None when no line available)
  cover_prob_total?: number | null;
  // Residual sigma from the manifest, driving the above
  margin_sigma?: number | null;
  total_sigma?: number | null;
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

/** `GET /api/signals/{game_id}`'s envelope. */
export interface SignalsResponse {
  sport: string;
  id: string;
  signals: Signal[];
}

export interface GameDetail extends Game {
  markets: MarketPrediction[];
  head_to_head: HeadToHeadMeeting[];
  home_recent_form: string[];
  away_recent_form: string[];
  /** The availability gate's own sentence. */
  injury_summary?: string | null;
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
  /**
   * In-sample mean absolute error for THIS stat -- the HEADLINE, over every
   * counted pick: one per (game, player, stat), the earliest recorded,
   * whenever it was made.
   *
   * `null` when nothing has been resolved for the stat yet, which is "never
   * measured" and not "measured at zero": the site says so in words rather than
   * drawing a ±0, which would claim the model has never missed by a tenth of a
   * point. It is a per-stat aggregate, so it must never be read as this one
   * player's error -- there is no per-player graded record for NBA props.
   *
   * The name is unchanged from before 2026-10-01, when it was the pre-tip-only
   * figure: same field, fuller record behind it.
   */
  mae?: number | null;
  /**
   * The same error estimate over the picks made BEFORE tip-off only: what the
   * model would have said on the night. Published beside `mae` because the two
   * differ, and a reader weighing the model against a book needs to know which
   * one they are reading. `null`, never 0, when no counted pick for this stat
   * was made in time.
   */
  mae_pre_tip?: number | null;
  /** The n behind `mae`. An error estimate with no count is unweighable. */
  mae_n?: number;
  /** The n behind `mae_pre_tip`, which is the figure most likely to be thin. */
  mae_n_pre_tip?: number;
}

/** A player the availability gate removed from this game's ranking. Served by
 *  its own sibling route rather than wrapped onto the props response, because
 *  the props fetch is typed as `PlayerProp[]` and the two answer different
 *  questions: what the model projects, and who cannot be projected for. */
export interface OutPlayer {
  player_id: string;
  player_name: string;
  team: string;
  status: string;
  /** Where the out claim came from, in words. Never an id. */
  source: string;
  /** When it was known, in words. Never an epoch. */
  dated: string;
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

/** The secondary figure: the same tally over the picks made BEFORE tip-off. */
export interface TrackRecordTally {
  total_predictions: number;
  correct_predictions: number;
  /** null when nothing in the subset graded — a dash, never a 0% claim. */
  hit_rate: number | null;
  n_push?: number;
  weekly?: TrackRecordWeek[];
}

/** One recorded pick for one market, with when it was made. */
export interface TrackRecordPick {
  game_id: string;
  market: string;
  /** What the model backed, with the line it was priced at. */
  pick: string;
  /** What actually happened, in the same words. */
  actual: string;
  /** null for a push or a row with no line: nobody won it, so it is not a miss. */
  hit: boolean | null;
  /** Derived on every read from `created_at` vs tip-off, as UTC instants. */
  made_before_tip: boolean;
  created_at: string;
  /** False for a rerun that lost the earliest-pick contest. */
  counted: boolean;
  gameday: string | null;
  /** The line the pick was priced at. Present for spreads and totals, null
   *  for moneylines and markets without a line. Used to distinguish a push
   *  (hit=null but point present) from a row with no line (hit=null, no point).
   */
  point: number | null;
}

export interface TrackRecord {
  market: string;
  total_predictions: number;
  correct_predictions: number;
  /** null when nothing was graded — a dash, never a 0% claim. */
  hit_rate: number | null;
  /**
   * How stale this row is, in seconds, when a cached value is served because the
   * refresh failed. `null` means it was computed for this request — an ordinary
   * in-TTL cache hit is fresh, not stale, and saying otherwise would train a
   * reader to ignore the field.
   *
   * Set only on the cached `player_props` aggregate, and only on failure. No
   * surface renders player-props detail on this page yet, so the notice has no
   * visible home; it is here so the disclosure is not silently dropped.
   */
  served_stale_seconds?: number | null;
  /**
   * RENAMED IN MEANING, name kept: it used to count finals LEFT OUT of the
   * record. It is now the number of graded counted picks made at or after their
   * own tip-off, so `total_predictions === pre_tip.total_predictions +
   * n_rebuilt`. Subtract it to see how much of the headline is the rerun
   * rather than the night.
   */
  n_rebuilt?: number;
  /** Picks left out of the rate: the result landed on the line, or there was no line. */
  n_push?: number;
  /** Counted picks the schedule cannot date, so they appear in no week row. */
  n_unplaced?: number;
  /** The size of the pre-tip subset; equal to `pre_tip.total_predictions`. */
  n_pre_tip?: number;
  /** The pre-tip subset beside the headline. Absent only where `settled` is false. */
  pre_tip?: TrackRecordTally;
  /** Every recorded pick for this market, counted or not. */
  per_pick?: TrackRecordPick[];
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
  // null, not a status string, when the schedule says there is no race to be in
  // (off-season or pre-season). The backend drops the label rather than
  // inventing one, so the type has to admit its absence.
  playoff_status: "clinched" | "play-in" | "eliminated" | "in-hunt" | null;
  season_state: "offseason" | "preseason" | "regular" | "postseason";
}

export interface Manifest {
  model_version: string;
  trained_at: string;
  models: string[];
  metrics: Record<string, Record<string, number | null>>;
  training?: {
    n_train_games?: number;
    n_holdout_games?: number;
    n_current_season_games?: number;
  };
}

export interface PlayerPropsManifest {
  model_version: string;
  trained_at: string;
  models: string[];
  metrics: Record<string, Record<string, number | null>>;
  training?: {
    n_train_player_games?: number;
    in_sample_metrics?: boolean;
  };
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
  /** Players the gate removed from this game's ranking: shown once, below the
   *  lists, attributed and dated. A sibling route, not a field on the props
   *  response -- see `OutPlayer`. */
  getGameOutPlayers: (gameId: string) => fetchJson<OutPlayer[]>(`/games/${gameId}/players/out`),
  /** Spec §3's per-fixture signal payloads, rendered by the shared `SignalRows`.
   *  `[]` is a complete answer, never absent -- spec §2's "no data, no row". */
  getGameSignals: (gameId: string) => fetchJson<SignalsResponse>(`/api/signals/${encodeURIComponent(gameId)}`),
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
  getPlayerPropsManifest: () => fetchJson<PlayerPropsManifest>("/player-props-manifest"),
  getCalibration: () => fetchJson<CalibrationBin[]>("/calibration"),
};