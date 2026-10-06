/**
 * The REAL `GameDetailModal`, with the api stubbed, to photograph WHERE the signal
 * rows sit.
 *
 * `signals-harness.tsx` draws the row in isolation, which proves the visual and
 * nothing about its position. This change moved the block ABOVE the "Player
 * projections" section -- #36's own comment claimed that position while the code
 * put the rows last -- and PL's `FixtureModal` and Sports' `GameDetailModal` both
 * already put them there. A position is a claim about the rendered page, so the
 * capture has to be the page.
 *
 * The stubbed numbers are invented and labelled as such in `signals-harness.html`'s
 * sibling page. No real fixture's status is altered to produce this: the real
 * endpoint answers `[]` for every game on the slate, which is the honest answer and
 * the reason no real game page can carry this row today.
 *
 * Not part of the deployed bundle -- vite builds `index.html` only.
 */
import { createRoot } from "react-dom/client";

import GameDetailModal from "./components/GameDetailModal";
import { api, type GameDetail } from "./api/client";

const GAME_ID = "401";

const DETAIL = {
  game_id: GAME_ID,
  game_date: "2026-10-05",
  home_team: "BOS",
  away_team: "LAL",
  completed: false,
  home_pts: null,
  away_pts: null,
  markets: [],
  head_to_head: [],
  home_recent_form: [],
  away_recent_form: [],
} as unknown as GameDetail;

/** Enough players that BOTH the box score and the projections section render. */
const PLAYERS = [
  { player_id: "p1", player_name: "A. Player", stat: "points", predicted_value: 28.0, actual_value: null },
  { player_id: "p2", player_name: "B. Player", stat: "points", predicted_value: 24.5, actual_value: null },
  { player_id: "p3", player_name: "C. Player", stat: "rebounds", predicted_value: 11.0, actual_value: null },
];

const SIGNAL = {
  kind: "absence",
  sport: "nba",
  game_id: GAME_ID,
  headline: {
    text: "Out: A. Player, our #1 Points projection (31.5 pts)",
    figures: { projection: 31.5 },
  },
  n: 1,
  source: "ESPN injury report, 2026-10-04 · 1 player out",
  as_of: "2026-10-04",
  strength: 1.0,
  pre_kickoff_only: true,
  visual: "absence_strip",
};

// Every feed the modal reads. Stubs, not mocks of the module: this harness runs in
// the browser, so `api` is a real object whose methods are replaced.
Object.assign(api, {
  getGameDetail: async () => DETAIL,
  getGamePlayers: async () => PLAYERS,
  getGameOutPlayers: async () => [],
  getHubPlayers: async () => [],
  getHubTeams: async () => [],
  getHubRankings: async () => [],
  getHubStandings: async () => [],
  getTrackRecord: async () => [],
  getVsMarket: async () => null,
  getManifest: async () => null,
  getCalibration: async () => [],
  explainGame: async () => null,
  getGameSignals: async () => ({ sport: "nba", id: GAME_ID, signals: [SIGNAL] }),
});

const el = document.getElementById("root");
if (!el) throw new Error("no #root to mount into");
createRoot(el).render(<GameDetailModal gameId={GAME_ID} onClose={() => {}} />);
