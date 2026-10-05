/**
 * NBA's game modal fetches and renders its `absence` row.
 *
 * `/api/signals/{game_id}` shipped in #33 and is **live in production** — but
 * nothing on this page asked for it, so the endpoint served rows to no reader. The
 * vendored `SignalRows` can draw `absence_strip`; this is the hop between them, and
 * it mirrors Sports' `GameDetailModal` and PL's `FixtureModal` so "a fixture page
 * shows its signals" stays one implementation across all three.
 *
 * ## Why this is worth more on NBA than on the other two
 *
 * NBA is the sport where the absence was *invisible*: `api/routes.get_game_players`
 * drops an out player's rows before serialising, and `OutPlayerOut` carries no
 * projection, no stat and no rank. So today the page shows the out player as a
 * one-line mention under the lists and **nothing else** — while every player he
 * outranks silently gains a place. This row is the first thing on the page that says
 * what he was expected to do.
 *
 * ## EVERY failure is silence, and the failures are indistinguishable
 *
 * A 404, a network error and an honest empty list all leave `[]`. Spec §2's "no data,
 * no row" forbids a placeholder, and a signal is an *enhancement* on this page: it
 * must never become the page's error state.
 *
 * ## The gate is `detail.game_status`, not a truthiness test
 *
 * `/api/signals/{game_id}` answers `[]` for a game that has started, and the reason
 * is the same one that empties the props feed: a started game's roster is the one
 * known *before* tip-off, so quoting it afterwards is hindsight. Asking is skipped
 * outright, once per started game, rather than asking for an answer already known.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

import GameDetailModal from "./GameDetailModal";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    getGameDetail: vi.fn(),
    getGamePlayers: vi.fn(),
    getGameOutPlayers: vi.fn(),
    getHubPlayers: vi.fn(),
    getHubTeams: vi.fn(),
    getHubRankings: vi.fn(),
    getHubStandings: vi.fn(),
    getTrackRecord: vi.fn(),
    getVsMarket: vi.fn(),
    getManifest: vi.fn(),
    getCalibration: vi.fn(),
    explainGame: vi.fn(),
    // The signal fetch. A bare `vi.fn()` returns `undefined`, which the modal
    // calls `.then` on -- the same trap PL's five client mocks fell into.
    getGameSignals: vi.fn(async () => ({ sport: "nba", id: "401", signals: [] })),
  },
}));

/** The one absence row NBA's adapter can produce, in the shape it sends. */
const ABSENCE = {
  kind: "absence",
  sport: "nba",
  game_id: "401",
  headline: { text: "Out: Big, our #1 Points projection (31.5 pts)", figures: { projection: 31.5 } },
  n: 1,
  source: "ESPN injury report, 2026-10-04",
  as_of: "2026-10-04",
  strength: 1.0,
  pre_kickoff_only: true,
  visual: "absence_strip",
};

const detail = {
  game_id: "401",
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
};

/** Props with enough players that the projections section renders at all. */
const PLAYERS = [
  { player_id: "p1", player_name: "A. Player", stat: "points", predicted_value: 28.0, actual_value: null },
  { player_id: "p2", player_name: "B. Player", stat: "points", predicted_value: 24.0, actual_value: null },
];

function mockApi(over: Record<string, unknown> = {}) {
  vi.mocked(api.getGameDetail).mockResolvedValue({ ...detail } as never);
  vi.mocked(api.getGamePlayers).mockResolvedValue(PLAYERS as never);
  vi.mocked(api.getGameOutPlayers).mockResolvedValue([] as never);
  vi.mocked(api.getHubPlayers).mockResolvedValue([] as never);
  vi.mocked(api.getHubTeams).mockResolvedValue([] as never);
  vi.mocked(api.getHubRankings).mockResolvedValue([] as never);
  vi.mocked(api.getHubStandings).mockResolvedValue([] as never);
  vi.mocked(api.getTrackRecord).mockResolvedValue([] as never);
  vi.mocked(api.getVsMarket).mockResolvedValue(null as never);
  vi.mocked(api.getManifest).mockResolvedValue(null as never);
  vi.mocked(api.getCalibration).mockResolvedValue([] as never);
  vi.mocked(api.explainGame).mockRejectedValue(new Error("not needed"));
  Object.assign(api, over);
}

async function openModal() {
  render(<GameDetailModal gameId="401" onClose={() => {}} />);
  // The props have to be in before the absence row means anything: this page hides
  // its projections entirely when the availability feed has not loaded.
  await waitFor(() => expect(api.getGamePlayers).toHaveBeenCalled());
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

describe("NBA's absence row on the game modal", () => {
  it("asks for the signals when an upcoming game opens", async () => {
    mockApi();
    await openModal();
    await waitFor(() => expect(api.getGameSignals).toHaveBeenCalledWith("401"));
  });

  it("renders the row the endpoint sent", async () => {
    mockApi();
    vi.mocked(api.getGameSignals).mockResolvedValue({ sport: "nba", id: "401", signals: [ABSENCE] } as never);
    await openModal();
    // Read back out of the DOM: a screenshot proves a thing was drawn, only the
    // text says whether it was the RIGHT thing.
    expect(await screen.findByText(/Out: Big/)).toBeInTheDocument();
  });

  it("draws the figure beside the headline", async () => {
    mockApi();
    vi.mocked(api.getGameSignals).mockResolvedValue({ sport: "nba", id: "401", signals: [ABSENCE] } as never);
    await openModal();
    // `SignalRows` REFUSES a row whose words do not state this figure, so its
    // presence proves the marker and the headline agree.
    const marker = await screen.findByTestId("signal-absence");
    expect(marker).toHaveTextContent("31.5");
  });

  it("renders NO row and no empty state when the endpoint has none", async () => {
    mockApi();
    await openModal();
    await waitFor(() => expect(api.getGameSignals).toHaveBeenCalled());
    expect(screen.queryByTestId("signal-absence")).not.toBeInTheDocument();
    expect(screen.queryByText(/Out:/)).not.toBeInTheDocument();
  });

  it("renders nothing when the fetch rejects, and the page survives", async () => {
    mockApi();
    vi.mocked(api.getGameSignals).mockRejectedValue(new Error("404"));
    await openModal();
    await waitFor(() => expect(api.getGameSignals).toHaveBeenCalled());
    expect(screen.queryByTestId("signal-absence")).not.toBeInTheDocument();
    // The page itself is still there -- the row is an enhancement, never the page's
    // error state.
    expect(await screen.findByText(/Player projections/)).toBeInTheDocument();
  });

  it("does not ask about a completed game", async () => {
    // The endpoint answers [] for those: a completed game's roster is the one known
    // before tip-off, and quoting it afterwards is hindsight.
    //
    // `completed`, not a `game_status` this API does not send. An earlier version of
    // this fixture set `game_status: "final"`, and the test passed -- because the
    // COMPONENT was reading that same non-existent field, so the two agreed on
    // fiction and the gate looked like it worked. `tsc --noEmit` caught it: the
    // property is not on `GameDetail`. Against the real backend the gate was inert
    // and every completed game would have been asked.
    mockApi();
    vi.mocked(api.getGameDetail).mockResolvedValue({ ...detail, completed: true } as never);
    vi.mocked(api.getGameSignals).mockResolvedValue({ sport: "nba", id: "401", signals: [ABSENCE] } as never);
    await openModal();
    await waitFor(() => expect(api.getGameDetail).toHaveBeenCalled());
    expect(api.getGameSignals).not.toHaveBeenCalled();
    expect(screen.queryByTestId("signal-absence")).not.toBeInTheDocument();
  });

  it("does not read one game's completion to decide about the next", async () => {
    // The stale-status defect, and it only shows when the modal is REUSED for a
    // different game -- which it is, from the schedule list. The reset is a
    // `setDetail(null)` inside another effect, and an effect reads the value
    // captured by its own render, so on the render where `gameId` changes `detail`
    // is still the PREVIOUS game's.
    //
    // Asserted as a CALL COUNT because that is the whole of it: the fetch is keyed
    // on `gameId` and every write is guarded by `cancelled`, so no previous game's
    // data can reach the page. A `!detail` gate reads the old game's `completed`
    // and fires for the new id immediately, then again once its detail lands.
    mockApi();
    const { rerender } = render(<GameDetailModal gameId="401" onClose={() => {}} />);
    await waitFor(() => expect(api.getGameSignals).toHaveBeenCalledTimes(1));

    // A COMPLETED previous game, and a new id whose detail never lands.
    vi.mocked(api.getGameDetail).mockResolvedValue({ ...detail, completed: true } as never);
    vi.mocked(api.getGameDetail).mockReturnValue(new Promise(() => {}) as never);
    rerender(<GameDetailModal gameId="402" onClose={() => {}} />);
    await waitFor(() => expect(api.getGameDetail).toHaveBeenCalled());
    // Still exactly one: the completed "401" detail did not authorise a fetch, and
    // "402" has not arrived.
    expect(api.getGameSignals).toHaveBeenCalledTimes(1);
    expect(api.getGameSignals).toHaveBeenCalledWith("401");
  });

  it("does not ask while the game detail is still in flight", async () => {
    // PL's effect had this bug: a `detail?.status` gate is `undefined` -- falsy --
    // on the first render, so the fetch fired for every game including completed
    // ones, and only the re-run after the detail landed suppressed it.
    mockApi();
    vi.mocked(api.getGameDetail).mockReturnValue(new Promise(() => {}) as never);
    render(<GameDetailModal gameId="401" onClose={() => {}} />);
    await waitFor(() => expect(api.getGamePlayers).toHaveBeenCalled());
    expect(api.getGameSignals).not.toHaveBeenCalled();
  });
});
