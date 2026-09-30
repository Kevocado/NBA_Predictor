import { describe, expect, it, vi, afterEach, beforeEach } from "vitest";
import { render, screen, within } from "@testing-library/react";
import GameDetailModal from "./GameDetailModal";
import { api, type TrackRecord } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    getGameDetail: vi.fn(),
    getGamePlayers: vi.fn(),
    getHubPlayers: vi.fn(),
    getTrackRecord: vi.fn(),
    explainGame: vi.fn(),
  },
}));

/** The winner-pick row, as `GET /hub/track-record` sends it.
 *
 *  `game_outcome` is the market that settles the model's own win/loss call, and
 *  `hub_service._settle_game_outcome` builds it from `pre_tip_picks` only — the
 *  picks rebuilt after tip-off are counted in `n_rebuilt` and left out of the
 *  counts. So this row is already a pre-tip-only record, which is why it is the
 *  one the block may quote. (`h2h` is a different claim — the same probability
 *  measured against the bookmaker's price — and quoting it here would put a
 *  different figure under the verdict's name.) */
const winnerPick = (total: number, correct: number): TrackRecord => ({
  market: "game_outcome",
  total_predictions: total,
  correct_predictions: correct,
  hit_rate: total ? Number((correct / total).toFixed(3)) : null,
  n_rebuilt: 3,
});

/** The bookmaker-price row, present to prove it is not the one being quoted. */
const vsMarket: TrackRecord = {
  market: "h2h",
  total_predictions: 38,
  correct_predictions: 21,
  hit_rate: 0.553,
};

const preTip = {
  game_id: "g1",
  game_date: "2026-11-01",
  home_team: "BOS",
  away_team: "MIA",
  prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
  completed: false,
  home_pts: null,
  away_pts: null,
  markets: [
    { market: "h2h", selection: "BOS", model_probability: 0.62, market_probability: 0.55, edge: 0.07, bookmaker: "DraftKings", american_odds: -130, point: null },
    { market: "spread", selection: "BOS", model_probability: 0.52, market_probability: 0.5, edge: 0.02, bookmaker: "DraftKings", american_odds: -110, point: -4.5 },
  ],
  head_to_head: [],
  home_recent_form: [],
  away_recent_form: [],
};

const completed = { ...preTip, completed: true, home_pts: 113, away_pts: 105 };

afterEach(() => {
  vi.restoreAllMocks();
});

// `vi.restoreAllMocks()` wipes a mockResolvedValue set at module scope, and the
// box score's team split is a join against this feed, so re-arm per test.
beforeEach(() => {
  vi.mocked(api.getHubPlayers).mockResolvedValue([]);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);
  vi.mocked(api.getTrackRecord).mockResolvedValue([winnerPick(40, 22)]);
});

/** Render with the network hard-blocked: every `fetch` rejects. The block's
 *  whole claim is that it needs none of it, so the parity rule is that the
 *  figures are on screen anyway. */
async function renderOffline(detail: unknown) {
  const fetchSpy = vi.fn(() => Promise.reject(new Error("network blocked")));
  vi.stubGlobal("fetch", fetchSpy);
  vi.mocked(api.getGameDetail).mockResolvedValue(detail as never);
  render(<GameDetailModal gameId="g1" onClose={() => {}} />);
  await screen.findByTestId("instant-block");
  return fetchSpy;
}

/** The record strip, found by its label — the label is the strip's own, and it
 *  is what tells two records apart if this page ever grows a second one. */
function recordStrip() {
  return screen.getByText("Winner pick made before tip-off").parentElement as HTMLElement;
}

describe("GameDetailModal instant block", () => {
  it("renders the block from the bundle with the network blocked", async () => {
    const fetchSpy = await renderOffline(preTip);

    // The tiles NBA already had, unchanged.
    expect(screen.getByTestId("tile-moneyline")).toBeInTheDocument();
    // The verdict, from the bundle's own pick.
    expect(screen.getByText("Celtics is the pick.")).toBeInTheDocument();
    // The record, from the winner-pick row.
    expect(screen.getByTestId("record-fill")).toBeInTheDocument();
    expect(screen.getByText("22/40")).toBeInTheDocument();

    // No request was made, and in particular not for the AI summary: the
    // button is still there to press.
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(api.explainGame).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /ai summary/i })).toBeInTheDocument();
    vi.unstubAllGlobals();
  });

  it("says what the AI button adds, under the button", async () => {
    await renderOffline(preTip);
    // The promise is what the button buys, and it has to be on screen before
    // the button is pressed or the reader has nothing to decide against.
    const promise = screen.getByTestId("ai-promise");
    expect(promise).toHaveTextContent(/model vs line/i);
    expect(promise).toHaveTextContent(/who's out/i);
    // ...and after the block, not inside it: the block is facts, the button is
    // the one thing it cannot state.
    expect(
      screen.getByTestId("instant-block").compareDocumentPosition(promise) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    vi.unstubAllGlobals();
  });

  it("keeps the markets table", async () => {
    // Decision 3: NBA's table is priced against the bookmaker, per market, and
    // is not a second copy of the block. Asserted so a later de-duplication
    // cannot quietly delete it.
    await renderOffline(preTip);

    const rows = screen.getAllByTestId("market-row");
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveTextContent("h2h");
    expect(rows[1]).toHaveTextContent("spread");
    expect(screen.getByRole("columnheader", { name: "Bookmaker" })).toBeInTheDocument();
    vi.unstubAllGlobals();
  });

  it("keeps the box score's team split out of the block", async () => {
    // The other half of "do not touch": the split belongs to the box score, and
    // the block must not grow a second copy of it.
    vi.mocked(api.getGamePlayers).mockResolvedValue([
      { player_id: "p1", player_name: "Jayson Tatum", stat: "points", predicted_value: 27.5, actual_value: null },
    ]);
    vi.mocked(api.getHubPlayers).mockResolvedValue([
      { player_id: "p1", player_name: "Jayson Tatum", team: "BOS", position: "G", rating: 0, live_form_rating: 0, points_per_game: 0, rebounds_per_game: 0, assists_per_game: 0, fg_pct: 0, three_pt_pct: 0, ft_pct: 0, usage_rate: 0, minutes_per_game: 0 },
    ]);
    await renderOffline(preTip);

    expect(screen.getByRole("rowheader", { name: "Jayson Tatum" })).toBeInTheDocument();
    expect(within(screen.getByTestId("instant-block")).queryByRole("rowheader")).toBeNull();
    vi.unstubAllGlobals();
  });

  it("replaces the PregamePick sentence with the timing badge, and never both", async () => {
    await renderOffline({ ...completed, rebuilt: true });

    // The badge says it once: rebuilt, and the sport's own moment.
    expect(screen.getByText("Rebuilt after tip-off")).toBeInTheDocument();
    expect(screen.getByText(/not counted/i)).toBeInTheDocument();
    // The old prose is gone. If it were still here a reader would meet the
    // same fact twice, in two different sentences.
    expect(screen.queryByText(/Rebuilt after tip-off:/)).toBeNull();
    expect(screen.queryByText(/Not counted in the record\./)).toBeNull();
    // ...and a rebuilt pick is still not judged.
    expect(screen.queryByTestId("post-match-verdict")).toBeNull();
    vi.unstubAllGlobals();
  });

  it("shows the quiet chip for a pick made before tip-off", async () => {
    await renderOffline(completed);

    expect(screen.getByText("Made before tip-off")).toBeInTheDocument();
    // A pick made in time is not a badge case, and must not be dressed as one.
    expect(screen.queryByText("Rebuilt after tip-off")).toBeNull();
    expect(screen.queryByText(/not counted/i)).toBeNull();
    vi.unstubAllGlobals();
  });

  it("carries the winner-pick record", async () => {
    await renderOffline(preTip);

    const strip = recordStrip();
    expect(strip).toHaveTextContent("Winner pick made before tip-off");
    expect(strip).toHaveTextContent("22/40");
    // The record is about picks, so it is not a percentage of anything else.
    expect(strip).not.toHaveTextContent("55%");
    vi.unstubAllGlobals();
  });

  it("never quotes the bookmaker-price row as the pick's record", async () => {
    // `h2h` (38 graded, 21 right) is a real row in the same response. It
    // measures the same probability against a price, so putting it under the
    // verdict would state one pick two ways.
    vi.mocked(api.getTrackRecord).mockResolvedValue([vsMarket]);
    await renderOffline(preTip);

    expect(screen.queryByTestId("record-fill")).toBeNull();
    expect(screen.queryByText("21/38")).toBeNull();
    vi.unstubAllGlobals();
  });

  it("shows no record strip when the track record has no winner-pick row", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([]);
    await renderOffline(preTip);

    // No row, no strip. What must never appear is a fabricated 0/0.
    expect(screen.queryByTestId("record-fill")).toBeNull();
    expect(screen.queryByText("0/0")).toBeNull();
    vi.unstubAllGlobals();
  });

  it("shows a dash, not 0/0, when the winner-pick row has settled nothing", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([winnerPick(0, 0)]);
    await renderOffline(preTip);

    // A row that exists and has graded nothing is a real row with no numbers
    // in it: the dash. `0/0` would be a claim about a record that does not
    // exist, and the whole reason this rule is written down.
    expect(screen.queryByTestId("record-fill")).toBeNull();
    expect(screen.queryByText("0/0")).toBeNull();
    const strip = recordStrip();
    expect(strip).toHaveTextContent("Winner pick made before tip-off");
    expect(strip).toHaveTextContent("—");
    vi.unstubAllGlobals();
  });

  it("shows no record strip while the track record is still loading", async () => {
    // A strip that renders 0/0 for a moment and then corrects itself is a lie
    // a reader can catch. Nothing arrives until the numbers do.
    vi.mocked(api.getTrackRecord).mockReturnValue(new Promise(() => {}));
    await renderOffline(preTip);

    expect(screen.queryByTestId("record-fill")).toBeNull();
    expect(screen.queryByText("0/0")).toBeNull();
    expect(screen.getByTestId("instant-block")).toBeInTheDocument();
    vi.unstubAllGlobals();
  });

  it("loses the record without losing the game", async () => {
    vi.mocked(api.getTrackRecord).mockRejectedValue(new Error("track record down"));
    await renderOffline(preTip);

    expect(screen.getByTestId("instant-block")).toBeInTheDocument();
    expect(screen.getByTestId("tile-moneyline")).toBeInTheDocument();
    expect(screen.queryByText(/couldn.t load this game/i)).toBeNull();
    expect(screen.queryByText("0/0")).toBeNull();
    vi.unstubAllGlobals();
  });

  it("still says plainly that no pick was made", async () => {
    await renderOffline({ ...preTip, prediction: null });

    expect(screen.getByText("Celtics is the pick.")).toBeNull();
    expect(screen.getByText(/No pick was made for this fixture\./)).toBeInTheDocument();
    // No pick is not no facts: the block still carries what the site has.
    expect(screen.getByTestId("instant-block")).toBeInTheDocument();
    vi.unstubAllGlobals();
  });
});
