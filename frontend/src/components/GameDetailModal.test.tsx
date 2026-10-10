import { describe, expect, it, vi, afterEach, beforeEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import GameDetailModal from "./GameDetailModal";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    // `GameDetailModal` fetches its signal rows separately, so a mock that omits
    // it leaves it `undefined` and the modal's effect throws on open -- 41 failures
    // across four files when this row landed.
    //
    // Default: a valid EMPTY list -- spec §2's "no data, no row". A bare `vi.fn()`
    // returns `undefined` and the modal calls `.then` on it.
    getGameSignals: vi.fn(async () => ({ sport: "nba", id: "", signals: [] })),
    getGameDetail: vi.fn(),
    getGamePlayers: vi.fn(),
    // The gate's own sibling route, fetched separately. These tests are about
    // the game's figures, so it resolves empty: an unreadable feed would (rightly)
    // withhold the ranking block, and that is TopCalls' own test to cover.
    getGameOutPlayers: vi.fn().mockResolvedValue([]),
    getHubPlayers: vi.fn().mockResolvedValue([]),
    // The block's record strip reads this. These tests are about the game's own
    // figures, so it stays empty here — the strip is covered in
    // GameDetailModal.instant.test.tsx.
    getTrackRecord: vi.fn().mockResolvedValue([]),
    explainGame: vi.fn(),
    loadContext: vi.fn(async () => ({ matchups: [] })),
  },
}));

afterEach(() => {
  vi.restoreAllMocks();
});

// The box score's team split is a join against the season hub feed, so the
// modal fetches it too. Every test here restores mocks afterwards, which wipes
// a mockResolvedValue set at module scope -- so it has to be re-armed per test
// or the second test onwards calls undefined.then() and every render throws.
beforeEach(() => {
  vi.mocked(api.getHubPlayers).mockResolvedValue([]);
  vi.mocked(api.getTrackRecord).mockResolvedValue([]);
  // Same re-arm for the gate's sibling route: a wiped mock would make every
  // render call undefined.then() and take the whole modal down.
  vi.mocked(api.getGameOutPlayers).mockResolvedValue([]);
});

const completedDetail = {
  game_id: "g2", game_date: "2026-01-05", home_team: "BOS", away_team: "MIA",
  completed: true, home_pts: 113, away_pts: 105,
  prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
  markets: [], head_to_head: [], home_recent_form: [], away_recent_form: [],
};

const detail = {
  game_id: "g1", game_date: "2026-11-01", home_team: "BOS", away_team: "MIA",
  prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
  completed: false, home_pts: null, away_pts: null,
  markets: [
    { market: "h2h", selection: "BOS", model_probability: 0.62, market_probability: 0.55, edge: 0.07, bookmaker: "DraftKings", american_odds: -130, point: null },
    { market: "spread", selection: "BOS", model_probability: 0.52, market_probability: 0.5, edge: 0.02, bookmaker: "DraftKings", american_odds: -110, point: -4.5 },
    { market: "total", selection: "over", model_probability: 0.48, market_probability: 0.5, edge: -0.02, bookmaker: "FanDuel", american_odds: -105, point: 224.5 },
  ],
  head_to_head: [
    { game_id: "g0", game_date: "2026-01-01", home_team: "BOS", away_team: "MIA", home_pts: 110, away_pts: 100 },
  ],
  home_recent_form: ["W", "L", "W"],
  away_recent_form: ["L", "L", "W"],
};

const players = [{ player_id: "203999", player_name: "Nikola Jokic", stat: "points", predicted_value: 27.5, actual_value: null }];

/** The box score splits by team, so its fixtures need a player on each side. */
const hubPlayer = (player_id: string, player_name: string, team: string) => ({
  player_id, player_name, team, position: "C",
  rating: 0, live_form_rating: 0, points_per_game: 0, rebounds_per_game: 0,
  assists_per_game: 0, fg_pct: 0, three_pt_pct: 0, ft_pct: 0, usage_rate: 0, minutes_per_game: 0,
});
const hub = [hubPlayer("203999", "Nikola Jokic", "MIA"), hubPlayer("p1", "Jayson Tatum", "BOS")];

describe("GameDetailModal", () => {
  it("shows a loading state before data arrives", () => {
    vi.mocked(api.getGameDetail).mockReturnValue(new Promise(() => {}));
    vi.mocked(api.getGamePlayers).mockReturnValue(new Promise(() => {}));

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it("renders player prop predictions as a box score split by team", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);
    vi.mocked(api.getHubPlayers).mockResolvedValue(hub);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    // One row per player, not one row per stat: the player is a row header and
    // the prediction is a cell in it. Scoped to the row because a one-player
    // team totals the same number, which is correct and would otherwise be an
    // ambiguous query.
    const head = await screen.findByRole("rowheader", { name: "Nikola Jokic" });
    expect(within(head.closest("tr")!).getByText("27.5")).toBeInTheDocument();
    // The stat names are column headers now, not per-row text.
    expect(screen.getAllByRole("columnheader", { name: "Pts" })).toHaveLength(2);
  });

  it("renders the model's projected margin and total, and no bookmaker table", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    await screen.findByRole("heading", { name: "Heat at Celtics" });
    expect(screen.getAllByText(/224\.5/).length).toBeGreaterThan(0);
    expect(screen.getByText("Projected margin")).toBeInTheDocument();
    // The odds-by-bookmaker table is gone: NBA stays independent of betting lines.
    expect(screen.queryByTestId("market-row")).not.toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "Bookmaker" })).not.toBeInTheDocument();
  });

  it("renders head-to-head history", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    expect(await screen.findByText("Thu 1 Jan")).toBeInTheDocument();
    expect(screen.getByText(/MIA 100 – 110 BOS/)).toBeInTheDocument();
  });

  it("renders recent form for both teams", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    expect(await screen.findAllByTestId("form-badge")).toHaveLength(6);
  });

  it("shows the final score instead of a live prediction for a completed game", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue({
      ...detail, completed: true, home_pts: 108, away_pts: 101, markets: [], head_to_head: [],
    });
    vi.mocked(api.getGamePlayers).mockResolvedValue([]);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    expect(await screen.findByText(/final/i)).toBeInTheDocument();
    expect(screen.getByText("108")).toBeInTheDocument();
    expect(screen.getByText("101")).toBeInTheDocument();
  });

  it("calls onClose when the close button is clicked", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);
    const onClose = vi.fn();

    render(<GameDetailModal gameId="g1" onClose={onClose} />);
    await screen.findByRole("heading", { name: /Celtics/ });

    await userEvent.click(screen.getByRole("button", { name: /close/i }));
    expect(onClose).toHaveBeenCalled();
  });

  it("shows an error state when either fetch fails", async () => {
    vi.mocked(api.getGameDetail).mockRejectedValue(new Error("boom"));
    vi.mocked(api.getGamePlayers).mockResolvedValue([]);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);
    expect(await screen.findByText(/couldn.t load this game/i)).toBeInTheDocument();
  });

  it("closes when Escape is pressed", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);
    const onClose = vi.fn();

    render(<GameDetailModal gameId="g1" onClose={onClose} />);
    await screen.findByRole("heading", { name: /Celtics/ });

    await userEvent.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();
  });
});

it("shows a correct winner-call verdict when the favorite actually won", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(completedDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  expect(await screen.findByText("Called it ✓")).toBeInTheDocument();
  // The pick is stated by the block, as a verdict. The old one-line
  // "Pick before tip-off: BOS · 62%" is gone: the block's quiet chip already
  // says the timing, and a second copy of the same fact in a second sentence is
  // how two figures end up disagreeing.
  expect(screen.getByText("Celtics is the pick.")).toBeInTheDocument();
});

it("shows an incorrect winner-call verdict when the underdog actually won", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue({ ...completedDetail, home_pts: 90, away_pts: 100 });
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  expect(await screen.findByText("Missed ✗")).toBeInTheDocument();
});

it("shows predicted vs actual margin with the absolute difference", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(completedDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  expect(await screen.findByText(/predicted margin/i)).toBeInTheDocument();
  expect(screen.getByText(/off by 4.5/i)).toBeInTheDocument();
});

it("shows predicted vs actual total with the absolute difference", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(completedDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  expect(await screen.findByText(/predicted total/i)).toBeInTheDocument();
  expect(screen.getByText(/off by 6.5/i)).toBeInTheDocument();
});

it("does not show a post-match verdict for an upcoming game", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(detail);
  vi.mocked(api.getGamePlayers).mockResolvedValue(players);

  render(<GameDetailModal gameId="g1" onClose={() => {}} />);

  await screen.findByRole("heading", { name: /Celtics/ });
  expect(screen.queryByTestId("post-match-verdict")).not.toBeInTheDocument();
});

const settledDetail = {
  ...completedDetail,
  markets: [
    { market: "h2h", selection: "BOS", model_probability: 0.62, market_probability: 0.55, edge: 0.07, bookmaker: "DraftKings", american_odds: -130, point: null },
    { market: "spread", selection: "MIA", model_probability: 0.4, market_probability: 0.45, edge: -0.05, bookmaker: "DraftKings", american_odds: 110, point: 4.5 },
    { market: "total", selection: "over", model_probability: 0.5, market_probability: 0.5, edge: 0.0, bookmaker: "FanDuel", american_odds: -105, point: 224.5 },
  ],
};

it("shows predicted vs actual and the error for a settled player prop", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(completedDetail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([
    { player_id: "203999", player_name: "Nikola Jokic", stat: "points", predicted_value: 27.5, actual_value: 24.0 },
  ]);
  vi.mocked(api.getHubPlayers).mockResolvedValue(hub);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  // Predicted on one line, actual and the miss beneath it. The old list said
  // "off by 3.5"; the number is the same, the delta is what carries it now.
  const head = await screen.findByRole("rowheader", { name: "Nikola Jokic" });
  const playerRow = within(head.closest("tr")!);
  expect(playerRow.getByText("27.5")).toBeInTheDocument();
  expect(playerRow.getByText(/24\.0 \(-3\.5\)|24 \(-3\.5\)/)).toBeInTheDocument();
  // And a real team total appears, because a per-player list makes the reader
  // add it up by hand.
  expect(screen.getByTestId("box-score-totals")).toBeInTheDocument();
});

it("shows only the predicted value for an unsettled player prop", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(detail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([
    { player_id: "203999", player_name: "Nikola Jokic", stat: "points", predicted_value: 27.5, actual_value: null },
  ]);
  vi.mocked(api.getHubPlayers).mockResolvedValue(hub);

  render(<GameDetailModal gameId="g1" onClose={() => {}} />);

  const head = await screen.findByRole("rowheader", { name: "Nikola Jokic" });
  const playerRow = within(head.closest("tr")!);
  // Predicted only, and no delta in parentheses -- the cell must not imply a
  // result that does not exist yet.
  expect(playerRow.getByText("27.5")).toBeInTheDocument();
  expect(playerRow.queryByText(/\(/)).toBeNull();
  expect(playerRow.queryByText(/actual/i)).toBeNull();
});

it("names the favoured side for the margin when both models agree", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue({ ...detail, prediction: { home_win_probability: 0.47, predicted_margin: -4.8, predicted_total: 221.3 } });
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);
  render(<GameDetailModal gameId="g1" onClose={() => {}} />);
  // The margin strip names the side it measures. The "TOR to win" line that
  // used to sit beside it is gone: the pick is the block's, stated once by its
  // tile, and the strip's own job is the margin.
  expect(await screen.findByText("MIA by 4.8")).toBeInTheDocument();
  // ...and the block still states the pick, so removing the header's copy cost
  // the page nothing: it is here, exactly once. The tile carries the LEADING
  // side, which at home_win_probability 0.47 is the away side at 53%.
  expect(screen.getByTestId("tile-moneyline")).toHaveTextContent("53%");
  expect(screen.getByTestId("tile-moneyline")).toHaveTextContent("MIA");
});

it("calls the margin a toss-up when it disagrees with the win pick or is under half a point", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue({ ...detail, prediction: { home_win_probability: 0.52, predicted_margin: -0.6, predicted_total: 221.3 } });
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);
  const { unmount } = render(<GameDetailModal gameId="g1" onClose={() => {}} />);
  // The block names the pick; the strip refuses to contradict it.
  expect(await screen.findByTestId("tile-moneyline")).toHaveTextContent("52%");
  expect(screen.getByText("Toss-up")).toBeInTheDocument();
  expect(screen.queryByText(/MIA by/)).not.toBeInTheDocument();
  unmount();

  vi.mocked(api.getGameDetail).mockResolvedValue({ ...detail, prediction: { home_win_probability: 0.6, predicted_margin: 0.03, predicted_total: 221.3 } });
  render(<GameDetailModal gameId="g1" onClose={() => {}} />);
  expect(await screen.findByText("Toss-up")).toBeInTheDocument();
  expect(screen.queryByText(/by 0\.0/)).not.toBeInTheDocument();
});

it("labels a pick rebuilt after tip-off and never judges it", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue({ ...completedDetail, rebuilt: true });
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  // The badge says it once. The old prose sentences are gone, and this is the
  // test that holds them gone.
  expect(await screen.findByText("Made after tip-off")).toBeInTheDocument();
  expect(screen.queryByText(/Rebuilt after tip-off:/)).toBeNull();
  expect(screen.queryByText(/not counted/i)).toBeNull();
  expect(screen.queryByText("Called it ✓")).not.toBeInTheDocument();
  expect(screen.queryByTestId("post-match-verdict")).not.toBeInTheDocument();
});

it("says so when no pick was made before a final", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue({ ...completedDetail, prediction: null });
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  expect(await screen.findByText("No pick was made for this fixture.")).toBeInTheDocument();
});

it("names the teams and writes dates in words", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue(detail);
  vi.mocked(api.getGamePlayers).mockResolvedValue(players);

  render(<GameDetailModal gameId="g1" onClose={() => {}} />);

  expect(await screen.findByRole("heading", { name: "Heat at Celtics" })).toBeInTheDocument();
  expect(screen.getByText("Thu 1 Jan")).toBeInTheDocument();
  expect(screen.queryByText("2026-01-01")).not.toBeInTheDocument();
});

it("is a labelled dialog, and a failed load offers Try again", async () => {
  vi.mocked(api.getGameDetail).mockRejectedValueOnce(new Error("x")).mockResolvedValue(detail);
  vi.mocked(api.getGamePlayers).mockResolvedValue(players);

  render(<GameDetailModal gameId="g1" onClose={() => {}} />);

  expect(await screen.findByRole("alert")).toHaveTextContent("We couldn't load this game.");
  await userEvent.click(screen.getByRole("button", { name: "Try again" }));
  // The dialog is already in the DOM; after retry the heading updates.
  // Find the heading (which gives the dialog its accessible name) and verify the dialog.
  const heading = await screen.findByRole("heading", { name: "Heat at Celtics" });
  const dialog = heading.closest('[role="dialog"]');
  expect(dialog).toHaveAttribute("aria-modal", "true");
});

it("labels rebuilt player props and never judges them", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue({
    ...completedDetail,
    markets: [{ market: "h2h", selection: "BOS", model_probability: 0.6, market_probability: 0.5, edge: 0.1, bookmaker: "DraftKings", american_odds: -120, point: null, rebuilt: true }],
  });
  vi.mocked(api.getGamePlayers).mockResolvedValue([
    { player_id: "p1", player_name: "Jayson Tatum", stat: "points", predicted_value: 27.456, actual_value: 31, rebuilt: true },
  ]);
  vi.mocked(api.getHubPlayers).mockResolvedValue(hub);

  render(<GameDetailModal gameId="g2" onClose={() => {}} />);

  // A rebuilt prop is shown with its actual, but never judged -- so no error delta.
  const prop = (await screen.findByRole("rowheader", { name: /Jayson Tatum/ })).closest("tr")!;
  expect(prop).toHaveTextContent("27.5");
  expect(prop).toHaveTextContent("31");
  // The label travels with the name, so a screen reader hears it too.
  expect(prop).toHaveTextContent("Rebuilt");
  // No delta: scoring a rebuilt projection would judge a forecast made with
  // the result already known.
  expect(prop).not.toHaveTextContent(/\(\s*[+-]/);
  expect(screen.getByText(/built after tip-off/i)).toBeInTheDocument();
});

it("labels a rebuilt pick on a game that has not finished", async () => {
  vi.mocked(api.getGameDetail).mockResolvedValue({ ...detail, rebuilt: true });
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);

  render(<GameDetailModal gameId="g1" onClose={() => {}} />);

  expect(await screen.findByText("Made after tip-off")).toBeInTheDocument();
  expect(screen.queryByText(/not counted/i)).toBeNull();
});

/** The plain-English panel: flow-first, thin by data. The game carries no
 *  market line (142 of 142 bundles), so the flow says the pick and stops and
 *  the summary state draws the model's own moneyline split and nothing else.
 *  That thinness is data, not a defect: do not file it, do not work around it. */
const nbaSummary = {
  verdict: "Boston is the pick at home.",
  band: "moderate",
  factors: [{ key: "moneyline", direction: "up", headline: "Home edge", text: "Boston at home." }],
  source: "template" as const,
  model: "",
  generated_at: new Date().toISOString(),
  sport: "nba",
  pick_timing: "pre_kickoff" as const,
};

function openPregame() {
  vi.mocked(api.getGameDetail).mockResolvedValue(detail);
  vi.mocked(api.getGamePlayers).mockResolvedValue([]);
}

describe("GameDetailModal and the plain-English panel", () => {
  it("says the pick with no line named, and words the moment for tip-off", async () => {
    openPregame();
    render(<GameDetailModal gameId="g1" onClose={() => {}} />);
    const flow = await screen.findByTestId("fixture-flow");
    // Before tip-off the flow has nothing to say that the block does not say
    // better, so it says nothing at all. It used to render the fixture's own
    // name as a heading here — a bare `BOS vs MIA` with no sentence under it,
    // directly above the AI button. Asserted as the absence of a heading, so a
    // rename of the teams cannot let it back through.
    expect(flow.querySelector("h1, h2, h3, h4, h5, h6")).toBeNull();
    expect(flow.textContent?.trim()).toBe("");
    expect(flow.innerHTML).not.toContain("Win probabilities");
    expect(flow.innerHTML).not.toContain("The model picks");
    // The timing is stated once, by the block, in basketball's words.
    expect(screen.getByText("Made before tip-off")).toBeInTheDocument();
    // No line is named, because the game carries none.
    expect(flow.innerHTML).not.toContain("market line");
    expect(flow.innerHTML).not.toContain("market-line");
  });

  it("makes no summary request until asked, then asks for this game", async () => {
    openPregame();
    const explain = vi.mocked(api.explainGame).mockResolvedValue(nbaSummary as never);
    render(<GameDetailModal gameId="g1" onClose={() => {}} />);
    await screen.findByTestId("fixture-flow");
    expect(explain).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: /ai summary/i }));
    expect(await screen.findByText("Boston is the pick at home.")).toBeInTheDocument();
    expect(explain).toHaveBeenCalledTimes(1);
  });

  it("hands the context loader this game's id as the fixtureId", async () => {
    openPregame();
    render(<GameDetailModal gameId="g1" onClose={() => {}} />);
    await screen.findByTestId("fixture-flow");
    await waitFor(() => expect(api.loadContext).toHaveBeenCalledWith("g1"));
  });

  it("leaves every figure to the block once the summary is in", async () => {
    // The summary is prose and nothing else. It used to re-render the tiles, the
    // bar, the legend and the record beside the block that already drew them,
    // so one figure was on screen twice — and a de-duplication pass would have
    // had to guess which copy was the real one.
    openPregame();
    vi.mocked(api.explainGame).mockResolvedValue(nbaSummary as never);
    render(<GameDetailModal gameId="g1" onClose={() => {}} />);
    await userEvent.click(await screen.findByRole("button", { name: /ai summary/i }));
    const summary = await screen.findByTestId("fixture-summary");

    // No figure inside the summary...
    expect(within(summary).queryByTestId("tile-moneyline")).toBeNull();
    expect(within(summary).queryAllByTestId("pbar-fill")).toHaveLength(0);
    expect(within(summary).queryByTestId("pbar-legend")).toBeNull();
    expect(within(summary).queryByTestId("record-fill")).toBeNull();
    // ...and exactly one of each in the panel, which the block owns. The game
    // carries no spread or total line, so those tiles are never drawn at all.
    const block = screen.getByTestId("instant-block");
    expect(within(block).getAllByTestId("tile-moneyline")).toHaveLength(1);
    expect(screen.getAllByTestId("tile-moneyline")).toHaveLength(1);
    expect(within(block).queryByTestId("tile-spread")).toBeNull();
    expect(within(block).queryByTestId("tile-total")).toBeNull();
  });

  it("shows the flow with no request made when the explainer is unreachable", async () => {
    openPregame();
    const explain = vi.mocked(api.explainGame).mockRejectedValue(new Error("unreachable"));
    render(<GameDetailModal gameId="g1" onClose={() => {}} />);
    const flow = await screen.findByTestId("fixture-flow");
    // The panel survives a failed request: the facts are still on screen and the
    // button offers a retry. The flow is mounted and empty here, which is the
    // same shape it has pre-game whether or not the summary can be fetched.
    expect(flow).toBeInTheDocument();
    expect(flow.querySelector("h1, h2, h3, h4, h5, h6")).toBeNull();
    expect(explain).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /ai summary/i })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByTestId("fixture-summary")).toBeNull();
  });
});
