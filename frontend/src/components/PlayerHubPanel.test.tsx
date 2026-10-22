import { describe, expect, it, vi, afterEach, beforeEach } from "vitest";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, useLocation } from "react-router-dom";
import PlayerHubPanel from "./PlayerHubPanel";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    getHubPlayers: vi.fn(),
    // The page's extras: the next games and the model's error bands. Neither
    // may stop the table from drawing.
    getHubPlayersNextGames: vi.fn().mockResolvedValue({}),
    getPlayerPropsManifest: vi.fn().mockResolvedValue({ model_version: "v1", trained_at: "2026-10-01", models: [], metrics: {} }),
    getHubPlayer: vi.fn().mockResolvedValue({ player_id: "1", team: "BOS", games: [], status: null }),
  },
}));
afterEach(() => {
  // Unmount first: restoring wipes the mocks' implementations, and a panel still mounted at that moment can run an
  // effect against a bare mock (`undefined.then`), which surfaced as a failure in the NEXT test, intermittently in CI.
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

// The page's extras: the next games and the model's error bands. `restoreAllMocks`
// wipes a mock's implementation, so they are re-armed per test.
beforeEach(() => {
  vi.mocked(api.getHubPlayersNextGames).mockResolvedValue({});
  vi.mocked(api.getPlayerPropsManifest).mockResolvedValue({
    model_version: "v1",
    trained_at: "2026-10-01",
    models: [],
    metrics: {},
  });
  vi.mocked(api.getHubPlayer).mockResolvedValue({
    player_id: "1",
    team: "BOS",
    games: [],
    status: null,
  });
});

function makePlayer(i: number, over: Record<string, unknown> = {}) {
  return {
    player_id: `${i}`, player_name: `Player ${i}`, team: "BOS", position: "G",
    rating: 100 - i, live_form_rating: 90, points_per_game: 20, rebounds_per_game: 5,
    assists_per_game: 4, fg_pct: 0.47, three_pt_pct: 0.37, ft_pct: 0.85, usage_rate: 0.24,
    minutes_per_game: 32,
    ...over,
  };
}

/** jsdom has no matchMedia; this is the narrow-screen query the panel reads. */
function setNarrow(narrow: boolean) {
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches: narrow,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
}

function Where() {
  return <span data-testid="where">{useLocation().search}</span>;
}

function renderPanel(initialEntry = "/") {
  return render(<MemoryRouter initialEntries={[initialEntry]}><PlayerHubPanel /><Where /></MemoryRouter>);
}

const playerNames = () =>
  screen
    .getAllByRole("row")
    .slice(1)
    // The first cell carries the row's expand toggle, so it starts with its
    // marker; the name is what is left.
    .map((row) => within(row).getAllByRole("cell")[0].textContent?.replace(/^▸/, "") ?? "");

/** A player's name, in the table. The leaders strip above names them too. */
const inTable = (name: string) => within(screen.getByRole("table")).getByText(name);

describe("PlayerHubPanel", () => {
  it("shows the first 20 players and paginates to the next page", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue(Array.from({ length: 25 }, (_, i) => makePlayer(i)));

    renderPanel();

    await waitFor(() => expect(inTable("Player 0")).toBeInTheDocument());
    expect(within(screen.getByRole("table")).queryByText("Player 20")).not.toBeInTheDocument();

    await userEvent.click(screen.getByTestId("player-hub-next"));
    expect(await screen.findByText("Player 20")).toBeInTheDocument();
  });

  it("says which page it is on, and disables the pager at the ends", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue(Array.from({ length: 25 }, (_, i) => makePlayer(i)));

    renderPanel();
    await waitFor(() => expect(inTable("Player 0")).toBeInTheDocument());

    // The first page: there is no page before it.
    expect(screen.getByTestId("player-hub-prev")).toBeDisabled();
    expect(screen.getByTestId("player-hub-next")).not.toBeDisabled();
    expect(screen.getByTestId("player-hub-page")).toHaveTextContent("Page 1 of 2");

    await userEvent.click(screen.getByTestId("player-hub-next"));

    expect(await screen.findByText("Player 20")).toBeInTheDocument();
    expect(screen.getByTestId("player-hub-prev")).not.toBeDisabled();
    // The last page: there is no page after it.
    expect(screen.getByTestId("player-hub-next")).toBeDisabled();
    expect(screen.getByTestId("player-hub-page")).toHaveTextContent("Page 2 of 2");
  });

  it("does not cut a player's name off with an ellipsis", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue([
      makePlayer(1, { player_name: "Shai Gilgeous-Alexander" }),
    ]);

    renderPanel();
    await waitFor(() => expect(inTable("Shai Gilgeous-Alexander")).toBeInTheDocument());

    // The leader board's name: it wraps, it is not cut short.
    const name = within(screen.getByTestId("player-leaders")).getAllByText("Shai Gilgeous-Alexander")[0];
    expect(name.className).not.toMatch(/truncate/);
  });

  it("gives the player's record a Close button, and puts the focus in it", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue([makePlayer(1)]);
    vi.mocked(api.getHubPlayer).mockResolvedValue({
      player_id: "1",
      team: "BOS",
      status: null,
      games: [],
    });

    renderPanel();
    await waitFor(() => expect(inTable("Player 1")).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: /Player 1/i }));

    const record = await screen.findByTestId("player-detail-1");
    // The focus is in the record, so a reader is reading the record.
    await waitFor(() => expect(record).toHaveFocus());
    expect(screen.getByTestId("player-record-close")).toBeInTheDocument();

    // And the Close button puts it away.
    await userEvent.click(screen.getByTestId("player-record-close"));
    await waitFor(() => expect(screen.queryByTestId("player-detail-1")).toBeNull());
  });

  it("shows an empty state when no player data is cached yet", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue([]);
    renderPanel();
    await waitFor(() => expect(screen.getByText(/no player data/i)).toBeInTheDocument());
  });

  it("starts sorted by rating and re-sorts when a header is clicked", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue([
      makePlayer(1, { rating: 70, points_per_game: 26 }),
      makePlayer(2, { rating: 90, points_per_game: 12 }),
      makePlayer(3, { rating: 80, points_per_game: 19 }),
    ]);

    renderPanel();

    await waitFor(() => expect(inTable("Player 2")).toBeInTheDocument());
    expect(playerNames()).toEqual(["Player 2", "Player 3", "Player 1"]);

    await userEvent.click(screen.getByRole("button", { name: /PPG/ }));
    expect(playerNames()).toEqual(["Player 1", "Player 3", "Player 2"]);
    const ppgHeader = screen.getByRole("columnheader", { name: /PPG/ });
    expect(ppgHeader).toHaveAttribute("aria-sort", "descending");
  });

  it("filters to one team and resets to the first page", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue(
      Array.from({ length: 22 }, (_, i) => makePlayer(i, { team: i < 20 ? "BOS" : "LAL" })),
    );

    renderPanel();
    await waitFor(() => expect(inTable("Player 0")).toBeInTheDocument());
    await userEvent.click(screen.getByTestId("player-hub-next"));
    expect(await screen.findByText("Player 20")).toBeInTheDocument();

    await userEvent.selectOptions(screen.getByLabelText("Team"), "LAL");

    expect(await waitFor(() => expect(inTable("Player 20")).toBeInTheDocument())).toBeTruthy();
    expect(within(screen.getByRole("table")).queryByText("Player 0")).not.toBeInTheDocument();
    expect(screen.getByText(/Players 1–2 of 2/)).toBeInTheDocument();
  });

  it("keeps six columns on a narrow screen and puts the rest behind More stats", async () => {
    setNarrow(true);
    vi.mocked(api.getHubPlayers).mockResolvedValue([makePlayer(1)]);

    renderPanel();
    await waitFor(() => expect(inTable("Player 1")).toBeInTheDocument());

    // The six the spec names: Player, Team, PPG, RPG, APG and Next.
    for (const label of ["Player", "Team", "PPG", "RPG", "APG", "Next"]) {
      expect(screen.getByRole("columnheader", { name: new RegExp(label) })).toBeInTheDocument();
    }
    for (const label of ["Pos", "Rating", "FG%", "3P%", "FT%"]) {
      expect(screen.queryByRole("columnheader", { name: new RegExp(label.replace("%", "\\%")) })).toBeNull();
    }

    const more = screen.getByRole("button", { name: "More stats" });
    expect(more).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(more);

    expect(screen.getByRole("columnheader", { name: /FG%/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Fewer stats" })).toHaveAttribute("aria-expanded", "true");
  });


  it("searches by name", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue([
      makePlayer(1, { player_name: "Jayson Tatum" }),
      makePlayer(2, { player_name: "Jaylen Brown" }),
    ]);

    renderPanel();
    await waitFor(() => expect(inTable("Jayson Tatum")).toBeInTheDocument());

    await userEvent.type(screen.getByLabelText("Search"), "tatum");

    await waitFor(() => expect(screen.queryByText("Jaylen Brown")).not.toBeInTheDocument());
    expect(inTable("Jayson Tatum")).toBeInTheDocument();
  });

  it("filters by position chip", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue([
      makePlayer(1, { position: "PG" }),
      makePlayer(2, { position: "C" }),
    ]);

    renderPanel();
    await waitFor(() => expect(inTable("Player 1")).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: "C" }));

    await waitFor(() => expect(screen.queryByText("Player 1")).not.toBeInTheDocument());
    expect(inTable("Player 2")).toBeInTheDocument();
  });

  it("hides the players who do not play, until the switch is turned off", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue([
      makePlayer(1, { minutes_per_game: 32 }),
      makePlayer(2, { minutes_per_game: 4 }),
    ]);

    renderPanel();
    await waitFor(() => expect(inTable("Player 1")).toBeInTheDocument());
    expect(screen.queryByText("Player 2")).not.toBeInTheDocument();

    await userEvent.click(screen.getByLabelText(/min 15 minutes/i));

    await waitFor(() => expect(inTable("Player 2")).toBeInTheDocument());
  });

  it("shows the next game with the model's projection, and a dash when there is none", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue([makePlayer(1), makePlayer(2)]);
    vi.mocked(api.getHubPlayersNextGames).mockResolvedValue({
      "1": {
        game_id: "g2",
        game_date: "2026-11-02",
        opponent: "MIA",
        home: true,
        projections: { points: 27.5, rebounds: 6.1, assists: 4.2 },
      },
    });

    renderPanel();
    await waitFor(() => expect(inTable("Player 1")).toBeInTheDocument());

    const next = await screen.findByTestId("player-next");
    expect(next.textContent).toContain("27.5");
    expect(next.textContent).toContain("6.1");
    expect(next.textContent).toContain("4.2");
    // The player with no stored projection has no projection: a dash, never 0.
    expect(screen.getAllByTestId("player-next")).toHaveLength(1);
  });

  it("opens a player's detail with their last ten games", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue([makePlayer(1)]);
    vi.mocked(api.getHubPlayer).mockResolvedValue({
      player_id: "1",
      team: "BOS",
      status: { status: "Out", source: "ESPN injury report", dated: "2026-10-19" },
      games: [
        {
          game_id: "g1",
          game_date: "2026-10-30",
          opponent: "MIA",
          home: true,
          score: "110–100",
          actual: { points: 30, rebounds: 8, assists: 5, threes: 2 },
          projection: { points: 22, rebounds: 7, assists: 4, threes: 1 },
        },
      ],
      averages: {},
    });

    renderPanel();
    await waitFor(() => expect(inTable("Player 1")).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: /Player 1/i }));

    const detail = await screen.findByTestId("player-detail-1");
    expect(within(detail).getByTestId("player-chart-points")).toBeInTheDocument();
    expect(within(detail).getByTestId("player-status")).toHaveTextContent("Out");
    expect(within(detail).getByTestId("player-averages")).toHaveTextContent("20.0 ppg");
  });

  it("shows every column on a wide screen without a toggle", async () => {
    setNarrow(false);
    vi.mocked(api.getHubPlayers).mockResolvedValue([makePlayer(1)]);

    renderPanel();
    await waitFor(() => expect(inTable("Player 1")).toBeInTheDocument());

    expect(screen.getByRole("columnheader", { name: /FG%/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "More stats" })).toBeNull();
  });

  it("reads the team filter from the URL query, so a link keeps its filters", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue([
      makePlayer(1, { team: "BOS" }),
      makePlayer(2, { team: "LAL" }),
    ]);
    renderPanel("/hub?team=LAL");
    await waitFor(() => expect(inTable("Player 2")).toBeInTheDocument());
    expect(screen.queryByText("Player 1")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Team")).toHaveValue("LAL");
  });

  it("writes the filters to the URL as they change, without a render-time update", async () => {
    const errors = vi.spyOn(console, "error").mockImplementation(() => {});
    vi.mocked(api.getHubPlayers).mockResolvedValue([
      makePlayer(1, { team: "BOS" }),
      makePlayer(2, { team: "LAL" }),
    ]);
    renderPanel();
    await waitFor(() => expect(inTable("Player 1")).toBeInTheDocument());

    await userEvent.selectOptions(screen.getByLabelText("Team"), "LAL");
    await waitFor(() => expect(screen.getByTestId("where")).toHaveTextContent("team=LAL"));
    expect(screen.queryByText("Player 1")).not.toBeInTheDocument();
    expect(errors.mock.calls.filter((c) => String(c[0]).includes("while rendering"))).toHaveLength(0);
  });
});
