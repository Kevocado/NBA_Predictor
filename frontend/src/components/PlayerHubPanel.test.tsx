import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import PlayerHubPanel from "./PlayerHubPanel";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { getHubPlayers: vi.fn() } }));
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
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

const playerNames = () =>
  screen.getAllByRole("row").slice(1).map((row) => within(row).getAllByRole("cell")[0].textContent);

describe("PlayerHubPanel", () => {
  it("shows the first 20 players and paginates to the next page", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue(Array.from({ length: 25 }, (_, i) => makePlayer(i)));

    render(<PlayerHubPanel />);

    await waitFor(() => expect(screen.getByText("Player 0")).toBeInTheDocument());
    expect(screen.queryByText("Player 20")).not.toBeInTheDocument();

    await userEvent.click(screen.getByTestId("player-hub-next"));
    expect(await screen.findByText("Player 20")).toBeInTheDocument();
  });

  it("shows an empty state when no player data is cached yet", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue([]);
    render(<PlayerHubPanel />);
    await waitFor(() => expect(screen.getByText(/no player data/i)).toBeInTheDocument());
  });

  it("starts sorted by rating and re-sorts when a header is clicked", async () => {
    vi.mocked(api.getHubPlayers).mockResolvedValue([
      makePlayer(1, { rating: 70, points_per_game: 26 }),
      makePlayer(2, { rating: 90, points_per_game: 12 }),
      makePlayer(3, { rating: 80, points_per_game: 19 }),
    ]);

    render(<PlayerHubPanel />);

    await screen.findByText("Player 2");
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

    render(<PlayerHubPanel />);
    await screen.findByText("Player 0");
    await userEvent.click(screen.getByTestId("player-hub-next"));
    expect(await screen.findByText("Player 20")).toBeInTheDocument();

    await userEvent.selectOptions(screen.getByLabelText("Team"), "LAL");

    expect(await screen.findByText("Player 20")).toBeInTheDocument();
    expect(screen.queryByText("Player 0")).not.toBeInTheDocument();
    expect(screen.getByText(/Players 1–2 of 2/)).toBeInTheDocument();
  });

  it("keeps six columns on a narrow screen and puts the rest behind More stats", async () => {
    setNarrow(true);
    vi.mocked(api.getHubPlayers).mockResolvedValue([makePlayer(1)]);

    render(<PlayerHubPanel />);
    await screen.findByText("Player 1");

    for (const label of ["Player", "Team", "Pos", "PPG", "RPG", "APG"]) {
      expect(screen.getByRole("columnheader", { name: new RegExp(label) })).toBeInTheDocument();
    }
    for (const label of ["Rating", "FG%", "3P%", "FT%"]) {
      expect(screen.queryByRole("columnheader", { name: new RegExp(label.replace("%", "\\%")) })).toBeNull();
    }

    const more = screen.getByRole("button", { name: "More stats" });
    expect(more).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(more);

    expect(screen.getByRole("columnheader", { name: /FG%/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Fewer stats" })).toHaveAttribute("aria-expanded", "true");
  });

  it("shows every column on a wide screen without a toggle", async () => {
    setNarrow(false);
    vi.mocked(api.getHubPlayers).mockResolvedValue([makePlayer(1)]);

    render(<PlayerHubPanel />);
    await screen.findByText("Player 1");

    expect(screen.getByRole("columnheader", { name: /FG%/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "More stats" })).toBeNull();
  });
});
