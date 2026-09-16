import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import PlayerHubPanel from "./PlayerHubPanel";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { getHubPlayers: vi.fn() } }));
afterEach(() => vi.restoreAllMocks());

function makePlayer(i: number) {
  return {
    player_id: `${i}`, player_name: `Player ${i}`, team: "BOS", position: "G",
    rating: 100 - i, live_form_rating: 90, points_per_game: 20, rebounds_per_game: 5,
    assists_per_game: 4, fg_pct: 0.47, three_pt_pct: 0.37, ft_pct: 0.85, usage_rate: 0.24,
    minutes_per_game: 32,
  };
}

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
});