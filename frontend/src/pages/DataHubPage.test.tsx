import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import DataHubPage from "./DataHubPage";

vi.mock("../api/client", () => ({
  api: {
    getHubTeams: vi.fn().mockResolvedValue([]),
    getHubPlayers: vi.fn().mockResolvedValue([]),
    getHubRankings: vi.fn().mockResolvedValue([]),
    getHubStandings: vi.fn().mockResolvedValue([]),
    getTrackRecord: vi.fn().mockResolvedValue([]),
    getVsMarket: vi.fn().mockResolvedValue({
      market: "h2h",
      n: 0,
      mean_model_probability: null,
      mean_market_probability: null,
      mean_edge_points: null,
      disagreement_n: 0,
      disagreement_hit_rate: null,
      disagreement_game_ids: [],
      weekly: [],
      scope: {
        population: "finished games with a pre-tip moneyline price",
        weekly_from: null,
        weekly_through: null,
        n_games_total: 0,
        n_games_in_weekly: 0,
        n_games_outside_weekly: 0,
      },
      method: {},
    }),
  },
}));

describe("DataHubPage", () => {
  it("shows the Team Hub panel by default", async () => {
    render(<DataHubPage />);
    await waitFor(() => expect(screen.getByText(/no team data/i)).toBeInTheDocument());
  });

  it("switches to the Player Hub panel when its tab is clicked", async () => {
    render(<DataHubPage />);
    await userEvent.click(screen.getByTestId("hub-tab-player-hub"));
    await waitFor(() => expect(screen.getByText(/no player data/i)).toBeInTheDocument());
    expect(screen.getByTestId("hub-tab-player-hub")).toHaveAttribute("aria-current", "true");
    expect(screen.getByTestId("hub-tab-team-hub")).not.toHaveAttribute("aria-current");
  });

  it("switches to the Track Record panel when its tab is clicked", async () => {
    render(<DataHubPage />);
    await userEvent.click(screen.getByTestId("hub-tab-track-record"));
    await waitFor(() => expect(screen.getByText(/no tracked predictions/i)).toBeInTheDocument());
  });
});