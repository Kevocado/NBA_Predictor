import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import TeamHubPanel from "./TeamHubPanel";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { getHubTeams: vi.fn() } }));
afterEach(() => vi.restoreAllMocks());

describe("TeamHubPanel", () => {
  it("groups teams by conference", async () => {
    vi.mocked(api.getHubTeams).mockResolvedValue([
      { abbreviation: "BOS", conference: "East", division: "Atlantic", wins: 10, losses: 2, points_per_game: 118, opp_points_per_game: 108, net_rating: 10, pace: 99, streak: 3 },
      { abbreviation: "LAL", conference: "West", division: "Pacific", wins: 8, losses: 4, points_per_game: 115, opp_points_per_game: 110, net_rating: 5, pace: 101, streak: -1 },
    ]);

    render(<TeamHubPanel />);

    await waitFor(() => expect(screen.getByText("BOS")).toBeInTheDocument());
    expect(screen.getByText("East")).toBeInTheDocument();
    expect(screen.getByText("West")).toBeInTheDocument();
  });

  it("shows an empty state when no team data is cached yet", async () => {
    vi.mocked(api.getHubTeams).mockResolvedValue([]);
    render(<TeamHubPanel />);
    await waitFor(() => expect(screen.getByText(/no team data/i)).toBeInTheDocument());
  });
});