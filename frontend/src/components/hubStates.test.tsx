import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import TeamHubPanel from "./TeamHubPanel";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { getHubTeams: vi.fn(), getHubStandings: vi.fn() } }));

const team = (abbreviation: string, wins: number, losses: number) => ({
  abbreviation, conference: "East", division: "Atlantic", wins, losses,
  points_per_game: 110, opp_points_per_game: 108, net_rating: 2, pace: 100, streak: 1,
});

describe("Data Hub panels", () => {
  beforeEach(() => vi.clearAllMocks());

  it("offers a retry when a table fails to load, and recovers", async () => {
    vi.mocked(api.getHubTeams).mockRejectedValueOnce(new Error("down")).mockResolvedValue([team("BOS", 3, 2)] as never);
    render(<TeamHubPanel />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/couldn't load team data/i);
    await userEvent.click(screen.getByRole("button", { name: /try again/i }));
    expect(await screen.findByText("BOS")).toBeInTheDocument();
  });

  it("says which season the Teams table is from", async () => {
    vi.mocked(api.getHubTeams).mockResolvedValue([team("BOS", 74, 37)] as never);
    render(<TeamHubPanel />);
    expect(await screen.findByText(/last season's records, playoffs included/i)).toBeInTheDocument();
  });
});
