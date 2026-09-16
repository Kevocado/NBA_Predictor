import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import StandingsPanel from "./StandingsPanel";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { getHubStandings: vi.fn() } }));
afterEach(() => vi.restoreAllMocks());

describe("StandingsPanel", () => {
  it("splits standings into East and West columns by seed", async () => {
    vi.mocked(api.getHubStandings).mockResolvedValue([
      { conference: "East", seed: 1, abbreviation: "BOS", wins: 50, losses: 20, win_pct: 0.714, games_back: 0, playoff_status: "clinched" },
      { conference: "West", seed: 1, abbreviation: "LAL", wins: 48, losses: 22, win_pct: 0.686, games_back: 0, playoff_status: "clinched" },
    ]);

    render(<StandingsPanel />);

    await waitFor(() => expect(screen.getByText("BOS")).toBeInTheDocument());
    expect(screen.getByText("East")).toBeInTheDocument();
    expect(screen.getByText("West")).toBeInTheDocument();
  });

  it("shows an empty state when no standings are cached yet", async () => {
    vi.mocked(api.getHubStandings).mockResolvedValue([]);
    render(<StandingsPanel />);
    await waitFor(() => expect(screen.getByText(/no standings data/i)).toBeInTheDocument());
  });
});