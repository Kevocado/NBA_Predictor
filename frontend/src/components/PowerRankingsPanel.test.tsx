import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import PowerRankingsPanel from "./PowerRankingsPanel";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { getHubRankings: vi.fn() } }));
afterEach(() => vi.restoreAllMocks());

describe("PowerRankingsPanel", () => {
  it("renders ranked teams with trend glyphs", async () => {
    vi.mocked(api.getHubRankings).mockResolvedValue([
      { rank: 1, abbreviation: "BOS", power_rating: 1620, trend: "up" },
      { rank: 2, abbreviation: "LAL", power_rating: 1590, trend: "down" },
    ]);

    render(<PowerRankingsPanel />);

    await waitFor(() => expect(screen.getByText("BOS")).toBeInTheDocument());
    expect(screen.getByText("▲")).toBeInTheDocument();
    expect(screen.getByText("▼")).toBeInTheDocument();
  });

  it("shows an empty state when no rankings are cached yet", async () => {
    vi.mocked(api.getHubRankings).mockResolvedValue([]);
    render(<PowerRankingsPanel />);
    await waitFor(() => expect(screen.getByText(/no ranking data/i)).toBeInTheDocument());
  });
});