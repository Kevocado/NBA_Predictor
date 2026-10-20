import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import GamesPage from "./GamesPage";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    getGamesWeek: vi.fn(async () => []),
    getSeasonFirstWeek: vi.fn(async () => ({ first_week_start: "2026-02-10" })),
    getTrackRecord: vi.fn(),
    getGameSignals: vi.fn(async () => ({ sport: "nba", id: "", signals: [] })),
  },
}));

afterEach(() => vi.restoreAllMocks());

describe("GamesPage trust strip", () => {
  it("puts the pre-tip winner record under the week navigator", async () => {
    // The h2h row has a better-looking pre-tip record. The line must not quote it.
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      {
        market: "h2h",
        total_predictions: 20,
        correct_predictions: 19,
        hit_rate: 0.95,
        settled: true,
        pre_tip: { total_predictions: 20, correct_predictions: 19, hit_rate: 0.95 },
      },
      {
        market: "game_outcome",
        total_predictions: 80,
        correct_predictions: 50,
        hit_rate: 0.625,
        settled: true,
        pre_tip: { total_predictions: 66, correct_predictions: 33, hit_rate: 0.5 },
      },
    ]);

    render(<GamesPage />);

    const line = await screen.findByText(/Model, picks made before tip-off:/);
    expect(line).toHaveTextContent("Model, picks made before tip-off: 33 of 66");
    expect(line).not.toHaveTextContent("19 of 20");
    expect(screen.getByRole("link", { name: "Track record" })).toHaveAttribute("href", "/track-record");
  });

  it("stays silent when the record cannot be read", async () => {
    vi.mocked(api.getTrackRecord).mockRejectedValue(new Error("record down"));

    render(<GamesPage />);

    await waitFor(() => expect(api.getTrackRecord).toHaveBeenCalled());
    expect(screen.queryByText(/Model, picks made before tip-off:/)).toBeNull();
    // The page still works without it.
    expect(await screen.findByText("No games this week.")).toBeInTheDocument();
  });

  it("stays silent while the record is still loading", async () => {
    vi.mocked(api.getTrackRecord).mockReturnValue(new Promise(() => {}));

    render(<GamesPage />);

    expect(await screen.findByText("No games this week.")).toBeInTheDocument();
    expect(screen.queryByText(/Model, picks made before tip-off:/)).toBeNull();
  });
});
