import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import TrackRecordPanel from "./TrackRecordPanel";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { getTrackRecord: vi.fn() } }));
afterEach(() => vi.restoreAllMocks());

describe("TrackRecordPanel", () => {
  it("renders hit rate as a percentage", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      { market: "h2h", total_predictions: 100, correct_predictions: 58, hit_rate: 0.58 },
    ]);

    render(<TrackRecordPanel />);

    await waitFor(() => expect(screen.getByText("Moneyline vs the market")).toBeInTheDocument());
    expect(screen.getByText("58%")).toBeInTheDocument();
  });

  it("says only pre-tip picks count, and how many finals were left out as rebuilt", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      { market: "game_outcome", total_predictions: 40, correct_predictions: 26, hit_rate: 0.65, n_rebuilt: 1200 },
      { market: "spread", total_predictions: 12, correct_predictions: 0, hit_rate: 0 },
    ]);
    render(<TrackRecordPanel />);
    expect(await screen.findByText("Winner pick")).toBeInTheDocument();
    expect(screen.getByText(/Only picks made before tip-off count/)).toBeInTheDocument();
    expect(screen.getByText(/1,200 finals had only a pick rebuilt after tip-off/)).toBeInTheDocument();
    // Spread picks can't be settled from what is stored: never a fake 0%.
    expect(screen.getAllByText("Not settled yet").length).toBeGreaterThan(0);
    expect(screen.queryByText("0%")).not.toBeInTheDocument();
  });

  it("shows an empty state when no predictions are tracked yet", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([]);
    render(<TrackRecordPanel />);
    await waitFor(() => expect(screen.getByText(/no tracked predictions/i)).toBeInTheDocument());
  });
});