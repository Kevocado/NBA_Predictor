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

    await waitFor(() => expect(screen.getByText("h2h")).toBeInTheDocument());
    expect(screen.getByText("58%")).toBeInTheDocument();
  });

  it("marks the 50% break-even line on each hit-rate bar", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      { market: "h2h", total_predictions: 100, correct_predictions: 58, hit_rate: 0.58 },
      { market: "spread", total_predictions: 100, correct_predictions: 51, hit_rate: 0.51 },
    ]);

    render(<TrackRecordPanel />);

    await waitFor(() => expect(screen.getAllByTestId("break-even-50")).toHaveLength(2));
    expect(screen.getAllByText("50%")).toHaveLength(2);
  });

  it("renders a dash instead of a misleading 0% for the player-props row", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([
      { market: "player_props", total_predictions: 200, correct_predictions: 0, hit_rate: 0.0 },
    ]);

    render(<TrackRecordPanel />);

    await waitFor(() => expect(screen.getByText("player_props")).toBeInTheDocument());
    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.queryByText("0%")).not.toBeInTheDocument();
  });

  it("shows an empty state when no predictions are tracked yet", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([]);
    render(<TrackRecordPanel />);
    await waitFor(() => expect(screen.getByText(/no tracked predictions/i)).toBeInTheDocument());
  });
});