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

  it("shows an empty state when no predictions are tracked yet", async () => {
    vi.mocked(api.getTrackRecord).mockResolvedValue([]);
    render(<TrackRecordPanel />);
    await waitFor(() => expect(screen.getByText(/no tracked predictions/i)).toBeInTheDocument());
  });
});