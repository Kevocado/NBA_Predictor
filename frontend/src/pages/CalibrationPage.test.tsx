import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import CalibrationPage from "./CalibrationPage";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { getCalibration: vi.fn() } }));
afterEach(() => vi.restoreAllMocks());

describe("CalibrationPage", () => {
  it("renders a row per calibration bin", async () => {
    vi.mocked(api.getCalibration).mockResolvedValue([
      { bin_start: 0.8, bin_end: 0.9, predicted_rate: 0.85, actual_rate: 0.82, count: 40 },
    ]);

    render(<CalibrationPage />);

    await waitFor(() => expect(screen.getByTestId("calibration-row")).toBeInTheDocument());
    expect(screen.getByText("85%")).toBeInTheDocument();
    expect(screen.getByText("82%")).toBeInTheDocument();
    expect(screen.getByText("40")).toBeInTheDocument();
  });

  it("shows an empty state when no settled predictions exist", async () => {
    vi.mocked(api.getCalibration).mockResolvedValue([]);
    render(<CalibrationPage />);
    await waitFor(() => expect(screen.getByText(/no settled predictions/i)).toBeInTheDocument());
  });

  it("shows an error state when the fetch fails", async () => {
    vi.mocked(api.getCalibration).mockRejectedValue(new Error("network error"));
    render(<CalibrationPage />);
    await waitFor(() => expect(screen.getByText(/couldn't load calibration/i)).toBeInTheDocument());
  });
});
