import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import ModelSummaryPage from "./ModelSummaryPage";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { getManifest: vi.fn() } }));
afterEach(() => vi.restoreAllMocks());

describe("ModelSummaryPage", () => {
  it("renders model version and metrics", async () => {
    vi.mocked(api.getManifest).mockResolvedValue({
      model_version: "v20261101120000",
      trained_at: "2026-11-01T12:00:00",
      models: ["win_probability", "margin", "total"],
      metrics: { win_probability: { accuracy: 0.64 } },
    });

    render(<ModelSummaryPage />);

    await waitFor(() => expect(screen.getByText("Nov 1 model")).toBeInTheDocument());
    expect(screen.getByText("Win probability")).toBeInTheDocument();
    expect(screen.getByText("0.64")).toBeInTheDocument();
    expect(screen.queryByText("v20261101120000")).not.toBeInTheDocument();
  });

  it("survives a manifest without a model list", async () => {
    vi.mocked(api.getManifest).mockResolvedValue({ model_version: "v1", trained_at: "", models: undefined, metrics: {} } as never);
    render(<ModelSummaryPage />);
    expect(await screen.findByText(/no model has been trained/i)).toBeInTheDocument();
  });

  it("shows an empty state when no model has been trained yet", async () => {
    vi.mocked(api.getManifest).mockRejectedValue(new Error("404"));
    render(<ModelSummaryPage />);
    await waitFor(() => expect(screen.getByText(/no model has been trained/i)).toBeInTheDocument());
  });
});