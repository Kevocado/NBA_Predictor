import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import ModelSummaryPage from "./ModelSummaryPage";
import { api } from "../api/client";

vi.mock("../api/client", () => ({ api: { getManifest: vi.fn(), getPlayerPropsManifest: vi.fn() } }));
afterEach(() => vi.restoreAllMocks());

describe("ModelSummaryPage", () => {
  it("renders model version and metrics", async () => {
    vi.mocked(api.getManifest).mockResolvedValue({
      model_version: "v20261101120000",
      trained_at: "2026-11-01T12:00:00",
      models: ["win_probability", "margin", "total"],
      metrics: { win_probability: { accuracy: 0.64 } },
    });
    vi.mocked(api.getPlayerPropsManifest).mockRejectedValue(new Error("404"));

    render(<ModelSummaryPage />);

    await waitFor(() => expect(screen.getByText("Nov 1 model")).toBeInTheDocument());
    expect(screen.getByText("Win probability")).toBeInTheDocument();
    expect(screen.getByText("0.64")).toBeInTheDocument();
    expect(screen.queryByText("v20261101120000")).not.toBeInTheDocument();
    expect(screen.getByText("Trained 1 Nov 2026")).toBeInTheDocument();
  });

  it("survives a manifest without a model list", async () => {
    vi.mocked(api.getManifest).mockResolvedValue({ model_version: "v1", trained_at: "", models: undefined, metrics: {} } as never);
    render(<ModelSummaryPage />);
    expect(await screen.findByText(/no model has been trained/i)).toBeInTheDocument();
  });

  it("shows an empty state when no model has been trained yet", async () => {
    vi.mocked(api.getManifest).mockRejectedValue(new Error("404"));
    vi.mocked(api.getPlayerPropsManifest).mockRejectedValue(new Error("404"));
    render(<ModelSummaryPage />);
    await waitFor(() => expect(screen.getByText(/no model has been trained/i)).toBeInTheDocument());
  });
});

describe("ModelSummaryPage training data", () => {
  it("renders training counts and the current-season explainer", async () => {
    vi.mocked(api.getManifest).mockResolvedValue({
      model_version: "v9",
      trained_at: "2026-09-20T00:00:00+00:00",
      models: ["win_probability", "margin", "total"],
      metrics: {},
      training: { n_train_games: 4000, n_holdout_games: 1000, n_current_season_games: 350 },
    });
    vi.mocked(api.getPlayerPropsManifest).mockResolvedValue({
      model_version: "v9",
      trained_at: "2026-09-20T00:00:00+00:00",
      models: ["points", "rebounds", "assists", "threes"],
      metrics: {},
      training: { n_train_player_games: 120000, in_sample_metrics: true },
    });

    render(<ModelSummaryPage />);
    await waitFor(() => expect(screen.getByText(/training data/i)).toBeInTheDocument());
    expect(screen.getByText("4,000")).toBeInTheDocument();
    expect(screen.getByText("1,000")).toBeInTheDocument();
    expect(screen.getByText("350")).toBeInTheDocument();
    expect(
      screen.getByText(/current-season matches count only once their stats are final/i)
    ).toBeInTheDocument();
    expect(screen.getByText("120,000")).toBeInTheDocument();
  });

  it("shows fallback copy when manifests are missing", async () => {
    vi.mocked(api.getManifest).mockResolvedValue({
      model_version: "v9",
      trained_at: "t",
      models: [],
      metrics: {},
      // no training key: manifest predates training counts
    });
    vi.mocked(api.getPlayerPropsManifest).mockRejectedValue(new Error("404"));
    render(<ModelSummaryPage />);
    await waitFor(() =>
      expect(screen.getAllByText(/training data not published yet/i)).toHaveLength(2)
    );
  });
});