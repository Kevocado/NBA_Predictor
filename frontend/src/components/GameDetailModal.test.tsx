import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import GameDetailModal from "./GameDetailModal";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: { getGameDetail: vi.fn(), getGamePlayers: vi.fn() },
}));

afterEach(() => {
  vi.restoreAllMocks();
});

const detail = {
  game_id: "g1", game_date: "2026-11-01", home_team: "BOS", away_team: "MIA",
  prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
  markets: [
    { market: "h2h", selection: "home", model_probability: 0.62, market_probability: 0.55, edge: 0.07, bookmaker: "DraftKings", american_odds: -130 },
    { market: "spread", selection: "home", model_probability: 0.52, market_probability: 0.5, edge: 0.02, bookmaker: "DraftKings", american_odds: -110 },
  ],
};

const players = [{ player_id: "203999", player_name: "203999", stat: "points", predicted_value: 27.5 }];

describe("GameDetailModal", () => {
  it("shows a loading state before data arrives", () => {
    vi.mocked(api.getGameDetail).mockReturnValue(new Promise(() => {}));
    vi.mocked(api.getGamePlayers).mockReturnValue(new Promise(() => {}));

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it("renders markets sorted by edge descending", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    const rows = await screen.findAllByTestId("market-row");
    expect(rows[0]).toHaveTextContent("h2h");
    expect(rows[1]).toHaveTextContent("spread");
  });

  it("renders player prop predictions", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    expect(await screen.findByText("points")).toBeInTheDocument();
    expect(screen.getByText("27.5")).toBeInTheDocument();
  });

  it("calls onClose when the close button is clicked", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);
    const onClose = vi.fn();

    render(<GameDetailModal gameId="g1" onClose={onClose} />);
    await screen.findByRole("heading", { name: /BOS/ });

    await userEvent.click(screen.getByRole("button", { name: /close/i }));
    expect(onClose).toHaveBeenCalled();
  });

  it("shows an error state when either fetch fails", async () => {
    vi.mocked(api.getGameDetail).mockRejectedValue(new Error("boom"));
    vi.mocked(api.getGamePlayers).mockResolvedValue([]);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);
    expect(await screen.findByText(/couldn't load game details/i)).toBeInTheDocument();
  });
});