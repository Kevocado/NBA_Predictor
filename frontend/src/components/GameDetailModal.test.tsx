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
  completed: false, home_pts: null, away_pts: null,
  markets: [
    { market: "h2h", selection: "BOS", model_probability: 0.62, market_probability: 0.55, edge: 0.07, bookmaker: "DraftKings", american_odds: -130, point: null },
    { market: "spread", selection: "BOS", model_probability: 0.52, market_probability: 0.5, edge: 0.02, bookmaker: "DraftKings", american_odds: -110, point: -4.5 },
    { market: "total", selection: "over", model_probability: 0.48, market_probability: 0.5, edge: -0.02, bookmaker: "FanDuel", american_odds: -105, point: 224.5 },
  ],
  head_to_head: [
    { game_id: "g0", game_date: "2026-01-01", home_team: "BOS", away_team: "MIA", home_pts: 110, away_pts: 100 },
  ],
  home_recent_form: ["W", "L", "W"],
  away_recent_form: ["L", "L", "W"],
};

const players = [{ player_id: "203999", player_name: "Nikola Jokic", stat: "points", predicted_value: 27.5, actual_value: null }];

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

  it("renders bookmaker and american odds for each market row", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    await screen.findAllByTestId("market-row");
    const bookmakers = screen.getAllByText(/DraftKings|FanDuel/);
    expect(bookmakers).toHaveLength(3);
    expect(screen.getByText(/-130/)).toBeInTheDocument();
  });

  it("renders the spread and total lines", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    await screen.findAllByTestId("market-row");
    expect(screen.getAllByText(/-4\.5/)).toHaveLength(1);
    expect(screen.getAllByText(/224\.5/)).toHaveLength(2);
  });

  it("renders head-to-head history", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    expect(await screen.findByText(/2026-01-01/)).toBeInTheDocument();
    expect(screen.getByText(/MIA 100 – 110 BOS/)).toBeInTheDocument();
  });

  it("renders recent form for both teams", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    expect(await screen.findAllByTestId("form-badge")).toHaveLength(6);
  });

  it("shows the final score instead of a live prediction for a completed game", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue({
      ...detail, completed: true, home_pts: 108, away_pts: 101, markets: [], head_to_head: [],
    });
    vi.mocked(api.getGamePlayers).mockResolvedValue([]);

    render(<GameDetailModal gameId="g1" onClose={() => {}} />);

    expect(await screen.findByText(/final/i)).toBeInTheDocument();
    expect(screen.getByText("108")).toBeInTheDocument();
    expect(screen.getByText("101")).toBeInTheDocument();
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

  it("closes when Escape is pressed", async () => {
    vi.mocked(api.getGameDetail).mockResolvedValue(detail);
    vi.mocked(api.getGamePlayers).mockResolvedValue(players);
    const onClose = vi.fn();

    render(<GameDetailModal gameId="g1" onClose={onClose} />);
    await screen.findByRole("heading", { name: /BOS/ });

    await userEvent.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();
  });
});
