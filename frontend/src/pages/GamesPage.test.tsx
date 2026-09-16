import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import GamesPage from "./GamesPage";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: { getGamesWeek: vi.fn() },
}));

afterEach(() => {
  vi.restoreAllMocks();
});

describe("GamesPage", () => {
  it("shows a loading state before games arrive", () => {
    vi.mocked(api.getGamesWeek).mockReturnValue(new Promise(() => {}));
    render(<GamesPage />);
    expect(screen.getByText(/loading games/i)).toBeInTheDocument();
  });

  it("groups games by day and renders a card per game", async () => {
    vi.mocked(api.getGamesWeek).mockResolvedValue([
      {
        game_id: "g1", game_date: "2026-02-16", home_team: "BOS", away_team: "MIA",
        prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
      },
      {
        game_id: "g2", game_date: "2026-02-18", home_team: "LAL", away_team: "GSW",
        prediction: null,
      },
    ]);

    render(<GamesPage />);

    await waitFor(() => expect(screen.getByTestId("game-card-g1")).toBeInTheDocument());
    expect(screen.getByTestId("game-card-g2")).toBeInTheDocument();
    expect(screen.getByText(/62%/)).toBeInTheDocument();
  });

  it("shows an empty state when no games are scheduled this week", async () => {
    vi.mocked(api.getGamesWeek).mockResolvedValue([]);
    render(<GamesPage />);

    await waitFor(() => expect(screen.getByText(/no games scheduled this week/i)).toBeInTheDocument());
  });

  it("shows an error state when the fetch fails", async () => {
    vi.mocked(api.getGamesWeek).mockRejectedValue(new Error("network error"));
    render(<GamesPage />);

    await waitFor(() => expect(screen.getByText(/couldn't load games/i)).toBeInTheDocument());
  });

  it("re-fetches the following week when the next-week button is clicked", async () => {
    vi.mocked(api.getGamesWeek).mockResolvedValue([]);
    render(<GamesPage />);

    await waitFor(() => expect(api.getGamesWeek).toHaveBeenCalledTimes(1));
    const firstCallArg = vi.mocked(api.getGamesWeek).mock.calls[0][0];

    await userEvent.click(screen.getByRole("button", { name: /next week/i }));

    await waitFor(() => expect(api.getGamesWeek).toHaveBeenCalledTimes(2));
    const secondCallArg = vi.mocked(api.getGamesWeek).mock.calls[1][0];

    const firstDate = new Date(firstCallArg + "T00:00:00");
    const secondDate = new Date(secondCallArg + "T00:00:00");
    const diffDays = (secondDate.getTime() - firstDate.getTime()) / (1000 * 60 * 60 * 24);
    expect(diffDays).toBe(7);
  });

  it("re-fetches the previous week when the previous-week button is clicked", async () => {
    vi.mocked(api.getGamesWeek).mockResolvedValue([]);
    render(<GamesPage />);

    await waitFor(() => expect(api.getGamesWeek).toHaveBeenCalledTimes(1));
    const firstCallArg = vi.mocked(api.getGamesWeek).mock.calls[0][0];

    await userEvent.click(screen.getByRole("button", { name: /previous week/i }));

    await waitFor(() => expect(api.getGamesWeek).toHaveBeenCalledTimes(2));
    const secondCallArg = vi.mocked(api.getGamesWeek).mock.calls[1][0];

    const firstDate = new Date(firstCallArg + "T00:00:00");
    const secondDate = new Date(secondCallArg + "T00:00:00");
    const diffDays = (secondDate.getTime() - firstDate.getTime()) / (1000 * 60 * 60 * 24);
    expect(diffDays).toBe(-7);
  });
});
