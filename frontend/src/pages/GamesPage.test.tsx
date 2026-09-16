import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import GamesPage from "./GamesPage";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: { getGames: vi.fn() },
}));

afterEach(() => {
  vi.restoreAllMocks();
});

describe("GamesPage", () => {
  it("shows a loading state before games arrive", () => {
    vi.mocked(api.getGames).mockReturnValue(new Promise(() => {}));
    render(<GamesPage />);
    expect(screen.getByText(/loading games/i)).toBeInTheDocument();
  });

  it("renders a game card per returned game", async () => {
    vi.mocked(api.getGames).mockResolvedValue([
      {
        game_id: "g1", game_date: "2026-11-01", home_team: "BOS", away_team: "MIA",
        prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
      },
    ]);

    render(<GamesPage />);

    await waitFor(() => expect(screen.getByTestId("game-card-g1")).toBeInTheDocument());
    expect(screen.getByText(/62%/)).toBeInTheDocument();
  });

  it("shows an empty state when no games are scheduled", async () => {
    vi.mocked(api.getGames).mockResolvedValue([]);
    render(<GamesPage />);

    await waitFor(() => expect(screen.getByText(/no games scheduled/i)).toBeInTheDocument());
  });

  it("shows an error state when the fetch fails", async () => {
    vi.mocked(api.getGames).mockRejectedValue(new Error("network error"));
    render(<GamesPage />);

    await waitFor(() => expect(screen.getByText(/couldn't load games/i)).toBeInTheDocument());
  });

  it("re-fetches games when the date input changes", async () => {
    vi.mocked(api.getGames).mockResolvedValue([]);
    render(<GamesPage />);

    await waitFor(() => expect(api.getGames).toHaveBeenCalledTimes(1));

    const dateInput = screen.getByLabelText(/date/i);
    await userEvent.clear(dateInput);
    await userEvent.type(dateInput, "2026-12-25");

    await waitFor(() => expect(api.getGames).toHaveBeenLastCalledWith("2026-12-25"));
  });
});