import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import GameCard from "./GameCard";
import type { Game } from "../api/client";

const baseGame: Game = {
  game_id: "g1", game_date: "2026-11-01", home_team: "BOS", away_team: "MIA",
  prediction: { home_win_probability: 0.62, predicted_margin: 3.5, predicted_total: 224.5 },
  completed: false, home_pts: null, away_pts: null,
};

describe("GameCard", () => {
  it("shows the win-probability prediction for an upcoming game", () => {
    render(<GameCard game={baseGame} onSelect={() => {}} />);
    expect(screen.getByText(/62%/)).toBeInTheDocument();
  });

  it("shows the final score instead of a prediction for a completed game", () => {
    const finished: Game = { ...baseGame, completed: true, home_pts: 110, away_pts: 102 };
    render(<GameCard game={finished} onSelect={() => {}} />);

    const scoreContainer = screen.getByText(/110/).parentElement;
    expect(scoreContainer).toHaveTextContent("102 – 110");
    expect(screen.getByText(/final/i)).toBeInTheDocument();
    expect(screen.queryByText(/62%/)).not.toBeInTheDocument();
  });

  it("calls onSelect with the game id when clicked", async () => {
    const onSelect = vi.fn();
    render(<GameCard game={baseGame} onSelect={onSelect} />);

    await userEvent.click(screen.getByTestId("game-card-g1"));
    expect(onSelect).toHaveBeenCalledWith("g1");
  });

  it("leads with the favourite, even when that is the away team", () => {
    const awayFav: Game = { ...baseGame, prediction: { home_win_probability: 0.47, predicted_margin: -4.8, predicted_total: 220 } };
    render(<GameCard game={awayFav} onSelect={() => {}} />);
    expect(screen.getByText("53%")).toBeInTheDocument();
    expect(screen.getByText("MIA to win")).toBeInTheDocument();
  });

  it("picks home at exactly 50%", () => {
    const even: Game = { ...baseGame, prediction: { home_win_probability: 0.5, predicted_margin: 0, predicted_total: 220 } };
    render(<GameCard game={even} onSelect={() => {}} />);
    expect(screen.getByText("BOS to win")).toBeInTheDocument();
  });

  it("says 'No pick yet' rather than 'Pending' and drops the raw date", () => {
    render(<GameCard game={{ ...baseGame, prediction: null }} onSelect={() => {}} />);
    expect(screen.getByText("No pick yet")).toBeInTheDocument();
    expect(screen.queryByText("Pending")).not.toBeInTheDocument();
    expect(screen.queryByText("2026-11-01")).not.toBeInTheDocument();
  });
});
