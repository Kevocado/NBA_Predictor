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
    expect(scoreContainer).toHaveTextContent("110 – 102");
    expect(screen.getByText(/final/i)).toBeInTheDocument();
    expect(screen.queryByText(/62%/)).not.toBeInTheDocument();
  });

  it("calls onSelect with the game id when clicked", async () => {
    const onSelect = vi.fn();
    render(<GameCard game={baseGame} onSelect={onSelect} />);

    await userEvent.click(screen.getByTestId("game-card-g1"));
    expect(onSelect).toHaveBeenCalledWith("g1");
  });
});
