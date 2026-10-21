import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import PlayerAvatar from "./PlayerAvatar";

describe("PlayerAvatar", () => {
  it("shows the player's headshot from ESPN", () => {
    render(<PlayerAvatar playerId="203999" name="Nikola Jokic" />);

    const img = screen.getByRole("presentation", { hidden: true }) as HTMLImageElement;
    expect(img.getAttribute("src")).toBe(
      "https://a.espncdn.com/i/headshots/nba/players/full/203999.png",
    );
  });

  it("falls back to the initial when ESPN has no picture", () => {
    render(<PlayerAvatar playerId="203999" name="Nikola Jokic" />);

    const img = screen.getByRole("presentation", { hidden: true }) as HTMLImageElement;
    fireEvent.error(img);

    expect(screen.getByRole("img", { name: "Nikola Jokic (no picture)" }).textContent).toBe("N");
  });

  it("falls back to the initial when there is no player id at all", () => {
    render(<PlayerAvatar playerId="" name="Nikola Jokic" />);

    expect(screen.getByRole("img", { name: "Nikola Jokic (no picture)" }).textContent).toBe("N");
  });
});
