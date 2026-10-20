import { describe, expect, it } from "vitest";
import { earlySeasonNotice, EARLY_SEASON_GAMES } from "./earlySeason";

describe("earlySeasonNotice", () => {
  it("says plainly that the model has not seen a game of this season", () => {
    expect(earlySeasonNotice(0)).toMatch(/has not seen a game of this season/i);
    expect(earlySeasonNotice(0)).toMatch(/less reliable/i);
  });

  it("states the count while the season is still young", () => {
    expect(earlySeasonNotice(40)).toMatch(/40 of this season's games/);
  });

  it("goes away once about ten games per team are in", () => {
    expect(earlySeasonNotice(EARLY_SEASON_GAMES)).toBeNull();
    expect(earlySeasonNotice(900)).toBeNull();
  });

  it("says nothing when the count is unknown (never guess)", () => {
    expect(earlySeasonNotice(undefined)).toBeNull();
    expect(earlySeasonNotice(null)).toBeNull();
    expect(earlySeasonNotice(Number.NaN)).toBeNull();
  });
});
