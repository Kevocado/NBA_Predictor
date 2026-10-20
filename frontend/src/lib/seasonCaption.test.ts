import { describe, expect, it } from "vitest";
import { standingsStatus, teamsCaption } from "./seasonCaption";

describe("teamsCaption", () => {
  it("recognises last season by games played (no team plays more than 82 regular-season games)", () => {
    expect(teamsCaption([{ wins: 74, losses: 37 }, { wins: 40, losses: 42 }])).toMatch(/Last season's records, playoffs included/);
  });
  it("says how far into this season the table is", () => {
    expect(teamsCaption([{ wins: 3, losses: 2 }, { wins: 2, losses: 3 }])).toBe("This season so far: about 5 games per team.");
  });
  it("says nothing for an empty table", () => {
    expect(teamsCaption([])).toBeNull();
  });
});

describe("standingsStatus", () => {
  it("softens clinched/out in the first games of a season", () => {
    expect(standingsStatus("clinched", "Clinched", 4)).toBe("Leading");
    expect(standingsStatus("eliminated", "Out", 4)).toBe("Trailing");
    expect(standingsStatus("play-in", "Play-in", 4)).toBe("Play-in");
  });
  it("uses the real labels once the season is underway", () => {
    expect(standingsStatus("clinched", "Clinched", 60)).toBe("Clinched");
    expect(standingsStatus("eliminated", "Out", 60)).toBe("Out");
  });
});
