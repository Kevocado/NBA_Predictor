import { describe, expect, it } from "vitest";
import { TEAM_LOGO_URLS, teamLogoUrl } from "./teamLogos";

const EXPECTED_ABBREVS = [
  "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
  "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
  "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
];

describe("teamLogos", () => {
  it("maps all 30 NBA abbreviations to ESPN CDN URLs", () => {
    expect(Object.keys(TEAM_LOGO_URLS)).toHaveLength(30);
    for (const abbr of EXPECTED_ABBREVS) {
      const url = TEAM_LOGO_URLS[abbr];
      expect(url, abbr).toMatch(/^https:\/\/a\.espncdn\.com\/i\/teamlogos\/nba\/500\/.+\.png$/);
    }
  });

  it("teamLogoUrl returns the CDN URL for a known team", () => {
    expect(teamLogoUrl("BOS")).toContain("bos.png");
  });

  it("teamLogoUrl returns undefined for an unknown identifier", () => {
    expect(teamLogoUrl("XYZ")).toBeUndefined();
  });
});
