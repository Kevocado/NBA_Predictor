// ESPN CDN slug for each NBA abbreviation.
const ESPN_NBA_SLUGS: Record<string, string> = {
  ATL: "atl",
  BKN: "bkn",
  BOS: "bos",
  CHA: "cha",
  CHI: "chi",
  CLE: "cle",
  DAL: "dal",
  DEN: "den",
  DET: "det",
  GSW: "gsw",
  HOU: "hou",
  IND: "ind",
  LAC: "lac",
  LAL: "lal",
  MEM: "mem",
  MIA: "mia",
  MIL: "mil",
  MIN: "min",
  NOP: "no",
  NYK: "ny",
  OKC: "okc",
  ORL: "orl",
  PHI: "phi",
  PHX: "phx",
  POR: "por",
  SAC: "sac",
  SAS: "sa",
  TOR: "tor",
  UTA: "utah",
  WAS: "wsh",
};

const CDN_BASE = "https://a.espncdn.com/i/teamlogos/nba/500";

/** Logo URL keyed by NBA abbreviation. */
export const TEAM_LOGO_URLS: Record<string, string> = Object.fromEntries(
  Object.entries(ESPN_NBA_SLUGS).map(([abbr, slug]) => [abbr, `${CDN_BASE}/${slug}.png`]),
);

/** Returns the logo URL for a team abbreviation, or undefined if unknown. */
export function teamLogoUrl(abbreviation: string): string | undefined {
  return TEAM_LOGO_URLS[abbreviation];
}
