// Team identity without logos: each club's real primary colour (from its own
// brand guide) and nickname. Chips nudge dark colours to stay readable.
export const TEAMS: Record<string, { name: string; color: string }> = {
  ATL: { name: "Hawks", color: "#E03A3E" },
  BOS: { name: "Celtics", color: "#007A33" },
  BKN: { name: "Nets", color: "#000000" },
  CHA: { name: "Hornets", color: "#1D1160" },
  CHI: { name: "Bulls", color: "#CE1141" },
  CLE: { name: "Cavaliers", color: "#860038" },
  DAL: { name: "Mavericks", color: "#00538C" },
  DEN: { name: "Nuggets", color: "#0E2240" },
  DET: { name: "Pistons", color: "#C8102E" },
  GSW: { name: "Warriors", color: "#1D428A" },
  HOU: { name: "Rockets", color: "#CE1141" },
  IND: { name: "Pacers", color: "#002D62" },
  LAC: { name: "Clippers", color: "#C8102E" },
  LAL: { name: "Lakers", color: "#552583" },
  MEM: { name: "Grizzlies", color: "#5D76A9" },
  MIA: { name: "Heat", color: "#98002E" },
  MIL: { name: "Bucks", color: "#00471B" },
  MIN: { name: "Timberwolves", color: "#0C2340" },
  NOP: { name: "Pelicans", color: "#0C2340" },
  NYK: { name: "Knicks", color: "#006BB6" },
  OKC: { name: "Thunder", color: "#007AC1" },
  ORL: { name: "Magic", color: "#0077C0" },
  PHI: { name: "76ers", color: "#006BB6" },
  PHX: { name: "Suns", color: "#1D1160" },
  POR: { name: "Trail Blazers", color: "#E03A3E" },
  SAC: { name: "Kings", color: "#5A2D81" },
  SAS: { name: "Spurs", color: "#C4CED4" },
  TOR: { name: "Raptors", color: "#CE1141" },
  UTA: { name: "Jazz", color: "#002B5C" },
  WAS: { name: "Wizards", color: "#002B5C" },
};

export const teamName = (code: string) => TEAMS[code]?.name ?? code;
export const teamColor = (code: string): string | undefined => TEAMS[code]?.color;
