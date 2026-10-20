const REGULAR_SEASON_GAMES = 82;

/**
 * Which season a team table is from, read off the records themselves: no team plays more than
 * 82 regular-season games, so a larger total can only be last season with its playoffs.
 */
export function teamsCaption(rows: { wins: number; losses: number }[]): string | null {
  if (rows.length === 0) return null;
  const games = rows.map((r) => r.wins + r.losses);
  if (Math.max(...games) > REGULAR_SEASON_GAMES) return "Last season's records, playoffs included. This season's record is on the Standings tab.";
  const avg = Math.round(games.reduce((a, b) => a + b, 0) / games.length);
  return `This season so far: about ${avg} games per team.`;
}

/** Until about game 20 "clinched" and "out" are claims no fan would make, so say what the table shows. */
export const EARLY_STANDINGS_GAMES = 20;
const EARLY_STATUS: Record<string, string> = { clinched: "Leading", eliminated: "Trailing" };

export function standingsStatus(status: string, label: string, mostGamesPlayed: number): string {
  return mostGamesPlayed < EARLY_STANDINGS_GAMES ? EARLY_STATUS[status] ?? label : label;
}
