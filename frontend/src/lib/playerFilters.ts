import type { PlayerHubRow } from "../api/client";

/**
 * The Players page's filters, as one pure function so the rules can be
 * tested without a browser.
 *
 * The position chips are coarse on purpose: the hub feed says "G" for 240
 * players and "PG" for a few, so a chip per position would be a long list of
 * mostly-empty choices. A player whose position starts with the letter is in
 * that group, and a position nobody recognises is in "All" and nowhere else.
 *
 * The minutes filter is on by default: early in a season a bench player's
 * averages rest on a handful of minutes, and a table of them is mostly noise.
 */

export type Position = "All" | "G" | "F" | "C";

export const POSITIONS: Position[] = ["All", "G", "F", "C"];

/** The minutes a player has to be on the floor for the table to show them. */
export const MIN_MINUTES = 15;

export interface PlayerFilters {
  search: string;
  team: string;
  position: Position;
  /** The "min 15 minutes" switch. On by default. */
  minMinutes: boolean;
}

export const DEFAULT_FILTERS: PlayerFilters = {
  search: "",
  team: "",
  position: "All",
  minMinutes: true,
};

/** Which position codes each chip covers. */
const POSITION_GROUPS: Record<string, string[]> = {
  G: ["G", "PG", "SG"],
  F: ["F", "SF", "PF"],
  C: ["C"],
};

/** Does this player belong to the chip's group? */
export function matchesPosition(position: string, wanted: Position): boolean {
  if (wanted === "All") return true;
  const code = (position || "").trim().toUpperCase();
  if (!code) return false;
  return (POSITION_GROUPS[wanted] ?? [wanted]).includes(code);
}

/** The rows the filters leave. */
export function filterPlayers(rows: PlayerHubRow[], filters: PlayerFilters): PlayerHubRow[] {
  const needle = filters.search.trim().toLowerCase();
  return rows.filter((row) => {
    if (filters.team && row.team !== filters.team) return false;
    if (!matchesPosition(row.position, filters.position)) return false;
    if (filters.minMinutes && (row.minutes_per_game ?? 0) < MIN_MINUTES) return false;
    if (needle) {
      const haystack = `${row.player_name} ${row.team}`.toLowerCase();
      if (!haystack.includes(needle)) return false;
    }
    return true;
  });
}

/** The five best players on a stat, biggest first. */
export function leadersOn(
  rows: PlayerHubRow[],
  value: (row: PlayerHubRow) => number | null | undefined,
  limit = 5,
): PlayerHubRow[] {
  return rows
    .filter((row) => {
      const number = value(row);
      return typeof number === "number" && Number.isFinite(number);
    })
    .sort((a, b) => (value(b) as number) - (value(a) as number))
    .slice(0, limit);
}
