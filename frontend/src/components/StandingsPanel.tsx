import { useEffect, useState } from "react";
import { api, type StandingsRow } from "../api/client";
import TeamLogo from "./TeamLogo";

const STATUS: Record<NonNullable<StandingsRow["playoff_status"]>, string> = {
  clinched: "Clinched",
  "play-in": "Play-in",
  eliminated: "Out",
  "in-hunt": "In the hunt",
};

// The race does not exist outside a season, so the label says so instead of
// leaving a bare table that looks like a live one. The wins and losses below it
// are real arithmetic on real games -- they are just last season's.
const OFFSEASON_NOTE: Record<Exclude<StandingsRow["season_state"], "regular" | "postseason">, string> = {
  offseason: "Off-season — last season's record. No playoff race yet.",
  preseason: "Pre-season — schedule loaded, nobody has played.",
};

export default function StandingsPanel() {
  const [rows, setRows] = useState<StandingsRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getHubStandings().then(setRows).catch(() => setError("Couldn't load standings."));
  }, []);

  if (error) return <p className="text-[var(--color-shotclock)]">{error}</p>;
  if (rows === null) return <p>Loading standings…</p>;
  if (rows.length === 0) return <p>No standings data cached yet.</p>;

  const conferences: Array<"East" | "West"> = ["East", "West"];

  // Every row carries the same state, so any row will do -- and there has to be
  // a row, because `rows.length === 0` already returned above.
  const state = rows[0].season_state;
  const note = state === "regular" || state === "postseason" ? null : OFFSEASON_NOTE[state];

  return (
    <>
      {/* Once, above both conferences -- not per column, which printed it
          twice. */}
      {note && (
        <p className="mb-3 text-sm text-[var(--color-net-faint)]">{note}</p>
      )}
      <div className="grid gap-6 sm:grid-cols-2">
      {conferences.map((conference) => (
        <div key={conference}>
          <h3 className="mb-2 font-semibold">{conference}</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[var(--color-net-faint)]">
                  <th>Seed</th>
                  <th>Team</th>
                  <th>W-L</th>
                  <th>GB</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {rows
                  .filter((row) => row.conference === conference)
                  .sort((a, b) => a.seed - b.seed)
                  .map((row) => (
                    <tr key={row.abbreviation}>
                      <td>{row.seed}</td>
                      <td>
                        <span className="inline-flex items-center gap-1.5">
                          <TeamLogo team={row.abbreviation} size={18} />
                          {row.abbreviation}
                        </span>
                      </td>
                      <td>
                        {row.wins}-{row.losses}
                      </td>
                      <td>{row.games_back === 0 ? "—" : row.games_back.toFixed(1)}</td>
                      {/* No status outside a season. The backend sends null
                          rather than a label, so there is nothing to show. */}
                      <td>{row.playoff_status ? STATUS[row.playoff_status] : "—"}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
      </div>
    </>
  );
}