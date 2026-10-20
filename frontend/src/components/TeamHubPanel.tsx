import { api, type TeamHubRow } from "../api/client";
import HubRows from "./HubRows";
import { teamsCaption } from "../lib/seasonCaption";
import TeamLogo from "./TeamLogo";
import { signed, streak } from "../predictor-ui";

export default function TeamHubPanel() {
  return (
    <HubRows<TeamHubRow> load={api.getHubTeams} noun="team">
      {(rows) => <TeamTables rows={rows} />}
    </HubRows>
  );
}

function TeamTables({ rows }: { rows: TeamHubRow[] }) {
  const conferences: Array<"East" | "West"> = ["East", "West"];
  const caption = teamsCaption(rows);

  return (
    <div className="space-y-6">
      {caption && <p className="text-sm text-pr-text-dim">{caption}</p>}
      {conferences.map((conference) => (
        <div key={conference}>
          <h3 className="mb-2 font-semibold">{conference}</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[var(--color-net-faint)]">
                  <th>Team</th>
                  <th>W-L</th>
                  <th>PPG</th>
                  <th>Opp PPG</th>
                  <th>Net Rtg</th>
                  <th>Pace</th>
                  <th>Streak</th>
                </tr>
              </thead>
              <tbody>
                {rows
                  .filter((row) => row.conference === conference)
                  .sort((a, b) => b.points_per_game - a.points_per_game)
                  .map((row) => (
                    <tr key={row.abbreviation}>
                      <td>
                        <span className="inline-flex items-center gap-1.5">
                          <TeamLogo team={row.abbreviation} size={18} />
                          {row.abbreviation}
                        </span>
                      </td>
                      <td>
                        {row.wins}-{row.losses}
                      </td>
                      <td>{row.points_per_game.toFixed(1)}</td>
                      <td>{row.opp_points_per_game.toFixed(1)}</td>
                      <td>{signed(row.net_rating)}</td>
                      <td>{row.pace.toFixed(1)}</td>
                      <td>{streak(row.streak)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </div>
  );
}