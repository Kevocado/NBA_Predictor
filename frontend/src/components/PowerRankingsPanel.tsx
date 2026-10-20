import { api, type PowerRankingRow } from "../api/client";
import HubRows from "./HubRows";
import TeamLogo from "./TeamLogo";

const TREND_GLYPH: Record<PowerRankingRow["trend"], string> = { up: "▲", down: "▼", steady: "–" };
const TREND_COLOR: Record<PowerRankingRow["trend"], string> = {
  up: "text-[var(--color-win)]",
  down: "text-[var(--color-shotclock)]",
  steady: "text-[var(--color-net-faint)]",
};

export default function PowerRankingsPanel() {
  return (
    <HubRows<PowerRankingRow> load={api.getHubRankings} noun="ranking">
      {(rows) => (
        <>
          <p className="mb-3 max-w-prose text-sm text-pr-text-dim">Team strength rating: higher is stronger.</p>
          <ol className="space-y-1 text-sm">
            {[...rows]
              .sort((a, b) => a.rank - b.rank)
              .map((row) => (
                <li key={row.abbreviation} className="flex justify-between border-b border-[var(--color-line)] py-1">
                  <span className="flex items-center gap-1.5">
                    #{row.rank} <TeamLogo team={row.abbreviation} size={18} />
                    <span>{row.abbreviation}</span>
                  </span>
                  <span>
                    {row.power_rating.toFixed(0)} <span className={TREND_COLOR[row.trend]}>{TREND_GLYPH[row.trend]}</span>
                  </span>
                </li>
              ))}
          </ol>
        </>
      )}
    </HubRows>
  );
}
