import { useState } from "react";
import TeamHubPanel from "../components/TeamHubPanel";
import PlayerHubPanel from "../components/PlayerHubPanel";
import PowerRankingsPanel from "../components/PowerRankingsPanel";
import StandingsPanel from "../components/StandingsPanel";
import TrackRecordPanel from "../components/TrackRecordPanel";

const TABS = [
  { key: "team-hub", label: "Teams", panel: TeamHubPanel },
  { key: "player-hub", label: "Players", panel: PlayerHubPanel },
  { key: "power-rankings", label: "Power rankings", panel: PowerRankingsPanel },
  { key: "standings", label: "Standings", panel: StandingsPanel },
  { key: "track-record", label: "Track record", panel: TrackRecordPanel },
] as const;

export default function DataHubPage() {
  const [activeTab, setActiveTab] = useState<(typeof TABS)[number]["key"]>("team-hub");
  const ActivePanel = TABS.find((tab) => tab.key === activeTab)!.panel;

  return (
    <div>
      <nav aria-label="Data Hub sections" className="mb-4 flex gap-2 overflow-x-auto border-b border-pr-rule">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            data-testid={`hub-tab-${tab.key}`}
            type="button"
            aria-current={activeTab === tab.key || undefined}
            onClick={() => setActiveTab(tab.key)}
            className={`whitespace-nowrap border-b-2 px-3 py-2 text-sm font-semibold ${
              activeTab === tab.key ? "border-pr-accent text-pr-text" : "border-transparent text-pr-text-dim hover:text-pr-text"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </nav>
      <ActivePanel />
    </div>
  );
}