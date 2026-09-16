import { useState } from "react";
import TeamHubPanel from "../components/TeamHubPanel";
import PlayerHubPanel from "../components/PlayerHubPanel";
import PowerRankingsPanel from "../components/PowerRankingsPanel";
import StandingsPanel from "../components/StandingsPanel";
import TrackRecordPanel from "../components/TrackRecordPanel";

const TABS = [
  { key: "team-hub", label: "Team Hub", panel: TeamHubPanel },
  { key: "player-hub", label: "Player Hub", panel: PlayerHubPanel },
  { key: "power-rankings", label: "Power Rankings", panel: PowerRankingsPanel },
  { key: "standings", label: "Standings", panel: StandingsPanel },
  { key: "track-record", label: "Track Record", panel: TrackRecordPanel },
] as const;

export default function DataHubPage() {
  const [activeTab, setActiveTab] = useState<(typeof TABS)[number]["key"]>("team-hub");
  const ActivePanel = TABS.find((tab) => tab.key === activeTab)!.panel;

  return (
    <div>
      <div className="mb-4 flex gap-2 border-b border-[var(--color-line)]">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            data-testid={`hub-tab-${tab.key}`}
            onClick={() => setActiveTab(tab.key)}
            className={`px-3 py-2 text-sm ${
              activeTab === tab.key ? "border-b-2 border-[var(--color-hardwood)] font-semibold" : "text-[var(--color-net-dim)]"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <ActivePanel />
    </div>
  );
}