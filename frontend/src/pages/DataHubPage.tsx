import TeamHubPanel from "../components/TeamHubPanel";
import PlayerHubPanel from "../components/PlayerHubPanel";
import PowerRankingsPanel from "../components/PowerRankingsPanel";
import StandingsPanel from "../components/StandingsPanel";
import TrackRecordPanel from "../components/TrackRecordPanel";

const PANELS = {
  "team-hub": TeamHubPanel,
  "player-hub": PlayerHubPanel,
  "power-rankings": PowerRankingsPanel,
  standings: StandingsPanel,
  "track-record": TrackRecordPanel,
} as const;

/**
 * One Data Hub panel per route. The page tabs in AppFrame are the only navigation:
 * the panel is chosen by the route (`initialTab`), never by a second tab row here.
 */
export default function DataHubPage({ initialTab = "team-hub" }: { initialTab?: keyof typeof PANELS }) {
  const Panel = PANELS[initialTab];
  return <Panel />;
}
