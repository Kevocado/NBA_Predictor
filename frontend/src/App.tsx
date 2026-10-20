import { Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { AppFrame } from "./predictor-ui";
import { SITES } from "./lib/sites";
import GamesPage from "./pages/GamesPage";
import DataHubPage from "./pages/DataHubPage";
import ModelSummaryPage from "./pages/ModelSummaryPage";
import CalibrationPage from "./pages/CalibrationPage";

// Page tabs map onto the existing routes, so deep links and refreshes keep working.
// The aggregate "Data Hub" tab is dropped; its panels each have their own route.
// Teams lives at /hub (the old Data Hub URL): /teams is an API route, so a refresh or shared
// link to it would return JSON instead of this app.
const TABS = [
  { id: "/", label: "Games" },
  { id: "/hub", label: "Teams" },
  { id: "/players", label: "Players" },
  { id: "/power-rankings", label: "Power rankings" },
  { id: "/standings", label: "Standings" },
  { id: "/track-record", label: "Track record" },
  { id: "/calibration-report", label: "Calibration" },
  { id: "/model", label: "Model" },
];

export default function App() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const active = TABS.find((tab) => tab.id !== "/" && pathname.startsWith(tab.id))?.id ?? "/";

  return (
    <AppFrame
      sport="nba"
      sportName="NBA"
      sites={SITES}
      tabs={TABS}
      activeTab={active}
      paperTradingHref="https://40-160-91-131.sslip.io/paper.html"
      onTab={(id) => navigate(id)}
    >
      <Routes>
        <Route path="/" element={<GamesPage />} />
        <Route path="/hub" element={<DataHubPage initialTab="team-hub" />} />
        <Route path="/players" element={<DataHubPage initialTab="player-hub" />} />
        <Route path="/power-rankings" element={<DataHubPage initialTab="power-rankings" />} />
        <Route path="/standings" element={<DataHubPage initialTab="standings" />} />
        <Route path="/track-record" element={<DataHubPage initialTab="track-record" />} />
        <Route path="/model" element={<ModelSummaryPage />} />
        <Route path="/calibration-report" element={<CalibrationPage />} />
      </Routes>
    </AppFrame>
  );
}
