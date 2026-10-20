import { useEffect, useState } from "react";
import { api } from "../api/client";
import { earlySeasonNotice } from "../lib/earlySeason";

/** Says so while the model is young; renders nothing mid-season or when the manifest cannot be read. */
export default function EarlySeasonNotice() {
  const [text, setText] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    Promise.resolve()
      .then(() => api.getManifest())
      .then((m) => { if (!cancelled) setText(earlySeasonNotice(m.training?.n_current_season_games)); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);
  if (!text) return null;
  return (
    <p role="note" className="mb-4 max-w-prose rounded-pr border border-pr-rule bg-pr-panel px-4 py-3 text-sm text-pr-text-dim">
      {text}
    </p>
  );
}
