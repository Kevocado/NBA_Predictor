import { useEffect, useState } from "react";
import { api, type Manifest, type PlayerPropsManifest } from "../api/client";

const FALLBACK_COPY = "Training data not published yet — it appears after the next retrain.";

export default function ModelSummaryPage() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [propsManifest, setPropsManifest] = useState<PlayerPropsManifest | null>(null);
  const [notTrained, setNotTrained] = useState(false);

  useEffect(() => {
    api
      .getManifest()
      .then(setManifest)
      .catch(() => setNotTrained(true));
    api
      .getPlayerPropsManifest()
      .then(setPropsManifest)
      .catch(() => setPropsManifest(null));
  }, []);

  if (notTrained) return <p>No model has been trained yet.</p>;
  if (!manifest) return <p>Loading model summary…</p>;

  const training = manifest.training;
  const propsTraining = propsManifest?.training;

  return (
    <div>
      <p className="mb-1 text-sm text-[var(--color-net-dim)]">Model version</p>
      <p className="mb-4 font-mono tracking-tight">{manifest.model_version}</p>
      <p className="mb-1 text-sm text-[var(--color-net-dim)]">Trained at</p>
      <p className="mb-4">{manifest.trained_at}</p>

      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-[var(--color-net-faint)]">
            <th>Model</th>
            <th>Metrics</th>
          </tr>
        </thead>
        <tbody>
          {manifest.models.map((modelName) => (
            <tr key={modelName}>
              <td>{modelName}</td>
              <td className="flex gap-3">
                {Object.entries(manifest.metrics[modelName] ?? {}).map(([key, value]) => (
                  <span key={key}>
                    {key}: <span>{value}</span>
                  </span>
                ))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2 className="mb-2 mt-6 text-lg font-semibold">Training data</h2>
      {training?.n_train_games != null ? (
        <div className="text-sm">
          <p className="mb-1">
            <span className="text-[var(--color-net-dim)]">Training matches: </span>
            <span>{training.n_train_games.toLocaleString()}</span>
          </p>
          {training.n_current_season_games != null && (
            <p className="mb-1">
              <span className="text-[var(--color-net-dim)]">Current-season matches: </span>
              <span>{training.n_current_season_games.toLocaleString()}</span>
            </p>
          )}
          {training.n_holdout_games != null && (
            <p className="mb-1">
              <span className="text-[var(--color-net-dim)]">Held-out validation matches: </span>
              <span>{training.n_holdout_games.toLocaleString()}</span>
            </p>
          )}
          <p className="mt-2 text-[var(--color-net-dim)]">
            Current-season matches count only once their stats are final and the model has
            retrained.
          </p>
        </div>
      ) : (
        <p className="text-sm text-[var(--color-net-dim)]">{FALLBACK_COPY}</p>
      )}

      <h3 className="mb-2 mt-4 text-base font-semibold">Player props</h3>
      {propsTraining?.n_train_player_games != null ? (
        <p className="text-sm">
          <span className="text-[var(--color-net-dim)]">Player-games used: </span>
          <span>{propsTraining.n_train_player_games.toLocaleString()}</span>
          {propsTraining.in_sample_metrics && (
            <span className="text-[var(--color-net-dim)]"> (in-sample)</span>
          )}
        </p>
      ) : (
        <p className="text-sm text-[var(--color-net-dim)]">{FALLBACK_COPY}</p>
      )}
    </div>
  );
}