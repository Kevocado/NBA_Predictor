import { useEffect, useState } from "react";
import { api, type Manifest } from "../api/client";

export default function ModelSummaryPage() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [notTrained, setNotTrained] = useState(false);

  useEffect(() => {
    api
      .getManifest()
      .then(setManifest)
      .catch(() => setNotTrained(true));
  }, []);

  if (notTrained) return <p>No model has been trained yet.</p>;
  if (!manifest) return <p>Loading model summary…</p>;

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
    </div>
  );
}