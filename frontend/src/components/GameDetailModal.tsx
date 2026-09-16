import { useEffect, useState } from "react";
import { api, type GameDetail, type PlayerProp } from "../api/client";

interface GameDetailModalProps {
  gameId: string;
  onClose: () => void;
}

export default function GameDetailModal({ gameId, onClose }: GameDetailModalProps) {
  const [detail, setDetail] = useState<GameDetail | null>(null);
  const [players, setPlayers] = useState<PlayerProp[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDetail(null);
    setPlayers(null);
    setError(null);
    Promise.all([api.getGameDetail(gameId), api.getGamePlayers(gameId)])
      .then(([detailResult, playersResult]) => {
        setDetail(detailResult);
        setPlayers(playersResult);
      })
      .catch(() => setError("Couldn't load game details."));
  }, [gameId]);

  const sortedMarkets = detail ? [...detail.markets].sort((a, b) => (b.edge ?? 0) - (a.edge ?? 0)) : [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70" onClick={onClose}>
      <div
        className="max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded border border-[var(--color-line)] bg-[var(--color-court-900)] p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-bold">{detail ? `${detail.away_team} at ${detail.home_team}` : "Game detail"}</h2>
          <button aria-label="Close" onClick={onClose} className="text-xl leading-none text-[var(--color-net-dim)]">
            ×
          </button>
        </div>

        {error && <p className="text-[var(--color-shotclock)]">{error}</p>}
        {!error && !detail && <p>Loading…</p>}

        {detail?.prediction && (
          <div className="mb-5 grid grid-cols-3 gap-4 border-b border-[var(--color-line)] pb-5">
            <div>
              <div className="stat-display text-2xl leading-none text-[var(--color-hardwood-bright)]">
                {Math.round(detail.prediction.home_win_probability * 100)}%
              </div>
              <div className="mt-1 text-xs text-[var(--color-net-faint)]">{detail.home_team} win probability</div>
            </div>
            <div>
              <div className="stat-display text-2xl leading-none">{detail.prediction.predicted_margin.toFixed(1)}</div>
              <div className="mt-1 text-xs text-[var(--color-net-faint)]">Predicted margin</div>
            </div>
            <div>
              <div className="stat-display text-2xl leading-none">{detail.prediction.predicted_total.toFixed(1)}</div>
              <div className="mt-1 text-xs text-[var(--color-net-faint)]">Predicted total</div>
            </div>
          </div>
        )}

        {sortedMarkets.length > 0 && (
          <table className="mb-4 w-full text-sm">
            <thead>
              <tr className="text-left text-[var(--color-net-faint)]">
                <th>Market</th>
                <th>Selection</th>
                <th>Model %</th>
                <th>Market %</th>
                <th>Edge</th>
              </tr>
            </thead>
            <tbody>
              {sortedMarkets.map((market, i) => (
                <tr key={i} data-testid="market-row">
                  <td>{market.market}</td>
                  <td>{market.selection}</td>
                  <td>{Math.round(market.model_probability * 100)}%</td>
                  <td>{market.market_probability !== null ? `${Math.round(market.market_probability * 100)}%` : "—"}</td>
                  <td
                    className={
                      market.edge === null
                        ? undefined
                        : market.edge > 0
                          ? "text-[var(--color-win)]"
                          : "text-[var(--color-shotclock)]"
                    }
                  >
                    {market.edge !== null ? `${(market.edge * 100).toFixed(1)}pp` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {players && players.length > 0 && (
          <ul className="text-sm">
            {players.map((player, i) => (
              <li key={i} className="flex justify-between border-b border-[var(--color-line)] py-1">
                <span>{player.player_name}</span>
                <span>
                  <span>{player.stat}</span>: <span>{player.predicted_value}</span>
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}