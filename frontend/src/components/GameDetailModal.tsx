import { useEffect, useState } from "react";
import { api, type GameDetail, type PlayerProp, type MarketPrediction } from "../api/client";

interface GameDetailModalProps {
  gameId: string;
  onClose: () => void;
}

interface FavoredTeam {
  team: string;
  value: number;
}

function favoredTeam(margin: number, homeTeam: string, awayTeam: string): FavoredTeam {
  return margin >= 0 ? { team: homeTeam, value: margin } : { team: awayTeam, value: -margin };
}

interface PostMatchVerdict {
  winnerCorrect: boolean;
  predictedMargin: FavoredTeam;
  actualMargin: FavoredTeam;
  marginDiff: number;
  predictedTotal: number;
  actualTotal: number;
  totalDiff: number;
}

export function computePostMatchVerdict(detail: GameDetail): PostMatchVerdict | null {
  if (!detail.completed || !detail.prediction || detail.home_pts === null || detail.away_pts === null) {
    return null;
  }
  const actualMarginValue = detail.home_pts - detail.away_pts;
  const actualTotal = detail.home_pts + detail.away_pts;
  return {
    winnerCorrect: detail.prediction.home_win_probability >= 0.5 === actualMarginValue > 0,
    predictedMargin: favoredTeam(detail.prediction.predicted_margin, detail.home_team, detail.away_team),
    actualMargin: favoredTeam(actualMarginValue, detail.home_team, detail.away_team),
    marginDiff: Math.abs(detail.prediction.predicted_margin - actualMarginValue),
    predictedTotal: detail.prediction.predicted_total,
    actualTotal,
    totalDiff: Math.abs(detail.prediction.predicted_total - actualTotal),
  };
}

export function marketVerdict(market: MarketPrediction, detail: GameDetail): boolean | null {
  if (!detail.completed || detail.home_pts === null || detail.away_pts === null) return null;

  if (market.market === "h2h") {
    const actualWinner = detail.home_pts > detail.away_pts ? detail.home_team : detail.away_team;
    return market.selection === actualWinner;
  }
  if (market.market === "spread") {
    if (market.point === null) return null;
    const teamMargin =
      market.selection === detail.home_team
        ? detail.home_pts - detail.away_pts
        : detail.away_pts - detail.home_pts;
    return teamMargin > -market.point;
  }
  if (market.market === "total") {
    if (market.point === null) return null;
    const actualTotal = detail.home_pts + detail.away_pts;
    return market.selection === "over" ? actualTotal > market.point : actualTotal < market.point;
  }
  return null;
}

function FormBadge({ result }: { result: string }) {
  return (
    <span
      data-testid="form-badge"
      className={
        result === "W"
          ? "inline-flex h-5 w-5 items-center justify-center rounded-full bg-[var(--color-win)]/20 text-xs text-[var(--color-win)]"
          : "inline-flex h-5 w-5 items-center justify-center rounded-full bg-[var(--color-shotclock)]/20 text-xs text-[var(--color-shotclock)]"
      }
    >
      {result}
    </span>
  );
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

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const sortedMarkets = detail ? [...detail.markets].sort((a, b) => (b.edge ?? 0) - (a.edge ?? 0)) : [];
  const verdict = detail ? computePostMatchVerdict(detail) : null;

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

        {detail?.completed ? (
          <div className="mb-5 flex items-center justify-center gap-6 border-b border-[var(--color-line)] pb-5">
            <div className="text-center">
              <div className="stat-display text-3xl leading-none">{detail.away_pts}</div>
              <div className="mt-1 text-xs text-[var(--color-net-faint)]">{detail.away_team}</div>
            </div>
            <div className="text-xs uppercase text-[var(--color-net-faint)]">Final</div>
            <div className="text-center">
              <div className="stat-display text-3xl leading-none">{detail.home_pts}</div>
              <div className="mt-1 text-xs text-[var(--color-net-faint)]">{detail.home_team}</div>
            </div>
          </div>
        ) : (
          detail?.prediction && (
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
          )
        )}

        {verdict && (
          <div className="mb-5 border-b border-[var(--color-line)] pb-5 text-sm" data-testid="post-match-verdict">
            <div className="mb-2 flex items-center gap-2">
              <span className={verdict.winnerCorrect ? "text-[var(--color-win)]" : "text-[var(--color-shotclock)]"}>
                {verdict.winnerCorrect ? "✓ Correct" : "✗ Incorrect"}
              </span>
              <span className="text-[var(--color-net-faint)]">winner call</span>
            </div>
            <div className="text-[var(--color-net-faint)]">
              Predicted margin: {verdict.predictedMargin.team} +{verdict.predictedMargin.value.toFixed(1)} —{" "}
              Actual: {verdict.actualMargin.team} +{verdict.actualMargin.value.toFixed(1)} (off by {verdict.marginDiff.toFixed(1)})
            </div>
            <div className="text-[var(--color-net-faint)]">
              Predicted total: {verdict.predictedTotal.toFixed(1)} — Actual: {verdict.actualTotal} (off by {verdict.totalDiff.toFixed(1)})
            </div>
          </div>
        )}

        {detail && (detail.home_recent_form.length > 0 || detail.away_recent_form.length > 0) && (
          <div className="mb-5 flex items-center justify-between border-b border-[var(--color-line)] pb-5 text-sm">
            <div>
              <div className="mb-1 text-xs text-[var(--color-net-faint)]">{detail.away_team} form</div>
              <div className="flex gap-1">
                {detail.away_recent_form.map((r, i) => (
                  <FormBadge key={i} result={r} />
                ))}
              </div>
            </div>
            <div className="text-right">
              <div className="mb-1 text-xs text-[var(--color-net-faint)]">{detail.home_team} form</div>
              <div className="flex justify-end gap-1">
                {detail.home_recent_form.map((r, i) => (
                  <FormBadge key={i} result={r} />
                ))}
              </div>
            </div>
          </div>
        )}

        {sortedMarkets.length > 0 && (
          <table className="mb-4 w-full text-sm">
            <thead>
              <tr className="text-left text-[var(--color-net-faint)]">
                <th>Market</th>
                <th>Selection</th>
                <th>Line</th>
                <th>Bookmaker</th>
                <th>Odds</th>
                <th>Model %</th>
                <th>Market %</th>
                <th>Edge</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {sortedMarkets.map((market, i) => {
                const verdict = detail ? marketVerdict(market, detail) : null;
                return (
                  <tr key={i} data-testid="market-row">
                  <td>{market.market}</td>
                  <td>{market.selection}</td>
                  <td>{market.point !== null ? market.point : "—"}</td>
                  <td>{market.bookmaker ?? "—"}</td>
                  <td>{market.american_odds !== null ? market.american_odds : "—"}</td>
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
                  <td>
                    {verdict === null ? (
                      "—"
                    ) : verdict ? (
                      <span className="text-[var(--color-win)]">✓</span>
                    ) : (
                      <span className="text-[var(--color-shotclock)]">✗</span>
                    )}
                  </td>
                </tr>
              );
              })}
            </tbody>
          </table>
        )}

        {detail && detail.head_to_head.length > 0 && (
          <div className="mb-4">
            <h3 className="mb-2 text-sm text-[var(--color-net-faint)]">Head to head</h3>
            <ul className="text-sm">
              {detail.head_to_head.map((meeting) => (
                <li key={meeting.game_id} className="flex justify-between border-b border-[var(--color-line)] py-1">
                  <span>{meeting.game_date}</span>
                  <span>
                    {meeting.away_team} {meeting.away_pts} – {meeting.home_pts} {meeting.home_team}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {players && players.length > 0 && (
          <ul className="text-sm">
            {players.map((player, i) => (
              <li key={i} className="flex justify-between border-b border-[var(--color-line)] py-1">
                <span>{player.player_name}</span>
                <span>
                  <span>{player.stat}</span>:{" "}
                  {player.actual_value !== null ? (
                    <>
                      Predicted: {player.predicted_value} — Actual: {player.actual_value} (off by{" "}
                      {Math.abs(player.predicted_value - player.actual_value).toFixed(1)})
                    </>
                  ) : (
                    <span>{player.predicted_value}</span>
                  )}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
