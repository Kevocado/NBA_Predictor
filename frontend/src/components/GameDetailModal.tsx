import { useEffect, useState } from "react";
import { api, type GameDetail, type PlayerProp, type MarketPrediction } from "../api/client";
import { favourite } from "../lib/pick";
import { teamName } from "../lib/teams";
import { kickoff, pct, statusWords } from "../predictor-ui";

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
  // A pick rebuilt after tip-off is shown for reference but never judged.
  if (!detail.completed || !detail.prediction || detail.rebuilt || detail.home_pts === null || detail.away_pts === null) {
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

/** What the model said before tip-off, and whether it counts. */
function PregamePick({ detail }: { detail: GameDetail }) {
  const p = detail.prediction;
  const line = !p
    ? "No pick was made before tip-off."
    : (() => {
        const fav = favourite(p, detail.home_team, detail.away_team);
        return detail.rebuilt
          ? `Rebuilt after tip-off: ${fav.team} · ${pct(fav.prob)}. Not counted in the record.`
          : `Pick before tip-off: ${fav.team} · ${pct(fav.prob)}`;
      })();
  return <p className="mb-3 text-sm font-semibold text-pr-text">{line}</p>;
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
  const pickFav = detail?.prediction ? favourite(detail.prediction, detail.home_team, detail.away_team) : null;
  const marginFav = detail?.prediction ? favoredTeam(detail.prediction.predicted_margin, detail.home_team, detail.away_team) : null;
  // The win and margin numbers come from separate models. When they point at
  // different teams, or the margin rounds to nothing, don't print a
  // contradiction ("BOS to win" beside "MIA by 0.6") or "BOS by 0.0".
  const marginLabel = marginFav && pickFav && marginFav.team === pickFav.team && marginFav.value >= 0.5
    ? `${marginFav.team} by ${marginFav.value.toFixed(1)}`
    : "Toss-up";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70" onClick={onClose}>
      <div
        className="max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded border border-[var(--color-line)] bg-[var(--color-court-900)] p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold uppercase tracking-wide">
              {detail ? `${teamName(detail.away_team)} at ${teamName(detail.home_team)}` : "Game detail"}
            </h2>
            {detail && <p className="text-xs text-pr-text-dim">{kickoff(detail.tip_off ?? detail.game_date)}</p>}
          </div>
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
                  {Math.round((pickFav?.prob ?? 0) * 100)}%
                </div>
                <div className="mt-1 text-xs text-[var(--color-net-dim)]">{pickFav?.team} to win</div>
              </div>
              <div>
                <div className="stat-display text-2xl leading-none">
                  {marginLabel}
                </div>
                <div className="mt-1 text-xs text-[var(--color-net-dim)]">Projected margin</div>
              </div>
              <div>
                <div className="stat-display text-2xl leading-none">{detail.prediction.predicted_total.toFixed(1)}</div>
                <div className="mt-1 text-xs text-[var(--color-net-dim)]">Projected total points</div>
              </div>
            </div>
          )
        )}

        {detail?.completed && <PregamePick detail={detail} />}

        {verdict && (
          <div className="mb-5 border-b border-[var(--color-line)] pb-5 text-sm" data-testid="post-match-verdict">
            <div className="mb-2 flex items-center gap-2">
              <span className={`font-pr-display font-semibold uppercase tracking-wide ${verdict.winnerCorrect ? "text-pr-win" : "text-pr-loss"}`}>
                {statusWords(verdict.winnerCorrect ? "called" : "missed")}
              </span>
              <span className="text-pr-text-dim">winner pick</span>
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
          <div className="overflow-x-auto">
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
                    <td>{pct(market.model_probability)}</td>
                    <td>{market.market_probability !== null ? pct(market.market_probability) : "—"}</td>
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
          </div>
        )}

        {detail && detail.head_to_head.length > 0 && (
          <div className="mb-4">
            <h3 className="mb-2 text-sm text-[var(--color-net-faint)]">Head to head</h3>
            <ul className="text-sm">
              {detail.head_to_head.map((meeting) => (
                <li key={meeting.game_id} className="flex justify-between border-b border-[var(--color-line)] py-1">
                  <span>{kickoff(meeting.game_date)}</span>
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
