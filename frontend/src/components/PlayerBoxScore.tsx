import { useMemo } from "react";
import type { PlayerProp, PlayerHubRow } from "../api/client";
import {
  STAT_KEYS,
  STAT_LABELS,
  attachHubRows,
  hasActuals,
  pivotProps,
  splitByTeam,
  sumStat,
  type BoxScoreRow,
  type StatKey,
} from "../lib/boxScore";
import { teamName } from "../lib/teams";
import { stat } from "../predictor-ui";

interface PlayerBoxScoreProps {
  playerProps: PlayerProp[];
  hubPlayers: PlayerHubRow[];
  homeTeam: string;
  awayTeam: string;
}

function StatCell({ row, stat: key }: { row: BoxScoreRow; stat: StatKey }) {
  const predicted = row.predicted[key];
  const actual = row.actual[key];
  // A prop rebuilt after tip-off is shown, never judged -- so no delta, which
  // would be a score against a projection made with the result already known.
  const showDelta = actual !== undefined && predicted !== undefined && !row.rebuilt;
  return (
    <td className="box-score-cell">
      {predicted === undefined ? (
        <span className="text-pr-text-dim">—</span>
      ) : (
        <span className="stat-display tabular-nums">{stat(predicted)}</span>
      )}
      {actual !== undefined && (
        <span
          className="box-score-actual tabular-nums"
          title={showDelta ? "Actual, and how far off the prediction was" : "Actual"}
        >
          {actual}
          {showDelta && (
            <>
              {" "}({actual - predicted > 0 ? "+" : ""}
              {(actual - predicted).toFixed(1)})
            </>
          )}
        </span>
      )}
    </td>
  );
}

function SideTotals({ rows }: { rows: BoxScoreRow[] }) {
  return (
    <>
      {STAT_KEYS.map((key) => (
        <td key={key} className="box-score-cell box-score-total">
          <span className="stat-display tabular-nums">{stat(sumStat(rows, key, "predicted"))}</span>
        </td>
      ))}
    </>
  );
}

function ActualTotals({ rows }: { rows: BoxScoreRow[] }) {
  // Only shown when the game is played and the API actually reported numbers.
  if (!hasActuals(rows)) return null;
  return (
    <>
      {STAT_KEYS.map((key) => (
        <td key={key} className="box-score-cell box-score-total">
          <span className="box-score-actual tabular-nums">{sumStat(rows, key, "actual")}</span>
        </td>
      ))}
    </>
  );
}

/**
 * The predicted box score, split down the middle by team.
 *
 * A real `<table>`, not a grid of divs: two `th scope="row"` in a row is valid
 * and is what makes a screen reader read the row as "MIA player, 16.5, ...".
 *
 * The scroll contract is deliberate and the thing most likely to be got wrong:
 * ONE container scrolls on both axes, with a bounded height so a 22-player
 * game does not grow the card. Sticky headers keep the column labels and both
 * team totals in view while it scrolls, which is what a reader needs to
 * compare the two sides of a split table.
 */
export default function PlayerBoxScore({ playerProps, hubPlayers, homeTeam, awayTeam }: PlayerBoxScoreProps) {
  const { away, home, unattributed } = useMemo(
    () => splitByTeam(attachHubRows(pivotProps(playerProps), hubPlayers), homeTeam, awayTeam),
    [playerProps, hubPlayers, homeTeam, awayTeam],
  );

  // Two sides of different lengths: pad the short one so the taller side keeps
  // its place and the totals row sits under both.
  const depth = Math.max(away.length, home.length);
  const rows: Array<{ away: BoxScoreRow | null; home: BoxScoreRow | null }> = Array.from(
    { length: depth },
    (_, i) => ({ away: away[i] ?? null, home: home[i] ?? null }),
  );

  if (depth === 0) return null;

  return (
    <div className="mt-5" data-testid="player-box-score">
      <div className="box-score-scroll" data-testid="box-score-scroll">
        <table className="box-score-table">
          <colgroup>
            <col className="box-score-col-name" />
            {STAT_KEYS.map((key) => (
              <col key={`a-${key}`} className="box-score-col-stat" />
            ))}
            <col className="box-score-col-divider" />
            {STAT_KEYS.map((key) => (
              <col key={`h-${key}`} className="box-score-col-stat" />
            ))}
            <col className="box-score-col-name" />
          </colgroup>
          <thead>
            <tr>
              <th scope="col" colSpan={5} className="box-score-team-head">
                {teamName(awayTeam)}
              </th>
              <th scope="col" className="box-score-divider" aria-hidden="true" />
              <th scope="col" colSpan={5} className="box-score-team-head box-score-team-head--home">
                {teamName(homeTeam)}
              </th>
            </tr>
            <tr>
              <th scope="col" className="box-score-col-head box-score-col-head--away">
                Player
              </th>
              {STAT_KEYS.map((key) => (
                <th key={`a-${key}`} scope="col" className="box-score-col-head">
                  {STAT_LABELS[key]}
                </th>
              ))}
              <th scope="col" className="box-score-divider" aria-hidden="true" />
              {STAT_KEYS.map((key) => (
                <th key={`h-${key}`} scope="col" className="box-score-col-head">
                  {STAT_LABELS[key]}
                </th>
              ))}
              <th scope="col" className="box-score-col-head box-score-col-head--home">
                Player
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ away: a, home: h }, i) => (
              <tr key={a?.player_id ?? h?.player_id ?? `gap-${i}`} data-testid="box-score-row">
                <th scope="row" className="box-score-name box-score-name--away">
                  {a ? a.player_name : ""}
                  {a?.rebuilt && (
                    <span className="box-score-rebuilt" title="Built after tip-off. Shown, never judged.">
                      Rebuilt
                    </span>
                  )}
                </th>
                {a ? (
                  STAT_KEYS.map((key) => <StatCell key={key} row={a} stat={key} />)
                ) : (
                  <td className="box-score-cell" colSpan={4} />
                )}
                <td className="box-score-divider" aria-hidden="true" />
                {h ? (
                  STAT_KEYS.map((key) => <StatCell key={key} row={h} stat={key} />)
                ) : (
                  <td className="box-score-cell" colSpan={4} />
                )}
                <th scope="row" className="box-score-name box-score-name--home">
                  {h ? h.player_name : ""}
                  {h?.rebuilt && (
                    <span className="box-score-rebuilt" title="Built after tip-off. Shown, never judged.">
                      Rebuilt
                    </span>
                  )}
                </th>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr data-testid="box-score-totals">
              <th scope="row" className="box-score-name box-score-name--away">
                Total
              </th>
              <SideTotals rows={away} />
              <td className="box-score-divider" aria-hidden="true" />
              <SideTotals rows={home} />
              <th scope="row" className="box-score-name box-score-name--home">
                Total
              </th>
            </tr>
            {hasActuals(away) && (
              <tr data-testid="box-score-actual-totals">
                <th scope="row" className="box-score-name box-score-name--away">
                  Actual
                </th>
                <ActualTotals rows={away} />
                <td className="box-score-divider" aria-hidden="true" />
                <ActualTotals rows={home} />
                <th scope="row" className="box-score-name box-score-name--home">
                  Actual
                </th>
              </tr>
            )}
          </tfoot>
        </table>
      </div>
      {unattributed.length > 0 && (
        <p className="mt-2 text-xs text-pr-text-dim">
          {unattributed.length} player{unattributed.length === 1 ? "" : "s"} could not be matched to{" "}
          {teamName(awayTeam)} or {teamName(homeTeam)} and {unattributed.length === 1 ? "is" : "are"} left out
          rather than guessed onto a side.
        </p>
      )}
    </div>
  );
}
