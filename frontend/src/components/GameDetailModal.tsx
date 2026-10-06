import { useEffect, useId, useMemo, useRef, useState } from "react";
import { api, type GameDetail, type OutPlayer, type PlayerProp, type PlayerHubRow, type MarketPrediction, type TrackRecord } from "../api/client";
import TopCalls from "./TopCalls";
import { favourite } from "../lib/pick";
import { teamName } from "../lib/teams";
import { ErrorState, FixtureExplainer, SignalRows, Skeleton, kickoff, pct, signalIsDrawn, stat, statusWords, type Signal } from "../predictor-ui";
import { panelFacts } from "../predictor-ui/lib/panelFacts";
import PlayerBoxScore from "./PlayerBoxScore";

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

  // Priced after tip-off: shown for reference, never judged.
  if (market.rebuilt) return null;
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
  // Who the availability gate removed from this game's ranking. `null` is a
  // third state on purpose, and it is not "nobody is out": while it is loading,
  // and after a failure, the ranking is not shown at all. A ranking whose
  // exclusions could not be read is a ranking nobody can vouch for, and the API
  // answers an unreadable feed with a 503 rather than an empty report precisely
  // so the two cases stay apart. Showing the top three of a list we cannot prove
  // is complete would be the dishonest reading, not the cautious one.
  const [outPlayers, setOutPlayers] = useState<OutPlayer[] | null>(null);
  // Spec §4's signal rows. `[]` and never null once settled -- and `[]` is also
  // what every failure leaves behind, see the effect below.
  const [signals, setSignals] = useState<Signal[]>([]);
  // The rows that will ACTUALLY render, computed by the same exported predicate
  // `SignalRows` filters with. Without this the section mounted on the raw count
  // and rendered an empty band whenever every row was undrawable; see the block's
  // comment. `useMemo` because `signalIsDrawn` walks every row on every render.
  const drawnSignals = useMemo(() => signals.filter((s) => signalIsDrawn(s)), [signals]);
  // The per-game player feed carries no team, so the box score's split comes
  // from the season hub feed. Fetched separately, and allowed to fail: losing
  // the split must not take the game detail down with it.
  const [hubPlayers, setHubPlayers] = useState<PlayerHubRow[]>([]);
  // The season's winner-pick record, for the block's record strip. Fetched once
  // with the modal and allowed to fail, like the hub feed above: losing the
  // record must not take the game detail down, and a record that cannot be read
  // is better absent than wrong. `null` while loading and after a failure, so
  // the block is passed no record at all rather than an empty one.
  const [trackRecord, setTrackRecord] = useState<TrackRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const titleId = useId();
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    setDetail(null);
    setPlayers(null);
    setOutPlayers(null);
    setError(null);
    Promise.all([api.getGameDetail(gameId), api.getGamePlayers(gameId)])
      .then(([detailResult, playersResult]) => {
        setDetail(detailResult);
        setPlayers(playersResult);
      })
      .catch(() => setError("We couldn't load this game. Check your connection and try again."));
  }, [gameId, reloadKey]);

  // The signal rows (spec §4). Its own fetch, allowed to fail on its own, and
  // GATED ON `detail.game_id === gameId` -- the same shape PL's `FixtureModal` uses,
  // and for PL's reason: this modal keeps the PREVIOUS game in state until the new
  // one lands (the reset is a `setDetail(null)` inside the effect above, and an
  // effect reads the value captured by ITS OWN render), so a `!detail` or
  // `detail?.game_status` gate would read the OLD game's status to decide about the
  // new one.
  //
  // A COMPLETED game is not asked about at all. The endpoint answers `[]` for one,
  // because a completed game's roster is the one known BEFORE tip-off -- the same
  // reason `get_game_players` withholds those rows and the same reason this page's
  // own `outPlayers` list stops being information. Asking for an answer already known
  // is a request per game for nothing.
  //
  // **`completed` AND NOT "has started".** The endpoint's own rule is
  // `facts._status(game, now) in ("live", "final")`, and `completed` is only the
  // `final` half of it: a LIVE game has `completed=False` with a tip-off in the
  // past. So a live game is still asked, and the endpoint answers `[]`. That is
  // deliberate -- reproducing `_status` on the client means comparing `tip_off`
  // against the browser's clock, which duplicates the server's rule on the wrong
  // side of a network boundary for the sake of one avoided request. One extra
  // request per live game is cheaper than a status rule that can disagree with the
  // endpoint it is mirroring.
  //
  // EVERY failure is silence, and the failures are deliberately indistinguishable:
  // a 404, a network error and an honest empty list all leave `[]`. Spec §2 forbids a
  // placeholder, and a signal is an enhancement on this page -- it must never become
  // the page's error state. Sports' `GameDetailModal` does the same at the same
  // place, so this is one implementation and not one per sport.
  useEffect(() => {
    let cancelled = false;
    setSignals([]);
    if (!detail || detail.game_id !== gameId) return;
    if (detail.completed) return;
    api
      .getGameSignals(gameId)
      .then((response) => {
        if (!cancelled) setSignals(response?.signals ?? []);
      })
      // Swallow only. The `[]` at this effect's head already covers every re-run, so
      // repeating it here would be a second place to keep the same rule -- PL's
      // proved that redundant, by passing its whole suite with the clear deleted.
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [gameId, detail, reloadKey]);

  // Its own fetch, allowed to fail on its own. Deliberately NOT folded into the
  // Promise.all above: that one failing would take the whole game detail down,
  // and losing the exclusion list must not cost the reader the box score.
  useEffect(() => {
    let cancelled = false;
    api
      .getGameOutPlayers(gameId)
      .then((result) => {
        if (!cancelled) setOutPlayers(result);
      })
      .catch(() => {
        if (!cancelled) setOutPlayers(null);
      });
    return () => {
      cancelled = true;
    };
  }, [gameId, reloadKey]);

  useEffect(() => {
    api.getHubPlayers()
      .then(setHubPlayers)
      .catch(() => setHubPlayers([]));
  }, []);

  useEffect(() => {
    let cancelled = false;
    api
      .getTrackRecord()
      .then((rows) => { if (!cancelled) setTrackRecord(rows); })
      .catch(() => { if (!cancelled) setTrackRecord([]); });
    return () => { cancelled = true; };
  }, []);

  // Focus moves into the dialog on open, and back to the card on close.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    return () => opener?.focus?.();
  }, []);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const sortedMarkets = detail ? [...detail.markets].sort((a, b) => (b.edge ?? 0) - (a.edge ?? 0)) : [];
  const verdict = detail ? computePostMatchVerdict(detail) : null;
  // The flow's facts: what this modal already holds, no request. The away
  // probability is the site's own convention (`favourite` derives the pair the
  // same way for the pick it prints), and the game carries no market line --
  // 142 of 142 bundles have none -- so the flow says the pick and stops. That
  // thinness is data, not a defect in the panel.
  const finite = (x: unknown): number | undefined =>
    typeof x === "number" && Number.isFinite(x) ? x : undefined;
  const flowBundle = useMemo(() => {
    if (!detail) return null;
    const hw = finite(detail.prediction?.home_win_probability);
    const aw = hw !== undefined ? 1 - hw : undefined;
    const fav = hw !== undefined && detail.prediction ? favourite(detail.prediction, detail.home_team, detail.away_team) : null;
    const wasRight = verdict ? verdict.winnerCorrect : undefined;
    const score =
      detail.home_pts != null && detail.away_pts != null
        ? { home: detail.home_pts, away: detail.away_pts }
        : undefined;
    /* THE BARE HEADING, and why the sides are spelled two ways here.
       Measured across the three states rather than assumed:
         - pre-game  — the flow's ONLY row was the fixture's own name, so the
                       live modal showed an `MIA vs TOR` heading with nothing
                       under it, directly above the AI button.
         - finished  — two REAL sentences (the result, and the pick's
                       rightness). These are live content and they stay.
       So this is not a deletion of the flow; it is the heading being
       conditional on rows existing beneath it, expressed in the only place a
       site may express it -- the bundle it hands over.

       `FixtureFlow` reads the sides under `home_team`/`away_team` for BOTH its
       pre-game name row AND its finished result sentence, so one spelling
       cannot suppress the first without breaking the second. `bundleFacts`
       (which is what the block's verdict sentence goes through) reads
       `home_team ?? team_home`, and both spellings are part of that package's
       documented contract -- its own header names PL's `team_home` and Sports'
       `home_team` as the two shapes it exists to bridge. So:

         - ALWAYS carry `team_home`/`team_away`, which the block's
           `fullTeamName` reads to expand "TOR" into "Raptors is the pick."
           Without this the block would fall back to the bare code.
         - carry `home_team`/`away_team` ONLY when the flow has a sentence that
           needs them -- i.e. once there is a score or a result. Pre-game there
           is no such sentence, and withholding the spelling the name row reads
           is what leaves the flow empty instead of bare.

       The block's figures do not depend on either key: `panelFacts` is handed
       this site's own game object, not the bundle, so the tile and the bar are
       unchanged by which spelling is present. */
    // The one condition the pre-game name row keys on, and nothing else: only
    // `home_team`/`away_team` make it exist, and only `in-play`/`finished` rows
    // need those keys to speak. Pre-game has neither, so withholding the keys
    // is what makes that row impossible rather than merely unlikely.
    const namesForSentences = Boolean(score) || detail.completed;
    return {
      team_home: detail.home_team,
      team_away: detail.away_team,
      ...(namesForSentences
        ? { home_team: detail.home_team, away_team: detail.away_team }
        : {}),
      // The codes are this modal's own vocabulary, and the title above already
      // spells them out with the same helper. The block's verdict sentence reads
      // far better as "Celtics is the pick." than as "BOS is the pick." — the
      // shared `verdictSentence` uses these when the pick matches a side, and
      // falls back to the code when the site does not carry a full name.
      // Spelled under BOTH keys for the reason above: the block reads whichever
      // alias `fullTeamName` resolves, and the flow's finished sentences read
      // the `home_team` one.
      team_home_full: teamName(detail.home_team),
      team_away_full: teamName(detail.away_team),
      home_team_full: teamName(detail.home_team),
      away_team_full: teamName(detail.away_team),
      home_win_prob: hw,
      away_win_prob: aw,
      pick: fav
        ? { label: fav.team, prob: fav.prob, ...(typeof wasRight === "boolean" ? { was_right: wasRight } : {}) }
        : undefined,
      pick_timing: detail.rebuilt ? "rebuilt" : undefined,
      score,
      result: !score
        ? undefined
        : score.home === score.away
          ? "draw"
          : score.home > score.away
            ? "home_win"
            : "away_win",
    };
  }, [detail, verdict]);
  // The panel's figures, from the SHARED adapter: moneyline segments for the
  // bar, and nothing else, because the game carries no market line for a tile.
  // The away probability is the site's own convention -- `favourite` derives
  // the pair the same way for the pick it prints -- so the bar this draws
  // agrees with the pick above it by construction, not by coincidence.
  const panel = useMemo(() => {
    if (!detail) return { tiles: [], segments: [], legend: undefined };
    const hw = finite(detail.prediction?.home_win_probability);
    return panelFacts({
      kind: "SP",
      game: { home_team: detail.home_team, away_team: detail.away_team, spread_line: null, total_line: null },
      prediction: detail.prediction
        ? { home_win_prob: hw ?? null, away_win_prob: hw !== undefined ? 1 - hw : null }
        : null,
    });
  }, [detail]);
  // The record the block may quote, or null when there is nothing true to say.
  //
  // `game_outcome` is the only row in `/hub/track-record` that settles the pick
  // the block just named: `hub_service._settle_game_outcome` grades
  // `home_win_prob >= 0.5` against the real result over `pre_tip_picks`, so
  // picks rebuilt after tip-off are already excluded and this row is a
  // pre-tip-only record by construction — which is exactly the record the spec
  // §F allows to be described as the record. `h2h` is a different claim (the
  // same probability measured against the bookmaker's price), so quoting it
  // here would state one pick two ways.
  //
  // No row, no record: the block is passed none and draws no strip, which is
  // the truthful "we have no record for this yet". A row that exists but has
  // graded nothing is a real record with no numbers in it, and `RecordStrip`
  // reads that as the dash. Neither path can print a 0/0.
  const winnerRecord = useMemo(() => {
    const row = trackRecord?.find((r) => r.market === "game_outcome");
    return row
      ? { label: "Winner pick made before tip-off", hits: row.correct_predictions, settled: row.total_predictions }
      : null;
  }, [trackRecord]);
  // The flow's state, derived ONCE and used for both the bundle and the prop,
  // because the bare-heading rule is a rule about the PAIR: the bundle withholds
  // `home_team` when the flow has nothing to say, and the prop has to be asking
  // the flow for that same nothing. Two separate derivations of "is this
  // finished?" is how the two drifted apart in the first place -- a live game
  // (`completed: false`, score present) took the PRE-GAME branch while the
  // bundle had already handed over the team names the pre-game name row reads,
  // which put the bare heading back on screen for exactly the games that have a
  // score to say.
  //
  // `in-play` is not decoration: `FixtureFlow`'s in-play rows are the score
  // sentence and (where a line exists) how it stands, and this site's data does
  // carry a partial score on an un-finished game. Those are real sentences, so
  // a live game asks for them rather than asking for a name.
  const flowState = !detail
    ? ("pre-game" as const)
    : detail.completed
      ? ("finished" as const)
      : detail.home_pts != null && detail.away_pts != null
        ? ("in-play" as const)
        : ("pre-game" as const);
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
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded border border-[var(--color-line)] bg-[var(--color-court-900)] p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 id={titleId} className="text-xl font-bold uppercase tracking-wide">
              {detail ? `${teamName(detail.away_team)} at ${teamName(detail.home_team)}` : "Game detail"}
            </h2>
            {detail && <p className="text-xs text-pr-text-dim">{kickoff(detail.tip_off ?? detail.game_date)}</p>}
          </div>
          <button ref={closeRef} aria-label="Close" onClick={onClose} className="text-xl leading-none text-[var(--color-net-dim)]">
            ×
          </button>
        </div>

        {error && <ErrorState message={error} onRetry={() => setReloadKey((k) => k + 1)} />}
        {!error && !detail && <Skeleton label="Loading…" />}

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
            /* THE FIELD-BY-FIELD AUDIT, and the reason this strip is two cells
               rather than three or none.
               This used to read `53% TOR to win | TOR by 4.8 | 230.6` ABOVE the
               instant block, so the pick probability was on the page three
               times: here, in the block's `moneyline` tile, and on the bar. Per
               the one-source rule the block owns the figures, so that cell is
               GONE -- the block's tile states the probability and which side it
               is, which is what this cell was saying in two pieces.

               The other two are KEPT, and the reason is measured rather than
               assumed: the game carries no market line (142 of 142 bundles), so
               `panelFacts` builds no spread tile and no total tile for it. The
               block therefore has no copy of the projected margin or the
               projected total, and this strip is their only source on the page.
               Deleting the strip wholesale would have removed two figures that
               exist nowhere else -- which is the opposite of the duplication
               rule, which is about a figure appearing twice, not once.

               Finished games do not render this strip at all (the Final block
               above owns that state, and the post-match review below states the
               prediction against what actually happened, which is a different
               set of figures entirely). */
            <div className="mb-5 grid grid-cols-2 gap-4 border-b border-[var(--color-line)] pb-5">
              <div>
                <div className="stat-display text-2xl leading-none">{marginLabel}</div>
                <div className="mt-1 text-xs text-[var(--color-net-dim)]">Projected margin</div>
              </div>
              <div>
                <div className="stat-display text-2xl leading-none">{detail.prediction.predicted_total.toFixed(1)}</div>
                <div className="mt-1 text-xs text-[var(--color-net-dim)]">Projected total points</div>
              </div>
            </div>
          )
        )}

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

        {detail && (
          <div className="mb-5 border-b border-[var(--color-line)] pb-5">
            {/* In plain English: the flow renders from facts this modal already
                holds, with no request; the AI summary sits behind the button
                and costs nothing until a reader asks. Thin by data, not by
                defect: the game carries no market line, so the flow says the
                pick and stops, and the bar below is the model's own split.

                The block above the button carries the same figures as figures:
                timing (from `pick_timing`, so a rebuilt pick is badged rather
                than described twice), the verdict, the tiles and the record.

                FixtureExplainer renders the flow itself and adds the button —
                mounting a bare FixtureFlow alongside it produced two flows and
                no button, which is how three tests here spent a cycle failing
                to find "Get the AI summary". */}
            <FixtureExplainer
              sport="nba"
              state={flowState}
              bundle={flowBundle}
              request={() => api.explainGame(gameId)}
              extras={{ tiles: panel.tiles, segments: panel.segments, record: winnerRecord ?? undefined, moment: "tip-off" }}
            />
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
                      {market.rebuilt ? (
                        <span className="text-pr-text-dim">Rebuilt</span>
                      ) : verdict === null ? (
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
          <section aria-labelledby={`${titleId}-box`}>
            <h3 id={`${titleId}-box`} className="mb-2 text-sm text-[var(--color-net-faint)]">
              Projected box score
            </h3>
            {players.some((p) => p.rebuilt) && (
              <p className="mb-2 text-xs text-pr-text-dim">
                Rows marked Rebuilt were built after tip-off, by a later retrain. They are shown for reference and never
                judged.
              </p>
            )}
            <PlayerBoxScore
              playerProps={players}
              hubPlayers={hubPlayers}
              homeTeam={detail?.home_team ?? ""}
              awayTeam={detail?.away_team ?? ""}
            />
          </section>
        )}

        {/* The signal rows (spec §4), ABOVE the projections -- the same position
            Sports' `GameDetailModal` and PL's `FixtureModal` put them, so a fixture
            page reads the same way in all three.

            On NBA this is the only place the out player's own projection appears at
            all: `/games/{id}/players` withholds his rows before serialising, so the
            mention under the lists below is all he gets today. This row is the first
            thing on the page that says what the model expected of him.

            **Gated on `drawnSignals`, NOT on `signals`.** `signals.length > 0` counts
            the RAW list, and `SignalRows` drops any row it cannot draw and returns
            `null` when none survive -- so the raw guard mounted a `<section>` with no
            heading and no rows inside it, which is the empty state spec §2 forbids
            ("no data, no row"). The filter is exported by the shared component
            precisely so a caller can gate on the same answer the renderer computes,
            rather than re-deriving it and getting it subtly wrong. Caught by
            CodeRabbit on #36.

            **The heading exists because `aria-labelledby` names it.** The first
            version carried `aria-labelledby={`${titleId}-signals`}` with no element
            of that id anywhere, so the reference dangled and the section was
            announced with no name at all -- worse than an unlabelled section, because
            it reads as labelled to anyone auditing the attribute. The sibling
            `-box` and `-calls` sections each pair the attribute with a real `<h3>`;
            this one now does too. Caught by CodeRabbit on #36. */}
        {drawnSignals.length > 0 && (
          <section aria-labelledby={`${titleId}-signals`} className="mt-4">
            <h3 id={`${titleId}-signals`} className="mb-2 text-sm text-[var(--color-net-faint)]">
              Signals
            </h3>
            <SignalRows signals={drawnSignals} />
          </section>
        )}

        {/* The ranked calls, as a block of their own rather than another column
            of the box score: a ranking that a reader could mistake for the full
            roster is the thing this must not be, and one list per category with
            at most three rows each cannot be read that way. Gated on the out
            feed having loaded -- see `outPlayers`. */}
        {players && players.length > 0 && outPlayers && (
          <section aria-labelledby={`${titleId}-calls`} className="mt-4">
            <h3 id={`${titleId}-calls`} className="mb-2 text-sm text-[var(--color-net-faint)]">
              Player projections
            </h3>
            <TopCalls props={players} out={outPlayers} />
          </section>
        )}
        {players && players.length > 0 && !outPlayers && (
          // No "0 players out", and no empty ranking: see `outPlayers` for why
          // an unreadable feed is not an empty report.
          <p data-testid="top-calls-unavailable" className="mt-4 text-xs text-pr-text-dim">
            The model's top calls are hidden for this game: we could not read the availability report, so we can't
            confirm who is fit to be ranked.
          </p>
        )}
      </div>
    </div>
  );
}
