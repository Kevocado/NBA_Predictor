import pathlib
p = pathlib.Path("frontend/src/components/GameDetailModal.tsx")
t = p.read_text()

old = """        {verdict && (
          <div className="mb-5 border-b border-[var(--color-line)] pb-5 text-sm" data-testid="post-match-verdict">
            <div className="mb-2 flex items-center gap-2">
              <span className={`font-pr-display font-semibold uppercase tracking-wide ${verdict.winnerCorrect ? "text-pr-win" : "text-pr-loss"}`}>
                {statusWords(verdict.winnerCorrect ? "called" : "missed")}
              </span>
              <span className="text-pr-text-dim">winner pick</span>
            </div>
            <div className="text-[var(--color-net-faint)]">
              Predicted margin: {verdict.predictedMargin.team} +{verdict.predictedMargin.value.toFixed(1)} \u2014{" "}
              Actual: {verdict.actualMargin.team} +{verdict.actualMargin.value.toFixed(1)} (off by {verdict.marginDiff.toFixed(1)})
            </div>
            <div className="text-[var(--color-net-faint)]">
              Predicted total: {verdict.predictedTotal.toFixed(1)} \u2014 Actual: {verdict.actualTotal} (off by {verdict.totalDiff.toFixed(1)})
            </div>
          </div>
        )}

        {detail && ("""

new = """        {verdict && (
          <div className="mb-5 border-b border-[var(--color-line)] pb-5 text-sm" data-testid="post-match-verdict">
            <div className="mb-2 flex items-center gap-2">
              <span className={`font-pr-display font-semibold uppercase tracking-wide ${verdict.winnerCorrect ? "text-pr-win" : "text-pr-loss"}`}>
                {statusWords(verdict.winnerCorrect ? "called" : "missed")}
              </span>
              <span className="text-pr-text-dim">winner pick</span>
            </div>
            <div className="text-[var(--color-net-faint)]">
              Predicted margin: {verdict.predictedMargin.team} +{verdict.predictedMargin.value.toFixed(1)} \u2014{" "}
              Actual: {verdict.actualMargin.team} +{verdict.actualMargin.value.toFixed(1)} (off by {verdict.marginDiff.toFixed(1)})
            </div>
            <div className="text-[var(--color-net-faint)]">
              Predicted total: {verdict.predictedTotal.toFixed(1)} \u2014 Actual: {verdict.actualTotal} (off by {verdict.totalDiff.toFixed(1)})
            </div>
          </div>
        )}

        {/* Other model markets: cover chances and projected margin \u00b1 */}
        {detail && detail.prediction && (
          <div className="mb-5 border-b border-[var(--color-line)] pb-5 text-sm">
            <h3 className="mb-2 text-xs uppercase tracking-wide text-[var(--color-net-faint)]">
              Other model markets
            </h3>
            <div className="grid gap-2 sm:grid-cols-2">
              {detail.prediction.cover_prob_spread !== null && detail.prediction.cover_prob_spread !== undefined && (
                <div className="p-2 rounded border border-[var(--color-line)] bg-[var(--color-court-950)]">
                  <div className="text-xs text-[var(--color-net-faint)]">Spread cover chance</div>
                  <div className="text-lg font-pr-display font-semibold">
                    {pct(detail.prediction.cover_prob_spread!)}
                  </div>
                  {detail.prediction.margin_sigma && (
                    <div className="text-xs text-[var(--color-net-dim)]">
                      \u03c3 = {detail.prediction.margin_sigma!.toFixed(1)} pts
                    </div>
                  )}
                </div>
              )}
              {detail.prediction.cover_prob_total !== null && detail.prediction.cover_prob_total !== undefined && (
                <div className="p-2 rounded border border-[var(--color-line)] bg-[var(--color-court-950)]">
                  <div className="text-xs text-[var(--color-net-faint)]">Over {detail.prediction.predicted_total.toFixed(1)} chance</div>
                  <div className="text-lg font-pr-display font-semibold">
                    {pct(detail.prediction.cover_prob_total!)}
                  </div>
                  {detail.prediction.total_sigma && (
                    <div className="text-xs text-[var(--color-net-dim)]">
                      \u03c3 = {detail.prediction.total_sigma!.toFixed(1)} pts
                    </div>
                  )}
                </div>
              )}
              <div className="p-2 rounded border border-[var(--color-line)] bg-[var(--color-court-950)]">
                <div className="text-xs text-[var(--color-net-faint)]">Projected margin \u00b1</div>
                <div className="text-lg font-pr-display font-semibold">
                  {detail.prediction.predicted_margin >= 0
                    ? `\${detail.home_team} +\${detail.prediction.predicted_margin.toFixed(1)}`
                    : `\${detail.away_team} +\${Math.abs(detail.prediction.predicted_margin).toFixed(1)}`}
                </div>
                {detail.prediction.margin_sigma && (
                  <div className="text-xs text-[var(--color-net-dim)]">
                    \u00b1{detail.prediction.margin_sigma.toFixed(1)} pts (1 \u03c3)
                  </div>
                )}
              </div>
              <div className="p-2 rounded border border-[var(--color-line)] bg-[var(--color-court-950)]">
                <div className="text-xs text-[var(--color-net-faint)]">Projected total \u00b1</div>
                <div className="text-lg font-pr-display font-semibold">
                  {detail.prediction.predicted_total.toFixed(1)}
                </div>
                {detail.prediction.total_sigma && (
                  <div className="text-xs text-[var(--color-net-dim)]">
                    \u00b1{detail.prediction.total_sigma.toFixed(1)} pts (1 \u03c3)
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Injury report line */}
        {detail && detail.injury_summary && (
          <div className="mb-5 border-b border-[var(--color-line)] pb-5 text-sm">
            <div className="text-xs text-[var(--color-net-faint)]">{detail.injury_summary}</div>
          </div>
        )}

        {detail && ("""

if old in t:
    t = t.replace(old, new)
    p.write_text(t)
    print("GameDetailModal updated")
else:
    print("OLD NOT FOUND")
    # Debug
    idx = t.find("verdict &&")
    if idx >= 0:
        print(t[idx:idx+500])