# Paste-ready prompt for Kevin's coding agent — NBA parity build

Copy everything below the line into a fresh agent session.

---

You are implementing the NBA predictor parity program for Kevin (Kevocado).

**Repo:** `Kevocado/NBA_Predictor` — **branch:** `nba-parity-review` (pull it; it is
ahead of main and is where all work happens — never merge to `main` without
Kevin's explicit approval; open a PR per phase).

**Read first, in order:**
1. `docs/superpowers/specs/2026-10-04-nba-parity-design.md` — the full design spec
2. `docs/superpowers/plans/2026-10-04-nba-parity.md` — the implementation plan

**How to work (all three apply, no exceptions):**
- **Superpowers framework** — follow the plan task-by-task in order, exactly as written.
- **Ponytail skill** (DietrichGebert/ponytail — the anti-over-engineering ladder;
  spec §15 states the ladder inline) — climb the ladder before writing any code:
  YAGNI → reuse existing code (the spec tells you exactly which NFL/PL patterns
  to copy verbatim) → stdlib → native → existing deps (do NOT install new
  packages) → one-liner → minimum viable diff.
- **Impeccable skill** (pbakaus/impeccable; spec §16) — for every frontend task:
  `/impeccable craft` or `polish` to build, `/impeccable audit` each touched
  surface before committing, fix detector findings in the same task. If
  impeccable isn't installed, match the repo's incumbent dark theme and
  `predictor-ui` components — no new design language.
- **TDD always** — failing test first for every task, no production code without
  one. Commit after every task.

**Hard constraints:** $0 data spend (no new APIs/keys). Additive changes only —
new model artifacts are versioned, never overwrite production pickles until
Kevin approves. Every rolling feature is shift(1)-then-rolling (no leakage).
Snapshots are immutable and strictly pre-tip-off. No injury fabrication, ever —
injury features come only from the verified ESPN availability feed. Nothing
publishes, posts, or places bets.

**Phasing and gates:** Start at Phase 0, then Phase A (model quality — must land
before the NBA regular season tips off late October). **STOP after Phase A:**
open a PR and wait for Kevin's review of the honest evaluation report before
starting Phase B. Same gate after Phase B before Phase C.

**If you get stuck:** the spec's §14 (Review Focus) lists the five failure modes
most likely to bite — check there first. If a phase's offline gate fails
honestly (e.g. a model doesn't beat naive), report the honest numbers — that is
a successful outcome, not a failure. Do not weaken a gate to make it pass.
