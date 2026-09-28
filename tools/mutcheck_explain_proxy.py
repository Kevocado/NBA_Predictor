#!/usr/bin/env python3
"""Mutation-check NBA's explainer proxy.

Run from the repo root: `uv run python tools/mutcheck_explain_proxy.py`

Why this exists as a file rather than a shell loop: harnesses in this project
reported "all passed" for the wrong reason three different ways, and this one is
built to make each of them impossible.

* **It separates failed, passed and errored**, and treats all three as a bite. A
  mutation that breaks collection *did* change behaviour; reporting it as neither
  BITES nor SILENT made it invisible.
* **It refuses to print a table on a red baseline.** A failing test makes every
  mutation report BITES for free, and the run then congratulates itself. This
  happened for real: the walk-out test asserted a status code that only holds
  when `frontend/dist` is absent, so in the production topology -- which is what
  `Dockerfile:19` ships -- two tests were red, all nine mutations "bit", and the
  harness exited 0 on a failing suite. That is the single worst outcome this file
  could produce, because the harness is the evidence that the security guards
  work.
* **It carries a canary** that mutates something nothing covers and must stay
  silent, so a harness which never applies its mutations is distinguishable from
  a set of guards that never fail.
* **It distinguishes the two kinds of survival.** A mutation that survives
  because a *stronger* guard already covers the same input is a pass; one that
  survives because *nothing* covers it is a failure. Those must not share a
  verdict, or the table stops meaning anything.
* **It purges `__pycache__` and sets `PYTHONDONTWRITEBYTECODE`** before every run.
  Python treats a `.pyc` as current on the source's mtime *and size*, and the
  stored mtime has one-second resolution, so a mutation that keeps the file the
  same length inside the same second can be silently ignored and pytest imports
  the unmutated bytecode.

  **How far that is actually established, stated precisely:** the mechanism is
  real and the sibling harness in `predictor-hub` hit it -- one guard reported
  SILENT on four runs of five and BITES on the fifth, purely on where the second
  boundary fell, and a debug `read_text()` "fixed" it because a few more
  milliseconds was enough. Disabling the purge *here* did not reproduce it: the
  two same-length mutations below still bit. So the purge is a cheap guard
  against a failure that is intermittent by nature, not one demonstrated in this
  repo. The two `"15"` mutations are same-length precisely so the question stays
  answerable on any run rather than being assumed.
* **It keeps a `.bak` beside the source and restores under `try/finally`.** Also
  demonstrated for real: a bug in an earlier version of this script's own
  unpacking raised mid-table, and the `git checkout` used to recover reverted
  three unrelated fixes along with the mutation.

Exit code is 0 only when the baseline was green, every mutation bit or was
documented as expected-silent, the canary stayed silent, and the file was
restored.
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys

TARGET = pathlib.Path("src/nba_predictor/api/explain.py")
TEST = "tests/test_explain_proxy.py"
KEEP = TARGET.read_text()

#: Expected to be silent, and the reason is written down rather than left as a
#: mystery. Surviving because a stronger control covers the same input is a pass;
#: surviving because nothing covers it is a failure.
EXPECTED_SILENT_MARK = "expected silent"
CANARY = "CANARY: nothing covers this"

MUTATIONS = [
    # --- deletions: the obvious weakenings ---
    ("the whole walk-out guard removed",
     r'^    if "\.\." in sport.*?explainer_id\.startswith\("/"\):$', "    if False:"),
    ("redirects followed (requests' default)", r"allow_redirects=False", "allow_redirects=True"),
    ("non-2xx check removed",
     r"^        if not \(200 <= response\.status_code < 300\):$", "        if False:"),
    ("non-2xx returned instead of 502",
     r'            raise requests\.HTTPError\(f"upstream \{response\.status_code\}"\)',
     "            return body"),
    ("upstream body echoed in the 502",
     r'        raise HTTPException\(status_code=502, detail=_UNAVAILABLE\) from exc',
     '        raise HTTPException(status_code=502, '
     'detail=str(getattr(getattr(exc, "response", None), "text", exc))) from exc'),
    ("sport not quoted into the url", r"quote\(sport, safe=''\)", "sport"),
    ("id not quoted into the url", r"quote\(explainer_id, safe=''\)", "explainer_id"),
    ("timeout dropped from the request", r"timeout=EXPLAINER_TIMEOUT_S, ", ""),
    ("the non-dict 2xx check removed",
     r"^        if not isinstance\(body, dict\):$", "        if False:"),
    ("the timeout range check removed",
     r"^    if not 0 < value <= _MAX_TIMEOUT_S:$", "    if False:"),
    ("the timeout parse fallback removed",
     r"^    except \(TypeError, ValueError\):$", "    except ZeroDivisionError:"),

    # --- weakenings a review demonstrated survive. All whole-guard
    # --- substitutions rather than deletions, which is the shape that hides. ---
    ("walk-out: the sport half dropped", r'"\.\." in sport or ', ""),
    ("walk-out: `in` weakened to `startswith`",
     r'"\.\." in sport or "\.\." in explainer_id',
     'sport.startswith("..") or explainer_id.startswith("..")'),
    ("walk-out: absolute-path id allowed again",
     r' or "/" in sport or explainer_id\.startswith\("/"\)', ""),
    ("2xx window narrowed from a range to == 200",
     r"if not \(200 <= response\.status_code < 300\):", "if response.status_code != 200:"),
    ("`{explainer_id:path}` narrowed to a single segment",
     r"\{explainer_id:path\}", "{explainer_id}"),
    ("the guard's own 502 body echoes the sport and id",
     r"detail=_UNAVAILABLE\)\n    url = ",
     'detail=f"no summary for {sport}/{explainer_id}")\n    url = '),
    ("the 502 body gains the upstream URL",
     r"        raise HTTPException\(status_code=502, detail=_UNAVAILABLE\) from exc",
     '        raise HTTPException(status_code=502, detail=f"upstream {url} failed") from exc'),

    # --- same-length mutations, which only the bytecode purge defends against ---
    # Both same-length, which is the only kind the bytecode purge defends
    # against: a `.pyc` is current on mtime AND size, so a mutation that keeps the
    # file the same byte count inside the same second is silently ignored. `15` ->
    # `45` violates the `< 20` guard; `15` -> `25` was the original. A same-length
    # value that does NOT violate the guard (`16`) is correctly silent and is not
    # in the table -- a mutation that should pass is not evidence of anything.
    ("timeout raised above the browser's 15->45 [same length]",
     r'EXPLAINER_TIMEOUT_S", "15"', 'EXPLAINER_TIMEOUT_S", "45"'),
    ("timeout raised above the browser's 15->25 [same length]",
     r'EXPLAINER_TIMEOUT_S", "15"', 'EXPLAINER_TIMEOUT_S", "25"'),

    (f"quote(sport) left at its `safe='/'` default [{EXPECTED_SILENT_MARK}: "
     f"the `/`-in-sport guard refuses it first, so this is defence in depth]",
     r"quote\(sport, safe=''\)", "quote(sport)"),

    (CANARY, r'EXPLAINER_URL", "http://predictor-explainer:8090"',
     'EXPLAINER_URL", "http://predictor-explainer:8091"'),
]


def _purge_bytecode() -> None:
    """Delete every `__pycache__` so a mutation can never be masked by a stale
    `.pyc`. See the module docstring for why this is load-bearing."""
    for d in pathlib.Path(".").rglob("__pycache__"):
        shutil.rmtree(d, ignore_errors=True)


def run() -> tuple[int, int, int]:
    """(failed, passed, errored). Any of the first and third counts as a bite."""
    _purge_bytecode()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    p = subprocess.run(
        ["uv", "run", "pytest", TEST, "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, env=env,
    )
    out = p.stdout + p.stderr
    if p.returncode not in (0, 1):
        return -1, 0, 1

    def count(word: str) -> int:
        m = re.search(rf"(\d+) {word}", out)
        return int(m.group(1)) if m else 0

    return count("failed"), count("passed"), 0


def main() -> int:
    base_failed, base_passed, base_errored = run()
    if base_errored or base_failed:
        print(f"  BASELINE IS RED: {base_failed} failed, {base_errored} errored "
              f"({base_passed} passed).")
        print("  Every mutation would report BITES for free, so the table below would")
        print("  be meaningless. Fix the baseline first; do not read the rows below.")
        return 1
    print(f"  baseline: {base_passed} passed, 0 failed -- the table below is meaningful")
    print(f"  {len(MUTATIONS)} mutations\n")

    backup = TARGET.with_suffix(TARGET.suffix + ".bak")
    backup.write_text(KEEP)
    silent: list[str] = []
    expected: list[str] = []
    canary_ok = False

    try:
        for label, pattern, replacement in MUTATIONS:
            new, n = re.subn(pattern, replacement, KEEP, count=1, flags=re.M | re.S)
            if n != 1:
                print(f"  {label[:58]:<58} NOT APPLIED (matched {n}x)")
                silent.append(f"{label} [not applied -- the anchor does not exist]")
                continue
            TARGET.write_text(new)
            failed, passed, errored = run()
            TARGET.write_text(KEEP)
            if errored:
                verdict, bite = "BITES (collection error)", True
            elif failed > 0:
                verdict, bite = f"BITES ({failed} failed)", True
            else:
                verdict, bite = f"*** SILENT *** ({passed} passed)", False
            print(f"  {label[:58]:<58} {verdict}")
            if bite:
                if label == CANARY:
                    canary_ok = True
            elif label == CANARY:
                canary_ok = True
            elif EXPECTED_SILENT_MARK in label:
                expected.append(label.split(" [")[0])
            else:
                silent.append(label)
    finally:
        TARGET.write_text(KEEP)
        if TARGET.read_text() != KEEP:  # pragma: no cover - belt and braces
            TARGET.write_text(backup.read_text())
        backup.unlink(missing_ok=True)

    failed, passed, errored = run()
    print(f"\n  restored: {passed} passed, {failed} failed, {errored} errored")
    if errored or failed:
        print("  RESTORE FAILED -- the file was not put back. Do not trust the tree.")
        return 1

    if expected:
        print(f"  {len(expected)} mutation(s) silent as documented:")
        for s in expected:
            print(f"    - {s}")
    if silent:
        print(f"  {len(silent)} mutation(s) did not bite:")
        for s in silent:
            print(f"    - {s}")
        return 1
    if not canary_ok:
        print("  the canary BORE -- the harness is misreporting; distrust the table")
        return 1
    print("  every mutation bit or was documented as expected-silent, "
          "and the canary stayed silent as it should")
    return 0


if __name__ == "__main__":
    sys.exit(main())
