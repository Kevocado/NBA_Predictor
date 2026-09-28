#!/usr/bin/env python3
"""Mutation-check NBA's explainer proxy.

Run from the repo root: `uv run python tools/mutcheck_explain_proxy.py` A mutation "BITES" when pytest reports any failure. The
canary at the top mutates something nothing covers and must report SILENT --
without it, a harness that never applies its mutation looks exactly like a set of
guards that never fail, which is the confusion this script exists to remove.
"""
import pathlib
import re
import subprocess
import sys

TARGET = pathlib.Path("src/nba_predictor/api/explain.py")
KEEP = TARGET.read_text()

#: The canary is EXPECTED to be silent. It is the control: a mutation of
#: something no test covers, so if it ever reports BITES the harness is
#: misreporting and every other line in the table is suspect.
CANARY = "CANARY: nothing covers this"

MUTATIONS = [
    ("path-walk guard removed", r'    if "\.\." in sport or "\.\." in explainer_id:', "    if False:"),
    ("redirects followed (requests' default)", r"allow_redirects=False", "allow_redirects=True"),
    ("non-2xx check removed", r"        if not \(200 <= response\.status_code < 300\):", "        if False:"),
    ("non-2xx returned instead of 502", r'            raise requests\.HTTPError\(f"upstream \{response\.status_code\}"\)', "            return response.json()"),
    ("upstream body echoed in the 502", r'        raise HTTPException\(status_code=502, detail=_UNAVAILABLE\) from exc', '        raise HTTPException(status_code=502, detail=str(getattr(getattr(exc, "response", None), "text", exc))) from exc'),
    ("sport not quoted into the url", r"quote\(sport, safe=''\)", "sport"),
    ("id not quoted into the url", r"quote\(explainer_id, safe=''\)", "explainer_id"),
    ("timeout dropped from the request", r"timeout=EXPLAINER_TIMEOUT_S, ", ""),
    ("timeout raised above the browser's", r'EXPLAINER_TIMEOUT_S", "15"', 'EXPLAINER_TIMEOUT_S", "25"'),
    ("CANARY: nothing covers this", r'EXPLAINER_URL", "http://predictor-explainer:8090"', 'EXPLAINER_URL", "http://predictor-explainer:8091"'),
]


def run() -> tuple[int, int]:
    p = subprocess.run(
        ["uv", "run", "pytest", "tests/test_explain_proxy.py", "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True,
    )
    out = p.stdout
    m = re.search(r"(\d+) failed", out)
    failed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) passed", out)
    passed = int(m.group(1)) if m else 0
    if p.returncode not in (0, 1):
        return -1, -1
    return failed, passed


def main() -> int:
    silent = []
    canary_ok = False
    for label, pattern, replacement in MUTATIONS:
        text = KEEP
        new, n = re.subn(pattern, replacement, text, count=1)
        if n != 1:
            print(f"  {label:<44} NOT APPLIED (matched {n}x)")
            silent.append(label + " [not applied]")
            TARGET.write_text(KEEP)
            continue
        TARGET.write_text(new)
        failed, passed = run()
        TARGET.write_text(KEEP)
        if failed < 0:
            verdict = "ERROR (collection?)"
        elif failed > 0:
            verdict = f"BITES ({failed} failed)"
        else:
            verdict = f"*** SILENT *** ({passed} passed)"
            if label == CANARY:
                canary_ok = True
            else:
                silent.append(label)
        print(f"  {label:<44} {verdict}")

    failed, passed = run()
    print(f"\n  restored: {passed} passed, {failed} failed")
    if silent:
        print(f"  {len(silent)} mutation(s) did not bite:")
        for s in silent:
            print(f"    - {s}")
        return 1
    if not canary_ok:
        print("  the canary BORE -- the harness is misreporting; distrust the table")
        return 1
    print("  every mutation bit, and the canary stayed silent as it should")
    return 0


if __name__ == "__main__":
    sys.exit(main())
