"""Proxy for the plain-English explainer service.

NBA's FastAPI serves the site, so Caddy only ever reverse-proxies this app. The
browser therefore cannot reach the explainer directly: it calls
``/api/explain/{sport}/{id}`` here, and this route forwards to the service's own
``/explain/{sport}/{id}``.

The proxy is deliberately thin and deliberately unforgiving. A summary is a
nice-to-have on top of a fixture page, so a missing, slow or broken explainer
becomes a 502 with a fixed message and the site's own error state takes it from
there -- the fixture itself never depends on this route answering.

**This module was deleted on the strength of a refusal that was never checked.**
NBA was refused upstream with a 404, so a proxy "had exactly one possible
outcome: a 502 on every game detail" -- which is true, and was the correct
conclusion from a premise that was only true for F1. NBA serves now: its `/facts`
carries the same `moneyline`/`spread`/`total` the panel draws. See
`tests/test_explain_proxy.py`, which asserts the safety properties that the
absence never could.
"""
from __future__ import annotations

import os
from urllib.parse import quote

import requests
from fastapi import APIRouter, HTTPException

router = APIRouter()

# The explainer is reached over the internal compose network by service name.
EXPLAINER_URL = os.getenv("EXPLAINER_URL", "http://predictor-explainer:8090").rstrip("/")
# Deliberately BELOW the browser's 20 s read timeout for this site. This route is
# a sync def, so an in-flight request occupies one of Starlette's worker threads
# for its whole duration: a proxy that waits longer than the client spends a
# thread on an answer nobody is waiting for. The explainer's own 25 s model
# timeout is longer than this on purpose -- a first uncached match usually lands
# here and the panel shows its Try again state, which pre-generation exists to
# avoid.
EXPLAINER_TIMEOUT_S = float(os.getenv("EXPLAINER_TIMEOUT_S", "15"))

#: Every failure is this message. An upstream error can carry key material or an
#: internal path, and the browser is not the place to find out.
_UNAVAILABLE = "The summary service is not available."


@router.get("/api/explain/{sport}/{explainer_id:path}")
def explain(sport: str, explainer_id: str) -> dict:
    """Forward to the explainer, or say plainly that it could not be reached.

    No upstream body is ever returned, on any status: a 2xx is the only thing
    passed through. Redirects are not followed, and neither path segment is
    allowed to walk out of the explainer's own route.
    """
    if ".." in sport or ".." in explainer_id:
        # Rejected before any request goes out: a path segment that walks out of
        # the explainer's own route would make this a forwarder to any path on
        # that host. 502 like every other failure, so the site's error state takes
        # it and no upstream (or no request) is involved.
        raise HTTPException(status_code=502, detail=_UNAVAILABLE)
    url = f"{EXPLAINER_URL}/explain/{quote(sport, safe='')}/{quote(explainer_id, safe='')}"
    try:
        response = requests.get(url, timeout=EXPLAINER_TIMEOUT_S, allow_redirects=False)
        if not (200 <= response.status_code < 300):
            # A 404 from the explainer means "no summary for this fixture", which
            # the site renders as a fixture with no panel. Anything else is the
            # service being absent, slow or broken. Both are 502 here so the site
            # has exactly one failure shape to handle.
            raise requests.HTTPError(f"upstream {response.status_code}")
        return response.json()
    except requests.RequestException as exc:
        # This also covers "a 2xx that is not JSON", which is the case a separate
        # `except ValueError` used to be written for. It was unreachable:
        # `requests.exceptions.JSONDecodeError` subclasses
        # `requests.exceptions.InvalidJSONError`, which subclasses
        # `RequestException`, so the handler above catches it first. Verified
        # against the pinned requests (2.34.2) rather than assumed -- a mutation
        # check that replaced this branch with `except KeyError` still passed
        # every test, which is what a dead branch looks like from the outside.
        #
        # Both cases are the same failure from the browser's point of view: the
        # explainer is misconfigured, absent or slow, so the site shows its retry
        # state. One handler, one message.
        raise HTTPException(status_code=502, detail=_UNAVAILABLE) from exc
