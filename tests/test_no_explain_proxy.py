"""NBA does not serve a plain-English summary, and must not look like it does.

The explainer now refuses `nba` with a 404 (predictor-hub#10, merged and live).
Until this landed, NBA's game detail still rendered the v1 panel, which turned
that 404 into "The summary didn't come through. Try again…" — an error box on
every game, on a page whose real content was fine.

So the panel and the proxy are gone. The test is here because the *absence* is
the thing that regressed, and an absent route has no way to fail on its own: the
next person to add "a harmless proxy" would be adding a route that can only ever
502, and nothing in the suite would have objected.
"""
from fastapi.testclient import TestClient

from nba_predictor.api.app import create_app


def _paths() -> set[str]:
    """Every path this app answers, from OpenAPI rather than `app.routes`.

    Reading `getattr(route, "path", "")` off `app.routes` does not work on the
    FastAPI this repo pins: each `include_router` is wrapped in a router object
    with no `.path` of its own, so a filter over the top level silently finds
    nothing. The same guard written that way in F1_Predictor passed while the
    proxy was still registered. OpenAPI is the framework's own public rendering
    of the route table, and it does not depend on that storage layout.
    """
    return set(create_app().openapi()["paths"])


def test_no_route_serves_the_shared_explainer():
    offenders = sorted(p for p in _paths() if p.startswith("/api/explain"))
    assert not offenders, (
        f"the app still proxies the shared explainer at {offenders}. It is now "
        f"refused upstream with a 404, so this route can only ever be a 502."
    )


def test_the_proxy_module_is_gone():
    import importlib

    try:
        importlib.import_module("nba_predictor.api.explain")
    except ModuleNotFoundError:
        return
    raise AssertionError(
        "nba_predictor.api.explain still imports. A dead proxy is worse than "
        "none: it is an endpoint that looks supported."
    )


def test_a_request_for_a_summary_is_not_the_proxy_and_not_json():
    """NBA's honest version of "a request is a plain 404".

    This app mounts a `SPAStaticFiles` at `/` that **deliberately** falls back to
    `index.html` for any path that is not a real static file, so once the proxy
    route is gone `/api/explain/...` is answered by that fallback with a **200
    and the app shell**. That is the app's documented behaviour for unknown
    paths and it is not something this change introduced, so pretending it is a
    404 would be a test asserting something untrue.

    What must be true is the part that matters: the request no longer reaches a
    proxy (so no more 502), and it is not answered with a summary. Asserting on
    the body is what distinguishes "the SPA answered" from "something served a
    summary here", and asserting *not* 502 is what pins the actual bug.
    """
    with TestClient(create_app()) as c:
        res = c.get("/api/explain/nba/401585")
    assert res.status_code != 502, (
        "the proxy is still answering; a 502 here is the live bug"
    )
    assert "application/json" not in res.headers.get("content-type", ""), (
        f"something is still serving JSON at /api/explain: {res.headers.get('content-type')}"
    )
    body = res.text.strip().lower()
    assert not body.startswith("{") and not body.startswith("["), (
        "a JSON body came back from /api/explain, so a summary is still served"
    )


def test_the_apps_own_routes_still_answer():
    """So the guard above cannot be satisfied by breaking the app.

    Every version of "the explainer is gone" that also breaks `/facts` is not a
    removal, it is an outage, and a test that only checked for absence would pass
    it happily.

    `/facts/upcoming` and `/hub/teams` rather than `/facts/{id}` or `/games`:
    both of those need an argument this test would have to invent, and an
    invented one 404s or 422s for reasons that have nothing to do with the
    removal — the same trap as every other hand-shaped double in this project.
    Listing routes answer from what is actually there.
    """
    with TestClient(create_app()) as c:
        facts = c.get("/facts/upcoming")
        teams = c.get("/hub/teams")
    assert facts.status_code == 200, f"/facts broke: {facts.status_code}"
    assert teams.status_code == 200, f"/hub/teams broke: {teams.status_code}"
