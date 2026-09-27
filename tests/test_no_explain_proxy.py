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

from nba_predictor import config
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


def test_a_request_for_a_summary_is_not_a_summary(monkeypatch, tmp_path):
    """The live bug was a 502 from the proxy. This asserts what replaced it.

    What answers `/api/explain/*` genuinely **depends on whether
    `frontend/dist` exists**, and that is the whole of the difference:

    * **with `dist`** (the production image): the `SPAStaticFiles` mount
      deliberately falls back to `index.html` for any path that is not a real
      static file, so the answer is **200 and the app shell**.
    * **without `dist`** (local dev, and a clean checkout in CI): there is no
      mount, so FastAPI's own handler answers **404 with
      `{"detail": "Not Found"}`** — which is `application/json`.

    The first version of this test asserted "no JSON content-type", which is true
    only in the *first* environment, so it failed on a clean checkout. That was
    also the wrong question: a 404 body is JSON and is not a summary. The
    invariant that holds in both is the one that matters — the response is not an
    explanation, and it is not the proxy.

    Both environments are exercised explicitly through `config.PROJECT_ROOT`, so
    the result does not depend on whether the machine running the suite has run a
    frontend build. A test whose answer changes with a local build artefact is a
    test that will be green on one machine and red on another.
    """
    for dist in (False, True):
        root = tmp_path / ("with-dist" if dist else "without-dist")
        (root / "frontend").mkdir(parents=True)
        if dist:
            (root / "frontend" / "dist").mkdir()
            (root / "frontend" / "dist" / "index.html").write_text(
                "<!doctype html><html lang=en><body>SPA shell</body></html>")
        monkeypatch.setattr(config, "PROJECT_ROOT", root)

        with TestClient(create_app()) as c:
            res = c.get("/api/explain/nba/401585")

        where = "with dist" if dist else "without dist"
        assert res.status_code != 502, (
            f"[{where}] the proxy is still answering; a 502 here is the live bug"
        )
        assert "summary" not in res.text.lower(), (
            f"[{where}] something is still serving a summary: {res.text[:120]!r}"
        )
        assert "verdict" not in res.text and "factors" not in res.text, (
            f"[{where}] the response carries explanation fields: {res.text[:120]!r}"
        )


def test_both_environments_really_do_differ(monkeypatch, tmp_path):
    """So the test above is not passing because the two cases are identical.

    If the SPA mount stopped existing, or stopped falling back, the loop above
    would still pass — it just would not be covering two environments. This pins
    the difference so that a change to the mount is a loud failure here rather
    than a silent narrowing of what the other test checks.
    """
    seen = {}
    for dist in (False, True):
        root = tmp_path / str(dist)
        (root / "frontend").mkdir(parents=True)
        if dist:
            (root / "frontend" / "dist").mkdir()
            (root / "frontend" / "dist" / "index.html").write_text("<!doctype html><html></html>")
        monkeypatch.setattr(config, "PROJECT_ROOT", root)
        with TestClient(create_app()) as c:
            res = c.get("/api/explain/nba/401585")
        seen[dist] = (res.status_code, res.headers.get("content-type", ""))
    assert seen[False][0] == 404, f"without dist: expected FastAPI's own 404, got {seen[False]}"
    assert seen[True][0] == 200, f"with dist: expected the SPA shell, got {seen[True]}"
    assert "text/html" in seen[True][1], f"with dist: expected HTML, got {seen[True][1]!r}"


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
