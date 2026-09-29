"""NBA's site proxies the shared explainer, and the proxy is safe to expose.

**This file used to assert the opposite.** `tests/test_no_explain_proxy.py` read:

    There is deliberately no /api/explain proxy here. The shared explainer refuses
    nba with a 404 (predictor-hub#10), so a proxy had exactly one possible
    outcome: a 502 on every game detail.

That reasoning was checked for F1 and assumed for NBA. The explainer serves NBA
now, and NBA's `/facts` carries the same `moneyline`/`spread`/`total` the panel
draws. So the proxy is restored — and the tests below are the ones that keep it
honest, which the absence could never do: an absent route has no way to fail on
its own, and the next person to add "a harmless proxy" would have added an
endpoint that leaks an upstream error body to the browser.

What is asserted here, and why each of these is a real risk rather than a
formality:

* the route is registered — read from **OpenAPI**, not `app.routes`, because a
  filter over `getattr(route, "path", "")` finds nothing on the FastAPI this repo
  pins, and the same guard written that way in F1_Predictor passed while the
  proxy was still registered;
* a good upstream answer reaches the browser unchanged;
* **no upstream body is ever echoed** — an explainer error can carry key material
  or an internal path, and the browser is not the right place to find out;
* a `..` in either path segment is rejected *before* any request leaves, so the
  route cannot be turned into a forwarder to any path on the explainer's host;
* redirects are not followed, for the same reason.
"""
from __future__ import annotations

import importlib
import json

import pytest
import requests
from fastapi import HTTPException
from fastapi.testclient import TestClient

from nba_predictor import config
from nba_predictor.api import explain as explain_module
from nba_predictor.api.app import create_app

#: Deliberately shares no digit sequence with UPSTREAM_ERROR_STATUSES. The
#: original fixture id was `401585`, and the no-echo assertion
#: `str(status) not in res.text` was passing for the WRONG reason on `[401]`:
#: it found "401" inside "401585". Change the id to `777` and a mutation that
#: echoes the upstream URL into the 502 body stops failing any test at all.
GAME_ID = "777"

#: The 502 body, as a LITERAL rather than as `explain_module._UNAVAILABLE`.
#:
#: Asserting against the constant looks stronger and is weaker: it can only detect
#: a change at the raise site, never a change to the message. Appending the internal
#: hostname -- `_UNAVAILABLE = "The summary service is not available (upstream
#: predictor-explainer:8090)."` -- left all 55 tests green while the internal host
#: reached the browser. This is the same "guard relative to a live object" mistake
#: as reading the route table off a framework structure that hid the route, and as
#: the `401585`/`"401"` substring coincidence.
UNAVAILABLE_LITERAL = "The summary service is not available."

UPSTREAM_BODY = {
    "verdict": "BOS by 4.2 points, against a line of BOS -3.5.",
    "factors": [{"key": "spread", "text": "It rates BOS 4.2 points better."}],
}


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


@pytest.fixture
def upstream(monkeypatch):
    """Capture what the proxy sends, and answer with a canned summary."""
    sent: list[dict] = []

    def fake_get(url, **kwargs):
        sent.append({"url": url, **kwargs})
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(UPSTREAM_BODY).encode()
        return response

    monkeypatch.setattr(explain_module.requests, "get", fake_get)
    return sent


# --- the route is there ----------------------------------------------------


def test_the_proxy_route_is_registered():
    offenders = sorted(p for p in _paths() if p.startswith("/api/explain"))
    assert offenders, (
        "no /api/explain route is registered, so the panel cannot render on NBA's "
        "site. Caddy only reverse-proxies this app, so the browser has no other "
        "route to the explainer."
    )


def test_the_proxy_module_is_present():
    assert importlib.import_module("nba_predictor.api.explain") is explain_module


# --- a good answer reaches the browser ------------------------------------


def test_a_summary_is_forwarded_unchanged(upstream):
    with TestClient(create_app()) as c:
        res = c.get(f"/api/explain/nba/{GAME_ID}")

    assert res.status_code == 200, res.text
    assert res.json() == UPSTREAM_BODY
    assert len(upstream) == 1
    assert upstream[0]["url"].endswith(f"/explain/nba/{GAME_ID}"), upstream[0]["url"]


def test_the_request_carries_a_timeout_and_refuses_redirects(upstream):
    """Both of these are per-call arguments, so asserting the module constants
    proves nothing -- a future edit could drop them from the call and every
    constant-level test would still pass.

    Found by mutation: replacing the whole `requests.get(...)` call's `timeout=`
    with nothing left all 20 tests green.
    """
    with TestClient(create_app()) as c:
        c.get("/api/explain/nba/401585")

    assert len(upstream) == 1, upstream
    call = upstream[0]
    assert call.get("timeout") == explain_module.EXPLAINER_TIMEOUT_S, call
    assert call.get("allow_redirects") is False, (
        f"allow_redirects={call.get('allow_redirects')!r}; requests defaults to "
        f"True, so a 302 from the explainer would be followed"
    )


def test_a_path_segment_is_quoted_into_the_upstream_url(upstream):
    """`{explainer_id:path}` accepts slashes, and a segment containing `?`, `#`
    or a space would otherwise truncate or mangle the upstream URL -- turning the
    request into a *different* question than the one the browser asked.

    NBA's ids are numeric so this never fires in practice. It is here because
    `quote()` is in the code, and an unexercised call is one a future edit can
    drop without any test noticing. Found by mutation, same as the timeout.
    """
    with TestClient(create_app()) as c:
        res = c.get("/api/explain/nba/2026%2F10%2F20%20PHI%20at%20BOS")

    assert res.status_code == 200, res.text
    assert upstream, "no request was sent"
    sent = upstream[0]["url"]
    assert " " not in sent, f"an unencoded space reached the upstream URL: {sent}"
    assert sent.endswith("/explain/nba/2026%2F10%2F20%20PHI%20at%20BOS"), sent


#: A sport containing `/` is NOT here. The guard refuses it before any quoting
#: happens -- `or "/" in sport` in `explain()` -- so the honest assertion for that
#: input is a 502, made in `test_a_slash_in_the_sport_is_refused`, not a quoted
#: URL. It also means `quote(sport, safe='')` is defence in depth rather than the
#: primary control, which is why the mutation that drops `safe=''` is expected to
#: survive on this file. Belt and braces is the right shape; a mutation surviving
#: because a stronger guard already covers it is a different thing from a mutation
#: surviving because nothing covers it, and the harness says which is which only
#: if the note is here.
@pytest.mark.parametrize(
    ("sport", "expected"),
    [
        # A sport with a space or a comma is nonsense, but the route accepts any
        # string and forwards it; `quote()` is what stops a nonsense value from
        # becoming a *different, well-formed* upstream URL. `nba,cfb` is the
        # readable case: unquoted it is a literal comma on the wire, and there is
        # no downstream parser here to reject it, so the explainer is the only
        # thing standing between that and whatever it does with the path.
        ("nba,cfb", "nba%2Ccfb"),

        ("n ba", "n%20ba"),
        ("nba#x", "nba%23x"),
    ],
)
def test_the_sport_segment_is_quoted_too(monkeypatch, upstream, sport, expected):
    """Called directly rather than through HTTP, on purpose.

    Through the client, `{sport}` cannot contain a `/` (it is not a `:path`
    parameter) and the client normalises much of the rest before routing, so an
    end-to-end test can only ever use `nba` -- which needs no quoting, and so
    cannot tell a quoted value from an unquoted one. That is why the `sport`
    branch of the guard was uncovered while the `explainer_id` branch was not:
    the HTTP test could only reach the half that was already exercised.
    """
    result = explain_module.explain(sport, "401585")

    assert result == UPSTREAM_BODY
    assert upstream[0]["url"] == f"{explain_module.EXPLAINER_URL}/explain/{expected}/401585", upstream


def test_the_upstream_timeout_is_below_the_browsers():
    """A sync `def` route occupies a Starlette worker thread for its whole
    duration, so waiting longer than the browser's own read timeout spends a
    thread on an answer nobody is waiting for. The explainer's 25 s model timeout
    is deliberately longer than this: a first uncached match usually lands here
    and the panel shows its Try again state, which is what pre-generation is for.
    """
    assert explain_module.EXPLAINER_TIMEOUT_S < 20, (
        f"{explain_module.EXPLAINER_TIMEOUT_S}s is not below the browser's 20 s "
        f"read timeout"
    )


# --- nothing upstream is ever echoed ---------------------------------------

#: Every non-2xx the proxy must turn into a 502. **The 3xx range is here for a
#: reason:** with only 400-and-up, `< 300` -> `<= 300` and `< 300` -> `< 400` were
#: both silent, and a 3xx carrying a JSON body would be handed to the browser as a
#: summary -- contradicting both the module docstring and this file's headline
#: claim. "The window is a range" is proved by testing below 200; where it ENDS is a
#: separate question, and only a 3xx asks it.
#:
#: The fake RETURNS a response with
#: this status -- it does not raise. That distinction is the whole test: when the
#: fake raised `HTTPError` instead, `requests.get()` never returned and the
#: proxy's own `if not (200 <= response.status_code < 300)` check was never
#: executed, so removing that check left every test green.
UPSTREAM_ERROR_STATUSES = [300, 301, 302, 303, 304, 307, 308, 399,
                          400, 401, 403, 404, 422, 429, 500, 502, 503]


@pytest.mark.parametrize("status", UPSTREAM_ERROR_STATUSES)
def test_an_upstream_error_status_becomes_a_502_carrying_no_upstream_body(monkeypatch, status):
    secret = "sk-or-v1-should-never-reach-a-browser"

    def fake_get(url, **kwargs):
        response = requests.Response()
        response.status_code = status
        response._content = f'{{"detail": "{secret}", "path": "/srv/internal/{secret}"}}'.encode()
        return response  # returned, NOT raised -- see UPSTREAM_ERROR_STATUSES

    monkeypatch.setattr(explain_module.requests, "get", fake_get)

    with TestClient(create_app()) as c:
        res = c.get(f"/api/explain/nba/{GAME_ID}")

    assert res.status_code == 502, f"[upstream {status}] {res.text[:200]!r}"
    assert secret not in res.text, f"[upstream {status}] an upstream body was echoed"
    assert "/srv/internal" not in res.text, f"[upstream {status}] an upstream path was echoed"
    # The strongest form: the body IS the fixed message. Every assertion above is
    # a "not containing", and a mutation that appends the upstream URL to the
    # message satisfies all of them while disclosing the internal host and path.
    assert res.json() == {"detail": UNAVAILABLE_LITERAL}, res.text[:200]


@pytest.mark.parametrize("status", UPSTREAM_ERROR_STATUSES)
def test_a_raised_upstream_error_becomes_a_502_carrying_nothing(monkeypatch, status):
    """The other half, and it is a different branch. Real `requests` attaches the
    response to the exception, so the message is a candidate for leaking -- which
    is why the fake attaches one.

    The first version of this test raised a bare `HTTPError("upstream 404")` with
    no response, and a mutation that replaced the fixed message with
    `exc.response.text` left every test green. The assertion was a property of
    the fake, not of the proxy.
    """
    secret = "sk-or-v1-should-never-reach-a-browser"

    def fake_get(url, **kwargs):
        response = requests.Response()
        response.status_code = status
        response._content = f'{{"detail": "{secret}"}}'.encode()
        raise requests.HTTPError(f"upstream {status}", response=response)

    monkeypatch.setattr(explain_module.requests, "get", fake_get)

    with TestClient(create_app()) as c:
        res = c.get(f"/api/explain/nba/{GAME_ID}")

    assert res.status_code == 502, f"[upstream {status}] {res.text[:200]!r}"
    assert secret not in res.text, f"[upstream {status}] an upstream body was echoed"
    assert res.json() == {"detail": UNAVAILABLE_LITERAL}, res.text[:200]


def test_a_200_that_is_not_json_is_a_502(monkeypatch):
    """Something else is answering on the explainer's port, or it is
    misconfigured. Either way the site should show its retry state rather than a
    bare 500 it cannot interpret."""

    def fake_get(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = b"<html>not json</html>"
        return response

    monkeypatch.setattr(explain_module.requests, "get", fake_get)

    with TestClient(create_app()) as c:
        res = c.get(f"/api/explain/nba/{GAME_ID}")

    assert res.status_code == 502, res.text[:200]
    assert "not json" not in res.text


@pytest.mark.parametrize("status", [201, 202, 204, 299])
def test_any_2xx_is_forwarded_not_just_200(monkeypatch, status):
    """The window is a RANGE, and only 200 and non-2xx were exercised, so
    narrowing `200 <= status < 300` to `status == 200` left every test green.

    A 204 is worth calling out: it is a 2xx with no body, so `response.json()`
    raises and it takes the 502 path -- which is correct, because there is no
    summary to forward.
    """
    def fake_get(url, **kwargs):
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(UPSTREAM_BODY).encode() if status != 204 else b""
        return response

    monkeypatch.setattr(explain_module.requests, "get", fake_get)

    with TestClient(create_app()) as c:
        res = c.get(f"/api/explain/nba/{GAME_ID}")

    if status == 204:
        assert res.status_code == 502, res.text[:200]
    else:
        assert res.status_code == 200, f"[{status}] {res.text[:200]!r}"
        assert res.json() == UPSTREAM_BODY


def test_a_dead_explainer_is_a_502_not_a_500(monkeypatch):
    def fake_get(url, **kwargs):
        raise requests.ConnectionError("explainer is down")

    monkeypatch.setattr(explain_module.requests, "get", fake_get)

    with TestClient(create_app()) as c:
        res = c.get(f"/api/explain/nba/{GAME_ID}")

    assert res.status_code == 502, res.text[:200]
    assert "explainer is down" not in res.text, "an internal reason leaked to the browser"


# --- the route is not a forwarder to anything on that host -----------------


#: Cases that must never reach the explainer. The list is built to defeat three
#: plausible weakenings of the guard:
#:
#: * dropping the `sport` half -- only `..%2Fadmin/x` puts a `..` into `sport`
#:   (it decodes to `sport=".."`, `explainer_id="admin/x"`), and it is the ONLY
#:   case that reaches that half;
#: * `in` -> `startswith` -- `secrets/../x` has the `..` INTERIOR to the segment,
#:   so a prefix check satisfies every other case;
#: * an absolute-path id -- `nba//etc/passwd` contains no `..` at all, so a
#:   walk-out check alone forwards it.
#: (path, must_be_refused_outright).
#:
#: `must_be_refused_outright` is False only where the HTTP client normalises the
#: path BEFORE routing, so the route never matches and the request that leaves (if
#: any) is a well-formed shorter path. Asserting "nothing was sent" for those
#: would be asserting something about the client, not about the guard -- and the
#: first version of this file did exactly that, which is why it was red in the
#: production topology.
#:
#: The encoded cases are the ones that must be refused, and each defeats a
#: different weakening:
#:
#: * `..%2Fadmin/x`                -> `sport == ".."`, the ONLY case that reaches
#:                                     the `sport` half of the guard;
#: * `secrets%2F..%2Fx`            -> the `..` is INTERIOR, so `startswith("..")`
#:                                     passes while the walk-out is real;
#: * `%2Fetc%2Fpasswd`             -> no `..` at all, so no walk-out check can see
#:                                     it; `quote()` sends it as `%2F` and the
#:                                     explainer decodes it back to a deeper path.
WALK_OUT_PATHS = [
    ("/api/explain/nba/../../secrets", False),
    ("/api/explain/../admin/x", False),
    ("/api/explain/..%2Fadmin/x", True),
    ("/api/explain/nba/%2e%2e/%2e%2e/etc/passwd", True),
    ("/api/explain/nba/secrets%2F..%2Fx", True),
    ("/api/explain/nba/secrets/../x", False),
    # `True`, not `False`: the guard DOES refuse this (the id resolves to
    # `/etc/passwd`). The comment above presents it as the case that defeats a
    # walk-out-only guard, and with `False` the confinement loop alone would not
    # notice the guard being removed -- the outbound URL satisfies all four
    # confinement assertions either way.
    ("/api/explain/nba//etc/passwd", True),
    ("/api/explain/nba/%2Fetc%2Fpasswd", True),
]


@pytest.mark.parametrize(("path", "must_refuse"), WALK_OUT_PATHS,
                         ids=[f"{p}|refuse={r}" for p, r in WALK_OUT_PATHS])
@pytest.mark.parametrize("ship_dist", [False, True], ids=["no-dist", "with-dist"])
def test_a_path_that_walks_out_never_reaches_the_explainer(monkeypatch, tmp_path, path, must_refuse, ship_dist):
    """No request may leave for any of these, in EITHER deployment shape.

    **The status code is deliberately not asserted, and the two environments are
    the reason.** A literal `..` is normalised away by the HTTP client before
    routing, so those cases never reach the app -- and what answers an unknown
    path depends entirely on whether `frontend/dist` exists:

    * **without `dist`** (local dev, a clean checkout): FastAPI's own 404;
    * **with `dist`** (the production image -- `Dockerfile:19` COPYs it in):
      `SPAStaticFiles` deliberately falls back to `index.html`, so **200 and the
      app shell**.

    The first version of this test asserted `in (404, 502)` and was therefore
    RED in the production topology -- verified by injecting a `dist` and getting
    `200: '<!doctype html>...'`. Worse than a failing test: with two tests red
    regardless of mutation, every mutation in `tools/mutcheck_explain_proxy.py`
    reported BITES, so the harness printed "every mutation bit" and exited 0 on a
    red baseline. The evidence for the security guards was manufactured by a
    failure.

    Commit 67de375 is *"test: the explain-absence guard must not depend on a local
    build artefact"*, and the file this one replaced carried
    `test_both_environments_really_do_differ` for exactly this. Both properties
    are restored: the invariant is asserted, and both shapes are exercised, so a
    test whose answer changes with a build artefact cannot hide here again.
    """
    root = tmp_path / ("with-dist" if ship_dist else "no-dist")
    (root / "frontend").mkdir(parents=True)
    if ship_dist:
        (root / "frontend" / "dist").mkdir()
        (root / "frontend" / "dist" / "index.html").write_text(
            "<!doctype html><html lang=en><body>SPA shell</body></html>")
    monkeypatch.setattr(config, "PROJECT_ROOT", root)

    sent: list[str] = []

    def fake_get(url, **kwargs):
        sent.append(url)
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"verdict": "leaked"}'
        return response

    monkeypatch.setattr(explain_module.requests, "get", fake_get)

    with TestClient(create_app()) as c:
        res = c.get(path)

    if must_refuse:
        assert not sent, (
            f"[{path}] reached the wire but should have been refused before any "
            f"request: {sent}"
        )
        # The refusal's own body is pinned too. A mutation that made THIS 502
        # echo the sport and the id left every test green, because nothing
        # asserted the body of the guard's refusal -- only the bodies of the
        # failures that happen after a request was sent.
        assert res.json() == {"detail": UNAVAILABLE_LITERAL}, (
            f"[{path}] the refusal did not use the fixed message: {res.text[:200]!r}"
        )
    # The invariant is NOT "no request left". A path the client normalises
    # *before routing* -- `secrets/../x` becomes `x`, a literal `..` becomes a
    # shorter path -- never reaches the guard at all, and the request that leaves
    # is a well-formed one. Asserting "no request left" would be asserting that
    # the HTTP client does something it does not do.
    #
    # The invariant is that whatever leaves is confined to the explainer's own
    # route: under `/explain/{sport}/`, with no `..` and no extra path depth.
    # That is true whether the guard fired (502, nothing sent), the client
    # normalised it (a safe shorter path), or the route never matched.
    for url in sent:
        assert url.startswith(f"{explain_module.EXPLAINER_URL}/explain/"), (
            f"[{path}] a request escaped the explainer's route: {url}"
        )
        assert ".." not in url, f"[{path}] a traversal reached the wire: {url}"
        tail = url[len(f"{explain_module.EXPLAINER_URL}/explain/"):]
        assert tail.startswith("nba/"), f"[{path}] wrong sport segment: {url}"
        assert not tail[len("nba/"):].startswith("/"), (
            f"[{path}] an absolute-path id was forwarded, so the explainer's ASGI "
            f"layer decodes it back to a deeper path: {url}"
        )
    # No assertion on the RESPONSE body here. For a path the client normalised
    # before routing, serving the summary for the normalised id is correct -- the
    # browser asked for it -- so "leaked" is not the invariant. The confinement
    # of the outbound URL is, and that is what the loop checks.


def test_the_two_deployment_shapes_really_do_answer_differently(tmp_path, monkeypatch):
    """Keeps the loop above honest.

    Without this, "both environments are exercised" is a claim rather than a
    fact: if a future change made `SPAStaticFiles` 404 like plain FastAPI, the
    two parametrisations would collapse into one and the topology dependence this
    file just got burned by would silently return. This is the assertion the
    deleted file called `test_both_environments_really_do_differ`.
    """
    answers = {}
    for ship_dist in (False, True):
        root = tmp_path / f"root-{ship_dist}"
        (root / "frontend").mkdir(parents=True)
        if ship_dist:
            (root / "frontend" / "dist").mkdir()
            (root / "frontend" / "dist" / "index.html").write_text("<!doctype html><html>SPA shell</html>")
        monkeypatch.setattr(config, "PROJECT_ROOT", root)
        with TestClient(create_app()) as c:
            answers[ship_dist] = (c.get("/definitely-not-a-route").status_code,
                                  c.get("/definitely-not-a-route").text[:40])

    assert answers[False][0] == 404, answers
    assert answers[True][0] == 200, answers
    assert "SPA shell" in answers[True][1], answers
    assert answers[False] != answers[True], (
        "the two shapes answered identically, so the walk-out loop is only "
        f"covering one of them: {answers}"
    )


def test_a_2xx_that_is_not_an_object_is_a_502(monkeypatch):
    """`response.json()` can return a list, a string, a number or `null`.

    With a `-> dict` return annotation, FastAPI infers `response_model=dict` and
    raises `ResponseValidationError`, which has no registered handler -- so the
    browser gets a bare **500 text/plain**, a third failure shape on a route whose
    docstring promises the site has "exactly one failure shape to handle", and the
    offending upstream value lands in the server traceback. Neither the annotation
    nor the behaviour was pinned: dropping the annotation entirely, and widening
    it to `dict | list | str | int | None`, each passed 34/34.
    """
    for payload in (b"[1, 2, 3]", b'"just a string"', b"null", b"7"):
        def fake_get(url, _payload=payload, **kwargs):
            response = requests.Response()
            response.status_code = 200
            response._content = _payload
            return response

        monkeypatch.setattr(explain_module.requests, "get", fake_get)
        with TestClient(create_app()) as c:
            res = c.get(f"/api/explain/nba/{GAME_ID}")
        assert res.status_code == 502, f"[{payload!r}] {res.status_code}: {res.text[:200]!r}"
        assert payload.decode() not in res.text or payload == b"null", res.text[:200]


@pytest.mark.parametrize("ship_dist", [False, True], ids=["no-dist", "with-dist"])
def test_the_apps_own_routes_still_answer(monkeypatch, tmp_path, ship_dist):
    """So the guard above cannot be satisfied by breaking a neighbour.

    This existed in the file this one replaced. Without it, a change that made the
    app fail to start, or that shadowed `/facts/*`, would leave the route-table
    assertion passing on an app that answers nothing useful.

    **The status code is not enough, and in the production shape it is not even
    true.** `SPAStaticFiles` answers 200 with the app shell for any path it does not
    recognise, so with `frontend/dist` present, deleting
    `app.include_router(facts_router)` makes `/facts/upcoming` answer **200** --
    and this guard passed, while failing in the no-dist shape. That is the
    Critical's own defect class (a status assertion satisfied by the static-file
    fallback) reintroduced in the guard this commit restored, in a file that had
    just learned it twice. So the BODY is asserted: a route that exists returns
    JSON, and a shell does not.
    """
    root = tmp_path / f"root-{ship_dist}"
    (root / "frontend").mkdir(parents=True)
    if ship_dist:
        (root / "frontend" / "dist").mkdir()
        (root / "frontend" / "dist" / "index.html").write_text(
            "<!doctype html><html lang=en><body>SPA shell</body></html>")
    monkeypatch.setattr(config, "PROJECT_ROOT", root)

    with TestClient(create_app()) as c:
        health = c.get("/health")
        assert health.status_code == 200, health.text[:200]
        assert isinstance(health.json(), dict), (
            f"the SPA shell answered /health: {health.text[:120]!r}")
        facts_res = c.get("/facts/upcoming")
        assert facts_res.status_code == 200, facts_res.text[:200]
        assert isinstance(facts_res.json(), dict), (
            f"the SPA shell answered /facts/upcoming: {facts_res.text[:120]!r}")


def test_an_unparseable_timeout_does_not_take_the_app_down(monkeypatch):
    """`float(os.getenv(...))` at import raises on a typo, and `app.py` imports
    this module -- so `EXPLAINER_TIMEOUT_S=` (set-but-empty, a realistic compose
    accident) or `=abc` would take the WHOLE API down, not just this route. And
    `=0` parses fine, then makes `requests` raise a bare `ValueError` that is not
    a `RequestException`, escaping the handler as a 500: a fourth failure shape.

    So the value is validated at import, in both directions.
    """
    for bad in ("", "abc", "0", "-1", "600", "1e400"):
        monkeypatch.setenv("EXPLAINER_TIMEOUT_S", bad)
        try:
            # It must FALL BACK, not raise. The first version of this test accepted
            # either, so replacing the `except (TypeError, ValueError)` clause with
            # `except ZeroDivisionError` left all 55 tests green -- and that
            # mutation is the one that takes the whole API down, which is the
            # entire reason the fallback exists. Falling back costs one panel;
            # raising costs the site.
            reloaded = importlib.reload(explain_module)
            assert reloaded.EXPLAINER_TIMEOUT_S == explain_module._DEFAULT_TIMEOUT_S, (
                f"EXPLAINER_TIMEOUT_S={bad!r} resolved to "
                f"{reloaded.EXPLAINER_TIMEOUT_S}, expected the default "
                f"{explain_module._DEFAULT_TIMEOUT_S}"
            )
        finally:
            monkeypatch.delenv("EXPLAINER_TIMEOUT_S", raising=False)
            importlib.reload(explain_module)


def test_a_slash_in_the_sport_is_refused(monkeypatch):
    """The one character that changes the URL's STRUCTURE rather than its content.

    `quote`'s default is `safe='/'`, so `quote(sport)` would leave a slash intact
    and the sport segment would become two segments -- a request for a different
    path than the caller named. Through HTTP `sport` is `[^/]+` so this is
    unreachable, but the route function is also called directly (by the tests
    above, and by anything that imports it), so the guard is what makes the
    quoting defence in depth rather than the only control.
    """
    sent: list[str] = []

    def fake_get(url, **kwargs):
        sent.append(url)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(UPSTREAM_BODY).encode()
        return response

    monkeypatch.setattr(explain_module.requests, "get", fake_get)

    from fastapi import HTTPException

    # Plain "nba/x", NOT "nba/../x". The latter also contains "..", so it trips the
    # walk-out half as well and the `/` half is never isolated -- deleting
    # `or "/" in sport` left every test green. One token, and it is the difference
    # between this guard existing and not.
    with pytest.raises(HTTPException) as caught:
        explain_module.explain("nba/x", GAME_ID)
    assert caught.value.status_code == 502
    assert not sent, f"a request was sent for a refused sport: {sent}"


def test_the_explainer_url_is_read_from_the_environment(monkeypatch):
    """Read at *import*, not per request.

    Worth pinning because it is easy to assume the opposite: the first version of
    the quoting test called `monkeypatch.setenv("EXPLAINER_URL", ...)` after the
    module was already imported and then asserted against that value. It failed
    for the right reason (the quoting was correct) and the wrong reason (the host
    in the URL was the compiled-in default).

    Note for the next reader: `importlib.reload` re-executes the module, so
    `explain_module.router` becomes a DIFFERENT object from the one `app.py` holds.
    `create_app()` closed over the original at import, so behaviour is
    deterministic -- but do not `from nba_predictor.api.explain import router`
    after reloading and expect the same object.
    """
    monkeypatch.setenv("EXPLAINER_URL", "http://elsewhere.test:9999/")
    reloaded = importlib.reload(explain_module)
    try:
        assert reloaded.EXPLAINER_URL == "http://elsewhere.test:9999", (
            f"EXPLAINER_URL={reloaded.EXPLAINER_URL!r}; the trailing slash should "
            f"be stripped so the join does not produce a double slash"
        )
    finally:
        monkeypatch.delenv("EXPLAINER_URL", raising=False)
        importlib.reload(explain_module)


def test_an_empty_id_is_refused():
    """`/api/explain/nba/` forwarded to `/explain/nba/`.

    Not a traversal and not a leak -- the path is the one the caller named -- but the
    guard's own comment says "an id is one relative segment; anything else is
    refused", and an empty string is not one. A comment that overstates what the code
    does is the same defect as a test that asserts less than its name says, so the
    code was brought in line rather than the comment weakened.
    """
    sent: list[str] = []

    def fake_get(url, **kwargs):
        sent.append(url)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(UPSTREAM_BODY).encode()
        return response

    m = pytest.MonkeyPatch()
    m.setattr(explain_module.requests, "get", fake_get)
    try:
        with pytest.raises(HTTPException) as caught:
            explain_module.explain("nba", "")
        assert caught.value.status_code == 502
        assert not sent, f"a request was sent for an empty id: {sent}"
    finally:
        m.undo()


def test_the_timeout_helper_survives_a_non_string():
    """`except (TypeError, ValueError)` looks redundant -- `os.getenv` with a default
    always returns a `str` -- and narrowing it to `except ValueError` was silent.

    Kept, and pinned. `_positive_float` is a module-level helper with a typed
    parameter, so a future caller passing `None` is the kind of change that looks
    safe and takes the whole API down at import, which is the failure the helper
    exists to prevent. A defensive branch nobody exercises is indistinguishable from
    a redundant one until it is exercised.
    """
    assert explain_module._positive_float(None, "X") == explain_module._DEFAULT_TIMEOUT_S
    assert explain_module._positive_float(float("nan"), "X") == explain_module._DEFAULT_TIMEOUT_S
    assert explain_module._positive_float(float("inf"), "X") == explain_module._DEFAULT_TIMEOUT_S
    assert explain_module._positive_float("abc", "X") == explain_module._DEFAULT_TIMEOUT_S
