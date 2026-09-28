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
from fastapi.testclient import TestClient

from nba_predictor.api import explain as explain_module
from nba_predictor.api.app import create_app

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
        res = c.get("/api/explain/nba/401585")

    assert res.status_code == 200, res.text
    assert res.json() == UPSTREAM_BODY
    assert len(upstream) == 1
    assert upstream[0]["url"].endswith("/explain/nba/401585"), upstream[0]["url"]


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

#: A non-2xx the proxy must turn into a 502. The fake RETURNS a response with
#: this status -- it does not raise. That distinction is the whole test: when the
#: fake raised `HTTPError` instead, `requests.get()` never returned and the
#: proxy's own `if not (200 <= response.status_code < 300)` check was never
#: executed, so removing that check left every test green.
UPSTREAM_ERROR_STATUSES = [400, 401, 403, 404, 422, 429, 500, 502, 503]


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
        res = c.get("/api/explain/nba/401585")

    assert res.status_code == 502, f"[upstream {status}] {res.text[:200]!r}"
    assert secret not in res.text, f"[upstream {status}] an upstream body was echoed"
    assert "/srv/internal" not in res.text, f"[upstream {status}] an upstream path was echoed"


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
        res = c.get("/api/explain/nba/401585")

    assert res.status_code == 502, f"[upstream {status}] {res.text[:200]!r}"
    assert secret not in res.text, f"[upstream {status}] an upstream body was echoed"
    assert str(status) not in res.text, f"[upstream {status}] the status was echoed"


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
        res = c.get("/api/explain/nba/401585")

    assert res.status_code == 502, res.text[:200]
    assert "not json" not in res.text


def test_a_dead_explainer_is_a_502_not_a_500(monkeypatch):
    def fake_get(url, **kwargs):
        raise requests.ConnectionError("explainer is down")

    monkeypatch.setattr(explain_module.requests, "get", fake_get)

    with TestClient(create_app()) as c:
        res = c.get("/api/explain/nba/401585")

    assert res.status_code == 502, res.text[:200]
    assert "explainer is down" not in res.text, "an internal reason leaked to the browser"


# --- the route is not a forwarder to anything on that host -----------------


@pytest.mark.parametrize(
    "path",
    [
        "/api/explain/nba/../../secrets",
        "/api/explain/../admin/x",
        "/api/explain/nba/%2e%2e/%2e%2e/etc/passwd",
    ],
)
def test_a_path_that_walks_out_never_reaches_the_explainer(monkeypatch, path):
    """A `..` in either segment would make this a forwarder to *any path* on the
    explainer's host, so no request may leave for one.

    **The status code is not pinned to 502, and that is the point.** A literal
    `..` is normalised away by the HTTP client before routing, so those cases are
    answered 404 -- the route never matches, which is *stronger* than the guard
    firing. A percent-encoded `%2e%2e` survives normalisation and does reach the
    route, which is where the `if ".." in ...` check earns its keep. The first
    version of this test asserted 502 for all three and failed on the two that
    never arrive, which is how a stricter-looking test can encode a weaker
    invariant: pinning the status would have forced the two cases to be *made*
    to return 502, i.e. to be routed, i.e. to reach the guard at all.

    So the invariant asserted is the one that is actually true and actually
    matters: nothing was sent, and nothing that looks like a summary came back.
    """
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

    assert not sent, f"a request was sent for {path!r} -> {sent}"
    assert "leaked" not in res.text, res.text[:200]
    assert res.status_code in (404, 502), f"[{path}] {res.status_code}: {res.text[:200]!r}"


def test_a_walk_out_that_does_reach_the_route_is_caught_by_the_guard(monkeypatch):
    """The percent-encoded case, isolated -- because it is the only one that
    reaches the route, so it is the only one that proves the guard exists. If the
    two 404 cases above ever became 502s, this is the test that says why."""
    sent: list[str] = []

    def fake_get(url, **kwargs):
        sent.append(url)
        raise AssertionError("the guard let a walk-out through")

    monkeypatch.setattr(explain_module.requests, "get", fake_get)

    with TestClient(create_app()) as c:
        res = c.get("/api/explain/nba/%2e%2e/%2e%2e/etc/passwd")

    assert res.status_code == 502, f"{res.status_code}: {res.text[:200]!r}"
    assert not sent, sent


def test_the_explainer_url_is_read_from_the_environment(monkeypatch):
    """Read at *import*, not per request.

    Worth pinning because it is easy to assume the opposite: the first version of
    the quoting test called `monkeypatch.setenv("EXPLAINER_URL", ...)` after the
    module was already imported and then asserted against that value. It failed
    for the right reason (the quoting was correct) and the wrong reason (the host
    in the URL was the compiled-in default), which is a bad way to learn
    something. The override is a deployment-time knob, so import-time is
    correct -- but a test that cannot redirect the host must not pretend to.
    """
    import importlib

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
