from site_gate import COOKIE_NAME, DEFAULT_TTL, make_token, verify_token

SECRET = "test-secret"


def test_fresh_token_verifies():
    assert verify_token(make_token(SECRET), SECRET) is True


def test_token_signed_with_other_secret_is_rejected():
    assert verify_token(make_token("other-secret"), SECRET) is False


def test_expired_token_is_rejected():
    assert verify_token(make_token(SECRET, ttl_seconds=-1), SECRET) is False


def test_tampered_signature_is_rejected():
    expiry, sig = make_token(SECRET).rsplit(".", 1)
    flipped = "0" if sig[0] != "0" else "1"
    assert verify_token(f"{expiry}.{flipped}{sig[1:]}", SECRET) is False


def test_tampered_expiry_is_rejected():
    expiry, sig = make_token(SECRET, ttl_seconds=-1).rsplit(".", 1)
    assert verify_token(f"{int(expiry) + 99999}.{sig}", SECRET) is False


def test_malformed_tokens_are_rejected():
    for bad in (None, "", "nodot", "abc.def", "..", "9999999999."):
        assert verify_token(bad, SECRET) is False


def test_cookie_name_and_ttl_constants():
    assert COOKIE_NAME == "quantlab_session"
    assert DEFAULT_TTL == 86400


from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route, WebSocketRoute

from site_gate import SiteGate

PASSWORD = "hunter2"
ORIGIN = "https://tobibui1904.github.io"


async def _protected(request):
    return PlainTextResponse("protected")


async def _ws_endpoint(websocket):
    await websocket.accept()
    await websocket.send_text("connected")
    await websocket.close()


def _stub_app():
    """Stands in for the hub: one HTTP route, one websocket route."""
    return Starlette(
        routes=[Route("/", _protected), WebSocketRoute("/ws", _ws_endpoint)]
    )


def _client(**kw):
    kw.setdefault("password", PASSWORD)
    kw.setdefault("secret", SECRET)
    kw.setdefault("allow_origin", ORIGIN)
    return TestClient(SiteGate(_stub_app(), **kw))


def test_healthz_is_public_while_logged_out():
    r = _client().get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_healthz_allows_the_pages_origin():
    r = _client().get("/healthz", headers={"Origin": ORIGIN})
    assert r.headers["access-control-allow-origin"] == ORIGIN


def test_healthz_denies_any_other_origin():
    r = _client().get("/healthz", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_healthz_preflight_is_answered():
    r = _client().options("/healthz", headers={"Origin": ORIGIN})
    assert r.status_code == 204
    assert r.headers["access-control-allow-origin"] == ORIGIN


def test_login_page_renders_a_form():
    r = _client().get("/login")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert 'name="password"' in r.text


def test_wrong_password_sets_no_cookie():
    r = _client().post("/login", data={"password": "wrong"})
    assert r.status_code == 401
    assert COOKIE_NAME not in r.cookies


def test_correct_password_sets_a_valid_cookie():
    r = _client().post(
        "/login", data={"password": PASSWORD}, follow_redirects=False
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/"
    assert verify_token(r.cookies[COOKIE_NAME], SECRET) is True


def test_cookie_carries_the_security_flags():
    r = _client().post(
        "/login", data={"password": PASSWORD}, follow_redirects=False
    )
    raw = r.headers["set-cookie"].lower()
    assert "httponly" in raw
    assert "secure" in raw
    assert "samesite=lax" in raw


def test_login_page_needs_no_external_assets():
    """Must render before auth, so it cannot depend on gated resources."""
    body = _client().get("/login").text
    assert "http://" not in body
    assert "https://" not in body


def test_oversized_login_body_is_rejected():
    """A body past the cap must be rejected even when it carries the correct
    password -- proves the read loop bails out instead of accumulating and
    parsing an unbounded body. (A body with only a wrong password would 401
    either way and wouldn't prove the cap is doing anything.)"""
    padded = f"password={PASSWORD}&pad=" + ("x" * 3000)
    r = _client().post(
        "/login",
        content=padded.encode(),
        headers={"content-type": "application/x-www-form-urlencoded"},
        follow_redirects=False,
    )
    assert r.status_code == 401
    assert COOKIE_NAME not in r.cookies


def test_invalid_utf8_login_body_yields_401_not_500():
    """Malformed bytes in the POST body must fail gracefully, not raise."""
    r = _client().post(
        "/login",
        content=b"password=\xff\xfe",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 401
    assert COOKIE_NAME not in r.cookies


import pytest
from starlette.websockets import WebSocketDisconnect


def _authed_client(**kw):
    client = _client(**kw)
    client.cookies.set(COOKIE_NAME, make_token(SECRET))
    return client


def test_http_without_cookie_redirects_to_login():
    r = _client().get("/", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_http_with_cookie_reaches_the_app():
    r = _authed_client().get("/")
    assert r.status_code == 200
    assert r.text == "protected"


def test_http_with_expired_cookie_redirects_to_login():
    client = _client()
    client.cookies.set(COOKIE_NAME, make_token(SECRET, ttl_seconds=-1))
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303


def test_websocket_is_rejected_without_a_cookie():
    """The trap: HTTP-only middleware would let this straight through."""
    with pytest.raises(WebSocketDisconnect) as caught:
        with _client().websocket_connect("/ws"):
            pass
    assert caught.value.code == 1008


def test_websocket_is_accepted_with_a_cookie():
    with _authed_client().websocket_connect("/ws") as ws:
        assert ws.receive_text() == "connected"


def test_gate_disabled_when_no_password_is_set():
    """localhost:8000 must behave exactly as before."""
    client = TestClient(SiteGate(_stub_app(), password=None))
    assert client.get("/").text == "protected"
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_text() == "connected"


def test_healthz_still_public_when_gate_enabled():
    assert _client().get("/healthz").json() == {"ok": True}


def test_lifespan_passes_through():
    """The equity sampler starts in the hub's lifespan -- it must survive.

    The installed Starlette (1.3.1) has dropped Router.on_startup in favor
    of the lifespan=<asynccontextmanager> style -- which is also exactly
    what main.py's real hub uses (see `_lifespan` in main.py). This test
    is adapted to that API rather than the brief's `on_startup.append(...)`
    snippet, which no longer exists on this version's Router.
    """
    from contextlib import asynccontextmanager

    started = []

    @asynccontextmanager
    async def lifespan(app):
        started.append(True)
        yield

    stub = Starlette(
        routes=[Route("/", _protected), WebSocketRoute("/ws", _ws_endpoint)],
        lifespan=lifespan,
    )
    with TestClient(SiteGate(stub, password=PASSWORD, secret=SECRET)):
        pass
    assert started == [True]


# --- Review round 1 -----------------------------------------------------
#
# 1. `SITE_PASSWORD=` in a filled-out .env.example loads as password="",
#    which must disable the gate exactly like password=None -- not leave it
#    reporting as "enabled" while actually admitting any blank submission.
# 2. `_authed`'s cookie-name match must be exact equality, not a suffix or
#    substring match; this is a regression net on that exact property.


def test_blank_password_does_not_redirect_to_login():
    """A blank password must not leave the gate looking armed (redirecting
    unauthenticated visitors to /login) while actually being bypassable by
    an empty form submission. It must simply be off, like password=None."""
    client = TestClient(SiteGate(_stub_app(), password="", secret=SECRET))
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 200
    assert r.text == "protected"


def test_blank_password_behaves_identically_to_no_password():
    """password="" and password=None must be indistinguishable: both fully
    transparent, for http and websocket alike."""
    client = TestClient(SiteGate(_stub_app(), password="", secret=SECRET))
    assert client.get("/").text == "protected"
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_text() == "connected"


def test_cookie_name_must_match_exactly_not_just_suffix():
    """`evil_quantlab_session` carrying an otherwise-valid token must not
    authenticate -- only an exact COOKIE_NAME match may. Guards against a
    future `_authed` regressing to `endswith`/`in` cookie-name matching."""
    client = _client()
    client.cookies.set("evil_" + COOKIE_NAME, make_token(SECRET))
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_duplicate_cookie_first_match_is_verified_not_assumed():
    """When the raw Cookie header carries two same-named entries, the gate
    must verify whichever it matches rather than treating any occurrence of
    the right name as automatically good."""
    client = _client()
    valid = make_token(SECRET)
    r = client.get(
        "/",
        headers={"Cookie": f"{COOKIE_NAME}=garbage; {COOKIE_NAME}={valid}"},
        follow_redirects=False,
    )
    assert r.status_code == 303


def test_main_exposes_a_gated_app(monkeypatch):
    """main.app is what uvicorn serves; it must be the gated wrapper."""
    import importlib

    monkeypatch.setenv("SITE_PASSWORD", "plan-test")
    monkeypatch.setenv("SESSION_SECRET", SECRET)
    import main

    importlib.reload(main)
    assert isinstance(main.app, SiteGate)
    assert main.app.app is main.hub

    client = TestClient(main.app)
    assert client.get("/healthz").json() == {"ok": True}
    assert client.get("/", follow_redirects=False).status_code == 303


# --- Final fix wave (2026-07-28) -----------------------------------------
#
# Finding 1+2: nothing told an operator whether the gate was armed. The
# main.py startup print is exercised manually (see final-fix-report.md) --
# it can't be unit tested without either running uvicorn.run() for real or
# restructuring main.py, both out of scope here. This test instead pins the
# one behavior that print line depends on: a whitespace-only password must
# leave the gate disabled, so app.password reads falsy and the message
# correctly says DISABLED. Without this test, a future weakening of
# `password.strip() or None` to `password or None` would leave "   " truthy
# (gate reports ARMED, but any blank/whitespace submission is rejected by
# compare_digest against a non-empty stripped secret -- a confusing
# half-broken state) and nothing would catch it.
def test_whitespace_only_password_disables_gate():
    client = TestClient(SiteGate(_stub_app(), password="   ", secret=SECRET))
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 200
    assert r.text == "protected"
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_text() == "connected"


# Finding 3(a): a stolen session cookie could not be revoked short of a
# restart. /logout must clear the cookie and send the caller to /login,
# whether or not they currently hold a valid one.
def test_logout_clears_cookie_and_redirects_to_login():
    client = _authed_client()
    r = client.get("/logout", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"
    raw = r.headers["set-cookie"].lower()
    assert f"{COOKIE_NAME}=".lower() in raw
    assert "max-age=0" in raw


def test_logout_works_without_a_cookie():
    client = _client()
    r = client.get("/logout", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_logged_out_cookie_no_longer_authenticates():
    """The cookie /logout hands back must actually fail verification --
    proves this isn't just a redirect that leaves the old cookie usable."""
    client = _authed_client()
    logout_resp = client.get("/logout", follow_redirects=False)
    cleared = logout_resp.cookies.get(COOKIE_NAME, "")
    assert not cleared or verify_token(cleared, SECRET) is False


# Finding 4: the configured password was stripped but the submitted one
# was not, so SITE_PASSWORD=" s3cret " locked out a user typing the spaces.
def test_password_matches_when_submitted_with_surrounding_whitespace():
    client = TestClient(SiteGate(_stub_app(), password=" s3cret ", secret=SECRET))
    r = client.post(
        "/login", data={"password": "  s3cret  "}, follow_redirects=False
    )
    assert r.status_code == 303
    assert verify_token(r.cookies[COOKIE_NAME], SECRET) is True


# Finding 6: a cache between a visitor and the tunnel could serve a stale
# {"ok": true} after the PC sleeps, so /healthz must forbid caching.
def test_healthz_forbids_caching():
    r = _client().get("/healthz")
    assert r.headers["cache-control"] == "no-store"


# Finding 5: the real hub's gated surface (beyond "/" and "/healthz") had
# almost no test coverage. The endpoints below close real trading
# positions; a notebook path stands in for the twelve mounted kernels.
# Kept last/adjacent to test_main_exposes_a_gated_app so module reloading
# cannot corrupt earlier tests. Imports main -- slow, and expected to be.
def test_real_hub_gates_close_endpoints_and_notebooks(monkeypatch):
    import importlib

    monkeypatch.setenv("SITE_PASSWORD", "plan-test")
    monkeypatch.setenv("SESSION_SECRET", SECRET)
    import main

    importlib.reload(main)

    client = TestClient(main.app)
    for method, path in (
        ("post", "/api/oanda/close/EUR_USD"),
        ("post", "/api/alpaca/close/AAPL"),
        ("get", "/forex"),
    ):
        r = getattr(client, method)(path, follow_redirects=False)
        assert r.status_code == 303, f"{method.upper()} {path} was not gated"
        assert r.headers["location"] == "/login"
