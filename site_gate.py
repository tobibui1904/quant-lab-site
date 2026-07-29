"""Public edge for the Quant Lab hub.

Owns the only three things an outside visitor can touch: /healthz (liveness,
always open), /login (the password form), and the gate standing in front of
everything else -- including marimo's websockets.

Pure ASGI on purpose. Starlette's BaseHTTPMiddleware only sees
scope["type"] == "http", so websockets would slip past an HTTP-only gate
unauthenticated and every notebook kernel would be open to the public.
"""
import hashlib
import hmac
import json
import secrets
import time
import urllib.parse

COOKIE_NAME = "quantlab_session"
DEFAULT_TTL = 86400  # 24h
MAX_LOGIN_BODY_BYTES = 2048  # a login form needs at most a few KB

LOGIN_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Quant Lab</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{min-height:100vh;display:flex;align-items:center;justify-content:center;
background:#0C110F;color:#E9EEEB;font-family:Georgia,serif}
form{width:300px;padding:34px;border:1px solid #1F2A24;border-radius:10px;
background:#0F1512}
h1{font-size:23px;font-style:italic;font-weight:400;margin-bottom:6px}
p{font-size:12px;color:#5F6F66;margin-bottom:20px;letter-spacing:.16em;
text-transform:uppercase;font-family:monospace}
input{width:100%;padding:11px;margin-bottom:14px;border:1px solid #2C3B33;
border-radius:6px;background:#0C110F;color:#E9EEEB;font-size:14px}
button{width:100%;padding:11px;border:0;border-radius:6px;background:#1D9E75;
color:#06130E;font-size:12px;letter-spacing:.14em;text-transform:uppercase;
font-family:monospace;cursor:pointer}
.err{color:#E4796B;font-size:12px;margin-bottom:12px;font-family:monospace}
</style></head><body>
<form method="post" action="/login">
<h1>Quant Lab</h1><p>Desk access</p>
__ERROR__
<input type="password" name="password" placeholder="Password" autofocus>
<button type="submit">Enter</button>
</form></body></html>"""


def _sign(expiry, secret):
    return hmac.new(
        secret.encode(), str(expiry).encode(), hashlib.sha256
    ).hexdigest()


def make_token(secret, ttl_seconds=DEFAULT_TTL):
    """Return "{expiry_epoch}.{hex_hmac}"."""
    expiry = int(time.time()) + ttl_seconds
    return f"{expiry}.{_sign(expiry, secret)}"


def verify_token(token, secret):
    """True only for an untampered, unexpired token."""
    if not token or "." not in token:
        return False
    expiry_str, sig = token.rsplit(".", 1)
    try:
        expiry = int(expiry_str)
    except ValueError:
        return False
    if not hmac.compare_digest(sig, _sign(expiry, secret)):
        return False
    return expiry > int(time.time())


async def _send_response(send, status, body, headers=None):
    raw = [(b"content-type", b"application/json")]
    for key, value in (headers or {}).items():
        raw.append((key.encode(), value.encode()))
    payload = json.dumps(body).encode()
    raw.append((b"content-length", str(len(payload)).encode()))
    await send({"type": "http.response.start", "status": status, "headers": raw})
    await send({"type": "http.response.body", "body": payload})


def _header(scope, name):
    wanted = name.lower().encode()
    for key, value in scope.get("headers", []):
        if key.lower() == wanted:
            return value.decode("latin-1")
    return None


class SiteGate:
    """Wraps the hub. Everything public goes through here."""

    def __init__(
        self,
        app,
        password=None,
        secret=None,
        allow_origin=None,
        ttl_seconds=DEFAULT_TTL,
    ):
        self.app = app
        # A blank/whitespace-only password (e.g. SITE_PASSWORD= in a filled
        # -out .env.example, which load_dotenv turns into "" rather than
        # leaving unset) must disable the gate exactly like password=None --
        # never leave it looking armed while actually admitting any blank
        # login submission via `compare_digest(submitted, self.password or "")`.
        self.password = (password.strip() or None) if password else None
        self.secret = secret or secrets.token_urlsafe(32)
        self.allow_origin = allow_origin
        self.ttl_seconds = ttl_seconds

    def _cors(self, scope):
        """Echo the allowed origin only when it matches exactly."""
        origin = _header(scope, "origin")
        if origin and self.allow_origin and origin == self.allow_origin:
            return {
                "access-control-allow-origin": origin,
                "vary": "Origin",
            }
        return {}

    async def _healthz(self, scope, receive, send):
        headers = self._cors(scope)
        headers["cache-control"] = "no-store"
        if scope.get("method") == "OPTIONS":
            headers["access-control-allow-methods"] = "GET, OPTIONS"
            await send(
                {
                    "type": "http.response.start",
                    "status": 204,
                    "headers": [
                        (k.encode(), v.encode()) for k, v in headers.items()
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return
        await _send_response(send, 200, {"ok": True}, headers)

    async def _login(self, scope, receive, send):
        if scope.get("method") == "POST":
            body = b""
            oversized = False
            while True:
                message = await receive()
                body += message.get("body", b"")
                if len(body) > MAX_LOGIN_BODY_BYTES:
                    oversized = True
                    break
                if not message.get("more_body"):
                    break

            if oversized:
                return await self._login_page(
                    send, status=401, error="Incorrect password"
                )

            try:
                decoded_body = body.decode()
            except UnicodeDecodeError:
                return await self._login_page(
                    send, status=401, error="Incorrect password"
                )

            submitted = urllib.parse.parse_qs(decoded_body).get(
                "password", [""]
            )[0].strip()
            if secrets.compare_digest(submitted, self.password or ""):
                token = make_token(self.secret, self.ttl_seconds)
                cookie = (
                    f"{COOKIE_NAME}={token}; Path=/; Max-Age={self.ttl_seconds}; "
                    "HttpOnly; Secure; SameSite=Lax"
                )
                await send(
                    {
                        "type": "http.response.start",
                        "status": 303,
                        "headers": [
                            (b"location", b"/"),
                            (b"set-cookie", cookie.encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": b""})
                return
            return await self._login_page(send, status=401, error="Incorrect password")
        await self._login_page(send, status=200, error="")

    async def _login_page(self, send, status, error):
        markup = LOGIN_HTML.replace(
            "__ERROR__", f'<div class="err">{error}</div>' if error else ""
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"text/html; charset=utf-8"),
                    (b"content-length", str(len(markup)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": markup})

    def _authed(self, scope):
        raw = _header(scope, "cookie") or ""
        for part in raw.split(";"):
            name, _, value = part.strip().partition("=")
            if name == COOKIE_NAME:
                return verify_token(value, self.secret)
        return False

    async def _logout(self, send):
        """Clear the session cookie and send the caller back to /login.

        Works whether or not the caller currently holds a valid cookie --
        expiring an absent or already-invalid cookie is harmless.

        This does NOT revoke a stolen cookie. Tokens are stateless HMAC with
        no server-side record, so logout only deletes the copy in *this*
        browser; a token someone else already holds stays valid until its
        expiry (DEFAULT_TTL, 24h). The sole way to invalidate every
        outstanding session is to restart the hub with SESSION_SECRET unset,
        which mints a fresh random signing key.
        """
        cookie = (
            f"{COOKIE_NAME}=; Path=/; Max-Age=0; "
            "HttpOnly; Secure; SameSite=Lax"
        )
        await send(
            {
                "type": "http.response.start",
                "status": 303,
                "headers": [
                    (b"location", b"/login"),
                    (b"set-cookie", cookie.encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    async def _redirect_to_login(self, send):
        await send(
            {
                "type": "http.response.start",
                "status": 303,
                "headers": [(b"location", b"/login")],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            return await self.app(scope, receive, send)
        if scope.get("path") == "/healthz" and scope["type"] == "http":
            return await self._healthz(scope, receive, send)
        if scope.get("path") == "/login" and scope["type"] == "http":
            return await self._login(scope, receive, send)
        if scope.get("path") == "/logout" and scope["type"] == "http":
            return await self._logout(send)
        if self.password is None:
            return await self.app(scope, receive, send)
        if self._authed(scope):
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            await receive()  # consume websocket.connect before closing
            return await send({"type": "websocket.close", "code": 1008})
        await self._redirect_to_login(send)
