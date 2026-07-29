<div align="center">

# 🖥️ The Hub

**FastAPI application serving every desk behind one password gate.**

![FastAPI](https://img.shields.io/badge/FastAPI-1D9E75?style=flat-square&labelColor=0C110F) ![marimo](https://img.shields.io/badge/marimo-1D9E75?style=flat-square&labelColor=0C110F) ![ASGI](https://img.shields.io/badge/ASGI-1D9E75?style=flat-square&labelColor=0C110F) ![Tailscale](https://img.shields.io/badge/Tailscale-1D9E75?style=flat-square&labelColor=0C110F)

<sub>A desk of <a href="https://tobibui1904.github.io/quant-lab-site/">Quant Lab</a> · branch <code>hub</code></sub>

</div>

---

> Twelve notebooks, one origin, one login — and a gate that covers websockets, not just pages.

`main.py` mounts every desk as a marimo ASGI app under one FastAPI server. `site_gate.py` wraps it with a password gate, a public health endpoint, and scoped CORS. 38 tests cover the gate.

### The gate, and why it is pure ASGI

marimo is **entirely websocket-driven** — every cell execution and widget
update travels over a persistent socket. Starlette's `BaseHTTPMiddleware` only
sees `scope["type"] == "http"`, so a gate built on it would present a working
login page while leaving every notebook kernel open to anyone with the URL.

`site_gate.py` is therefore raw ASGI and handles all three scope types:

| Scope | Behaviour |
|---|---|
| `lifespan` | always forwarded — the hub starts a background equity sampler there |
| `http` | 303 to `/login` without a valid cookie |
| `websocket` | closed with code 1008 without a valid cookie |

Exempt paths are exact matches only: `/healthz`, `/login`, `/logout`. No prefix
matching, so `/healthz-x` and `/login/../forex` fail closed.

### Sessions

Cookies carry `{expiry}.{hmac_sha256(expiry, secret)}` — stateless, standard
library only, `HttpOnly` + `Secure` + `SameSite=Lax`.

A blank or whitespace `SITE_PASSWORD` disables the gate entirely and the hub
says so loudly at startup:

```
Site gate: DISABLED -- the hub is UNPROTECTED. Do NOT expose this publicly.
```

Logout clears **your** cookie; because tokens are stateless there is no
blocklist, so revoking a stolen cookie means restarting with `SESSION_SECRET`
unset to mint a fresh signing key.

### Publishing

The hub runs locally and reaches the internet through a Tailscale Funnel — free,
stable HTTPS, no open firewall port. The static front pages are on `main` and
poll `/healthz` cross-origin to show "Enter the Lab" or "Desk closed".

### Files on this branch

| |
|---|
| `layouts/` |
| `main.py` |
| `pyproject.toml` |
| `site_gate.py` |
| `test_site_gate.py` |
| `theme.css` |
| `theme_head.html` |

<sub>This branch also carries the published site from `main`.</sub>

---

<details>
<summary><b>Running this desk</b></summary>

<br>

These are [marimo](https://marimo.io) notebooks, not scripts. Each one is a
reactive dashboard: cells re-execute when their inputs change, so there is no
hidden run order to remember.

```bash
pip install marimo
marimo run main.py          # read-only dashboard
marimo edit main.py         # editable notebook
```

Served together, all desks live behind one FastAPI hub with a password gate
(branch [`hub`](../../tree/hub)). Credentials come from a local `.env` that is
never committed -- see `.env.example` on the `hub` branch for the variable
names.

</details>

<details>
<summary><b>Honest limitations</b></summary>

<br>

The gate is a single shared password, not per-user accounts — it is appropriate for a personal portfolio site, not for multi-user access. Sessions cannot be revoked individually. The tunnel only carries traffic while the machine is awake, so the site is designed to degrade to a "Desk closed" state rather than a broken link.

</details>

---

<div align="center">
<sub>Part of <b>Quant Lab</b> by <a href="https://github.com/tobibui1904">Tobi Bui</a> ·
<a href="https://tobibui1904.github.io/quant-lab-site/">Live site</a></sub>
</div>
