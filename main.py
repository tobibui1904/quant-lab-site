import asyncio
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import time
from collections import deque
from contextlib import asynccontextmanager

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import marimo
import otc_watchlist
import hub_events
from site_gate import SiteGate

load_dotenv(pathlib.Path(__file__).resolve().parent / ".env")

HOST = "localhost"
PORT = 8000

# Notebooks are served from the same origin as the hub, so "path" doubles as
# the iframe src -- no absolute URL, no second port.
# "group" and the list order drive the ledger index in the sidebar; the
# printed numbers follow list position, so inserting an entry renumbers the
# ones below it. Nothing keys off those numbers.
# ids are frontend-internal (DOM ids + /api/status names for the gated pair).
NOTEBOOKS = [
    {
        "id":          "fundamental",
        "label":       "Fundamental",
        "icon":        "ti-building-bank",
        "group":       "Research",
        "description": "Filings, ratios & the option-strategy verdict for the active pair",
        "path":        "/fundamental",
    },
    {
        "id":          "macro_research",
        "label":       "Macro Research",
        "icon":        "ti-flame",
        "group":       "Research",
        "description": "FRED factor models — sector betas under macro stress",
        "path":        "/macro_research",
    },
    {
        "id":          "otc_track",
        "label":       "OTC Track",
        "icon":        "ti-eye-off",
        "group":       "Research",
        "description": "Off-exchange tape — short volume, dark pools & short interest",
        "path":        "/otc_track",
        "locked":      True,
    },
    {
        "id":          "forex",
        "label":       "Forex Dashboard",
        "icon":        "ti-currency-dollar",
        "group":       "Markets",
        "description": "EUR/USD ML ensemble, executed through OANDA",
        "path":        "/forex",
    },
    {
        "id":          "letf",
        "label":       "LETF Backtest",
        "icon":        "ti-chart-candle",
        "group":       "Markets",
        "description": "Leveraged-ETF hedges — SQQQ overlays, backtested",
        "path":        "/letf",
    },
    {
        "id":          "crypto_inference",
        "label":       "Crypto",
        "icon":        "ti-coins",
        "group":       "Markets",
        "description": "MAPPO-LSTM agents trading BTC · ETH on Alpaca",
        "path":        "/crypto_inference",
    },
    {
        "id":          "pred_market",
        "label":       "Prediction Markets",
        "icon":        "ti-crystal-ball",
        "group":       "Markets",
        "description": "Polymarket desk driven by an LLM broker",
        "path":        "/pred_market",
    },
    {
        "id":          "treasury",
        "label":       "Treasury Securities",
        "icon":        "ti-certificate",
        "group":       "Markets",
        "description": "Bills, TIPS, FRNs & STRIPS — curves fitted with rateslib",
        "path":        "/treasury",
    },
    {
        "id":          "corporate",
        "label":       "Corporate Bonds",
        "icon":        "ti-building-bank",
        "group":       "Markets",
        "description": "Per-CUSIP IG bonds priced off issuer disclosure",
        "path":        "/corporate",
    },
    {
        "id":          "muni",
        "label":       "Municipal Bonds",
        "icon":        "ti-building-community",
        "group":       "Markets",
        "description": "Per-CUSIP munis with tax-equivalent yield",
        "path":        "/muni",
    },
    {
        "id":          "pair_trading",
        "label":       "Pair Trading",
        "icon":        "ti-wallet",
        "group":       "Strategies",
        "description": "Cointegration screens & live spread execution",
        "path":        "/known_pair_trading",
    },
    {
        "id":          "bull_call_spread",
        "label":       "Bull Call Spread",
        "icon":        "ti-trending-up",
        "group":       "Strategies",
        "description": "Defined-risk call debit spread on the bullish leg",
        "path":        "/bull_call_spread",
        "locked":      True,
    },
    {
        "id":          "bear_put_spread",
        "label":       "Bear Put Spread",
        "icon":        "ti-trending-down",
        "group":       "Strategies",
        "description": "Defined-risk put debit spread on the bearish leg",
        "path":        "/bear_put_spread",
        "locked":      True,
    },
]

# Reports written by fundamental.py, one per leg of the pair
REPORT_FILES = ["data/report_stock1.json", "data/report_stock2.json"]

# option_strategy.name is free-form LLM text, so match on keyword pairs
# ("Bear Put Debit Spread", "bear put spread", ... all resolve the same).
STRATEGY_TABS = [
    (("bull", "call"), "bull_call_spread"),
    (("bear", "put"),  "bear_put_spread"),
]


def strategy_tab_for(name):
    """Map an option_strategy name onto a notebook tab id, or None."""
    norm = "".join(c if c.isalnum() else " " for c in (name or "").lower())
    words = set(norm.split())
    for keywords, tab_id in STRATEGY_TABS:
        if all(k in words for k in keywords):
            return tab_id
    return None


def suggested_tabs():
    """Union of the tabs suggested across both legs of the pair.

    Both legs bearish -> ["bear_put_spread"] only.
    One bullish, one bearish -> both tabs.
    """
    tabs = []
    for fname in REPORT_FILES:
        p = pathlib.Path(fname)
        if not p.exists():
            continue
        try:
            report = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # mid-write or malformed; next poll picks it up
        tab_id = strategy_tab_for((report.get("option_strategy") or {}).get("name"))
        if tab_id and tab_id not in tabs:
            tabs.append(tab_id)
    return tabs


marimo_asgi = (
    marimo.create_asgi_app()
    .with_app(path="/fundamental",        root="desks/equities/fundamental.py")
    .with_app(path="/macro_research",        root="desks/macro/macro_research.py")
    .with_app(path="/otc_track",          root="desks/otc/OTC_Track.py")
    .with_app(path="/forex",              root="desks/forex/forex.py")
    .with_app(path="/letf",               root="desks/letf/letf.py")
    .with_app(path="/crypto_inference",               root="desks/crypto/crypto_inference.py")
    .with_app(path="/pred_market",               root="desks/prediction_markets/pred_market.py")
    .with_app(path="/treasury",       root="desks/fixed_income/treasury.py")
    .with_app(path="/corporate",      root="desks/fixed_income/corporate.py")
    .with_app(path="/muni",           root="desks/fixed_income/muni.py")
    .with_app(path="/known_pair_trading", root="desks/equities/known_pair_trading.py")
    .with_app(path="/bull_call_spread",   root="desks/options/bull_call_spread.py")
    .with_app(path="/bear_put_spread",   root="desks/options/bear_put_spread.py")
    
).build()


def build_sidebar():
    """Ledger index: notebooks numbered in order, ruled into their groups.

    Numbering follows list position, so inserting an entry renumbers every
    notebook below it — the numbers index the ledger, they are not IDs.
    """
    items = []
    current_group = None
    for i, nb in enumerate(NOTEBOOKS):
        if nb["group"] != current_group:
            current_group = nb["group"]
            items.append(f'<div class="section-label"><span>{current_group}</span></div>')
        locked = nb.get("locked", False)
        locked_attr = 'data-locked="true"' if locked else ""
        locked_class = " locked" if locked else ""
        lock_icon = "<i class='ti ti-lock' style='font-size:11px'></i>" if locked else ""
        # title= carries the full label: .nav-label ellipsises at the 236px
        # rail, and the longest names are clipped there.
        items.append(f"""<button class="nav-item{locked_class}" id="nav-{nb['id']}"
            data-label="{nb['label']}" data-path="{nb['path']}"
            data-id="{nb['id']}" data-icon="{nb['icon']}"
            title="{nb['label']}" {locked_attr}>
            <span class="nav-num">{i + 1:02d}</span>
            <span class="nav-label">{nb['label']}</span>
            <span class="nav-badge" id="badge-{nb['id']}">{lock_icon}</span>
            <span class="status-dot" id="dot-{nb['id']}"></span>
        </button>""")
    return "\n".join(items)


# The shell is a plain template (not an f-string): its CSS/JS is brace-heavy,
# so placeholders (__NAV__, __PORT__) are substituted at the end instead of
# doubling every { }.
_HUB_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Quant Lab</title>
<link rel="icon" href='data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="%230C110F"/><text x="16" y="17" font-family="Georgia" font-style="italic" font-size="15" fill="%23E9EEEB" text-anchor="middle" dominant-baseline="middle">Q</text><path d="M9 24.5h14" stroke="%231D9E75" stroke-width="1"/><path d="M9 27h14" stroke="%231D9E75" stroke-width="2"/></svg>'/>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=DM+Sans:opsz,wght@9..40,400;9..40,500&display=swap" rel="stylesheet">
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/dist/tabler-icons.min.css"/>
<style>
  /* ── Quant Lab · "The Ledger" ──────────────────────────────────────────
     Green-cast ink, ruled hairlines, serif italic display — the same
     system theme.css applies inside every notebook iframe. */
  :root {
    --ink:       #0C110F;   /* page */
    --surface:   #101613;   /* sidebar · topbar */
    --raised:    #131B16;   /* hover surfaces */
    --border:    #1F2A24;   /* hairlines */
    --rule:      #26332C;   /* stronger hairline */
    --text:      #E9EEEB;
    --muted:     #93A49B;
    --faint:     #5F6F66;
    --green:     #1D9E75;   /* heritage accent, kept from the notebooks */
    --green-hi:  #3AC493;
    --gold:      #C9A961;   /* seals · locked */
    --serif:     'DM Serif Display', Georgia, serif;
    --mono:      'DM Mono', ui-monospace, SFMono-Regular, monospace;
    --sans:      'DM Sans', -apple-system, 'Segoe UI', sans-serif;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  html { color-scheme: dark; }
  html, body { height: 100%; overflow: hidden; }

  body {
    font-family: var(--mono);
    background: var(--ink);
    color: var(--text);
    display: flex;
    flex-direction: row;
    height: 100vh;
    -webkit-font-smoothing: antialiased;
  }

  button { font: inherit; color: inherit; background: none; border: none; }

  /* ── Sidebar: the ledger's index column ── */
  :root { --sbw: 236px; }
  #sidebar {
    width: var(--sbw);
    flex-shrink: 0;
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--rule) transparent;
    /* hide/show: slide out via margin so children never reflow */
    transition: margin-left .28s cubic-bezier(.4,0,.2,1);
  }
  body.sb-hidden #sidebar { margin-left: calc(-1 * var(--sbw)); visibility: hidden; transition: margin-left .28s cubic-bezier(.4,0,.2,1), visibility 0s .28s; }

  .tb-toggle {
    display: flex; align-items: center;
    color: var(--faint); cursor: pointer;
    font-size: 16px; padding: 4px 6px; border-radius: 6px;
    transition: color .15s, background .15s;
  }
  .tb-toggle:hover { color: var(--green-hi); background: rgba(29,158,117,.08); }
  .tb-toggle:focus-visible { outline: 1px solid var(--green-hi); outline-offset: 1px; }

  .sb-header { padding: 20px 18px 16px; flex-shrink: 0; }
  .wordmark {
    display: block;
    font-family: var(--serif);
    font-style: italic;
    font-size: 22px;
    letter-spacing: .005em;
    color: var(--text);
    cursor: pointer;
    text-align: left;
    transition: color .15s;
  }
  .wordmark:hover { color: var(--green-hi); }
  .wordmark:focus-visible { outline: 1px solid var(--green-hi); outline-offset: 3px; }
  .sb-sub {
    font-size: 9px; letter-spacing: .24em; text-transform: uppercase;
    color: var(--faint); margin-top: 5px;
  }
  /* signature: the accountant's totals rule */
  .sb-rule { width: 64px; margin-top: 13px; border-bottom: 5px double var(--green); }

  .section-label {
    display: flex; align-items: center; gap: 10px;
    font-size: 9px; letter-spacing: .2em; text-transform: uppercase;
    color: var(--faint); padding: 16px 18px 6px; flex-shrink: 0;
  }
  .section-label::after { content: ""; flex: 1; height: 1px; background: var(--border); }

  .nav-item {
    display: flex; align-items: center; gap: 10px;
    width: 100%; padding: 8px 16px 8px 14px;
    border-left: 2px solid transparent;
    color: var(--muted); font-size: 12.5px;
    cursor: pointer; text-align: left;
    transition: color .15s, border-color .15s, background .15s;
  }
  .nav-num { font-size: 10px; color: var(--faint); width: 18px; flex-shrink: 0; transition: color .15s; }
  .nav-item:hover { color: var(--text); background: rgba(29,158,117,.05); }
  .nav-item:hover .nav-num { color: var(--green); }
  .nav-item:focus-visible { outline: 1px solid var(--green-hi); outline-offset: -1px; }
  .nav-item.active {
    color: var(--text); border-left-color: var(--green);
    background: linear-gradient(90deg, rgba(29,158,117,.10), transparent 70%);
  }
  .nav-item.active .nav-num { color: var(--green-hi); }
  .nav-item.locked { opacity: .45; cursor: not-allowed; }
  .nav-label { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .nav-badge .ti-lock { color: var(--gold); }

  .status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--rule); flex-shrink: 0; transition: background .3s;
  }
  .status-dot.live    { background: var(--green-hi); box-shadow: 0 0 6px rgba(58,196,147,.6); }
  .status-dot.loading { background: var(--gold); animation: pulse 1.1s ease-in-out infinite; }
  @keyframes pulse { 50% { opacity: .35; } }

  .sb-footer {
    margin-top: auto; padding: 12px 18px;
    border-top: 1px solid var(--border);
    font-size: 10px; color: var(--faint);
    display: flex; align-items: center; gap: 7px;
    flex-shrink: 0;
  }

  /* ── Main column ── */
  #main { flex: 1; display: flex; flex-direction: column; height: 100vh; min-width: 0; }

  #topbar {
    flex-shrink: 0; height: 46px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center;
    padding: 0 18px; gap: 10px;
  }
  #tb-icon { font-size: 15px; color: var(--faint); }
  .tb-crumb { font-size: 11px; color: var(--faint); cursor: pointer; transition: color .15s; }
  .tb-crumb:hover { color: var(--green-hi); }
  .tb-crumb:focus-visible { outline: 1px solid var(--green-hi); outline-offset: 2px; }
  #tb-title { font-size: 12px; color: var(--muted); letter-spacing: .02em; }
  #tb-path  { font-size: 10.5px; color: var(--faint); }
  #tb-right { margin-left: auto; display: flex; align-items: center; gap: 16px; }

  .lamp { display: flex; align-items: center; gap: 6px; font-size: 10px; letter-spacing: .12em; color: var(--faint); }
  .lamp-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--faint); }
  .lamp.open .lamp-dot { background: var(--green-hi); box-shadow: 0 0 6px rgba(58,196,147,.6); }
  .lamp.open { color: var(--muted); }
  #clock { font-size: 11.5px; color: var(--muted); font-variant-numeric: tabular-nums; }

  #tb-actions { display: flex; gap: 6px; }
  .tb-btn {
    display: none; align-items: center; gap: 5px;
    border: 1px solid var(--rule);
    color: var(--muted); font-size: 10.5px;
    padding: 4px 10px; border-radius: 6px; cursor: pointer;
    transition: color .15s, border-color .15s;
  }
  .tb-btn.on    { display: flex; }
  .tb-btn:hover { color: var(--text); border-color: var(--green); }
  .tb-btn:focus-visible { outline: 1px solid var(--green-hi); outline-offset: 1px; }
  .tb-btn .ti   { font-size: 13px; }

  #content { flex: 1; position: relative; overflow: hidden; min-height: 0; background: var(--ink); }

  /* ── Frontispiece: the ledger's title page ── */
  #placeholder {
    position: absolute; inset: 0;
    display: flex;
    overflow-y: auto;
    padding: clamp(20px, 4vh, 56px) 32px 44px;
    scrollbar-width: thin;
    scrollbar-color: var(--rule) transparent;
    /* faint ruled paper */
    background: repeating-linear-gradient(
      to bottom,
      transparent 0, transparent 31px,
      rgba(233,238,235,.016) 31px, rgba(233,238,235,.016) 32px);
  }
  /* margin:auto centers the frontispiece when it fits and still allows
     scrolling when it doesn't */
  .fp { max-width: 1010px; width: 100%; margin: auto; text-align: center; }
  .fp-eyebrow {
    display: flex; align-items: center; justify-content: center; gap: 16px;
    font-size: 10px; letter-spacing: .3em; text-transform: uppercase;
    color: var(--green);
  }
  .fp-eyebrow::before, .fp-eyebrow::after { content: ""; width: 48px; height: 1px; background: var(--rule); }
  .fp-title {
    font-family: var(--serif); font-style: italic; font-weight: 400;
    font-size: clamp(42px, 6vw, 66px); line-height: 1.02;
    letter-spacing: -.015em; color: var(--text);
    margin: 16px 0 14px;
  }
  .fp-rule { width: 88px; margin: 0 auto 16px; border-bottom: 5px double var(--green); }
  .fp-sub { font-size: 11px; letter-spacing: .06em; color: var(--muted); margin-bottom: 34px; }

  /* ── Account overview: the book, marked to market ── */
  .acct-row {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 14px; text-align: left;
  }
  @media (max-width: 980px) { .acct-row { grid-template-columns: 1fr; } }
  .acct {
    background: #111814; border: 1px solid var(--border); border-radius: 10px;
    padding: 16px 18px 14px;
    display: flex; flex-direction: column; gap: 12px;
  }
  .acct[hidden] { display: none; }
  .acct-head { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; }
  .acct-name { font-family: var(--serif); font-style: italic; font-size: 19px; color: var(--text); }
  .acct-tag {
    font-family: var(--mono); font-style: normal; font-size: 9px;
    letter-spacing: .18em; text-transform: uppercase; color: var(--faint); margin-left: 9px;
  }
  .acct-asof { font-size: 9.5px; letter-spacing: .1em; color: var(--faint); font-variant-numeric: tabular-nums; }
  .acct-hero { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .acct-nav { font-size: 27px; color: var(--text); font-variant-numeric: tabular-nums; letter-spacing: .01em; transition: color .5s; }
  .acct-delta { font-size: 12px; font-variant-numeric: tabular-nums; }
  .acct-delta .dl {
    font-size: 9px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--faint); margin-right: 6px;
  }
  .acct-delta .pos { color: var(--green-hi); }
  .acct-delta .neg { color: #E07B4F; }

  .spark-wrap { position: relative; }
  .spark { display: block; width: 100%; height: 64px; }
  .spark-base { stroke: #3A4A42; stroke-width: 1; stroke-dasharray: 2 4; vector-effect: non-scaling-stroke; }
  .spark-hair { stroke: #4A5A52; stroke-width: 1; vector-effect: non-scaling-stroke; }
  .spark-draw {
    stroke-dasharray: 1; stroke-dashoffset: 1;
    animation: sparkdraw 1.2s cubic-bezier(.4,0,.2,1) .35s forwards;
  }
  @keyframes sparkdraw { to { stroke-dashoffset: 0; } }
  .spark-tip {
    position: absolute; top: -24px; transform: translateX(-50%);
    padding: 2px 8px; white-space: nowrap; pointer-events: none;
    background: var(--raised); border: 1px solid var(--rule); border-radius: 5px;
    font-size: 10px; color: var(--muted); font-variant-numeric: tabular-nums;
  }

  .acct-stats {
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px 18px;
    border-top: 1px solid var(--border); padding-top: 11px;
  }
  .acct-stats > div { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
  .as-label { font-size: 9px; letter-spacing: .16em; text-transform: uppercase; color: var(--faint); }
  .as-val { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; transition: color .5s; }

  /* exchange-style tick flash: class lands (instant color), removal fades */
  .tick-up   { color: var(--green-hi) !important; transition: color 0s !important; }
  .tick-down { color: #E07B4F !important; transition: color 0s !important; }

  /* ── Blotters: live venue positions, entered like ledger lines ── */
  .blotter { margin-top: 42px; text-align: left; }
  .blotter + .blotter { margin-top: 28px; }
  .blotter[hidden] { display: none; }
  .bl-head { display: flex; align-items: center; gap: 14px; margin-bottom: 10px; }
  .bl-title {
    font-family: var(--serif); font-style: italic; font-weight: 400;
    font-size: 20px; color: var(--text); white-space: nowrap;
  }
  .bl-line { flex: 1; height: 1px; background: var(--rule); }
  .bl-src {
    display: flex; align-items: center; gap: 7px;
    font-size: 9.5px; letter-spacing: .18em; text-transform: uppercase;
    color: var(--faint); white-space: nowrap;
  }
  .bl-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--faint); }
  .bl-src.live { color: var(--muted); }
  .bl-src.live .bl-dot { background: var(--green-hi); box-shadow: 0 0 6px rgba(58,196,147,.6); }

  .bl-scroll {
    overflow-x: auto; background: #111814;
    border: 1px solid var(--border); border-radius: 10px;
  }
  .bl-table {
    width: 100%; border-collapse: collapse;
    font-size: 11.5px; font-variant-numeric: tabular-nums;
  }
  .bl-table th {
    font-size: 9px; letter-spacing: .16em; text-transform: uppercase;
    color: var(--faint); font-weight: 400; text-align: right;
    padding: 10px 14px 8px; border-bottom: 1px solid var(--rule); white-space: nowrap;
  }
  .bl-table td {
    text-align: right; color: var(--muted);
    padding: 9px 14px; border-bottom: 1px solid var(--border); white-space: nowrap;
  }
  .bl-table th.l, .bl-table td.l { text-align: left; }
  .bl-table tbody tr:last-child td { border-bottom: none; }
  .bl-table tbody tr { transition: background .15s; }
  .bl-table tbody tr:hover { background: rgba(29,158,117,.04); }
  /* Scoped under .bl-table so they outrank the `.bl-table td` base color —
     a bare `.pos` (one class) loses that specificity contest and the P/L
     columns silently stay muted. */
  .bl-table td.bl-market { color: var(--text); font-weight: 500; }
  .bl-table td.pos { color: var(--green-hi); }
  .bl-table td.neg { color: #E07B4F; }
  .bl-table td.bl-long { color: var(--green); }
  .bl-table td.bl-short { color: #E07B4F; }
  .bl-empty td { text-align: center; color: var(--faint); padding: 20px 14px; }

  .bl-x {
    border: 1px solid transparent; color: var(--faint); cursor: pointer;
    font-size: 13px; line-height: 1; padding: 3px 7px; border-radius: 5px;
    transition: color .15s, background .15s, border-color .15s;
  }
  .bl-x:hover { color: #E07B4F; background: rgba(216,104,60,.12); }
  .bl-x:focus-visible { outline: 1px solid #E07B4F; outline-offset: 1px; }
  .bl-x.armed {
    color: #E07B4F; border-color: rgba(216,104,60,.5);
    font-size: 9px; letter-spacing: .08em; text-transform: uppercase;
  }

  /* page-load: one orchestrated reveal, then quiet */
  @media (prefers-reduced-motion: no-preference) {
    .fp-eyebrow, .fp-title, .fp-rule, .fp-sub { animation: rise .55s cubic-bezier(.2,.7,.3,1) backwards; }
    .fp-title { animation-delay: .05s; }
    .fp-rule  { animation-delay: .11s; }
    .fp-sub   { animation-delay: .16s; }
    .acct     { animation: rise .55s cubic-bezier(.2,.7,.3,1) backwards; }
    #acct-oanda  { animation-delay: .22s; }
    #acct-alpaca { animation-delay: .28s; }
    .blotter  { animation: rise .55s cubic-bezier(.2,.7,.3,1) backwards; }
    #blotter  { animation-delay: .38s; }
    #blotter2 { animation-delay: .46s; }
  }
  @keyframes rise {
    from { opacity: 0; transform: translateY(10px); }
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
  }

  #loader {
    position: absolute; inset: 0;
    display: none; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 14px; color: var(--muted);
    pointer-events: none;
    z-index: 10;
  }
  #loader.on { display: flex; }
  .spinner {
    width: 22px; height: 22px;
    border: 2px solid var(--rule);
    border-top-color: var(--green);
    border-radius: 50%;
    animation: spin .7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  #loader p { font-size: 11px; letter-spacing: .08em; }

  /* compact: the index collapses to its numbers */
  @media (max-width: 720px) {
    :root { --sbw: 64px; }
    .nav-label, .nav-badge, .sb-sub, .section-label span, .sb-footer span { display: none; }
    .wordmark { font-size: 17px; }
    .sb-rule { width: 28px; }
    .section-label { padding: 14px 12px 4px; }
    #clock, .lamp { display: none; }
  }
</style>
</head>
<body>

<nav id="sidebar">
  <div class="sb-header">
    <button class="wordmark" id="home-link" title="Back to the index">Quant Lab</button>
    <div class="sb-sub">Private trading desk</div>
    <div class="sb-rule"></div>
  </div>
__NAV__
  <div class="sb-footer">
    <i class="ti ti-server" aria-hidden="true"></i>
    <span>localhost :__PORT__</span>
  </div>
</nav>

<main id="main">
  <div id="topbar">
    <button class="tb-toggle" id="btn-sidebar" title="Hide index (Ctrl+B)" aria-expanded="true" aria-controls="sidebar">
      <i class="ti ti-layout-sidebar-left-collapse" aria-hidden="true"></i>
    </button>
    <i class="ti ti-notebook" id="tb-icon" aria-hidden="true"></i>
    <button class="tb-crumb" id="crumb-home">Quant Lab /</button>
    <span id="tb-title">Select a desk</span>
    <span id="tb-path"></span>
    <div id="tb-right">
      <span id="clock" title="New York time">NY --:--:--</span>
      <span class="lamp closed" id="lamp" title="Regular hours Mon-Fri 9:30-16:00 ET">
        <span class="lamp-dot"></span><span id="lamp-text">NYSE</span>
      </span>
      <div id="tb-actions">
        <button class="tb-btn" id="btn-reload">
          <i class="ti ti-refresh" aria-hidden="true"></i> Reload
        </button>
        <button class="tb-btn" id="btn-tab">
          <i class="ti ti-external-link" aria-hidden="true"></i> Open in tab
        </button>
      </div>
    </div>
  </div>

  <div id="content">
    <div id="placeholder">
      <div class="fp">
        <div class="fp-eyebrow">Multi-strategy research</div>
        <h1 class="fp-title">Quant Lab</h1>
        <div class="fp-rule"></div>
        <p class="fp-sub">The book, marked to market &mdash; open a desk from the index.</p>
        <div class="acct-row">
          <section class="acct" id="acct-oanda" hidden>
            <div class="acct-head">
              <span class="acct-name">OANDA<span class="acct-tag">fx &middot; practice</span></span>
              <span class="acct-asof" id="ao-asof"></span>
            </div>
            <div class="acct-hero">
              <span class="acct-nav" id="ao-nav">&mdash;</span>
              <span class="acct-delta" id="ao-delta"></span>
            </div>
            <div class="spark-wrap">
              <svg class="spark" id="ao-spark" viewBox="0 0 100 30" preserveAspectRatio="none" aria-hidden="true"></svg>
              <div class="spark-tip" id="ao-tip" hidden></div>
            </div>
            <div class="acct-stats">
              <div><span class="as-label">Balance</span><span class="as-val" id="ao-balance">&mdash;</span></div>
              <div><span class="as-label">Margin used</span><span class="as-val" id="ao-mused">&mdash;</span></div>
              <div><span class="as-label">Margin available</span><span class="as-val" id="ao-mavail">&mdash;</span></div>
              <div><span class="as-label">Open trades</span><span class="as-val" id="ao-trades">&mdash;</span></div>
            </div>
          </section>
          <section class="acct" id="acct-alpaca" hidden>
            <div class="acct-head">
              <span class="acct-name">Alpaca<span class="acct-tag">equities &middot; paper</span></span>
              <span class="acct-asof" id="aa-asof"></span>
            </div>
            <div class="acct-hero">
              <span class="acct-nav" id="aa-nav">&mdash;</span>
              <span class="acct-delta" id="aa-delta"></span>
            </div>
            <div class="spark-wrap">
              <svg class="spark" id="aa-spark" viewBox="0 0 100 30" preserveAspectRatio="none" aria-hidden="true"></svg>
              <div class="spark-tip" id="aa-tip" hidden></div>
            </div>
            <div class="acct-stats">
              <div><span class="as-label">Cash</span><span class="as-val" id="aa-cash">&mdash;</span></div>
              <div><span class="as-label">Buying power</span><span class="as-val" id="aa-bp">&mdash;</span></div>
              <div><span class="as-label">Market value</span><span class="as-val" id="aa-mv">&mdash;</span></div>
              <div><span class="as-label">Prior close</span><span class="as-val" id="aa-prior">&mdash;</span></div>
            </div>
          </section>
        </div>
        <section class="blotter" id="blotter" hidden>
          <div class="bl-head">
            <span class="bl-title">FX Positions</span>
            <span class="bl-line"></span>
            <span class="bl-src" id="bl-status">
              <span class="bl-dot"></span><span id="bl-status-text">oanda</span>
            </span>
          </div>
          <div class="bl-scroll">
            <table class="bl-table">
              <thead><tr>
                <th class="l">Market</th><th>Units</th><th class="l">Type</th>
                <th>Margin (USD)</th><th>Price</th><th>Current</th>
                <th>Profit (USD)</th><th>Profit (Pips)</th><th>Profit (%)</th><th></th>
              </tr></thead>
              <tbody id="bl-body"></tbody>
            </table>
          </div>
        </section>
        <section class="blotter" id="blotter2" hidden>
          <div class="bl-head">
            <span class="bl-title">Alpaca Positions</span>
            <span class="bl-line"></span>
            <span class="bl-src" id="bl2-status">
              <span class="bl-dot"></span><span id="bl2-status-text">alpaca</span>
            </span>
          </div>
          <div class="bl-scroll">
            <table class="bl-table">
              <thead><tr>
                <th class="l">Asset</th><th>Qty</th><th class="l">Side</th>
                <th>Avg Entry</th><th>Price</th><th>Market Value</th>
                <th>Total P/L (USD)</th><th>P/L (%)</th><th></th>
              </tr></thead>
              <tbody id="bl2-body"></tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
    <div id="loader">
      <div class="spinner"></div>
      <p id="loader-text">Loading...</p>
    </div>
  </div>
</main>

<script>
  let currentPath  = null;
  let lastPairHash = null;
  const frameCache = {};

  function loadNotebook(btn) {
    if (btn.classList.contains("locked")) return;

    const label = btn.dataset.label;
    const path  = btn.dataset.path;
    const id    = btn.dataset.id;

    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tb-title').textContent = label;
    document.getElementById('tb-title').style.color = 'var(--text)';
    document.getElementById('tb-path').textContent  = path;
    document.getElementById('tb-icon').className    = 'ti ' + (btn.dataset.icon || 'ti-notebook');
    document.getElementById('btn-reload').classList.add('on');
    document.getElementById('btn-tab').classList.add('on');
    document.getElementById('placeholder').style.display = 'none';

    currentPath = path;

    // Hide all cached frames
    Object.values(frameCache).forEach(f => {
      f.style.opacity       = '0';
      f.style.pointerEvents = 'none';
    });

    if (frameCache[id]) {
      // Already loaded — just show it, no reload
      const f = frameCache[id];
      f.style.opacity       = '1';
      f.style.pointerEvents = 'auto';
      document.getElementById('loader').classList.remove('on');
      return;
    }

    // First visit — create a new iframe
    const content = document.getElementById('content');
    const frame   = document.createElement('iframe');
    frame.title   = label;
    frame.style.cssText = [
      'position:absolute', 'inset:0',
      'width:100%', 'height:100%',
      'border:none', 'opacity:0',
      'background:#0C110F',
      'pointer-events:none', 'transition:opacity .2s'
    ].join(';');
    content.appendChild(frame);
    frameCache[id] = frame;

    const dot    = document.getElementById('dot-' + id);
    const loader = document.getElementById('loader');
    if (dot) dot.className = 'status-dot loading';
    loader.classList.add('on');
    document.getElementById('loader-text').textContent = 'Opening ' + label + '...';

    frame.onload = () => {
      loader.classList.remove('on');
      frame.style.opacity       = '1';
      frame.style.pointerEvents = 'auto';
      if (dot) dot.className = 'status-dot live';
    };
    frame.src = path;
  }

  function goHome() {
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    Object.values(frameCache).forEach(f => {
      f.style.opacity       = '0';
      f.style.pointerEvents = 'none';
    });
    currentPath = null;
    document.getElementById('loader').classList.remove('on');
    document.getElementById('tb-title').textContent = 'Select a desk';
    document.getElementById('tb-title').style.color = 'var(--muted)';
    document.getElementById('tb-path').textContent  = '';
    document.getElementById('tb-icon').className    = 'ti ti-notebook';
    document.getElementById('btn-reload').classList.remove('on');
    document.getElementById('btn-tab').classList.remove('on');
    document.getElementById('placeholder').style.display = '';
    pollBlotter();  // refresh the book immediately on returning home
  }

  function reloadFrame() {
    if (!currentPath) return;
    const activeBtn = document.querySelector('.nav-item.active');
    if (!activeBtn) return;
    const id  = activeBtn.dataset.id;
    const old = frameCache[id];
    if (old) old.remove();
    delete frameCache[id];
    loadNotebook(activeBtn);
  }

  function openInTab() {
    if (currentPath) window.open(currentPath, '_blank');
  }

  function unlockTab(id) {
    const btn = document.getElementById("nav-" + id);
    if (!btn || !btn.classList.contains("locked")) return;

    btn.classList.remove("locked");
    btn.removeAttribute("data-locked");
    if (!btn.dataset.bound) {
      btn.addEventListener('click', () => loadNotebook(btn));
      btn.dataset.bound = "1";
    }

    const badge = document.getElementById("badge-" + id);
    if (badge) {
      badge.innerHTML = "<i class='ti ti-lock-open' style='font-size:11px;color:var(--green-hi)'></i>";
    }

    btn.style.transition = "background .4s";
    btn.style.background = "rgba(58,196,147,0.12)";
    setTimeout(() => { btn.style.background = ""; }, 1200);
  }

  function lockTab(id) {
    const btn = document.getElementById("nav-" + id);
    if (!btn || btn.classList.contains("locked")) return;

    btn.classList.add("locked");
    btn.setAttribute("data-locked", "true");

    const badge = document.getElementById("badge-" + id);
    if (badge) badge.innerHTML = "<i class='ti ti-lock' style='font-size:11px'></i>";
  }

  // Tabs gated behind the fundamental notebook finishing a run
  const FUNDAMENTAL_GATED = ["bull_call_spread", "bear_put_spread"];

  async function pollStatus() {
    try {
      const res  = await fetch("/api/status");
      const data = await res.json();

      // Unlock only the strategies fundamental.py actually suggested for the
      // pair; anything it didn't suggest stays (or goes back to) locked.
      const suggested = data.fundamental_done ? (data.strategies || []) : [];
      FUNDAMENTAL_GATED.forEach(id => {
        if (suggested.includes(id)) unlockTab(id); else lockTab(id);
      });
    } catch (e) {}
  }

  // ── OTC Track gate ────────────────────────────────────────────────
  // OTC Track charts whatever Pair Trading and LETF Backtest last published,
  // so until one of them has run it has nothing to say and stays locked.
  // Either desk unlocks it; run both and the tape covers both watchlists.
  let lastOtcStamp;   // stays undefined until the first poll answers

  function reloadCachedFrame(id) {
    const frame = frameCache[id];
    if (!frame) return;                 // never opened, nothing to refresh
    const btn = document.getElementById("nav-" + id);
    if (!btn) return;
    frame.src = btn.dataset.path;
    const dot = document.getElementById("dot-" + id);
    if (dot) dot.className = "status-dot loading";
    frame.onload = () => { if (dot) dot.className = "status-dot live"; };
  }

  async function pollOtcGate() {
    try {
      const res  = await fetch("/api/otc-ready");
      const data = await res.json();

      if (data.ready) unlockTab("otc_track"); else lockTab("otc_track");

      // The stamp folds in which desks are armed and what each picked, so a
      // new pair, a new triple, or a desk dropping out all move it. Reload so
      // the tape follows the picks without a manual refresh — skipped on the
      // first poll, which is only learning the current value.
      if (lastOtcStamp !== undefined && data.stamp !== lastOtcStamp) {
        reloadCachedFrame("otc_track");
      }
      lastOtcStamp = data.stamp;
    } catch (e) {}
  }

  let pairChangeDetected = false;
  async function pollPairHash() {
    try {
      const res  = await fetch("/api/pair-hash");
      const data = await res.json();

      if (data.hash === null) return;

      if (lastPairHash === null) {
        lastPairHash = data.hash;
        return;
      }

      if (data.hash !== lastPairHash) {
        lastPairHash = data.hash;
        pairChangeDetected = true;

        // Clear the ready flag immediately so we don't reload prematurely
        // (the notebook will re-touch it when done)
      }

      // Only reload dependents once the notebook signals it's done
      if (pairChangeDetected) {
        const readyRes  = await fetch("/api/pair-ready");
        const readyData = await readyRes.json();

        if (readyData.ready) {
          pairChangeDetected = false;  // reset

          const dependents = ["fundamental", ...FUNDAMENTAL_GATED];  // NOT known_pair_trading itself

          dependents.forEach(id => {
            const frame = frameCache[id];
            if (!frame) return;
            const btn = document.getElementById("nav-" + id);
            if (!btn) return;
            frame.src = btn.dataset.path;
            const dot = document.getElementById("dot-" + id);
            if (dot) dot.className = "status-dot loading";
            frame.onload = () => {
              if (dot) dot.className = "status-dot live";
            };
          });

          // Also reset the spread locks since fundamental needs to re-run
          FUNDAMENTAL_GATED.forEach(lockTab);

          // Show loader if a dependent is active
          const activeBtn = document.querySelector('.nav-item.active');
          if (activeBtn && dependents.includes(activeBtn.dataset.id)) {
            const loader = document.getElementById('loader');
            loader.classList.add('on');
            document.getElementById('loader-text').textContent = 'Reloading...';
            const activeFrame = frameCache[activeBtn.dataset.id];
            if (activeFrame) {
              activeFrame.onload = () => loader.classList.remove('on');
            }
          }
        }
      }
    } catch (e) {}
  }

  // ── Live venue blotters (FX via OANDA, equities/crypto via Alpaca) ──
  // Poll /api/{venue}/positions while the front page is visible. Each ✕ is
  // a two-click close: first click arms ("close?"), second click within 4s
  // sends the close; the arm state decays back to ✕ on its own.
  let blotterArmed = null;   // "venue:id" currently armed

  const fmtNum = (v, dec) => v.toLocaleString('en-US',
    { minimumFractionDigits: dec, maximumFractionDigits: dec });
  const fmtQty = v => v.toLocaleString('en-US', { maximumFractionDigits: 4 });
  const signed = (v, dec) => (v > 0 ? '+' : '') + fmtNum(v, dec);
  const plClass = v => v > 0 ? 'pos' : (v < 0 ? 'neg' : '');
  const sideCell = s => '<td class="l ' + (s === 'LONG' ? 'bl-long' : 'bl-short') + '">' + s + '</td>';

  function xCell(venue, id) {
    const armed = blotterArmed === venue + ':' + id;
    return '<td><button class="bl-x' + (armed ? ' armed' : '') + '" data-venue="' + venue
      + '" data-id="' + id + '" title="Close position">'
      + (armed ? 'close?' : '&times;') + '</button></td>';
  }

  const VENUES = {
    oanda: {
      sec: 'blotter', body: 'bl-body', lamp: 'bl-status', txt: 'bl-status-text',
      cols: 10, hideRe: /OANDA_TOKEN/,
      row: r => '<tr>'
        + '<td class="l bl-market">' + r.market + '</td>'
        + '<td>' + fmtNum(r.units, 0) + '</td>'
        + sideCell(r.side)
        + '<td>' + fmtNum(r.margin, 2) + '</td>'
        + '<td>' + r.avg + '</td>'
        + '<td>' + (r.current ?? '…') + '</td>'
        + '<td class="' + plClass(r.pl_usd) + '">' + signed(r.pl_usd, 2) + '</td>'
        + '<td class="' + plClass(r.pips ?? 0) + '">'
          + (r.pips == null ? '…' : signed(r.pips, 1)) + '</td>'
        + '<td class="' + plClass(r.pl_pct ?? 0) + '">'
          + (r.pl_pct == null ? '…' : signed(r.pl_pct, 2) + '%') + '</td>'
        + xCell('oanda', r.instrument) + '</tr>',
    },
    alpaca: {
      sec: 'blotter2', body: 'bl2-body', lamp: 'bl2-status', txt: 'bl2-status-text',
      cols: 9, hideRe: /ALPACA_API/,
      row: r => '<tr>'
        + '<td class="l bl-market">' + r.symbol + '</td>'
        + '<td>' + fmtQty(r.qty) + '</td>'
        + sideCell(r.side)
        + '<td>' + fmtNum(r.avg, 2) + '</td>'
        + '<td>' + fmtNum(r.price, 2) + '</td>'
        + '<td>' + fmtNum(r.value, 2) + '</td>'
        + '<td class="' + plClass(r.pl_usd) + '">' + signed(r.pl_usd, 2) + '</td>'
        + '<td class="' + plClass(r.pl_pct) + '">' + signed(r.pl_pct, 2) + '%</td>'
        + xCell('alpaca', r.symbol) + '</tr>',
    },
  };

  function setLamp(v, live, msg) {
    document.getElementById(v.lamp).className = 'bl-src' + (live ? ' live' : '');
    document.getElementById(v.txt).textContent = msg;
  }

  function renderRows(name, rows) {
    const v = VENUES[name];
    const body = document.getElementById(v.body);
    body.innerHTML = rows.length
      ? rows.map(v.row).join('')
      : '<tr class="bl-empty"><td colspan="' + v.cols + '">No open positions.</td></tr>';
    body.querySelectorAll('.bl-x').forEach(btn =>
      btn.addEventListener('click', () => blotterClose(btn.dataset.venue, btn.dataset.id, btn)));
  }

  async function blotterClose(venue, id, btn) {
    const key = venue + ':' + id;
    if (blotterArmed !== key) {
      blotterArmed = key;
      btn.classList.add('armed');
      btn.innerHTML = 'close?';
      setTimeout(() => {
        if (blotterArmed === key) { blotterArmed = null; pollVenue(venue); }
      }, 4000);
      return;
    }
    blotterArmed = null;
    btn.disabled = true;
    btn.innerHTML = '…';
    try {
      const res = await fetch('/api/' + venue + '/close/' + encodeURIComponent(id), { method: 'POST' });
      const data = await res.json();
      if (!data.ok) setLamp(VENUES[venue], false, (data.error || 'close failed').slice(0, 48).toLowerCase());
    } catch (e) {
      setLamp(VENUES[venue], false, venue + ' · offline');
    }
    pollVenue(venue);
  }

  async function pollVenue(name) {
    // only poll while the frontispiece is showing
    if (document.getElementById('placeholder').style.display === 'none' || document.hidden) return;
    const v = VENUES[name];
    try {
      const res  = await fetch('/api/' + name + '/positions');
      const data = await res.json();
      const sec = document.getElementById(v.sec);
      if (!data.ok) {
        // keys not configured -> no panel at all; transient errors keep the
        // panel up with an offline lamp
        if (v.hideRe.test(data.error || '')) { sec.hidden = true; return; }
        setLamp(v, false, name + ' · offline');
        return;
      }
      sec.hidden = false;
      setLamp(v, true, name + ' · live');
      renderRows(name, data.rows);
    } catch (e) {
      setLamp(v, false, name + ' · offline');
    }
  }

  // ── Account overview panels: hero number + real equity sparkline ──
  const money  = v => '$' + fmtNum(v, 2);
  const money0 = v => '$' + fmtNum(v, 0);

  // exchange-style flash when a value changes
  const _prevVals = {};
  function setVal(id, num, text) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    const was = _prevVals[id];
    _prevVals[id] = num;
    if (was == null || num == null || num === was) return;
    const cls = num > was ? 'tick-up' : 'tick-down';
    el.classList.add(cls);
    setTimeout(() => el.classList.remove(cls), 180);
  }

  function deltaHtml(label, v, pct) {
    return '<span class="dl">' + label + '</span><span class="' + plClass(v) + '">'
      + signed(v, 2) + (pct == null ? '' : ' &middot; ' + signed(pct, 2) + '%') + '</span>';
  }

  // Single-series sparkline: colored by polarity vs the baseline (prior
  // close where the venue provides one, else the first sample), dotted
  // baseline, draw-in animation on first render only.
  function drawSpark(id, hist, base) {
    const svg = document.getElementById(id);
    if (!svg) return;
    if (!hist || hist.length < 2) { svg.innerHTML = ''; delete svg.dataset.hist; return; }
    const vals = hist.map(p => p[1]);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (base != null) { lo = Math.min(lo, base); hi = Math.max(hi, base); }
    if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
    const pad = (hi - lo) * .1;
    lo -= pad; hi += pad;
    const X = i => (i / (vals.length - 1)) * 100;
    const Y = v => 30 - ((v - lo) / (hi - lo)) * 30;
    const pts = vals.map((v, i) => X(i).toFixed(2) + ',' + Y(v).toFixed(2)).join(' ');
    const col = vals[vals.length - 1] >= (base != null ? base : vals[0]) ? '#3AC493' : '#E07B4F';
    const first = !svg.dataset.drawn;
    svg.dataset.drawn = '1';
    svg.dataset.hist = JSON.stringify(hist);
    svg.innerHTML =
      (base != null
        ? '<line class="spark-base" x1="0" x2="100" y1="' + Y(base).toFixed(2) + '" y2="' + Y(base).toFixed(2) + '"/>'
        : '')
      + '<polyline points="' + pts + ' 100,30 0,30" fill="' + col + '" opacity=".08" stroke="none"/>'
      + '<polyline points="' + pts + '" fill="none" stroke="' + col + '" stroke-width="1.4" vector-effect="non-scaling-stroke"'
      + (first ? ' pathLength="1" class="spark-draw"' : '') + '/>';
  }

  // hover readout: hairline + value·time of the nearest sample
  function bindSpark(sparkId, tipId) {
    const svg = document.getElementById(sparkId);
    const tip = document.getElementById(tipId);
    if (!svg || !tip) return;
    svg.addEventListener('mousemove', e => {
      if (!svg.dataset.hist) return;
      const hist = JSON.parse(svg.dataset.hist);
      const r = svg.getBoundingClientRect();
      const fx = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
      const i = Math.round(fx * (hist.length - 1));
      const x = (i / (hist.length - 1)) * 100;
      const t = new Date(hist[i][0] * 1000);
      tip.hidden = false;
      tip.style.left = (x + '%');
      tip.textContent = money(hist[i][1]) + ' · '
        + t.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' });
      let hair = svg.querySelector('.spark-hair');
      if (!hair) {
        hair = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        hair.setAttribute('class', 'spark-hair');
        hair.setAttribute('y1', '0');
        hair.setAttribute('y2', '30');
        svg.appendChild(hair);
      }
      hair.setAttribute('x1', x);
      hair.setAttribute('x2', x);
    });
    svg.addEventListener('mouseleave', () => {
      tip.hidden = true;
      const hair = svg.querySelector('.spark-hair');
      if (hair) hair.remove();
    });
  }

  async function pollAccount(name) {
    if (document.getElementById('placeholder').style.display === 'none' || document.hidden) return;
    const sec = document.getElementById('acct-' + name);
    try {
      const res  = await fetch('/api/' + name + '/account');
      const data = await res.json();
      if (!data.ok) {
        if (/(OANDA_TOKEN|ALPACA_API)/.test(data.error || '')) sec.hidden = true;
        return;
      }
      sec.hidden = false;
      const a = data.acct;
      const asof = 'as of ' + new Date().toLocaleTimeString('en-US', { hour12: false });
      if (name === 'oanda') {
        setVal('ao-nav', a.nav, money(a.nav));
        document.getElementById('ao-delta').innerHTML =
          deltaHtml('unrealized', a.unrealized, 100 * a.unrealized / (a.balance || 1));
        setVal('ao-balance', a.balance, money(a.balance));
        setVal('ao-mused', a.margin_used, money0(a.margin_used));
        setVal('ao-mavail', a.margin_avail, money0(a.margin_avail));
        setVal('ao-trades', a.open_trades, String(a.open_trades));
        document.getElementById('ao-asof').textContent = asof;
        drawSpark('ao-spark', data.history, data.base);
      } else {
        const day = a.equity - a.last_equity;
        setVal('aa-nav', a.equity, money(a.equity));
        document.getElementById('aa-delta').innerHTML =
          deltaHtml('today', day, 100 * day / (a.last_equity || 1));
        setVal('aa-cash', a.cash, money(a.cash));
        setVal('aa-bp', a.buying_power, money0(a.buying_power));
        setVal('aa-mv', a.long_value, money(a.long_value));
        setVal('aa-prior', a.last_equity, money(a.last_equity));
        document.getElementById('aa-asof').textContent = asof;
        drawSpark('aa-spark', data.history, data.base);
      }
    } catch (e) {}
  }

  function pollBlotter() {
    pollVenue('oanda'); pollVenue('alpaca');
    pollAccount('oanda'); pollAccount('alpaca');
  }

  // ── Sidebar hide/show: toggle button, Ctrl+B, remembered per browser ──
  function setSidebar(hidden, save) {
    document.body.classList.toggle('sb-hidden', hidden);
    const btn = document.getElementById('btn-sidebar');
    btn.setAttribute('aria-expanded', String(!hidden));
    btn.title = (hidden ? 'Show index' : 'Hide index') + ' (Ctrl+B)';
    btn.querySelector('i').className =
      'ti ' + (hidden ? 'ti-layout-sidebar-left-expand' : 'ti-layout-sidebar-left-collapse');
    if (save) try { localStorage.setItem('qlab-sidebar-hidden', hidden ? '1' : '0'); } catch (e) {}
  }
  function toggleSidebar() {
    setSidebar(!document.body.classList.contains('sb-hidden'), true);
  }

  // New York clock + session lamp (client-side only; regular hours,
  // exchange holidays not modelled)
  const NY_FMT = new Intl.DateTimeFormat('en-GB', {
    timeZone: 'America/New_York', hour12: false,
    weekday: 'short', hour: '2-digit', minute: '2-digit', second: '2-digit'
  });
  function tickClock() {
    const parts = Object.fromEntries(
      NY_FMT.formatToParts(new Date()).map(p => [p.type, p.value]));
    document.getElementById('clock').textContent =
      'NY ' + parts.hour + ':' + parts.minute + ':' + parts.second;
    const mins = (+parts.hour) * 60 + (+parts.minute);
    const open = !['Sat', 'Sun'].includes(parts.weekday) && mins >= 570 && mins < 960;
    document.getElementById('lamp').className = 'lamp ' + (open ? 'open' : 'closed');
    document.getElementById('lamp-text').textContent = open ? 'NYSE OPEN' : 'NYSE CLOSED';
  }

  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.nav-item:not(.locked)').forEach(btn => {
      btn.addEventListener('click', () => loadNotebook(btn));
      btn.dataset.bound = "1";
    });
    document.getElementById('btn-reload').addEventListener('click', reloadFrame);
    document.getElementById('btn-tab').addEventListener('click', openInTab);
    document.getElementById('home-link').addEventListener('click', goHome);
    document.getElementById('crumb-home').addEventListener('click', goHome);

    bindSpark('ao-spark', 'ao-tip');
    bindSpark('aa-spark', 'aa-tip');

    document.getElementById('btn-sidebar').addEventListener('click', toggleSidebar);
    document.addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
        e.preventDefault();
        toggleSidebar();
      }
    });
    let sbHidden = false;
    try { sbHidden = localStorage.getItem('qlab-sidebar-hidden') === '1'; } catch (e) {}
    if (sbHidden) setSidebar(true, false);

    tickClock();
    setInterval(tickClock, 1000);
    setInterval(pollStatus,   3000);
    setInterval(pollPairHash, 3000);
    setInterval(pollOtcGate,  3000);
    setInterval(pollBlotter,  3000);
    pollStatus();
    pollOtcGate();
    pollBlotter();
  });
</script>
</body>
</html>
"""

HUB_HTML = (
    _HUB_TEMPLATE
    .replace("__NAV__", build_sidebar())
    .replace("__PORT__", str(PORT))
)

# ── OANDA blotter backing the front page ────────────────────────────────
# Same practice account the forex notebook trades. The positions endpoint is
# read-only; the close endpoint is invoked only by the blotter's two-click ✕.
OANDA_PRACTICE = True
OANDA_REST = (
    "https://api-fxpractice.oanda.com" if OANDA_PRACTICE
    else "https://api-fxtrade.oanda.com"
)
OANDA_ACCOUNT = os.environ.get("OANDA_ACCOUNT_ID")
OANDA_TOKEN = os.environ.get("OANDA_TOKEN")

_oanda_meta = {}  # instrument -> {pip, margin_rate, prec}; fetched once, lazily


def _oanda_get(path, **params):
    r = requests.get(
        f"{OANDA_REST}{path}",
        headers={"Authorization": f"Bearer {OANDA_TOKEN}"},
        params=params, timeout=10,
    )
    r.raise_for_status()
    return r.json()


def _oanda_instrument_meta():
    if not _oanda_meta:
        for i in _oanda_get(f"/v3/accounts/{OANDA_ACCOUNT}/instruments")["instruments"]:
            _oanda_meta[i["name"]] = {
                "pip": 10.0 ** int(i["pipLocation"]),
                "margin_rate": float(i["marginRate"]),
                "prec": int(i["displayPrecision"]),
            }
    return _oanda_meta


def _fetch_blotter():
    """openPositions + a pricing snapshot -> display rows for the blotter."""
    meta = _oanda_instrument_meta()
    positions = _oanda_get(f"/v3/accounts/{OANDA_ACCOUNT}/openPositions")["positions"]
    if not positions:
        return []
    px = {
        p["instrument"]: p
        for p in _oanda_get(
            f"/v3/accounts/{OANDA_ACCOUNT}/pricing",
            instruments=",".join(p["instrument"] for p in positions),
        )["prices"]
    }
    rows = []
    for p in positions:
        side = "long" if float(p["long"]["units"]) != 0 else "short"
        leg = p[side]
        inst = p["instrument"]
        m = meta.get(inst, {"pip": 1e-4, "margin_rate": 0.05, "prec": 5})
        avg = float(leg["averagePrice"])
        pl = float(leg["unrealizedPL"])
        margin = float(p.get("marginUsed", 0.0))
        q = px.get(inst)
        # Longs close on the bid, shorts on the ask — top-of-book, which is
        # what OANDA's own unrealizedPL marks at. The closeout prices run
        # wider off-hours and overstate the pip loss. Fall back to closeout
        # when the book is empty (halted market).
        cur = None
        if q:
            if side == "long":
                cur = float(q["bids"][0]["price"]) if q.get("bids") else float(q["closeoutBid"])
            else:
                cur = float(q["asks"][0]["price"]) if q.get("asks") else float(q["closeoutAsk"])
        pips = None if cur is None else ((cur - avg) if side == "long" else (avg - cur)) / m["pip"]
        # marginUsed / marginRate = USD notional by OANDA's own margin math
        notional = margin / m["margin_rate"] if m["margin_rate"] else 0.0
        rows.append({
            "instrument": inst,
            "market": inst.replace("_", "/"),
            "side": side.upper(),
            "units": abs(int(float(leg["units"]))),
            "margin": round(margin, 2),
            "avg": f"{avg:.{m['prec']}f}",
            "current": None if cur is None else f"{cur:.{m['prec']}f}",
            "pl_usd": round(pl, 2),
            "pips": None if pips is None else round(pips, 1),
            "pl_pct": round(100.0 * pl / notional, 2) if notional else None,
        })
    return rows


def _close_position(instrument):
    positions = _oanda_get(f"/v3/accounts/{OANDA_ACCOUNT}/openPositions")["positions"]
    target = next((p for p in positions if p["instrument"] == instrument), None)
    if target is None:
        return {"ok": False, "error": f"No open position in {instrument}"}
    side = "long" if float(target["long"]["units"]) != 0 else "short"
    r = requests.put(
        f"{OANDA_REST}/v3/accounts/{OANDA_ACCOUNT}/positions/{instrument}/close",
        headers={"Authorization": f"Bearer {OANDA_TOKEN}"},
        json={f"{side}Units": "ALL"}, timeout=15,
    )
    if r.status_code >= 400:
        try:
            msg = r.json().get("errorMessage", r.text[:200])
        except ValueError:
            msg = r.text[:200]
        return {"ok": False, "error": msg}
    return {"ok": True}


# ── Alpaca blotter (same paper account the notebooks trade) ─────────────
ALPACA_BASE = "https://paper-api.alpaca.markets"
ALPACA_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET = os.environ.get("ALPACA_API_SECRET")


def _alpaca_headers():
    return {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}


def _fetch_alpaca_blotter():
    """GET /v2/positions -> display rows. Alpaca computes price, market value
    and unrealized P&L server-side; qty is signed (negative = short)."""
    r = requests.get(f"{ALPACA_BASE}/v2/positions", headers=_alpaca_headers(), timeout=10)
    r.raise_for_status()
    rows = []
    for p in r.json():
        rows.append({
            "symbol": p["symbol"],
            "side": p["side"].upper(),
            "qty": abs(float(p["qty"])),
            "avg": float(p["avg_entry_price"]),
            "price": float(p["current_price"]),
            "value": abs(float(p["market_value"])),
            "pl_usd": round(float(p["unrealized_pl"]), 2),
            "pl_pct": round(100.0 * float(p["unrealized_plpc"]), 2),
        })
    return rows


def _close_alpaca_position(symbol):
    # DELETE /v2/positions/{symbol} liquidates the whole position with a
    # market order (rejected off-hours for equities; error passes through).
    r = requests.delete(
        f"{ALPACA_BASE}/v2/positions/{requests.utils.quote(symbol, safe='')}",
        headers=_alpaca_headers(), timeout=15,
    )
    if r.status_code >= 400:
        try:
            msg = r.json().get("message", r.text[:200])
        except ValueError:
            msg = r.text[:200]
        return {"ok": False, "error": msg}
    return {"ok": True}


# ── Account overview backing the front-page dashboard ───────────────────
# NAV/equity history for the sparklines is sampled by a background task, so
# the curves keep accumulating while the hub is up regardless of which page
# is open. Alpaca additionally has a real intraday equity curve served by
# its portfolio-history endpoint; OANDA has no equivalent, so its line is
# whatever this process has watched.
_HIST_MAX = 1440   # 24h at one sample per minute
_HIST_STEP = 60    # seconds between samples
_acct_hist = {"oanda": deque(maxlen=_HIST_MAX), "alpaca": deque(maxlen=_HIST_MAX)}
_alpaca_day_cache = {"t": 0.0, "base": None, "points": []}


def _fetch_oanda_account():
    a = _oanda_get(f"/v3/accounts/{OANDA_ACCOUNT}/summary")["account"]
    return {
        "nav": float(a["NAV"]),
        "balance": float(a["balance"]),
        "unrealized": float(a["unrealizedPL"]),
        "margin_used": float(a["marginUsed"]),
        "margin_avail": float(a["marginAvailable"]),
        "open_trades": int(a["openTradeCount"]),
    }


def _fetch_alpaca_account():
    r = requests.get(f"{ALPACA_BASE}/v2/account", headers=_alpaca_headers(), timeout=10)
    r.raise_for_status()
    a = r.json()
    return {
        "equity": float(a["equity"]),
        "last_equity": float(a["last_equity"]),
        "cash": float(a["cash"]),
        "buying_power": float(a["buying_power"]),
        "long_value": float(a["long_market_value"]),
    }


def _alpaca_day_curve():
    """Today's 5-minute equity curve + prior-close baseline; cached 2 min.

    Best-effort: portfolio-history occasionally 500s even when /v2/account is
    healthy, so failures keep the last good curve (possibly empty) instead of
    taking the whole account panel down, and retry no sooner than the cache
    window."""
    now = time.time()
    if now - _alpaca_day_cache["t"] > 120:
        _alpaca_day_cache["t"] = now
        try:
            r = requests.get(
                f"{ALPACA_BASE}/v2/account/portfolio/history",
                headers=_alpaca_headers(),
                params={"period": "1D", "timeframe": "5Min"}, timeout=10,
            )
            r.raise_for_status()
            h = r.json()
            pts = [
                [int(t), float(v)]
                for t, v in zip(h.get("timestamp") or [], h.get("equity") or [])
                if v is not None
            ]
            _alpaca_day_cache.update(base=h.get("base_value"), points=pts)
        except Exception:
            pass
    return _alpaca_day_cache


async def _history_sampler():
    while True:
        if OANDA_TOKEN:
            try:
                snap = await asyncio.to_thread(_fetch_oanda_account)
                _acct_hist["oanda"].append([int(time.time()), snap["nav"]])
            except Exception:
                pass
        if ALPACA_KEY and ALPACA_SECRET:
            try:
                snap = await asyncio.to_thread(_fetch_alpaca_account)
                _acct_hist["alpaca"].append([int(time.time()), snap["equity"]])
            except Exception:
                pass
        await asyncio.sleep(_HIST_STEP)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # NAV/equity sampler for the dashboard sparklines; cancelled on shutdown.
    sampler = asyncio.create_task(_history_sampler())
    try:
        yield
    finally:
        sampler.cancel()


hub = FastAPI(title="Quant Lab Hub", lifespan=_lifespan)

@hub.get("/", response_class=HTMLResponse)
async def serve_hub():
    return HUB_HTML

@hub.get("/api/pair-hash")
async def pair_hash():
    p = pathlib.Path("data/pair.txt")
    if not p.exists():
        return {"hash": None}
    return {"hash": hashlib.md5(p.read_bytes()).hexdigest()}

@hub.get("/api/status")
async def status():
    done = pathlib.Path(".fundamental_done").exists()
    # Only trust the reports once fundamental.py has signalled completion,
    # otherwise they may still be from the previous pair.
    return {
        "fundamental_done": done,
        "strategies": suggested_tabs() if done else [],
    }

@hub.get("/api/pair-ready")
async def pair_ready():
    return {"ready": pathlib.Path(".pair_ready").exists()}

@hub.get("/api/otc-ready")
async def otc_ready():
    """Gate state for OTC Track: which upstream desks are vouching for symbols.

    Ready as soon as either Pair Trading or LETF Backtest has completed a run;
    the stamp moves whenever their picks do, which is the sidebar's cue to
    reload the frame. A symbol file with no ready flag behind it is a leftover
    and counts for neither (see otc_watchlist.py).
    """
    symbols, armed = otc_watchlist.resolve_watchlist(".")
    return {
        "ready":   bool(symbols),
        "stamp":   otc_watchlist.watchlist_stamp("."),
        "sources": [src.label for src in armed],
        "symbols": symbols,
    }

@hub.get("/api/oanda/positions")
async def oanda_positions():
    if not OANDA_TOKEN:
        return {"ok": False, "error": "OANDA_TOKEN not set"}
    try:
        # to_thread: requests is blocking, and this loop also serves the
        # marimo notebook websockets
        return {"ok": True, "rows": await asyncio.to_thread(_fetch_blotter)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}

@hub.post("/api/oanda/close/{instrument}")
async def oanda_close(instrument: str):
    if not OANDA_TOKEN:
        return {"ok": False, "error": "OANDA_TOKEN not set"}
    try:
        return await asyncio.to_thread(_close_position, instrument)
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}

@hub.get("/api/oanda/account")
async def oanda_account():
    if not OANDA_TOKEN:
        return {"ok": False, "error": "OANDA_TOKEN not set"}
    try:
        acct = await asyncio.to_thread(_fetch_oanda_account)
        # sampled history + a live final point, so the line moves between samples
        history = list(_acct_hist["oanda"]) + [[int(time.time()), acct["nav"]]]
        return {"ok": True, "acct": acct, "history": history, "base": None}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}

@hub.get("/api/alpaca/account")
async def alpaca_account():
    if not (ALPACA_KEY and ALPACA_SECRET):
        return {"ok": False, "error": "ALPACA_API_KEY not set"}
    try:
        acct = await asyncio.to_thread(_fetch_alpaca_account)
        day = await asyncio.to_thread(_alpaca_day_curve)
        # Paper accounts report last_equity=0 until their first end-of-day
        # snapshot; portfolio-history's base_value is the same prior-close
        # number, so fall back to it for the day-P&L baseline. With neither
        # available, use current equity: day P&L reads 0.00 instead of the
        # account's whole value.
        if acct["last_equity"] <= 0:
            acct["last_equity"] = float(day["base"]) if day["base"] else acct["equity"]
        history = (day["points"] or list(_acct_hist["alpaca"]))
        history = history + [[int(time.time()), acct["equity"]]]
        return {"ok": True, "acct": acct, "history": history, "base": day["base"]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}

@hub.get("/api/alpaca/positions")
async def alpaca_positions():
    if not (ALPACA_KEY and ALPACA_SECRET):
        return {"ok": False, "error": "ALPACA_API_KEY not set"}
    try:
        return {"ok": True, "rows": await asyncio.to_thread(_fetch_alpaca_blotter)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}

@hub.post("/api/alpaca/close/{symbol}")
async def alpaca_close(symbol: str):
    if not (ALPACA_KEY and ALPACA_SECRET):
        return {"ok": False, "error": "ALPACA_API_KEY not set"}
    try:
        return await asyncio.to_thread(_close_alpaca_position, symbol)
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}

# Mounted last: Mount("/") matches every path, so the hub's own routes above
# must be registered first to win. Each notebook keeps its own prefix
# (/fundamental, /forex, ...), which is what marimo already renders its asset
# and websocket URLs against.
# Desk session events (desk + timing only) for Grafana; see hub_events.py.
hub.mount("/", hub_events.SessionMiddleware(marimo_asgi, hub_events.desk_map(NOTEBOOKS)))

# Public edge. With SITE_PASSWORD unset this is a transparent pass-through,
# so localhost development is unchanged. See site_gate.py.
PAGES_ORIGIN = "https://tobibui1904.github.io"
app = SiteGate(
    hub,
    password=os.environ.get("SITE_PASSWORD"),
    secret=os.environ.get("SESSION_SECRET"),
    allow_origin=PAGES_ORIGIN,
)


def parse_funnel_url(status_output, port):
    """Public https://...ts.net URL from `tailscale funnel status`, or None.

    Only returns a URL when Funnel is genuinely public AND proxying `port` --
    a "(tailnet only)" serve is reachable from your own devices, not from the
    internet, and reporting it as a public link would be a lie.
    """
    if not status_output:
        return None
    proxies_us = any(
        f"{host}:{port}" in status_output for host in ("127.0.0.1", "localhost")
    )
    if not proxies_us:
        return None
    m = re.search(r"(https://[^\s/]+\.ts\.net)\s*\(Funnel on\)", status_output)
    return m.group(1) if m else None


def funnel_url(port=None):
    """Ask the local tailscaled whether it is publishing this hub.

    Best effort: the tunnel is a separate daemon, so the hub can only ask, and
    a missing or slow tailscale must never delay startup.
    """
    exe = shutil.which("tailscale") or r"C:\Program Files\Tailscale\tailscale.exe"
    try:
        out = subprocess.run(
            [exe, "funnel", "status"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return None
    return parse_funnel_url(out, PORT if port is None else port)

if __name__ == "__main__":
    # Process-wide logging wiring. Here, not at import: tests import main.
    hub_events.install(NOTEBOOKS)
    pathlib.Path(".fundamental_done").unlink(missing_ok=True)
    # ASCII arrow: the console may be cp1252 when stdout is piped on Windows
    print(f"  Quant Lab -> http://{HOST}:{PORT}")
    # Read the gate's *actual* state off the constructed object, not the
    # environment again -- so this line cannot drift from what SiteGate is
    # really enforcing.
    if app.password:
        print("  Site gate: ARMED -- password required for every route.")
    else:
        print(
            "  Site gate: DISABLED -- the hub is UNPROTECTED. "
            "Do NOT expose this publicly."
        )
    # The tunnel is a separate daemon, so ask it rather than assume.
    public = funnel_url()
    if public:
        print(f"  Public     -> {public}")
    else:
        print(f"  Public     -> not exposed (tailscale funnel --bg {PORT})")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")