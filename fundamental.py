import marimo

__generated_with = "0.23.9"
app = marimo.App(css_file="../../theme.css", html_head_file="../../theme_head.html")


@app.cell
def _():
    with open('data/pair.txt', 'r') as f:
        content = f.read()
        lines = content.splitlines()
        pair = lines[0].split(" ")
    return (pair,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import requests
    from bs4 import BeautifulSoup
    import logging
    import calendar
    import time
    import yfinance as yf
    import json
    import re
    import sys
    import textwrap
    from datetime import datetime
    from typing import Any
    import ollama
    import marimo as mo

    # When stdout is a pipe rather than a console, Python encodes it with the
    # locale codec — cp1252 on this machine — and the report's ✓/✗/⚠/═ output
    # raises UnicodeEncodeError. Force UTF-8 on the real streams; marimo's
    # in-notebook capture stream has no reconfigure().
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")

    return (
        Any,
        BeautifulSoup,
        calendar,
        datetime,
        json,
        logging,
        mo,
        np,
        ollama,
        pd,
        re,
        requests,
        sys,
        textwrap,
        time,
        yf,
    )


@app.cell
def _(mo):
    def page_header(title, section_number, subtitle=None):
        subtitle_html = f"""
          <div style="font-family: 'DM Mono', monospace; font-size: 11px; color: var(--color-text-secondary); letter-spacing: 0.1em; margin-top: 8px;">{subtitle}</div>
        """ if subtitle else ""

        return mo.Html(f"""
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <div style="padding: 2rem 0 1.5rem;">
      <div style="display: flex; align-items: flex-start; gap: 16px;">
        <div style="padding-top: 6px;">
          <div style="width: 2px; height: 52px; background: #1D9E75;"></div>
        </div>
        <div>
          <div style="font-family: 'DM Mono', monospace; font-size: 11px; color: #1D9E75; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 6px;">Section {section_number}</div>
          <div style="font-family: 'DM Serif Display', serif; font-size: 42px; font-style: italic; font-weight: 400; line-height: 1.05; color: var(--color-text-primary);">{title}</div>
          {subtitle_html}
        </div>
      </div>
    </div>
    """)

    return (page_header,)


@app.cell
def _(mo):
    mo.Html("""
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

    <div style="padding: 2.5rem 0 2rem; text-align: center;">
      <div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; border-radius: 50%; border: 0.5px solid #1D9E75; margin-bottom: 1rem;">
        <i class="ti ti-chart-candle" style="font-size: 22px; color: #1D9E75;"></i>
      </div>
      <h1 style="font-family: 'DM Serif Display', serif; font-size: 36px; font-weight: 400; font-style: italic; margin: 0 0 6px; letter-spacing: -0.01em; color: var(--color-text-primary);">Financial Analysis Strategy</h1>
      <p style="font-family: 'DM Mono', monospace; font-size: 11px; color: var(--color-text-secondary); letter-spacing: 0.18em; text-transform: uppercase; margin: 0 0 1.25rem;">Designed optimally for Options</p>
      <div style="display: inline-flex; align-items: center; gap: 6px; font-family: 'DM Mono', monospace; font-size: 11px; color: #0F6E56; background: #E1F5EE; padding: 4px 14px; border-radius: 999px;">
        <span style="width: 6px; height: 6px; border-radius: 50%; background: #1D9E75; display: inline-block;"></span>
        Live
      </div>
      <div style="margin-top: 1.5rem; width: 40px; height: 0.5px; background: var(--color-border-tertiary); margin-left: auto; margin-right: auto;"></div>
    </div>
    """)
    return


@app.cell
def _(mo, pair):
    mo.Html(f"""
    <div style="padding:2rem 0;display:flex;justify-content:center;">
      <div style="display:inline-flex;align-items:center;gap:0;
        background:var(--color-background-primary);
        border:0.5px solid var(--color-border-tertiary);
        border-radius:var(--border-radius-lg);
        overflow:hidden;">

        <div style="padding:18px 28px;display:flex;flex-direction:column;align-items:center;gap:6px;
          background:#EEEDFE;">
          <span style="font-size:11px;font-weight:500;letter-spacing:.1em;text-transform:uppercase;
            color:#534AB7;">leg A</span>
          <span style="font-size:26px;font-weight:500;color:#26215C;font-family:var(--font-mono);">{pair[0]}</span>
          <span style="font-size:11px;color:#534AB7;">long</span>
        </div>

        <div style="padding:18px 16px;display:flex;flex-direction:column;align-items:center;gap:4px;
          border-left:0.5px solid var(--color-border-tertiary);
          border-right:0.5px solid var(--color-border-tertiary);">
          <i class="ti ti-arrows-exchange" style="font-size:20px;color:var(--color-text-secondary);" aria-hidden="true"></i>
          <span style="font-size:10px;letter-spacing:.08em;color:var(--color-text-secondary);text-transform:uppercase;">pair</span>
        </div>

        <div style="padding:18px 28px;display:flex;flex-direction:column;align-items:center;gap:6px;
          background:#E1F5EE;">
          <span style="font-size:11px;font-weight:500;letter-spacing:.1em;text-transform:uppercase;
            color:#0F6E56;">leg B</span>
          <span style="font-size:26px;font-weight:500;color:#04342C;font-family:var(--font-mono);">{pair[1]}</span>
          <span style="font-size:11px;color:#0F6E56;">short</span>
        </div>

      </div>
    </div>
    """)
    return


@app.cell
def _(page_header):
    page_header("Crawling Financial Statements", "01")
    return


@app.cell
def _(BeautifulSoup, calendar, logging, mo, np, pair, pd, requests, time, yf):
    """
    Extract Latest 20 Quarterly (10-Q) Financial Statements from SEC EDGAR
    Returns pandas DataFrames instead of saving to Excel
    """
    from fundamental_core import extract_latest_4_quarters


    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    pd.options.display.float_format = (
        lambda x: "{:,.0f}".format(x) if int(x) == x else "{:,.2f}".format(x)
    )

    # ============================================================================
    # CONFIGURATION - CHANGE YOUR EMAIL HERE
    # ============================================================================

    HEADERS = {"User-Agent": "buituannghia1904@gmail.com"}  # REQUIRED BY SEC

    # ============================================================================
    # USAGE
    # ============================================================================

    stock1_results = extract_latest_4_quarters(pair[0], HEADERS, show_filing_times=True)
    stock2_results = extract_latest_4_quarters(pair[1], HEADERS, show_filing_times=True)

    def make_tab(results, ticker):
        if results is None:
            return mo.md(f"❌ Could not load data for {ticker}")
        def tbl(df):
            return mo.ui.table(df) if df is not None else mo.md("_Not available_")
        return mo.vstack([
            mo.md("### Income Statement"),  tbl(results["income_statement_latest"]),
            mo.md("### Balance Sheet"),     tbl(results["balance_sheet_latest"]),
            mo.md("### Cash Flow"),         tbl(results["cash_flow_statement_latest"]),
        ])

    mo.ui.tabs({
        pair[0]: make_tab(stock1_results, pair[0]),
        pair[1]: make_tab(stock2_results, pair[1]),
    })
    return stock1_results, stock2_results


@app.cell
def _(
    Any,
    datetime,
    json,
    ollama,
    pair,
    pd,
    re,
    stock1_results,
    sys,
    textwrap,
):
    from fundamental_core import run_analysis

    if __name__ == "__main__":
        result1 = run_analysis(stock1_results, ticker = pair[0], save_json=True, filename="data/report_stock1.json")
    return result1, run_analysis


@app.cell
def _(pair, run_analysis, stock2_results):
    result2 = run_analysis(stock2_results, ticker = pair[1], save_json=True, filename="data/report_stock2.json")
    return (result2,)


@app.cell
def _(page_header):
    page_header("AI Analysis", "02")
    return


@app.cell
def _(mo, pair, pd, result1, result2):
    def render_report(result: dict) -> mo.Html:
        if not result:
            return mo.callout(mo.md("No result available"), kind="warn")

        company     = result.get("company", "Unknown")
        report_date = result.get("report_date", "")

        # ── Financial Health ──────────────────────────────────────────────
        health  = result.get("financial_health", {})
        rating  = health.get("rating", "N/A")
        rating_color = {"Strong": "success", "Moderate": "warn", "Weak": "danger"}.get(rating, "info")

        # ── Key Ratios table ──────────────────────────────────────────────
        ratios = result.get("key_ratios", {})
        ratio_rows = [
            {"Ratio": name, "Value": d.get("value"), "Unit": d.get("unit",""), "Interpretation": d.get("interpretation","")}
            for name, d in ratios.items() if isinstance(d, dict)
        ]

        # ── Trends ───────────────────────────────────────────────────────
        trends = result.get("trends", [])

        # ── Analysis ─────────────────────────────────────────────────────
        analysis = result.get("analysis", {})

        # ── Risks ────────────────────────────────────────────────────────
        risks = result.get("risks", [])

        # ── Option Strategy ───────────────────────────────────────────────
        s    = result.get("option_strategy", {})
        bias = s.get("bias", "")
        bias_color = {"Bullish": "success", "Bearish": "danger", "Neutral": "warn", "Hedge": "info"}.get(bias, "info")

        return mo.vstack([
            mo.md(f"# {company}  ·  {report_date}"),

            mo.callout(mo.md(f"**{rating}** — {health.get('summary','')}"), kind=rating_color),

            mo.md("## Key Ratios"),
            mo.ui.table(pd.DataFrame(ratio_rows)),

            mo.md("## Trends"),
            mo.vstack([mo.md(f"- {t}") for t in trends]),

            mo.md("## Analysis"),
            mo.vstack([
                mo.md(f"**Revenue & Earnings:** {analysis.get('revenue_and_earnings','')}"),
                mo.md(f"**Margins:** {analysis.get('margins','')}"),
                mo.md(f"**Balance Sheet:** {analysis.get('balance_sheet','')}"),
                mo.md(f"**Cash Flow:** {analysis.get('cash_flow','')}"),
                mo.md(f"**Red Flags:** {analysis.get('red_flags','')}"),
                mo.callout(mo.md(f"▶ {analysis.get('investment_thesis','')}"), kind="info"),
            ]),

            mo.md("## Key Risks"),
            mo.vstack([mo.md(f"- {r}") for r in risks]),

            mo.md("## Option Strategy"),
            mo.callout(mo.md(
                f"**{s.get('name','')}** [{bias}]\n\n"
                f"- Structure: {s.get('structure','')}\n"
                f"- Max Profit: {s.get('max_profit','')}\n"
                f"- Max Loss: {s.get('max_loss','')}\n"
                f"- IV Environment: {s.get('ideal_iv_environment','')}\n\n"
                f"**Rationale:** {s.get('rationale','')}\n\n"
                f"**Why best:** {s.get('why_best','')}"
            ), kind=bias_color),
        ])


    tabs = mo.ui.tabs({
        pair[0]: render_report(result1),
        pair[1]: render_report(result2),
    })
    return (tabs,)


@app.cell
def _(mo, tabs):
    import pathlib
    pathlib.Path(".fundamental_done").touch()

    mo.vstack([
        tabs,
        mo.md("✅ Fundamental analysis complete.")
    ])
    return


if __name__ == "__main__":
    app.run()
