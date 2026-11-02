import marimo

__generated_with = "0.23.14"
app = marimo.App(
    width="medium",
    css_file="../../theme.css",
    html_head_file="../../theme_head.html",
)


@app.cell
def _():
    import datetime as dt
    import os
    import pathlib

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import rateslib as rl

    from dotenv import load_dotenv

    import marimo as mo
    import sys as _sys

    # market_cache is shared across desks, so it lives in <root>/shared.
    _shared = str(pathlib.Path(mo.notebook_dir()).parents[1] / "shared")
    if _shared not in _sys.path:
        _sys.path.insert(0, _shared)

    # Credentials live in .env (gitignored), the same as every other notebook
    # here. os.environ alone never reads that file.
    _ = load_dotenv()

    return (
        dt,
        mo,
        np,
        os,
        pathlib,
        pd,
        rl,
    )


@app.cell
def _(pathlib, mo):
    import sys as _sys
    _here = str(pathlib.Path(mo.notebook_dir()))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    import fedinvest as _fedinvest

    session = _fedinvest.make_session()

    def fetch_latest(lookback=7):
        """Most recent business day with prices (see fedinvest.fetch_latest)."""
        return _fedinvest.fetch_latest(session, lookback=lookback)

    return fetch_latest, session


@app.cell
def _(session):
    import fedinvest as _fedinvest
    import market_cache as _market_cache

    def fetch_terms(cusips):
        """Per-CUSIP instrument terms (see fedinvest.fetch_terms)."""
        return _fedinvest.fetch_terms(session, cusips, _market_cache.get_cache())

    return (fetch_terms,)


@app.cell
def _(pd, session):
    _MSPD = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
             "/v1/debt/mspd")

    def fetch_mspd():
        """
        Latest monthly MSPD detail, one request per table, converted to dollars.

        Table 3 (marketable detail): authoritative outstanding_amt per CUSIP —
        fills the gaps the auction API leaves. Published in MILLIONS; each
        CUSIP carries its total on exactly one row (any extra rows for the
        same CUSIP have a null amount — hence the max(), which skips NaN).

        Table 5 (stripped form): the STRIPS data — how much of each security's
        face is held stripped, and the month's reconstitution flow. Keyed by
        the CORPUS (principal STRIP) CUSIP, so join it on corpus_cusip.
        Published in THOUSANDS.
        """
        def rows(table, latest, fields):
            r = session.get(f"{_MSPD}/{table}",
                             params={"filter": f"record_date:eq:{latest}",
                                     "fields": ",".join(fields),
                                     "page[size]": 10000},
                             headers={"Accept": "application/json"},
                             timeout=(10, 60))
            r.raise_for_status()
            j = r.json()
            if j["meta"]["total-pages"] > 1:
                raise RuntimeError(f"{table}: response paginated — raise page[size]")
            df = pd.DataFrame(j["data"])
            if df.empty:
                raise RuntimeError(f"{table}: no rows for record_date {latest}")
            return df

        # Resolve the record date ONCE and use it for both tables — during
        # publication week one table can be a month ahead of the other, which
        # would otherwise silently mix months.
        _probe = session.get(f"{_MSPD}/mspd_table_5",
                              params={"page[size]": 1, "sort": "-record_date"},
                              headers={"Accept": "application/json"},
                              timeout=(10, 30))
        _probe.raise_for_status()
        _latest_data = _probe.json()["data"]
        if not _latest_data:
            raise RuntimeError("MSPD probe returned no data")
        _latest = _latest_data[0]["record_date"]

        t3 = rows("mspd_table_3_market", _latest,
                  ["security_class2_desc", "outstanding_amt"])
        t3["outstanding_amt"] = pd.to_numeric(t3["outstanding_amt"], errors="coerce")
        outstanding = (t3.groupby("security_class2_desc")["outstanding_amt"].max()
                         .mul(1e6).rename("outstanding_mspd")
                         .rename_axis("cusip").reset_index())
        outstanding["outstanding_mspd_asof"] = pd.Timestamp(_latest)

        t5 = rows("mspd_table_5", _latest,
                  ["cusip", "portion_stripped_amt",
                   "portion_unstripped_amt", "reconstituted_amt"])
        # Subtotal rows carry the literal string "null" as their CUSIP today;
        # if the API ever emits real nulls, pandas would match NaN join keys
        # and fan the Grand Total onto every non-strippable security. Keep
        # only real 9-character CUSIPs.
        t5 = t5[t5["cusip"].str.fullmatch(r"[0-9A-Z]{9}", na=False)]
        for c in ("portion_stripped_amt", "portion_unstripped_amt",
                  "reconstituted_amt"):
            t5[c] = pd.to_numeric(t5[c], errors="coerce") * 1e3
        strips = t5.rename(columns={"cusip": "corpus_cusip",
                                    "portion_stripped_amt": "stripped",
                                    "portion_unstripped_amt": "unstripped",
                                    "reconstituted_amt": "reconstituted"})
        strips["strips_asof"] = pd.Timestamp(_latest)
        return outstanding, strips

    return (fetch_mspd,)


@app.cell
def _(fetch_latest, fetch_mspd, fetch_terms, mo):
    try:
        price_date, prices = fetch_latest()
    except Exception:
        import logging as _logging
        _logging.getLogger("treasury").exception("FedInvest price fetch failed")
        mo.stop(True, mo.md(
            "**Treasury quotes unavailable.** FedInvest could not be reached or returned "
            "an invalid price file. No prices or fitted analytics are being published. "
            "Retry when the data service is available."))
    # Closing market mid per 100 face. FedInvest buy/sell are the ask/bid of the
    # closing composite; averaging positive sides falls back to the one quoted
    # side (near-maturity bills lose their buy side, bills have no end_of_day).
    prices["mid"] = prices[["buy", "sell"]].where(prices[["buy", "sell"]] > 0).mean(axis=1)
    _terms, _failed = fetch_terms(prices["cusip"])
    prices = prices.merge(_terms.drop(columns=["latest_issue"]),
                          on="cusip", how="left", validate="m:1")

    # MSPD monthly detail: authoritative outstanding per CUSIP, and the STRIPS
    # panel (stripped/unstripped face, reconstitution flow) joined via the
    # corpus STRIP CUSIP. All in dollars.
    _outstanding_mspd, _strips = fetch_mspd()
    prices = prices.merge(_outstanding_mspd, on="cusip", how="left",
                          validate="m:1")
    # Auction currentlyOutstanding is a pre-issue announcement amount; using
    # it as a fallback understates reopenings and invents zero/old float for
    # new issues. Keep monthly balances explicitly dated, with missing = unknown.
    prices = prices.drop(columns=["outstanding"]).rename(columns={
        "outstanding_mspd": "outstanding", "outstanding_mspd_asof": "outstanding_asof"})
    prices = prices.merge(_strips, on="corpus_cusip", how="left",
                          validate="m:1")

    # The masthead. Same tokens and same devices as the hub's front page —
    # dot, rule to the margin, serif italic clamp, double green rule — so the
    # notebook opens as a section of the app rather than a separate document.
    _hero = mo.Html(f"""<style>
    .fh {{ --rule:#26332C; --text:#E9EEEB; --muted:#93A49B; --faint:#5F6F66;
      --green:#1D9E75; --green-hi:#3AC493;
      font-family: 'DM Sans', -apple-system, 'Segoe UI', sans-serif;
      max-width: 1040px; margin-bottom: 10px;
      font-variant-numeric: tabular-nums; }}
    .fh-brow {{ display: flex; align-items: center; gap: 11px;
      font-family: 'DM Mono', monospace; font-size: 9.5px; color: var(--green);
      letter-spacing: 0.18em; text-transform: uppercase; white-space: nowrap; }}
    .fh-brow::before {{ content: ""; width: 6px; height: 6px; flex: none;
      border-radius: 50%; background: var(--green-hi);
      box-shadow: 0 0 6px rgba(58,196,147,0.6); }}
    .fh-brow::after {{ content: ""; flex: 1 1 auto; height: 1px;
      background: var(--rule); }}
    .fh-t {{ font-family: 'DM Serif Display', Georgia, serif; font-style: italic;
      font-weight: 400; font-size: clamp(34px, 6vw, 58px); line-height: 1.02;
      margin: 14px 0 0; color: var(--text); letter-spacing: -0.015em; }}
    .fh-t::after {{ content: ""; display: block; width: 88px;
      margin: 16px 0 16px; border-bottom: 5px double var(--green); }}
    .fh-m {{ font-family: 'DM Mono', monospace; font-size: 10px;
      color: var(--faint); display: flex; gap: 26px; flex-wrap: wrap;
      letter-spacing: 0.14em; text-transform: uppercase; }}
    .fh-m b {{ color: var(--text); font-weight: 400; font-size: 12.5px;
      letter-spacing: 0.01em; margin-right: 5px; }}
    </style>
    <div class="fh">
      <div class="fh-brow">United States Treasury · marketable securities</div>
      <div class="fh-t">Close of {price_date:%d %B %Y}</div>
      <div class="fh-m">
    <span><b>{len(prices):,}</b> outstanding</span>
    <span><b>{prices['security_type'].nunique()}</b> instrument types</span>
    <span><b>{prices['strippable'].eq('Yes').sum():,}</b> strippable</span>
    <span>end of day · per 100 face</span>
      </div>
    </div>""")
    _notes = [_hero]
    if _failed:
        # A failed terms lookup leaves NaN terms, which quietly files a
        # strippable security into the wrong tab — say so instead.
        _notes.append(mo.md(
            f"⚠️ instrument-terms lookup failed for {len(_failed)} CUSIP(s), "
            f"e.g. {_failed[0]} — affected rows may miss term columns or sit "
            "in the wrong tab until the next run"))
    # The close being priced, and how many terms lookups failed.
    price_close = price_date
    terms_failed = len(_failed)

    mo.vstack(_notes)
    return price_close, prices, terms_failed


@app.cell
def _(mo, prices):
    # One table per security type, each carrying only the columns that mean
    # something for that type; plus a STRIPS view of the strippable universe.
    _PRICES = ["buy", "sell", "end_of_day", "mid"]
    _COMMON = ["security_term", "issue_date", "maturity_date"]

    def _tab(df, cols, sort="maturity_date", ascending=True):
        cols = list(cols)
        if "outstanding" in cols:
            cols.insert(cols.index("outstanding") + 1, "outstanding_asof")
        if "stripped" in cols:
            cols.insert(cols.index("stripped") + 1, "strips_asof")
        cols = ["cusip"] + [c for c in cols if c in df.columns]
        return mo.ui.table(
            df[cols].sort_values(sort, ascending=ascending, ignore_index=True),
            page_size=15)

    _st = prices["security_type"]
    # Every strippable security lives ONLY in the STRIPS tab; the per-type
    # tabs carry the unstrippable remainder, so each tab is distinct and no
    # CUSIP appears twice. (Deliberate owner preference — note that every
    # outstanding note is strippable, so the Notes tab stays empty.)
    _is_strip = prices["strippable"].eq("Yes")
    # Bills: zero-coupon discount paper — no rate, no end_of_day (always 0).
    bills = prices[(_st == "MARKET BASED BILL") & ~_is_strip]
    notes = prices[(_st == "MARKET BASED NOTE") & ~_is_strip]
    bonds = prices[(_st == "MARKET BASED BOND") & ~_is_strip]
    tips = prices[(_st == "TIPS") & ~_is_strip]
    frns = prices[(_st == "MARKET BASED FRN") & ~_is_strip]

    # Strippable securities with their STRIPS panel and share stripped.
    strips = prices[_is_strip].copy()
    strips["pct_stripped"] = strips["stripped"] / (strips["stripped"]
                                                   + strips["unstripped"])

    _coupon = ["rate", "pay_frequency"] + _COMMON + _PRICES + [
        "outstanding", "bid_to_cover", "n_auctions", "strippable"]
    _head = mo.Html("""<style>
    .fs { --rule:#26332C; --text:#E9EEEB; --muted:#93A49B; --green:#1D9E75;
      font-family: 'DM Sans', -apple-system, 'Segoe UI', sans-serif;
      max-width: 1040px; margin: 34px 0 14px; }
    .fs-brow { display: flex; align-items: center; gap: 11px;
      font-family: 'DM Mono', monospace; font-size: 9px; color: var(--green);
      letter-spacing: 0.16em; text-transform: uppercase; white-space: nowrap; }
    .fs-brow::after { content: ""; flex: 1 1 auto; height: 1px;
      background: var(--rule); }
    .fs-t { font-family: 'DM Serif Display', Georgia, serif; font-style: italic;
      font-weight: 400; font-size: 28px; margin: 10px 0 8px; color: var(--text);
      letter-spacing: -0.012em; }
    .fs-d { font-size: 13px; color: var(--muted); line-height: 1.65;
      max-width: 62ch; }
    </style>
    <div class="fs">
      <div class="fs-brow">The universe</div>
      <div class="fs-t">Every security, by kind</div>
      <div class="fs-d">Each tab carries only the columns that mean something for
    that instrument. Anything strippable lives in STRIPS with its stripping
    activity, so no security appears twice.</div>
    </div>""")
    _tabs = mo.ui.tabs({
        f"Bills ({len(bills)})": _tab(bills, _COMMON + [
            "buy", "sell", "mid", "outstanding", "bid_to_cover", "n_auctions"]),
        f"Notes ({len(notes)})": _tab(notes, _coupon),
        f"Bonds ({len(bonds)})": _tab(bonds, _coupon),
        f"TIPS ({len(tips)})": _tab(tips, ["rate", "pay_frequency"] + _COMMON + _PRICES + [
            "outstanding", "tips_ref_cpi_issue", "tips_ref_cpi_base",
            "tips_index_ratio_issue", "tips_cpi_base_period",
            "tips_adj_price", "tips_unadj_price",
            "tips_adj_accrued_per1000", "tips_unadj_accrued_per1000",
            "tips_conversion_factor"]),
        f"FRNs ({len(frns)})": _tab(frns, ["rate", "frn_spread", "pay_frequency"]
                                    + _COMMON + _PRICES
                                    + ["outstanding", "bid_to_cover"]),
        f"STRIPS ({len(strips)})": _tab(strips, [
            "security_type", "rate", "pay_frequency", "maturity_date",
            "mid", "outstanding",
            "stripped", "unstripped", "pct_stripped", "reconstituted",
            "corpus_cusip", "tint_cusip1", "tint_cusip1_due",
            "tint_cusip2", "tint_cusip2_due", "min_strip_amount"],
            sort="stripped", ascending=False),
    })
    mo.vstack([_head, _tabs])
    return bonds, tips


@app.cell
def _(dt, mo, np, pd, prices, rl):
    FI_CSS = """<style>
    .fi {
      --ink:#0C110F; --surface:#121917; --raised:#18221D;
      --border:#1F2A24; --rule:#26332C;
      --text:#E9EEEB; --muted:#93A49B; --faint:#5F6F66;
      --green:#1D9E75; --green-hi:#3AC493; --gold:#C9A961; --sienna:#D8683C;
      --serif:'DM Serif Display', Georgia, serif;
      --mono:'DM Mono', ui-monospace, SFMono-Regular, monospace;
      --sans:'DM Sans', -apple-system, 'Segoe UI', sans-serif;
      font-family: var(--sans); color: var(--text); max-width: 1040px;
      font-variant-numeric: tabular-nums;
    }
    /* Masthead. The eyebrow carries a live dot and a rule that runs to the
       margin — the hub's own section device, so a desk here reads as a
       section of the same publication. */
    .fi-eyebrow { display: flex; align-items: center; gap: 11px;
      flex-wrap: wrap; font-family: var(--mono); font-size: 9.5px;
      color: var(--green); letter-spacing: 0.18em; text-transform: uppercase;
      margin-bottom: 12px; }
    .fi-eyebrow::before { content: ""; width: 6px; height: 6px; flex: none;
      border-radius: 50%; background: var(--green-hi);
      box-shadow: 0 0 6px rgba(58,196,147,0.6); }
    .fi-eyebrow::after { content: ""; flex: 1 1 auto; height: 1px;
      background: var(--rule); }
    .fi-title { font-family: var(--serif); font-style: italic; font-weight: 400;
      font-size: clamp(30px, 4.4vw, 44px); line-height: 1.02; margin: 0;
      letter-spacing: -0.015em; }
    /* The double rule is the hub's front-page mark: a ledger's total line. */
    .fi-title::after { content: ""; display: block; width: 88px;
      margin: 14px 0 15px; border-bottom: 5px double var(--green); }
    .fi-deck { font-size: 13px; color: var(--muted); max-width: 62ch;
      line-height: 1.65; margin-bottom: 26px; }
    .fi-rail, .fi-panel { background: var(--surface); border: 1px solid var(--rule);
      border-radius: 10px; padding: 18px 20px 12px; margin-bottom: 18px; }
    .fi-cap { font-family: var(--mono); font-size: 9.5px; color: var(--faint);
      letter-spacing: 0.1em; margin-top: 8px; display: flex; gap: 14px;
      align-items: center; flex-wrap: wrap; }
    .fi-k { width: 16px; height: 0; display: inline-block; margin-right: -8px; }
    .fi-k-line { border-top: 2px solid var(--green); }
    .fi-k-dash { border-top: 2px dashed var(--muted); }
    .fi-k-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); }
    .fi-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
      background: var(--rule); border: 1px solid var(--rule);
      border-radius: 10px; overflow: hidden; }
    .fi-stat { background: var(--surface); padding: 15px 16px 13px; }
    .fi-lbl { font-family: var(--mono); font-size: 9px; color: var(--faint);
      letter-spacing: 0.16em; text-transform: uppercase; }
    .fi-num { font-family: var(--mono); font-size: 29px; line-height: 1.15;
      margin-top: 6px; letter-spacing: -0.01em; }
    .fi-num span { color: var(--muted); }
    .fi-num em { font-style: normal; font-size: 13px; color: var(--faint);
      margin-left: 3px; letter-spacing: 0.04em; }
    .fi-sub { font-family: var(--mono); font-size: 9.5px; color: var(--faint);
      letter-spacing: 0.04em; margin-top: 5px; }
    /* The verdict line. One green stem marks the sentence a reader should
       take away if they read nothing else on the desk. */
    .fi-slope { font-family: var(--mono); font-size: 11.5px; color: var(--muted);
      margin: 16px 0 22px; padding-left: 13px; line-height: 1.6;
      letter-spacing: 0.02em; border-left: 2px solid var(--green); }
    .fi-slope b { font-weight: 500; color: var(--text); }
    .fi-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 26px; }
    /* Section head: title, then a hairline to the margin. */
    .fi-h { display: flex; align-items: baseline; gap: 14px;
      font-family: var(--serif); font-style: italic; font-size: 20px;
      margin-bottom: 6px; }
    .fi-h::after { content: ""; flex: 1 1 auto; min-width: 20px; height: 1px;
      background: var(--rule); }
    .fi-note { font-size: 12px; color: var(--muted); line-height: 1.6;
      margin-bottom: 12px; max-width: 68ch; }
    .fi-note b { color: var(--text); font-weight: 500; }
    .fi-note code { font-family: var(--mono); font-size: 11px; color: var(--muted); }
    /* Tables scroll inside their own box. Nine to eleven columns of monospace
       cannot fit a phone, and the page itself must never scroll sideways. */
    .fi-scroll { overflow-x: auto; margin: 0 -2px; }
    .fi-tbl { width: 100%; border-collapse: collapse; font-family: var(--mono);
      font-size: 11.5px; }
    .fi-tbl th { font-weight: 400; font-size: 9px; color: var(--faint);
      letter-spacing: 0.1em; text-transform: uppercase;
      padding: 0 9px 8px; border-bottom: 1px solid var(--rule);
      white-space: nowrap; }
    .fi-tbl td { padding: 8px 9px; color: var(--muted);
      border-bottom: 1px solid var(--border); white-space: nowrap; }
    /* Alignment must be set on th AND td together, at a specificity that
       outranks `.fi-tbl th`. A bare `.fi-r` is one class and loses to
       `.fi-tbl th` — which silently left-aligns every numeric header over a
       right-aligned column. The hub's stylesheet carries the same warning. */
    .fi-tbl th, .fi-tbl td { text-align: left; }
    .fi-tbl th.fi-r, .fi-tbl td.fi-r { text-align: right; }
    .fi-tbl th:first-child, .fi-tbl td:first-child { padding-left: 0; }
    .fi-tbl th:last-child, .fi-tbl td:last-child { padding-right: 0; }
    .fi-tbl tbody tr:last-child td { border-bottom: none; }
    .fi-tbl tbody tr:hover td { background: rgba(29,158,117,0.05); }
    .fi-r { text-align: right; }
    .fi-cusip { color: var(--text); }
    .fi-dim { color: var(--faint); }
    .fi :focus-visible { outline: 2px solid var(--green-hi); outline-offset: 2px; }
    @media (max-width: 900px) {
      .fi-grid, .fi-stats { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 560px) {
      .fi-grid, .fi-stats { grid-template-columns: 1fr; }
      .fi-eyebrow { font-size: 9px; letter-spacing: 0.12em; }
    }
    </style>"""

    # ── Bill analytics, built on rateslib's Bill / Curve / Solver endpoints ──
    def _show(_html):
        mo.output.append(mo.Html(_html))

    def _run_panel():
        import ust_curve as _ust
        _settle = _ust.settlement(prices["price_date"].iloc[0])
        try:
            _bf = _ust.fit_bills(prices, _settle)
        except _ust.InsufficientQuotes:
            bill_desk = {"priced": 0, "status": "Unavailable: insufficient eligible maturities"}
            _show(FI_CSS + '<div class="fi"><h2 class="fi-title">Bill analytics unavailable</h2>'
                    '<p>At least three distinct, valid two-sided maturities are required.</p></div>')
            return FI_CSS, bill_desk
        bill_curve, _curve = _bf.bills, _bf.curve
        _dropped, repo_rate = _bf.dropped, _bf.funding_rate
        _maxd = int(bill_curve["days"].max())
        _n_universe = _bf.n_universe
        # Implied forwards include risk/liquidity premia; they are not forecasts
        # of realized short rates or the policy path.
        _grid = [_settle + dt.timedelta(days=int(_t))
                 for _t in range(0, _maxd - 13, 28)] + [_settle + dt.timedelta(days=_maxd)]
        _fwd = [float(_curve.rate(_a, _z)) for _a, _z in zip(_grid[:-1], _grid[1:])]

        # ── Presentation ────────────────────────────────────────────────────
        # Rendered as native SVG in the Ledger palette rather than a matplotlib
        # PNG: the theme frames images as light "paper exhibits", which fights a
        # dark instrument panel. Green reads cheap/bid, sienna rich/offer, gold
        # is par and fair value.
        _GREEN, _SIENNA, _GOLD = "#1D9E75", "#D8683C", "#C9A961"
        _SLATE, _RULE = "#93A49B", "#26332C"

        # Tokens are the hub's own (main.py :root), not a second palette invented
        # here — including --faint, the third text tier these desks were missing,
        # which is what lets a table read as three levels instead of two.


        # Only dislocations beyond one standard deviation earn a colour. Painting
        # every ±1bp residual green or sienna dresses quote noise up as conviction.
        _sig = float(bill_curve["resid_bp"].std())

        def _tone(v, gate=True):
            """Cheap (yields more than fair) reads green; rich reads sienna."""
            if pd.isna(v):
                return _SLATE
            if gate and abs(v) < _sig:
                return _SLATE
            return _GREEN if v > 0 else _SIENNA

        def _handle(y):
            """Split a yield the way a rates screen reads it: handle, then tail."""
            _s = f"{y:.3f}"
            return _s[:-1], _s[-1]

        # ── Signature: the par rail ─────────────────────────────────────────
        # A bill is a promise of 100 at a date, bought for less. Par is the gold
        # hairline; every bill hangs beneath it by its own discount, deepest at
        # the long end. Terminal dot carries the rich/cheap verdict.
        _W, _H, _PADL, _PADR = 1000, 132, 64, 24
        _span = _maxd or 1
        # Square-root tenor axis. Linear-in-days buries the front month — where a
        # bill curve actually moves — in the first 8% of the width.
        _xs = lambda _dd: _PADL + (np.sqrt(max(_dd, 0)) / np.sqrt(_span)) * (_W - _PADL - _PADR)
        _maxdisc = max(100.0 - bill_curve["mid"].min(), 0.01)
        _drops = []
        for _, _r in bill_curve.iterrows():
            _x = _xs(_r["days"])
            _y2 = 26 + (100.0 - _r["mid"]) / _maxdisc * 82
            _c = _tone(_r["resid_bp"])
            _drops.append(
                f'<line x1="{_x:.1f}" y1="26" x2="{_x:.1f}" y2="{_y2:.1f}" '
                f'stroke="{_GREEN}" stroke-width="1" opacity="0.28"/>'
                f'<circle cx="{_x:.1f}" cy="{_y2:.1f}" r="2.6" fill="{_c}"/>')
        _rail = (
            f'<svg viewBox="0 0 {_W} {_H}" width="100%" role="img" '
            f'aria-label="Every bill hanging below par by its discount">'
            f'<line x1="{_PADL}" y1="26" x2="{_W - _PADR}" y2="26" '
            f'stroke="{_GOLD}" stroke-width="1"/>'
            f'<text x="0" y="23" fill="{_GOLD}" font-family="DM Mono, monospace" '
            f'font-size="11">100.000</text>'
            f'<text x="0" y="{_H - 6}" fill="{_SLATE}" font-family="DM Mono, monospace" '
            f'font-size="10">{100 - _maxdisc:.2f}</text>'
            + "".join(_drops) + '</svg>')

        # ── The curve, as the working instrument ────────────────────────────
        _CH, _CT, _CB = 330, 26, 40
        _fwd_days = [(_g - _settle).days for _g in _grid]
        _lo = min(bill_curve["bey"].min(), min(_fwd)) - 0.08
        _hi = max(bill_curve["bey"].max(), max(_fwd)) + 0.08
        _ys = lambda _v: _CT + (_hi - _v) / (_hi - _lo) * (_CH - _CT - _CB)

        _gridlines, _tick = [], np.ceil(_lo * 4) / 4
        while _tick <= _hi:
            _gy = _ys(_tick)
            _gridlines.append(
                f'<line x1="{_PADL}" y1="{_gy:.1f}" x2="{_W - _PADR}" y2="{_gy:.1f}" '
                f'stroke="{_RULE}" stroke-width="1"/>'
                f'<text x="{_PADL - 10}" y="{_gy + 3:.1f}" fill="{_SLATE}" '
                f'text-anchor="end" font-family="DM Mono, monospace" font-size="10">'
                f'{_tick:.2f}</text>')
            _tick += 0.25

        _fit = " ".join(f"{_xs(_d):.1f},{_ys(_v):.1f}" for _d, _v
                        in zip(bill_curve["days"], bill_curve["curve_bey"]))
        _step = []
        for _i, _f in enumerate(_fwd):
            _x1, _x2 = _xs(_fwd_days[_i]), _xs(_fwd_days[_i + 1])
            _step.append(f"{_x1:.1f},{_ys(_f):.1f} {_x2:.1f},{_ys(_f):.1f}")
        _dots = "".join(
            f'<circle cx="{_xs(_r["days"]):.1f}" cy="{_ys(_r["bey"]):.1f}" r="3.1" '
            f'fill="{_tone(_r["resid_bp"])}" '
            f'stroke="#0C110F" stroke-width="1"><title>{_r["cusip"]} · {_r["days"]}d · '
            f'{_r["bey"]:.3f}%</title></circle>' for _, _r in bill_curve.iterrows())
        # A term structure is read in tenors, not calendar months.
        _months = "".join(
            f'<text x="{_xs(_d):.1f}" y="{_CH - 12}" fill="{_SLATE}" '
            f'text-anchor="middle" font-family="DM Mono, monospace" font-size="10">'
            f'{_lab}</text>'
            for _d, _lab in ((14, "2w"), (30, "1m"), (91, "3m"), (182, "6m"),
                             (273, "9m"), (346, "12m")) if _d <= _maxd)
        _chart = (
            f'<svg viewBox="0 0 {_W} {_CH}" width="100%" role="img" '
            f'aria-label="T-bill yield curve with implied forward path">'
            + "".join(_gridlines)
            + f'<polyline points="{" ".join(_step)}" fill="none" stroke="{_SLATE}" '
              f'stroke-width="1.6" stroke-dasharray="5 4" opacity="0.85"/>'
            + f'<polyline points="{_fit}" fill="none" stroke="{_GREEN}" '
              f'stroke-width="1.6" opacity="0.6"/>' + _dots + _months + '</svg>')

        # ── The day's read ──────────────────────────────────────────────────
        _front, _back = bill_curve.iloc[0], bill_curve.iloc[-1]
        _trough = bill_curve.loc[bill_curve["bey"].idxmin()]
        _stats = [("Front", _front["bey"], f'{int(_front["days"])}d'),
                  ("Trough", _trough["bey"], f'{int(_trough["days"])}d'),
                  ("One year", _back["bey"], f'{int(_back["days"])}d'),
                  ("Funding proxy", repo_rate, "nearest 4-week bill; not repo")]
        _stat_html = "".join(
            f'<div class="fi-stat"><div class="fi-lbl">{_n}</div>'
            f'<div class="fi-num">{_handle(_v)[0]}<span>{_handle(_v)[1]}</span>'
            f'<em>%</em></div><div class="fi-sub">{_s}</div></div>'
            for _n, _v, _s in _stats)
        _slope = (_back["bey"] - _front["bey"]) * 100

        def _rows(_df, _col, _unit):
            _out = []
            for _, _r in _df.iterrows():
                _v = _r[_col]
                _out.append(
                    f'<tr><td class="fi-cusip">{_r["cusip"]}</td>'
                    f'<td>{_r["maturity_date"]:%d %b %y}</td>'
                    f'<td class="fi-r">{int(_r["days"])}d</td>'
                    f'<td class="fi-r">{_r["bey"]:.3f}</td>'
                    f'<td class="fi-r" style="color:{_tone(_v)}">{_v:+.2f}</td>'
                    f'<td class="fi-r fi-dim">{_r["spread_bp"]:.2f}</td></tr>')
            return (f'<div class="fi-scroll"><table class="fi-tbl"><thead><tr><th>CUSIP</th><th>Matures</th>'
                    f'<th class="fi-r">Term</th><th class="fi-r">Yield</th>'
                    f'<th class="fi-r">{_unit}</th><th class="fi-r">Price bp</th>'
                    f'</tr></thead><tbody>' + "".join(_out) + '</tbody></table></div>')

        _cheap = bill_curve.nlargest(5, "resid_bp")
        _carry = bill_curve.nlargest(5, "carry_roll_bp")

        def _bill_rows(frame, column):
            return [{"cusip": _r["cusip"], "maturity": _r["maturity_date"].date(),
                     column: float(_r[column])} for _, _r in frame.head(2).iterrows()]

        bill_desk = {"priced": int(len(bill_curve)), "resid_sd_bp": _sig,
                     "funding_pct": repo_rate, "slope_bp": float(_slope),
                     "cheap": _bill_rows(_cheap, "resid_bp"),
                     "carry": _bill_rows(_carry, "carry_roll_bp")}
        # Never let the universe shrink silently — say which bills fell out.
        _excl = (f' <span style="color:{_SIENNA}">Excluded: '
                 f'{", ".join(_dropped)}</span> — one-sided, crossed or wide quotes, '
                 f'or inside seven days of settlement.' if _dropped else "")

        _show(FI_CSS + f"""
        <div class="fi">
          <div class="fi-eyebrow">FedInvest · settle {_settle:%d %b %Y}</div>
          <h2 class="fi-title">The bill desk</h2>
          <div class="fi-deck">{len(bill_curve)} of {_n_universe} bills priced off
            two-sided quotes and fitted to one discount curve. Colour is reserved for the
            bills that sit more than {_sig:.1f}bp from fair — green cheap, sienna rich.
            Everything else is noise at this spread.{_excl}</div>
          <div class="fi-rail">{_rail}
            <div class="fi-cap">Every bill hangs below par by its discount ·
              longest {int(_back["days"])} days</div></div>
          <div class="fi-stats">{_stat_html}</div>
          <div class="fi-slope">Front to one year
            <b style="color:{_tone(_slope)}">{_slope:+.0f} bp</b> ·
            minimum observed bill yield at {_trough["maturity_date"]:%d %b %Y}</div>
          <div class="fi-panel">{_chart}
            <div class="fi-cap"><span class="fi-k fi-k-line"></span>Fitted curve
              <span class="fi-k fi-k-dash"></span>Implied 4-week forwards
              <span class="fi-k fi-k-dot"></span>Quotes</div></div>
          <div class="fi-grid">
            <div><div class="fi-h">Cheapest to the curve</div>
              <div class="fi-note">Indicative mid-yield above the fitted curve.
                Edge is yield bp; spread is price bp, so these columns cannot be
                compared directly as a trading-cost test.</div>{_rows(_cheap, "resid_bp", "Yield bp")}</div>
            <div><div class="fi-h">Best four-week carry</div>
              <div class="fi-note">P&amp;L in bp of face against financing at
                a bill-yield funding proxy of {repo_rate:.3f}%, curve unchanged
                and residual converging to fair. Excludes bid-ask costs and actual repo.</div>
              {_rows(_carry, "carry_roll_bp", "P&L bp")}</div>
          </div>
        </div>""")
        return FI_CSS, bill_desk

    FI_CSS, bill_desk = _run_panel()
    return FI_CSS, bill_desk



@app.cell
def _(FI_CSS, bonds, dt, mo, np, pd, prices, rl):
    def _show(_html):
        mo.output.append(mo.Html(_html))

    def _run_panel():
        # ── The legacy long bonds ────────────────────────────────────────────
        # Fifteen 30-year bonds issued 1996-2008 that carry no stripping
        # activity — absent from the MSPD stripped-form table, no corpus CUSIP.
        # Everything else in the coupon universe has a STRIPS bid; these do not.
        #
        # They are priced against a curve fitted to the WHOLE coupon universe,
        # not to themselves: fifteen scattered bonds cannot define fair value.
        import ust_curve as _ust
        _settle = _ust.settlement(prices["price_date"].iloc[0])
        try:
            _nf = _ust.fit_nominal(prices, _settle)
        except _ust.InsufficientQuotes as _why:
            _status = ("Unavailable: too few quotes for curve nodes" if "nodes" in str(_why)
                       else "Unavailable: insufficient eligible quotes")
            nominal_desk = {"fitted": 0, "status": _status}
            ust_curve = None
            _show(FI_CSS + '<div class="fi">Nominal curve unavailable: ' + (
                'too few quotes for a stable curve fit.' if "nodes" in str(_why)
                else 'insufficient eligible two-sided quotes.') + '</div>')
            return nominal_desk, ust_curve
        # Exported: the TIPS cell reads breakevens off this same curve rather than
        # refitting 338 securities to a second, subtly different nominal curve.
        _u, ust_curve = _nf.bonds, _nf.curve
        legacy_bonds = _u[_u["cusip"].isin(set(bonds["cusip"]))].sort_values("maturity_date").copy()

        _peers = _u[(~_u["cusip"].isin(set(bonds["cusip"])))
                    & _u["days"].between(legacy_bonds["days"].min(),
                                         legacy_bonds["days"].max())]
        _gap = legacy_bonds["resid_bp"].median() - _peers["resid_bp"].median()

        _GREEN, _SIENNA, _SLATE = "#1D9E75", "#D8683C", "#93A49B"
        _sd = float(_u["resid_bp"].std())
        nominal_desk = {"fitted": int(len(_u)), "resid_sd_bp": _sd,
                        "legacy_bonds": int(len(legacy_bonds)), "legacy_gap_bp": float(_gap)}

        def _t2(_v):
            if pd.isna(_v) or abs(_v) < _sd:
                return _SLATE
            return _GREEN if _v > 0 else _SIENNA

        def _row2(_r):
            _sp = "—" if pd.isna(_r["spread_bp"]) else f'{_r["spread_bp"]:.1f}'
            return (f'<tr><td class="fi-cusip">{_r["cusip"]}</td>'
                    f'<td class="fi-r">{_r["rate"]:.3f}</td>'
                    f'<td>{_r["maturity_date"]:%b %Y}</td>'
                    f'<td class="fi-r">{_r["mid"]:.3f}</td>'
                    f'<td class="fi-r fi-dim">{_r["accrued"]:.3f}</td>'
                    f'<td class="fi-r">{_r["ytm"]:.3f}</td>'
                    f'<td class="fi-r" style="color:{_t2(_r["resid_bp"])}">'
                    f'{_r["resid_bp"]:+.1f}</td>'
                    f'<td class="fi-r">{_r["mod_dur"]:.2f}</td>'
                    f'<td class="fi-r fi-dim">{_sp}</td></tr>')

        _tr = "".join(_row2(_r) for _, _r in legacy_bonds.iterrows())

        _show(FI_CSS + f"""
        <div class="fi">
          <div class="fi-eyebrow">No strip bid · issued 1996–2008</div>
          <h2 class="fi-title">The legacy long bonds</h2>
          <div class="fi-deck">Fifteen 30-year bonds with no stripping activity —
            every other coupon security carries a STRIPS bid, these do not. Priced
            against a curve fitted to all {len(_u)} coupon securities, since fifteen
            scattered bonds cannot define their own fair value.</div>
          <div class="fi-stats">
            <div class="fi-stat"><div class="fi-lbl">Captured</div>
              <div class="fi-num">{len(legacy_bonds)}<span>/15</span></div>
              <div class="fi-sub">one matures inside 90 days</div></div>
            <div class="fi-stat"><div class="fi-lbl">Coupons</div>
              <div class="fi-num">{legacy_bonds["rate"].min():.2f}<span>–{legacy_bonds["rate"].max():.2f}</span><em>%</em></div>
              <div class="fi-sub">against ~4.4% today</div></div>
            <div class="fi-stat"><div class="fi-lbl">Median edge</div>
              <div class="fi-num" style="color:{_t2(_gap)}">{_gap:+.1f}<em>bp</em></div>
              <div class="fi-sub">vs tenor-matched peers</div></div>
            <div class="fi-stat"><div class="fi-lbl">Bid-ask</div>
              <div class="fi-num">{legacy_bonds["spread_bp"].median():.0f}<em>bp</em></div>
              <div class="fi-sub">bills trade at 0.2</div></div>
          </div>
          <div class="fi-slope">Median residual difference versus peers:
            <b style="color:{_t2(_gap)}">{_gap:+.1f} yield bp</b>.
            This comparison does not isolate the effect of stripping eligibility.</div>
          <div class="fi-scroll"><table class="fi-tbl"><thead><tr><th>CUSIP</th><th class="fi-r">Coupon</th>
            <th>Matures</th><th class="fi-r">Clean</th><th class="fi-r">Acc</th>
            <th class="fi-r">Yield</th><th class="fi-r">Edge</th>
            <th class="fi-r">Dur</th><th class="fi-r">Price bp</th></tr></thead>
            <tbody>{_tr}</tbody></table></div>
          <div class="fi-note" style="margin-top:14px">Edge is yield against the
            fitted curve — positive is cheap. Colour is reserved for moves beyond
            {_sd:.1f}bp. Quoted spreads are price bp and cannot be compared directly
            with yield residuals. These are indicative valuations, before costs.</div>
        </div>""")
        return nominal_desk, ust_curve

    nominal_desk, ust_curve = _run_panel()
    return nominal_desk, ust_curve



@app.cell
def _(FI_CSS, dt, mo, np, os, pd, prices, rl, session, tips, ust_curve):
    def _show(_html):
        mo.output.append(mo.Html(_html))

    def _run_panel():
        # ── The five TIPS with no strip bid ──────────────────────────────────
        # Every other TIPS carries a corpus CUSIP and stripping activity; these
        # five do not. They are priced against a real curve fitted to the WHOLE
        # 52-bond TIPS universe, not to themselves — five scattered bonds cannot
        # define their own fair value, the same reasoning the legacy nominal
        # bonds get one cell up.
        _settle = rl.add_tenor(prices["price_date"].iloc[0].to_pydatetime(),
                               "1b", "F", "nyc")

        # CPI-U NSA is the index TIPS accrete on. Registering it in rateslib's own
        # fixings store is what lets IndexFixedRateBond.index_ratio() apply the
        # Treasury convention itself — three-month lag, interpolated across the
        # month — instead of us reimplementing it. Every real-terms metric (yield,
        # duration, edge, breakeven) works without it, so a BLS outage costs the
        # index-ratio columns and nothing else.
        # CUUR0000SA0 = CPI-U, US city average, all items, NOT seasonally adjusted,
        # base 1982-84=100. The "U" in position three is load-bearing: TIPS index
        # to the unadjusted series, and TreasuryDirect's own tips_cpi_base_period
        # confirms the base. v2 with a registered key allows 500 calls a day
        # against v1's 25; the key rides in the POST body rather than a query
        # string, and the request still works (at v1 limits) if it is unset.
        _hole, _cpi_note = {}, ""
        try:
            _payload = {"seriesid": ["CUUR0000SA0"],
                        "startyear": str(_settle.year - 2),
                        "endyear": str(_settle.year)}
            if os.environ.get("BLS_API_KEY"):
                _payload["registrationkey"] = os.environ["BLS_API_KEY"]
            # Accept MUST be overridden per-request. The shared session advertises
            # "text/csv,text/html" because treasurydirect serves the marketing page
            # to anything else; BLS v2 answers only application/json and rejects
            # that session header outright with 406 Not Acceptable. (v1 GET did not
            # content-negotiate, so this only broke on the move to v2 — and the
            # fail-soft CPI path hid it until the error text was surfaced.)
            _bls = session.post("https://api.bls.gov/publicAPI/v2/timeseries/data/",
                                json=_payload, headers={"Accept": "application/json"},
                                timeout=(10, 30))
            _bls.raise_for_status()
            _body = _bls.json()
            # BLS answers a throttled or malformed request with HTTP 200 and
            # status REQUEST_NOT_PROCESSED, so the status field is the real check.
            if _body.get("status") != "REQUEST_SUCCEEDED":
                raise RuntimeError("; ".join(_body.get("message", ["BLS refused"])))
            # "M01" <= period <= "M12" excludes M13, the ANNUAL AVERAGE, which is
            # not a thirteenth month and would corrupt the interpolation.
            _mon = [_x for _x in _body["Results"]["series"][0]["data"]
                    if "M01" <= _x["period"] <= "M12"]
            _dt_of = lambda _x: pd.Timestamp(int(_x["year"]), int(_x["period"][1:]), 1)
            # A month BLS never published comes back as value "-" carrying a
            # footnote that says why — October 2025 is permanently absent, killed
            # by the lapse in appropriations. Read the footnote rather than let the
            # row drop out silently: a hole in CPI is a hole in every reference CPI
            # three months later, and inventing a value there is not an option.
            _hole = {_dt_of(_x): "; ".join(_f["text"] for _f in _x.get("footnotes", [])
                                           if _f.get("text")) or "not published"
                     for _x in _mon
                     if pd.isna(pd.to_numeric(_x["value"], errors="coerce"))}
            _cpi = pd.Series(dict(sorted(
                (_dt_of(_x), float(_x["value"])) for _x in _mon
                if _dt_of(_x) not in _hole)))
            # Reference CPI runs three months behind the print, so the index ratio
            # is already DETERMINED out to the first of the third month after the
            # last release. Nothing between here and there is a forecast.
            _known = (_cpi.index[-1] + pd.DateOffset(months=3)).to_pydatetime()
            # ...and by the same lag, any month from three before settle onward is
            # LIVE: a hole there would have rateslib interpolate straight across a
            # missing print and hand back a wrong ratio. Refuse instead.
            _live = (pd.Timestamp(_settle) - pd.DateOffset(months=3)).replace(day=1)
            # BLS can omit an entire row, not just publish a '-' placeholder.
            # A sparse Series otherwise lets rateslib interpolate over that gap.
            for _month in pd.date_range(_live, _cpi.index[-1], freq="MS"):
                if _month not in _cpi.index:
                    _hole.setdefault(_month, "monthly observation missing")
            _bad = sorted(_m for _m in _hole if _m >= _live)
            if _bad:
                raise RuntimeError("CPI unpublished for "
                                   + ", ".join(f"{_m:%b %Y}" for _m in _bad)
                                   + " — inside the window today's ratios read")
            rl.fixings.add("US_CPI", _cpi)
            _has_cpi = _known > _settle
        except Exception as _err:
            _cpi, _known, _has_cpi = None, None, False
            _cpi_note = str(_err) or type(_err).__name__

        _t = prices[prices["security_type"] == "TIPS"].copy()
        _t = _t[(_t["mid"] > 0) & _t["tips_ref_cpi_base"].notna()]
        _t = _t.sort_values("maturity_date")
        _t["days"] = (_t["maturity_date"] - pd.Timestamp(_settle)).dt.days
        _t = _t[(_t["days"] > 0) & _t["dated_date"].notna()
                & _t["rate"].notna() & np.isfinite(_t["mid"])].copy()
        _t["two_sided"] = ((_t["buy"] > 0) & (_t["sell"] > 0)
                           & (_t["buy"] >= _t["sell"]))
        _t["spread_bp"] = np.where(_t["two_sided"],
                                   (_t["buy"] - _t["sell"]) / _t["mid"] * 1e4, np.nan)
        _fit_days = _t.loc[_t["two_sided"] & (_t["days"] > 365), "days"]
        if len(_fit_days) < 3 or _fit_days.nunique() < 3 or ust_curve is None:
            tips_desk = {"count": int(len(_t)), "no_strip_bid": 0,
                         "ten_year_breakeven": np.nan,
                         "cpi_note": "TIPS curve unavailable: insufficient eligible quotes or nominal curve"}
            _show(FI_CSS + '<div class="fi"><h2 class="fi-title">TIPS analytics unavailable</h2>'
                    '<p>At least three distinct, two-sided maturities beyond one year and a '
                    'nominal curve are required. No fitted yields or breakevens are published.</p></div>')
            return (tips_desk,)

        def _mkt(_r):
            """A rateslib TIPS. index_base is Treasury's own reference CPI on the
            dated date — the denominator of every index ratio this bond will ever
            print — so it comes from the auction record, never inferred."""
            return rl.IndexFixedRateBond(
                effective=pd.Timestamp(_r["dated_date"]).to_pydatetime(),
                termination=_r["maturity_date"].to_pydatetime(), spec="us_gbi",
                fixed_rate=float(_r["rate"]),
                index_base=float(_r["tips_ref_cpi_base"]),
                index_fixings="US_CPI" if _has_cpi else rl.NoInput(0))

        # FedInvest quotes TIPS in REAL (unadjusted) price, so everything here is
        # a real-terms metric straight off IndexFixedRateBond.
        _obj = [_mkt(_r) for _, _r in _t.iterrows()]
        _t["ytm"] = [float(_o.ytm(price=float(_r["mid"]), settlement=_settle))
                     for _o, (_, _r) in zip(_obj, _t.iterrows())]
        _t["accrued"] = [float(_o.accrued(_settle)) for _o in _obj]
        _t["mod_dur"] = [float(_o.duration(ytm=_y, settlement=_settle,
                                           metric="modified"))
                         for _o, _y in zip(_obj, _t["ytm"])]
        _t["dv01"] = [float(_o.duration(ytm=_y, settlement=_settle, metric="risk"))
                      / 100 for _o, _y in zip(_obj, _t["ytm"])]
        # Treasury publishes reference CPI and index ratios to five decimals,
        # truncating to six decimals before rounding (31 CFR 356, Appendix B).
        # Unrounded rateslib ratios differ from the factors used for invoices.
        from decimal import Decimal as _Decimal, ROUND_DOWN as _DOWN, ROUND_HALF_UP as _HALF_UP

        def _treasury_round(_value):
            return _value.quantize(_Decimal("0.000001"), rounding=_DOWN).quantize(
                _Decimal("0.00001"), rounding=_HALF_UP)

        def _reference_cpi(_date):
            _day = pd.Timestamp(_date)
            _month = _day.replace(day=1) - pd.DateOffset(months=3)
            _first = _Decimal(str(_cpi.loc[_month]))
            if _day.day == 1:
                return _treasury_round(_first)
            _second = _Decimal(str(_cpi.loc[_month + pd.DateOffset(months=1)]))
            return _treasury_round(_first + (_second - _first)
                                   * _Decimal(_day.day - 1) / _Decimal(_day.days_in_month))

        if _has_cpi:
            _ref_settle = _reference_cpi(_settle)
            _t["ratio"] = [float(_treasury_round(_ref_settle / _Decimal(str(_base))))
                           for _base in _t["tips_ref_cpi_base"]]
            # What actually gets wired: the real quote plus real accrued, scaled by
            # every point of inflation since the bond was dated. And the risk you
            # actually run — a seasoned TIPS carries its real DV01 times that ratio.
            _t["invoice"] = (_t["mid"] + _t["accrued"]) * _t["ratio"]
            _t["adj_dv01"] = _t["dv01"] * _t["ratio"]
        else:
            _t["ratio"] = _t["invoice"] = _t["adj_dv01"] = np.nan

        # ── The real curve ───────────────────────────────────────────────────
        # Fitted with plain FixedRateBonds cloned off each TIPS. A TIPS real yield
        # IS the nominal-convention yield of its real cashflows — spec us_gbi
        # carries calc_mode "us_gb" — and the twins agree with
        # IndexFixedRateBond.ytm to machine precision (asserted below). Twins keep
        # the solve on one curve: IndexFixedRateBond.rate() requires an index
        # curve too, i.e. an inflation forecast, to return a number that is
        # inflation-free by definition.
        _twin = [rl.FixedRateBond(
            effective=pd.Timestamp(_r["dated_date"]).to_pydatetime(),
            termination=_r["maturity_date"].to_pydatetime(), spec="us_gb",
            fixed_rate=float(_r["rate"])) for _, _r in _t.iterrows()]
        assert max(abs(float(_a.ytm(price=float(_m), settlement=_settle))
                       - float(_b.ytm(price=float(_m), settlement=_settle)))
                   for _a, _b, _m in zip(_obj, _twin, _t["mid"])) < 1e-9

        # Weight by how tightly each quote pins a YIELD. One price tick is worth
        # 1.5625/duration bp, so a three-month TIPS locates its own yield ~100x
        # more loosely than a thirty-year; equal weights let the front drag the
        # whole curve. One-sided quotes never enter the fit at all — `mid` there
        # is a bare bid wearing a mid's clothes.
        _t["y_noise_bp"] = (_t["spread_bp"].fillna(20.0) + 1.5625) / _t["mod_dur"]
        # Front-year seasonal/known inflation effects are outside this smooth
        # real-curve model; they must not influence the fit either.
        _keep = (_t["two_sided"] & (_t["days"] > 365)).values
        _rk = sorted({int(_y * 365.25) for _y in [1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30]
                      if _y * 365.25 <= _fit_days.max()} | {0, int(_fit_days.max())})
        if len(_rk) - 1 >= len(_fit_days):
            tips_desk = {"count": int(len(_t)), "no_strip_bid": 0,
                         "ten_year_breakeven": np.nan,
                         "cpi_note": "TIPS curve unavailable: too few quotes for the curve nodes"}
            _show(FI_CSS + '<div class="fi">TIPS analytics unavailable: too few eligible '
                    'quotes for a stable curve fit.</div>')
            return (tips_desk,)
        _real = rl.Curve(nodes={_settle + dt.timedelta(days=int(_k)): 1.0 for _k in _rk},
                         interpolation="log_linear", convention="act365f",
                         calendar="nyc", id="real")
        _rpr = {"curves": _real, "metric": "ytm", "settlement": _settle}
        rl.Solver(curves=[_real],
                  instruments=[(_o, _rpr) for _o, _k in zip(_twin, _keep) if _k],
                  s=[float(_v) for _v, _k in zip(_t["ytm"], _keep) if _k],
                  weights=[1 / _v for _v, _k in zip(_t["y_noise_bp"], _keep) if _k],
                  id="realfit")
        _t["curve_ytm"] = [float(_o.rate(**_rpr)) for _o in _twin]
        # Inside a year a TIPS real yield is set by the CPI prints already banked,
        # not by the real curve — today's are deflationary, which lifts short real
        # yields ~150bp above the curve. A log-linear curve cannot bend into that,
        # so the residual there measures the model, not the bond. Left blank.
        _t["edge_bp"] = np.where(_keep,
                                 (_t["ytm"] - _t["curve_ytm"]) * 100, np.nan)

        def _par(_curve, _a, _b):
            """Par yield between two dates off a curve: iterate the coupon until
            the bond prices at par to itself. Converges to 1e-10 in two passes."""
            _y = 4.0
            for _ in range(6):
                _bd = rl.FixedRateBond(effective=_a, termination=_b, spec="us_gb",
                                       fixed_rate=_y)
                _n = float(_bd.rate(curves=_curve, metric="ytm", settlement=_a))
                if abs(_n - _y) < 1e-10:
                    return _n
                _y = _n
            return _y

        # Breakeven per bond: the nominal par yield at the SAME maturity, off the
        # coupon curve fitted upstream, less this bond's real yield.
        _t["nom_ytm"] = [_par(ust_curve, _settle, _m.to_pydatetime())
                         for _m in _t["maturity_date"]]
        _t["breakeven"] = _t["nom_ytm"] - _t["ytm"]

        # Headline breakevens between par yields at matched tenors. Curve.rate()
        # would be wrong here — it returns a SIMPLE rate over the window, which at
        # thirty years reads 12% instead of 5%.
        _tn = lambda _y: rl.add_tenor(_settle, f"{_y}y", "MF", "nyc")

        def _be(_a, _b):
            _da, _db = _tn(_a), _tn(_b)
            return _par(ust_curve, _da, _db) - _par(_real, _da, _db)

        _be10 = _be(0, 10)

        # The five, and the strippable TIPS that mature alongside them — the only
        # honest comparison, since edge drifts with tenor.
        five_tips = _t[_t["cusip"].isin(set(tips["cusip"]))].copy()
        tips_desk = {"count": int(len(_t)), "no_strip_bid": int(len(five_tips)),
                     "ten_year_breakeven": float(_be10), "cpi_note": _cpi_note}
        _peer = _t[~_t["cusip"].isin(set(tips["cusip"]))
                   & _t["days"].between(five_tips["days"].min(), five_tips["days"].max())]
        _gap = five_tips["edge_bp"].median() - _peer["edge_bp"].median()

        # ── Presentation ─────────────────────────────────────────────────────
        _GREEN, _SIENNA, _GOLD = "#1D9E75", "#D8683C", "#C9A961"
        _SLATE, _RULE = "#93A49B", "#26332C"
        _sig = float(np.nanstd(_t["edge_bp"]))

        def _tn3(_v):
            if pd.isna(_v) or abs(_v) < _sig:
                return _SLATE
            return _GREEN if _v > 0 else _SIENNA

        # ── Signature: the inflation wedge ───────────────────────────────────
        # Two par curves off the same desk — nominal in gold, real in green. The
        # band between them is the whole trade: what a buyer of nominal Treasuries
        # is paid for taking inflation risk, tenor by tenor.
        _W, _CH, _PADL, _PADR, _CT, _CB = 1000, 340, 64, 24, 26, 44
        # Draw all the way to the longest bond rather than to a round tenor — the
        # thirty-year is 29.6 years old-money and stopping at 25 would crop it.
        _maxt = _t["days"].max() / 365.25
        _grid_y = [_y for _y in [1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 25]
                   if _y < _maxt] + [_maxt]
        _nom = [_par(ust_curve, _settle, _tn(_y)) for _y in _grid_y]
        _rl_ = [_par(_real, _settle, _tn(_y)) for _y in _grid_y]
        _lo = min(min(_rl_), _t["ytm"].min()) - 0.15
        _hi = max(max(_nom), _t["ytm"].max()) + 0.15
        # Square-root tenor: linear-in-years buries the 1-10y sector, which is
        # where the TIPS curve actually has shape and where the auctions are.
        _xs = lambda _y: _PADL + (np.sqrt(max(_y, 0)) / np.sqrt(max(_grid_y))) * (_W - _PADL - _PADR)
        _ys = lambda _v: _CT + (_hi - _v) / (_hi - _lo) * (_CH - _CT - _CB)

        _rules, _tick = [], np.ceil(_lo * 2) / 2
        while _tick <= _hi:
            _gy = _ys(_tick)
            _rules.append(
                f'<line x1="{_PADL}" y1="{_gy:.1f}" x2="{_W - _PADR}" y2="{_gy:.1f}" '
                f'stroke="{_RULE}" stroke-width="1"/>'
                f'<text x="{_PADL - 10}" y="{_gy + 3:.1f}" fill="{_SLATE}" '
                f'text-anchor="end" font-family="DM Mono, monospace" font-size="10">'
                f'{_tick:.1f}</text>')
            _tick += 0.5

        _np = " ".join(f"{_xs(_y):.1f},{_ys(_v):.1f}" for _y, _v in zip(_grid_y, _nom))
        _rp = " ".join(f"{_xs(_y):.1f},{_ys(_v):.1f}" for _y, _v in zip(_grid_y, _rl_))
        _wedge = (f'<polygon points="{_np} '
                  + " ".join(f"{_xs(_y):.1f},{_ys(_v):.1f}"
                             for _y, _v in zip(reversed(_grid_y), reversed(_rl_)))
                  + f'" fill="{_GREEN}" opacity="0.10"/>')
        # The other 47 sit back as faint slate — they built the curve, they are not
        # the subject. The five carry the colour and the ring.
        _ctx = "".join(
            f'<circle cx="{_xs(_r["days"] / 365.25):.1f}" cy="{_ys(_r["ytm"]):.1f}" '
            f'r="2.4" fill="{_SLATE}" opacity="0.4"><title>{_r["cusip"]} · '
            f'{_r["maturity_date"]:%b %Y} · real {_r["ytm"]:.3f}%</title></circle>'
            for _, _r in _t.iterrows() if _r["cusip"] not in set(tips["cusip"]))
        _dots = "".join(
            f'<circle cx="{_xs(_r["days"] / 365.25):.1f}" cy="{_ys(_r["ytm"]):.1f}" '
            f'r="5" fill="none" stroke="{_tn3(_r["edge_bp"])}" stroke-width="1.4" '
            f'opacity="0.5"/>'
            f'<circle cx="{_xs(_r["days"] / 365.25):.1f}" cy="{_ys(_r["ytm"]):.1f}" '
            f'r="3.4" fill="{_tn3(_r["edge_bp"])}" stroke="#0C110F" stroke-width="1">'
            f'<title>{_r["cusip"]} · {_r["rate"]:.3f}% · {_r["maturity_date"]:%b %Y} · '
            f'real {_r["ytm"]:.3f}% · breakeven {_r["breakeven"]:.3f}%</title></circle>'
            for _, _r in five_tips.iterrows())
        _ticks = [(_y, f"{_y:g}y") for _y in (1, 2, 5, 10, 20) if _y < _maxt]
        _ticks.append((_maxt, f"{_maxt:.0f}y"))
        _axis = "".join(
            f'<text x="{_xs(_y):.1f}" y="{_CH - 14}" fill="{_SLATE}" '
            f'text-anchor="middle" font-family="DM Mono, monospace" font-size="10">'
            f'{_lab}</text>' for _y, _lab in _ticks)
        # Caliper on the ten-year, where the breakeven is quoted.
        _cx = _xs(10)
        _caliper = (
            f'<line x1="{_cx:.1f}" y1="{_ys(_par(ust_curve, _settle, _tn(10))):.1f}" '
            f'x2="{_cx:.1f}" y2="{_ys(_par(_real, _settle, _tn(10))):.1f}" '
            f'stroke="{_GREEN}" stroke-width="1.4" stroke-dasharray="3 3"/>'
            f'<text x="{_cx + 8:.1f}" y="{(_ys(_par(ust_curve, _settle, _tn(10))) + _ys(_par(_real, _settle, _tn(10)))) / 2 + 4:.1f}" '
            f'fill="{_GREEN}" font-family="DM Mono, monospace" font-size="11">'
            f'{_be10:.2f}% breakeven</text>')
        _chart = (
            f'<svg viewBox="0 0 {_W} {_CH}" width="100%" role="img" '
            f'aria-label="Nominal and real par curves, the gap being breakeven inflation">'
            + "".join(_rules) + _wedge
            + f'<polyline points="{_np}" fill="none" stroke="{_GOLD}" '
              f'stroke-width="1.7"/>'
            + f'<polyline points="{_rp}" fill="none" stroke="{_GREEN}" '
              f'stroke-width="1.7"/>' + _caliper + _ctx + _dots + _axis + '</svg>')

        # The locked window: accretion between here and the last date whose
        # reference CPI is already published. Identical for every TIPS — the index
        # base cancels — so it is a market fact, not a security-selection one.
        if _has_cpi:
            _lock = float(_reference_cpi(_known) / _ref_settle - 1)
            _ld = (_known - _settle).days
            _ann = ((1 + _lock) ** (365 / _ld) - 1) * 100
            _lockline = (
                f'<div class="fi-slope">Inflation accrual already fixed through '
                f'<b>{_known:%d %b}</b> — {_ld} days —'
                f'<b style="color:{_GREEN if _lock > 0 else _SIENNA}"> '
                f'{_lock * 1e4:+.1f} bp</b> of current indexed principal, '
                f'<b>{_ann:+.2f}%</b> annualised · every TIPS earns it equally</div>')
        else:
            _lockline = (f'<div class="fi-slope" style="color:{_SIENNA}">CPI series '
                         f'unusable — index ratios, invoice prices and accrual '
                         f'omitted; real yields and breakevens are unaffected'
                         f'{" · " + _cpi_note if _cpi_note else ""}</div>')
        # A hole outside the live window is survivable but worth saying out loud —
        # it will become live for anyone re-running this over an older settlement.
        if _hole and _has_cpi:
            _lockline += (
                '<div class="fi-note" style="margin:-12px 0 22px">No CPI print for '
                + ", ".join(f"{_m:%B %Y} ({_v.rstrip('.').lower()})"
                            for _m, _v in sorted(_hole.items()))
                + ' — outside the three-month window these ratios read, so today is '
                  'unaffected; reference CPI for the two months after each hole is '
                  'not recoverable from this series.</div>')

        def _num(_v, _fmt):
            return "—" if pd.isna(_v) else format(_v, _fmt)

        _rows = "".join(
            f'<tr><td class="fi-cusip">{_r["cusip"]}</td>'
            f'<td class="fi-r">{_r["rate"]:.3f}</td>'
            f'<td>{_r["maturity_date"]:%b %Y}</td>'
            f'<td class="fi-r">{_r["mid"]:.3f}</td>'
            f'<td class="fi-r">{_r["ytm"]:.3f}</td>'
            f'<td class="fi-r">{_num(_r["breakeven"], ".3f")}</td>'
            f'<td class="fi-r" style="color:{_tn3(_r["edge_bp"])}">'
            f'{_num(_r["edge_bp"], "+.1f")}</td>'
            f'<td class="fi-r" style="color:{_GOLD}">{_num(_r["ratio"], ".4f")}</td>'
            f'<td class="fi-r">{_num(_r["invoice"], ".2f")}</td>'
            f'<td class="fi-r">{_num(_r["adj_dv01"], ".3f")}</td>'
            f'<td class="fi-r fi-dim">{_num(_r["spread_bp"], ".1f")}</td></tr>'
            for _, _r in five_tips.iterrows())

        # Every index figure is NaN when the CPI fetch failed; say "—" rather than
        # printing "nan×" into a stat tile.
        _rmed = five_tips["ratio"].median()
        _rtile = "—" if pd.isna(_rmed) else f"{_rmed:.2f}"
        _runiv = ("" if pd.isna(_rmed)
                  else f'principal, versus {_t["ratio"].median():.2f}× typical')
        _accreted = ("" if pd.isna(_rmed) else
                     f' and principal up {(five_tips["ratio"].min() - 1) * 100:.0f}'
                     f'–{(five_tips["ratio"].max() - 1) * 100:.0f}%, so you settle'
                     f' the invoice, not the quote, and run the DV01 shown rather'
                     f' than the smaller real one')
        _one_sided = sorted(five_tips.loc[~five_tips["two_sided"], "cusip"])
        _excl = (f' <span style="color:{_SIENNA}">{", ".join(_one_sided)}</span> is '
                 f'quoted on one side only and stays off the fit.' if _one_sided else "")

        _show(FI_CSS + f"""
        <div class="fi">
          <div class="fi-eyebrow">No strip bid · issued 1998–2008</div>
          <h2 class="fi-title">The five unstripped TIPS</h2>
          <div class="fi-deck">Five inflation-linked bonds with no stripping
            activity — the other {len(_t) - len(five_tips)} carry a corpus CUSIP,
            these do not. Priced against a real curve fitted to all {len(_t)} TIPS,
            since five scattered bonds cannot define their own fair value. Colour
            is reserved for moves beyond {_sig:.1f}bp.{_excl}</div>
          <div class="fi-stats">
            <div class="fi-stat"><div class="fi-lbl">Median edge</div>
              <div class="fi-num" style="color:{_tn3(_gap)}">{_gap:+.1f}<em>bp</em></div>
              <div class="fi-sub">vs tenor-matched peers</div></div>
            <div class="fi-stat"><div class="fi-lbl">Real coupons</div>
              <div class="fi-num">{five_tips["rate"].min():.2f}<span>–{five_tips["rate"].max():.2f}</span><em>%</em></div>
              <div class="fi-sub">universe median {_t["rate"].median():.2f}%</div></div>
            <div class="fi-stat"><div class="fi-lbl">Index ratio</div>
              <div class="fi-num" style="color:{_GOLD}">{_rtile}<em>×</em></div>
              <div class="fi-sub">{_runiv}</div></div>
            <div class="fi-stat"><div class="fi-lbl">Bid-ask</div>
              <div class="fi-num">{five_tips["spread_bp"].median():.1f}<em>bp</em></div>
              <div class="fi-sub">peers trade at {_peer["spread_bp"].median():.1f}</div></div>
          </div>
          <div class="fi-slope">Median residual difference versus peers:
            <b style="color:{_tn3(_gap)}">{_gap:+.1f} real-yield bp</b>.
            Quoted spreads are price bp; this comparison does not establish
            an executable edge or a causal stripping premium.</div>
          {_lockline}
          <div class="fi-panel">{_chart}
            <div class="fi-cap"><span class="fi-k fi-k-line" style="border-color:{_GOLD}"></span>Nominal par
              <span class="fi-k fi-k-line"></span>Real par
              <span class="fi-k fi-k-dot"></span>The five
              <span class="fi-k fi-k-dot" style="background:{_SLATE};opacity:0.4"></span>Rest of the universe ·
              shaded band is breakeven, {_be10:.2f}% at ten years</div></div>
          <div class="fi-scroll"><table class="fi-tbl"><thead><tr><th>CUSIP</th><th class="fi-r">Coupon</th>
            <th>Matures</th><th class="fi-r">Real px</th>
            <th class="fi-r">Real yld</th><th class="fi-r">B/E</th>
            <th class="fi-r">Edge</th><th class="fi-r">Ratio</th>
            <th class="fi-r">Invoice</th><th class="fi-r">DV01</th>
            <th class="fi-r">Price bp</th></tr></thead>
            <tbody>{_rows}</tbody></table></div>
          <div class="fi-note" style="margin-top:14px">Edge is real yield against
            the fitted curve — positive is cheap. These are the first-generation
            TIPS: real coupons no auction has paid since{_accreted}. Anything
            maturing inside a year is left blank — its real yield is set by CPI
            already banked, which no log-linear curve reproduces.</div>
        </div>""")
        return (tips_desk,)

    tips_desk, = _run_panel()
    return tips_desk,



@app.cell
def _(FI_CSS, dt, mo, np, pd, prices, rl, ust_curve):
    # Treasury FRNs accrue daily at weekly auction rates, with coupon
    # lockouts and a zero floor. Today's coupon cannot recover past accrual.
    # A quarterly IBOR floater is not a valid substitute for this contract.
    _settle = rl.add_tenor(prices["price_date"].iloc[0].to_pydatetime(),
                           "1b", "F", "nyc")
    _f = prices[prices["security_type"] == "MARKET BASED FRN"].copy()
    _f["days"] = (_f["maturity_date"] - pd.Timestamp(_settle)).dt.days
    _f = _f[_f["days"] > 0].sort_values("maturity_date")
    _f["yrs"] = _f["days"] / 365.25
    _f["two_sided"] = ((_f["buy"] > 0) & (_f["sell"] > 0)
                       & (_f["buy"] >= _f["sell"]) & np.isfinite(_f["mid"]))
    _f["spread_bp"] = np.where(_f["two_sided"],
                               (_f["buy"] - _f["sell"]) / _f["mid"] * 1e4, np.nan)
    for _col in ("index", "accrued", "dm_bp", "fit_bp", "edge_bp"):
        _f[_col] = np.nan
    _f["valuation_status"] = "Unavailable: weekly auction fixings required"
    frn_desk = _f
    frn_stats = {"count": int(len(_f)),
                 "priced": int(_f["dm_bp"].notna().sum()),
                 "status": str(_f["valuation_status"].iloc[0]) if len(_f) else ""}

    def _display(_value, _fmt):
        return "N/A" if pd.isna(_value) else format(_value, _fmt)

    _rows = "".join(
        f'<tr><td class="fi-cusip">{_r["cusip"]}</td>'
        f'<td>{_r["maturity_date"]:%d %b %Y}</td>'
        f'<td class="fi-r">{_display(_r["rate"], ".3f")}</td>'
        f'<td class="fi-r">{_display(_r["frn_spread"] * 100, ".1f")}</td>'
        f'<td class="fi-r">{_display(_r["mid"], ".3f")}</td>'
        f'<td class="fi-r">{_display(_r["spread_bp"], ".2f")}</td></tr>'
        for _, _r in _f.iterrows())
    mo.Html(FI_CSS + f"""
    <div class="fi">
      <div class="fi-eyebrow">13-week bill index / weekly reset / settle {_settle:%d %b %Y}</div>
      <h2 class="fi-title">The floating-rate notes</h2>
      <div class="fi-deck">{len(_f)} outstanding FRNs. Coupon rates and contractual
        spreads below are reported by Treasury; mid prices may be one-sided.</div>
      <div class="fi-slope">Accrued interest, discount margins and relative-value
        rankings are unavailable. They require historical weekly auction rates,
        daily accrual, coupon lockouts and the zero coupon floor. The current
        coupon alone does not supply this history.</div>
      <div class="fi-scroll"><table class="fi-tbl"><thead><tr>
        <th>CUSIP</th><th>Matures</th><th class="fi-r">Coupon %</th>
        <th class="fi-r">Contract spread bp</th><th class="fi-r">Price</th>
        <th class="fi-r">Price spread bp</th></tr></thead><tbody>{_rows}</tbody></table></div>
    </div>""")
    return frn_desk, frn_stats


@app.cell
def _(FI_CSS, mo, np, pd, prices, rl, ust_curve):
    # ── STRIPS: where the zero-coupon bid actually lives ─────────────────
    # All 384 strippable securities, carrying MSPD Table 5's stripped and
    # unstripped face. Every bond metric below comes from rateslib off the
    # coupon curve already fitted upstream.
    _settle = rl.add_tenor(prices["price_date"].iloc[0].to_pydatetime(),
                           "1b", "F", "nyc")
    _s = prices[prices["strippable"].eq("Yes")].copy()
    _s["days"] = (_s["maturity_date"] - pd.Timestamp(_settle)).dt.days
    _s["yrs"] = _s["days"] / 365.25
    _s["pct"] = _s["stripped"] / (_s["stripped"] + _s["unstripped"])

    # Edge and corpus share need a fitted nominal curve, so they are defined
    # for the nominal coupon securities with a real quote and more than 90
    # days to run — the same population ust_curve was fitted to. TIPS are
    # left out of the valuation on purpose: they price off a real curve, and
    # as the numbers below show, not one of them is stripped anyway.
    _nom = _s["security_type"].isin(["MARKET BASED NOTE", "MARKET BASED BOND"])
    _live = (_nom & (ust_curve is not None) & (_s["buy"] > 0) & (_s["sell"] > 0)
             & (_s["buy"] >= _s["sell"]) & np.isfinite(_s["mid"])
             & (_s["mid"] > 0) & (_s["days"] > 90) & _s["rate"].notna())

    def _mkb(_r):
        _eff = (_r["dated_date"] if pd.notna(_r["dated_date"])
                else _r["maturity_date"] - pd.DateOffset(years=100))
        return rl.FixedRateBond(
            effective=pd.Timestamp(_eff).to_pydatetime(),
            termination=_r["maturity_date"].to_pydatetime(),
            spec="us_gb", fixed_rate=float(_r["rate"]))

    _pr = {"curves": ust_curve, "metric": "ytm", "settlement": _settle}
    _ytm, _edge, _dur, _dv01, _corpus = {}, {}, {}, {}, {}
    for _i, _r in _s[_live].iterrows():
        _b = _mkb(_r)
        _y = float(_b.ytm(price=float(_r["mid"]), settlement=_settle))
        _ytm[_i] = _y
        _edge[_i] = (_y - float(_b.rate(**_pr))) * 100
        _dur[_i] = float(_b.duration(ytm=_y, settlement=_settle, metric="modified"))
        _dv01[_i] = float(_b.duration(ytm=_y, settlement=_settle, metric="risk")) / 100
        # Corpus share: the redemption's present value over the whole bond's,
        # both straight out of rateslib's own cashflow table. This is what
        # decides WHAT stripping manufactures — at the long end the principal
        # is a quarter of the value and the other three quarters are coupons,
        # so stripping a 30-year mostly mints interest zeros, not principal.
        _cf = _b.cashflows(curves=ust_curve)
        _corpus[_i] = float(_cf.loc[_cf["Type"] == "Cashflow", "NPV"].sum()) \
            / float(_cf["NPV"].sum())
    for _c, _d in (("ytm", _ytm), ("edge_bp", _edge), ("mod_dur", _dur),
                   ("dv01", _dv01), ("corpus_share", _corpus)):
        _s[_c] = pd.Series(_d)

    _miss = sorted(_s.loc[_s["stripped"].isna(), "cusip"])
    _tot = _s["stripped"].sum(min_count=1)
    _recon = _s["reconstituted"].sum(min_count=1)
    _long = _s.loc[_s["yrs"] > 20, "stripped"].sum()
    _tips_n = int((_s["security_type"].eq("TIPS") & _s["stripped"].gt(0)).sum())

    # ── Does the strip bid make a bond rich? ─────────────────────────────
    # Stripping is a long-bond activity, and so is cheapness, so the raw
    # correlation would just be tenor twice. Take a quadratic in years out of
    # both sides first; whatever survives is the part tenor cannot explain.
    _lg = _s[(_s["yrs"] > 20) & _s["pct"].notna() & _s["edge_bp"].notna()].copy()
    _pr_, _er = _lg["pct"].values, _lg["edge_bp"].values
    _yv = _lg["yrs"].values
    _rho = np.nan
    # A quadratic uses three parameters. Fewer than five observations leave
    # at most one residual dimension and manufacture a +/-1 correlation.
    if len(_lg) >= 5 and np.unique(_yv).size >= 3:
        _x = (_yv - _yv.mean()) / _yv.std()
        _presid = _pr_ - np.polyval(np.polyfit(_x, _pr_, 2), _x)
        _eresid = _er - np.polyval(np.polyfit(_x, _er, 2), _x)
        if np.std(_presid) > 1e-12 and np.std(_eresid) > 1e-12:
            _rho = float(np.corrcoef(_presid, _eresid)[0, 1])
    strips_stats = {"stripped_usd": float(_tot), "reconstituted_usd": float(_recon),
                    "stripped_corr": _rho}
    _qt = pd.DataFrame(columns=["q", "n", "pct", "edge", "yrs"])
    _gap = np.nan
    if len(_lg) >= 4 and _lg["pct"].nunique() >= 4:
        # Do not split equal stripping shares arbitrarily to force four bins.
        _bins = pd.qcut(_lg["pct"], 4, labels=False, duplicates="drop")
        if _bins.nunique() == 4:
            _lg["q"] = _bins.map({0: "Least", 1: "Q2", 2: "Q3", 3: "Most"})
            _qt = _lg.groupby("q", sort=False).agg(
                n=("cusip", "size"), pct=("pct", "median"), edge=("edge_bp", "median"),
                yrs=("yrs", "median")).reindex(["Least", "Q2", "Q3", "Most"]).reset_index()
            _gap = float(_qt["edge"].iloc[-1] - _qt["edge"].iloc[0])
    strips_stats["quartile_gap_bp"] = _gap

    # ── Presentation ─────────────────────────────────────────────────────
    _GREEN, _SIENNA, _GOLD = "#1D9E75", "#D8683C", "#C9A961"
    _SLATE, _RULE = "#93A49B", "#26332C"

    # Signature: stripped face by maturity year, with corpus share riding
    # over it. The bars say the zero bid lives past twenty years; the line
    # says that is exactly where the principal stops being the point.
    _yr = _s.dropna(subset=["stripped"]).copy()
    _yr["y"] = _yr["maturity_date"].dt.year
    _agg = _yr.groupby("y").agg(bn=("stripped", lambda _v: _v.sum() / 1e9),
                                cs=("corpus_share", lambda _v: _v.dropna().median()
                                    if _v.notna().any() else np.nan)).reset_index()
    _W, _CH, _PADL, _PADR, _CT, _CB = 1000, 320, 58, 46, 24, 42
    _y0 = int(_agg["y"].min()) if len(_agg) else _settle.year
    _y1 = int(_agg["y"].max()) if len(_agg) else _y0 + 1
    _bmax = max(float(_agg["bn"].max()), 1e-9) if len(_agg) else 1.
    _xs = lambda _y: _PADL + (_y - _y0) / max(_y1 - _y0, 1) * (_W - _PADL - _PADR)
    _ys = lambda _v: _CT + (1 - _v / _bmax) * (_CH - _CT - _CB)
    _cy = lambda _v: _CT + (1 - _v) * (_CH - _CT - _CB)
    _bw = max((_W - _PADL - _PADR) / max(_y1 - _y0, 1) - 1.2, 1.6)

    _bars = "".join(
        f'<rect x="{_xs(_r["y"]) - _bw / 2:.1f}" y="{_ys(_r["bn"]):.1f}" '
        f'width="{_bw:.1f}" height="{max(_CH - _CB - _ys(_r["bn"]), 0.4):.1f}" '
        f'fill="{_GREEN}" opacity="{0.35 + 0.55 * _r["bn"] / _bmax:.2f}">'
        f'<title>{int(_r["y"])} · ${_r["bn"]:.1f}bn stripped</title></rect>'
        for _, _r in _agg.iterrows())
    _cline = " ".join(f"{_xs(_r['y']):.1f},{_cy(_r['cs']):.1f}"
                      for _, _r in _agg.dropna(subset=["cs"]).iterrows())
    _ticks = "".join(
        f'<text x="{_xs(_y):.1f}" y="{_CH - 14}" fill="{_SLATE}" '
        f'text-anchor="middle" font-family="DM Mono, monospace" font-size="10">'
        f'{_y}</text>' for _y in range(_y0 - _y0 % 5, _y1 + 1, 5) if _y >= _y0)
    _chart = (
        f'<svg viewBox="0 0 {_W} {_CH}" width="100%" role="img" '
        f'aria-label="Face held in stripped form by maturity year, with corpus share">'
        f'<line x1="{_PADL}" y1="{_CH - _CB}" x2="{_W - _PADR}" y2="{_CH - _CB}" '
        f'stroke="{_RULE}" stroke-width="1"/>' + _bars
        + f'<polyline points="{_cline}" fill="none" stroke="{_GOLD}" '
          f'stroke-width="1.6" stroke-dasharray="4 3" opacity="0.9"/>'
        + f'<text x="4" y="{_CT + 4}" fill="{_GREEN}" '
          f'font-family="DM Mono, monospace" font-size="10">${_bmax:.0f}bn</text>'
        + f'<text x="{_W - _PADR + 6}" y="{_cy(1.0) + 4:.1f}" fill="{_GOLD}" '
          f'font-family="DM Mono, monospace" font-size="10">100%</text>'
        + f'<text x="{_W - _PADR + 6}" y="{_cy(0.0) + 4:.1f}" fill="{_GOLD}" '
          f'font-family="DM Mono, monospace" font-size="10">0%</text>'
        + _ticks + '</svg>')

    _qrows = "".join(
        f'<tr><td class="fi-cusip">{_r["q"]}</td>'
        f'<td class="fi-r">{_r["n"]}</td>'
        f'<td class="fi-r">{_r["pct"] * 100:.1f}%</td>'
        f'<td class="fi-r">{_r["yrs"]:.1f}y</td>'
        f'<td class="fi-r" style="color:{_GREEN if _r["edge"] > 0 else _SIENNA}">'
        f'{_r["edge"]:+.2f}</td></tr>' for _, _r in _qt.iterrows())

    _top = _s.nlargest(10, "stripped")
    def _fmt(_v, _spec):
        return format(_v, _spec) if pd.notna(_v) and np.isfinite(_v) else "N/A"

    _rows = "".join(
        f'<tr><td class="fi-cusip">{_r["cusip"]}</td>'
        f'<td class="fi-r">{_r["rate"]:.3f}</td>'
        f'<td>{_r["maturity_date"]:%b %Y}</td>'
        f'<td class="fi-r">${_fmt(_r["stripped"] / 1e9, ".1f")}</td>'
        f'<td class="fi-r" style="color:{_GOLD}">{_fmt(_r["pct"] * 100, ".1f")}%</td>'
        f'<td class="fi-r fi-dim">{_fmt(_r["corpus_share"] * 100, ".0f")}%</td>'
        f'<td class="fi-r">{_fmt(_r["mod_dur"], ".1f")}</td>'
        f'<td class="fi-r" style="color:{_GREEN if _r["edge_bp"] > 0 else _SIENNA}">'
        f'{_fmt(_r["edge_bp"], "+.2f")}</td>'
        f'<td class="fi-r fi-dim">${_fmt(_r["reconstituted"] / 1e6, ",.0f")}m</td></tr>'
        for _, _r in _top.iterrows())

    _note = (f' <span style="color:{_SIENNA}">{", ".join(_miss)}</span> has no '
             f'matched Table 5 balance; missing data is not zero stripping.'
             if _miss else "")
    strips_desk = _s

    mo.Html(FI_CSS + f"""
    <div class="fi">
      <div class="fi-eyebrow">MSPD Table 5 · {len(_s)} strippable securities</div>
      <h2 class="fi-title">Where the zero bid lives</h2>
      <div class="fi-deck">Stripping turns one coupon bond into a principal
        zero and one zero per coupon. Almost none of it happens inside twenty
        years: the demand is for long duration that pays nothing until it
        pays everything.{_note}</div>
      <div class="fi-stats">
        <div class="fi-stat"><div class="fi-lbl">Held stripped</div>
          <div class="fi-num">{_fmt(_tot / 1e9, ",.0f")}<em>$bn</em></div>
          <div class="fi-sub">of face outstanding</div></div>
        <div class="fi-stat"><div class="fi-lbl">Past 20 years</div>
          <div class="fi-num" style="color:{_GREEN}">{_fmt(_long / _tot * 100 if _tot > 0 else np.nan, ".0f")}<em>%</em></div>
          <div class="fi-sub">of all stripped face</div></div>
        <div class="fi-stat"><div class="fi-lbl">Reconstituted</div>
          <div class="fi-num">{_fmt(_recon / 1e9, ",.1f")}<em>$bn</em></div>
          <div class="fi-sub">put back together, one month</div></div>
        <div class="fi-stat"><div class="fi-lbl">TIPS stripped</div>
          <div class="fi-num" style="color:{_SLATE}">{_tips_n}<span>/{int(_s["security_type"].eq("TIPS").sum())}</span></div>
          <div class="fi-sub">with positive reported stripped balance</div></div>
      </div>
      <div class="fi-panel">{_chart}
        <div class="fi-cap"><span class="fi-k fi-k-dot"></span>Face held stripped, by maturity year
          <span class="fi-k fi-k-dash" style="border-color:{_GOLD}"></span>Principal's share of value ·
          at thirty years the corpus is a quarter of the bond, so stripping mints coupon zeros</div></div>
      <div class="fi-grid">
        <div><div class="fi-h">Stripping and relative value</div>
          <div class="fi-note">Bonds past twenty years, sorted into quartiles by
            how much of them is held stripped. Positive edge is cheap. The most
            heavily stripped quartile has a <b>{_fmt(_gap, "+.1f")} yield-bp</b> residual
            difference versus the least. Correlation after removing a quadratic
            in tenor is <b>{_fmt(_rho, "+.2f")}</b>. N/A indicates insufficient
            observations or distinct values. This cross-sectional association does
            not establish causality or an executable trade; liquidity, coupon
            and issue effects remain uncontrolled.</div>
          <div class="fi-scroll"><table class="fi-tbl"><thead><tr><th>Quartile</th>
            <th class="fi-r">n</th><th class="fi-r">Stripped</th>
            <th class="fi-r">Tenor</th><th class="fi-r">Edge</th></tr></thead>
            <tbody>{_qrows}</tbody></table></div></div>
        <div><div class="fi-h">What stripping manufactures</div>
          <div class="fi-note">A two-year note is almost all principal, so
            stripping it produces one zero worth what the note was worth —
            pointless. A thirty-year is a quarter principal and three quarters
            coupons, so stripping mints sixty interest zeros spread across the
            curve. That asymmetry, not yield, is why the activity sits where it
            does; the gold line traces it.</div>
          <div class="fi-note">Reconstitution runs at
            <b>${_fmt(_recon / 1e9, ".1f")}bn</b> a month against
            <b>${_fmt(_tot / 1e9, ",.0f")}bn</b> outstanding — the pieces move back and
            forth as the relative bid shifts, which is the arbitrage keeping
            the zero curve tied to the coupon curve.</div></div>
      </div>
      <div style="margin-top:26px">
        <div class="fi-h">Most stripped, by face</div>
        <div class="fi-scroll"><table class="fi-tbl"><thead><tr><th>CUSIP</th><th class="fi-r">Coupon</th>
          <th>Matures</th><th class="fi-r">Stripped</th>
          <th class="fi-r">Share</th><th class="fi-r">Corpus</th>
          <th class="fi-r">Dur</th><th class="fi-r">Edge</th>
          <th class="fi-r">Recon</th></tr></thead>
          <tbody>{_rows}</tbody></table></div>
      </div>
    </div>""")
    return strips_desk, strips_stats


if __name__ == "__main__":
    app.run()
