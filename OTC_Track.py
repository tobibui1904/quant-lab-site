import marimo

__generated_with = "0.23.14"
app = marimo.App(
    width="medium",
    css_file="theme.css",
    html_head_file="theme_head.html",
)


@app.cell
def _():
    import os
    import sys
    from datetime import date, timedelta
    from pathlib import Path

    import pandas as pd
    import requests
    from dotenv import load_dotenv

    import marimo as mo

    _nb_dir = Path(__file__).resolve().parent

    # Load .env from the notebook's own directory, not the cwd marimo happens to
    # be launched from. Real environment variables still win over the file.
    _ = load_dotenv(_nb_dir / ".env")

    # Same reasoning for the watchlist module: the hub launches this notebook
    # from its own working directory, so put the notebook's directory on the
    # path rather than relying on cwd.
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    import otc_watchlist as wl

    return Path, date, mo, os, pd, requests, timedelta, wl


@app.cell
def _(os, requests):
    FINRA_TOKEN_URL = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"

    def finra_access_token():
        """OAuth2 client-credentials flow for the FINRA API Platform.

        Returns the token response dict ({"access_token": ..., "expires_in": ...}).
        Tokens are valid for ~30 minutes; call again to refresh.
        """
        client_id = os.environ.get("FINRA_CLIENT_ID")
        client_secret = os.environ.get("FINRA_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError("Set FINRA_CLIENT_ID and FINRA_CLIENT_SECRET in .env")

        resp = requests.post(
            FINRA_TOKEN_URL,
            params={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    return (finra_access_token,)


@app.cell
def _(finra_access_token, mo):
    try:
        finra_token = finra_access_token()["access_token"]
        _err = None
    except Exception as _e:
        finra_token = None
        _err = str(_e)
    # Quiet on success — the hero chip reports feed status.
    mo.md(f"**FINRA auth failed:** {_err}") if _err else None
    return (finra_token,)


@app.cell
def _(Path, mo, wl):
    # Watchlist = the active pair (Pair Trading) plus the selected LETF triple
    # (LETF Backtest), whichever of the two has actually run. A symbol file on
    # its own proves nothing — both writers leave one behind indefinitely — so
    # each source counts only while its ready flag stands. See otc_watchlist.py.
    _root = Path(__file__).resolve().parent
    SYMBOLS, ARMED = wl.resolve_watchlist(_root)

    mo.stop(
        not SYMBOLS,
        mo.Html("""
    <div style="max-width:560px;margin:5rem auto;text-align:center;">
      <div style="display:inline-flex;align-items:center;justify-content:center;width:48px;height:48px;border-radius:50%;border:0.5px solid #C9A961;margin-bottom:1.25rem;">
        <i class="ti ti-lock" style="font-size:22px;color:#C9A961;"></i>
      </div>
      <div style="font-family:'DM Serif Display',serif;font-size:30px;font-style:italic;color:var(--color-text-primary);margin-bottom:10px;">Nothing to track yet</div>
      <p style="font-family:'DM Mono',monospace;font-size:11.5px;line-height:1.9;color:var(--color-text-secondary);letter-spacing:.04em;">
        This desk reports the off-exchange tape for whatever the strategy
        notebooks are working on — it has no watchlist of its own.<br><br>
        Run <b>Pair Trading</b> to the end, or pick a triple in
        <b>LETF Backtest</b>. Either one unlocks this page; run both and the
        tape covers both.
      </p>
    </div>"""),
    )

    FOCUS = SYMBOLS  # dark-pool venue breakdown covers the whole watchlist
    return ARMED, FOCUS, SYMBOLS


@app.cell
def _(ARMED, SYMBOLS, finra_token):
    # Hub assistant run record: the watchlist and whether the FINRA feed is
    # live. Each section below adds its own figures to the same day's record;
    # this cell runs even when every section stops on empty data.
    import agent_diag.record as _agent_record

    _agent_record.record_otc(
        "auth", {}, symbols=SYMBOLS, sources=[s.label for s in ARMED],
        finra_live=finra_token is not None,
    )
    return


@app.cell
def _(ARMED, SYMBOLS, finra_token, mo):
    _pair = " · ".join(SYMBOLS)
    _src = " + ".join(f"{s.label} ({len(s.symbols)})" for s in ARMED)
    _live = finra_token is not None
    _chip_bg = "#E1F5EE" if _live else "#F7E5DC"
    _chip_fg = "#0F6E56" if _live else "#8A3D1F"
    _dot = "#1D9E75" if _live else "#D8683C"
    _label = "FINRA feed live" if _live else "FINRA feed offline"
    mo.Html(f"""
    <div style="padding: 2.5rem 0 2rem; text-align: center;">
      <div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; border-radius: 50%; border: 0.5px solid #1D9E75; margin-bottom: 1rem;">
        <i class="ti ti-eye-off" style="font-size: 22px; color: #1D9E75;"></i>
      </div>
      <h1 style="font-family: 'DM Serif Display', serif; font-size: 36px; font-weight: 400; font-style: italic; margin: 0 0 6px; letter-spacing: -0.01em; color: var(--color-text-primary);">OTC Track</h1>
      <p style="font-family: 'DM Mono', monospace; font-size: 11px; color: var(--color-text-secondary); letter-spacing: 0.18em; text-transform: uppercase; margin: 0 0 0.5rem;">Off-exchange tape · {_pair}</p>
      <p style="font-family: 'DM Mono', monospace; font-size: 10px; color: var(--color-text-tertiary, #93A49B); letter-spacing: 0.14em; text-transform: uppercase; margin: 0 0 1.25rem;">watchlist from {_src}</p>
      <div style="display: inline-flex; align-items: center; gap: 6px; font-family: 'DM Mono', monospace; font-size: 11px; color: {_chip_fg}; background: {_chip_bg}; padding: 4px 14px; border-radius: 999px;">
        <span style="width: 6px; height: 6px; border-radius: 50%; background: {_dot}; display: inline-block;"></span>
        {_label}
      </div>
      <div style="margin-top: 1.5rem; width: 40px; height: 0.5px; background: var(--color-border-tertiary); margin-left: auto; margin-right: auto;"></div>
    </div>
    """)
    return


@app.cell
def _(mo):
    def page_header(title, section_number, subtitle=None):
        sub = ""
        if subtitle:
            sub = (
                "<div style=\"font-family: 'DM Mono', monospace; font-size: 11px; "
                "color: var(--color-text-secondary); letter-spacing: 0.1em; "
                f'margin-top: 8px;">{subtitle}</div>'
            )
        return mo.Html(f"""
    <div style="padding: 2rem 0 1.25rem;">
      <div style="display: flex; align-items: flex-start; gap: 16px;">
        <div style="padding-top: 6px;">
          <div style="width: 2px; height: 52px; background: #1D9E75;"></div>
        </div>
        <div>
          <div style="font-family: 'DM Mono', monospace; font-size: 11px; color: #1D9E75; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 6px;">Section {section_number}</div>
          <div style="font-family: 'DM Serif Display', serif; font-size: 42px; font-style: italic; font-weight: 400; line-height: 1.05; color: var(--color-text-primary);">{title}</div>
          {sub}
        </div>
      </div>
    </div>""")

    def fmt_qty(n, prefix=""):
        for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(n) >= div:
                return f"{prefix}{n / div:.1f}{suffix}"
        return f"{prefix}{n:,.0f}"

    return fmt_qty, page_header


@app.cell
def _(finra_token, mo, pd, requests):
    mo.stop(finra_token is None, mo.md("No FINRA token — fix auth above first."))

    def finra_query(group, name, payload):
        """POST a filtered Query API request, DataFrame back.

        payload supports limit/offset (5000-row cap per response) plus
        compareFilters, domainFilters and dateRangeFilters.
        """
        resp = requests.post(
            f"https://api.finra.org/data/group/{group}/name/{name}",
            headers={
                "Authorization": f"Bearer {finra_token}",
                "Accept": "application/json",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return pd.DataFrame(resp.json())

    return (finra_query,)


@app.cell
def _(ARMED, SYMBOLS, date, finra_query, mo, page_header, timedelta):
    import agent_diag.record as _agent_record  # hub assistant run record

    _today = date.today()
    _raw = finra_query(
        "otcMarket",
        "regShoDaily",
        {
            "limit": 5000,
            "domainFilters": [
                {
                    "fieldName": "securitiesInformationProcessorSymbolIdentifier",
                    "values": SYMBOLS,
                }
            ],
            "dateRangeFilters": [
                {
                    "fieldName": "tradeReportDate",
                    "startDate": str(_today - timedelta(days=10)),
                    "endDate": str(_today),
                }
            ],
        },
    )
    mo.stop(_raw.empty, mo.md("No Reg SHO rows in the window."))

    # One row per reporting facility (TRFs, ADF); sum across them per symbol
    # and day before taking the ratio.
    _tape = (
        _raw.groupby(
            ["tradeReportDate", "securitiesInformationProcessorSymbolIdentifier"]
        )[["shortParQuantity", "totalParQuantity"]]
        .sum()
        .assign(ratio=lambda d: d.shortParQuantity / d.totalParQuantity)
        .reset_index()
        .rename(columns={"securitiesInformationProcessorSymbolIdentifier": "symbol"})
    )

    # Per-symbol latest figures, handed to the hub assistant's run record below.
    _record_rows = []

    _cards = []
    for _sym in SYMBOLS:
        _d = _tape[_tape.symbol == _sym].sort_values("tradeReportDate")
        if _d.empty:
            continue
        _latest_day = _d.tradeReportDate.max()
        _last = _d[_d.tradeReportDate == _latest_day].iloc[-1]
        _record_rows.append({
            "symbol": _sym,
            "short_ratio": round(float(_last.ratio), 4),
            "short_shares": float(_last.shortParQuantity),
            "total_shares": float(_last.totalParQuantity),
        })
        _rows, _prev = [], None
        for _r in _d.itertuples():
            if _prev is None:
                _delta = "<span style='color:#93A49B;'>—</span>"
            else:
                _chg = _r.ratio - _prev
                if abs(_chg) < 0.0005:
                    _delta = "<span style='color:#93A49B;'>· 0.000</span>"
                else:
                    _c = "#D8683C" if _chg > 0 else "#1D9E75"
                    _a = "▲" if _chg > 0 else "▼"
                    _delta = f"<span style='color:{_c};'>{_a} {abs(_chg):.3f}</span>"
            _prev = _r.ratio
            _is_latest = _r.tradeReportDate == _latest_day
            _row_bg = (
                "background:#18221D;border-radius:6px;padding:5px 8px;margin:0 -8px 4px;"
                if _is_latest
                else "padding:5px 0;margin-bottom:4px;"
            )
            _date_c = "#E9EEEB" if _is_latest else "#93A49B"
            _rows.append(f"""
        <div style="display:flex;align-items:center;gap:10px;{_row_bg}">
          <span style="font-family:'DM Mono',monospace;font-size:10px;color:{_date_c};width:40px;flex-shrink:0;">{_r.tradeReportDate[5:]}</span>
          <div style="position:relative;flex:1;height:5px;background:#26332C;border-radius:2px;">
            <div style="width:{max(_r.ratio * 100, 2):.1f}%;height:100%;background:#1D9E75;border-radius:2px;"></div>
            <div style="position:absolute;left:50%;top:-2px;width:1px;height:9px;background:#93A49B;opacity:.5;"></div>
          </div>
          <span style="font-family:'DM Mono',monospace;font-size:11px;color:#E9EEEB;width:44px;text-align:right;flex-shrink:0;">{_r.ratio:.3f}</span>
          <span style="font-family:'DM Mono',monospace;font-size:10px;width:62px;text-align:right;flex-shrink:0;">{_delta}</span>
        </div>""")
        _cards.append(f"""
      <div style="flex:1;min-width:240px;background:#121917;border:1px solid #26332C;border-radius:10px;padding:18px 20px;">
        <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:12px;">
          <span style="font-family:'DM Serif Display',serif;font-style:italic;font-size:22px;color:#E9EEEB;">{_sym}</span>
          <span style="font-family:'DM Mono',monospace;font-size:10px;color:#93A49B;letter-spacing:.14em;text-transform:uppercase;">latest {_d.ratio.iloc[-1]:.3f}</span>
        </div>
        {"".join(_rows)}
      </div>""")

    _view = mo.vstack(
        [
            page_header(
                "Short-Volume Tape",
                "01",
                subtitle=f"Reg SHO daily · {len(_tape.tradeReportDate.unique())} sessions · through {_tape.tradeReportDate.max()}",
            ),
            mo.Html(
                f"""
    <div style="display:flex;gap:14px;flex-wrap:wrap;">{"".join(_cards)}</div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#93A49B;letter-spacing:.06em;margin-top:12px;">
      TRF / off-exchange prints only · tick marks the 0.5 market-maker baseline · read the changes, not the levels
    </div>"""
            ),
        ]
    )

    _agent_record.record_otc(
        "reg_sho",
        {"latest_date": str(_tape.tradeReportDate.max()), "rows": int(len(_raw)),
         "symbols": _record_rows},
        symbols=SYMBOLS, sources=[s.label for s in ARMED], finra_live=True,
    )
    _view
    return


@app.cell
def _(ARMED, FOCUS, date, finra_query, fmt_qty, mo, page_header, timedelta):
    import agent_diag.record as _agent_record  # hub assistant run record

    _today = date.today()
    _ats = finra_query(
        "otcMarket",
        "weeklySummary",
        {
            "limit": 5000,
            "domainFilters": [
                {"fieldName": "issueSymbolIdentifier", "values": FOCUS}
            ],
            "compareFilters": [
                {
                    # Per-symbol-per-ATS rows; without this filter the result
                    # also mixes in market-wide aggregate records.
                    "fieldName": "summaryTypeCode",
                    "fieldValue": "ATS_W_SMBL_FIRM",
                    "compareType": "EQUAL",
                },
            ],
            "dateRangeFilters": [
                {
                    "fieldName": "weekStartDate",
                    "startDate": str(_today - timedelta(weeks=8)),
                    "endDate": str(_today),
                }
            ],
        },
    )
    mo.stop(_ats.empty, mo.md("No ATS rows in the window."))

    # Per-symbol weekly totals, handed to the hub assistant's run record below.
    _record_rows = []

    _cards = []
    for _sym in FOCUS:
        _d = _ats[_ats.issueSymbolIdentifier == _sym]
        if _d.empty:
            continue
        # Each symbol keeps its own newest week — Tier 1 NMS publishes two
        # weeks ahead of Tier 2 / OTCE, so a shared week would drop symbols.
        _week = _d.weekStartDate.max()
        _wk = _d[_d.weekStartDate == _week].sort_values(
            "totalWeeklyShareQuantity", ascending=False
        )
        _total_sh = _wk.totalWeeklyShareQuantity.sum()
        _total_ntl = _wk.totalNotionalSum.fillna(0).sum()
        _max_sh = _wk.totalWeeklyShareQuantity.max()
        _top = _wk.iloc[0]
        _record_rows.append({
            "symbol": _sym, "week": str(_week),
            "total_shares": float(_total_sh), "total_notional": float(_total_ntl),
            "top_venue": str(_top.MPID), "top_venue_shares": float(_top.totalWeeklyShareQuantity),
        })

        _bars = []
        for _r in _wk.head(5).itertuples():
            _name = (_r.marketParticipantName or "").removeprefix(_r.MPID or "").strip()
            _pct = 100 * _r.totalWeeklyShareQuantity / _total_sh
            _w = 100 * _r.totalWeeklyShareQuantity / _max_sh
            _bars.append(f"""
        <div style="margin-bottom:8px;">
          <div style="display:flex;justify-content:space-between;gap:10px;font-family:'DM Mono',monospace;font-size:10px;margin-bottom:3px;">
            <span style="color:#E9EEEB;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{_r.MPID} <span style="color:#93A49B;">· {_name}</span></span>
            <span style="color:#E9EEEB;white-space:nowrap;">{_pct:.1f}%</span>
          </div>
          <div style="height:5px;background:#26332C;border-radius:2px;overflow:hidden;">
            <div style="width:{max(_w, 2):.1f}%;height:100%;background:#1D9E75;border-radius:2px;"></div>
          </div>
        </div>""")
        _more = len(_wk) - 5
        _foot = (
            f"""<div style="font-family:'DM Mono',monospace;font-size:10px;color:#93A49B;margin-top:4px;">+ {_more} more venues</div>"""
            if _more > 0
            else ""
        )
        _cards.append(f"""
      <div style="flex:1;min-width:300px;background:#121917;border:1px solid #26332C;border-radius:10px;padding:16px 18px;">
        <div style="display:flex;justify-content:space-between;align-items:baseline;">
          <span style="font-family:'DM Serif Display',serif;font-style:italic;font-size:20px;color:#E9EEEB;">{_sym}</span>
          <span style="font-family:'DM Mono',monospace;font-size:10px;color:#93A49B;letter-spacing:.12em;text-transform:uppercase;">wk {_week[5:]}</span>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#93A49B;margin:2px 0 12px;">{fmt_qty(_total_sh)} sh · {fmt_qty(_total_ntl, "$")} · {len(_wk)} venues</div>
        {"".join(_bars)}
        {_foot}
      </div>""")

    _view = mo.vstack(
        [
            page_header(
                "Dark-Pool Venues",
                "02",
                subtitle="top ATS per symbol · latest published week (FINRA delays ATS data 2–4 weeks)",
            ),
            mo.Html(
                f'<div style="display:flex;gap:14px;flex-wrap:wrap;">{"".join(_cards)}</div>'
            ),
        ]
    )

    _agent_record.record_otc(
        "ats",
        {"latest_week": str(_ats.weekStartDate.max()), "rows": int(len(_ats)),
         "symbols": _record_rows},
        symbols=FOCUS, sources=[s.label for s in ARMED], finra_live=True,
    )
    _view
    return


@app.cell
def _(ARMED, SYMBOLS, date, finra_query, fmt_qty, mo, page_header, timedelta):
    import agent_diag.record as _agent_record  # hub assistant run record

    _si = finra_query(
        "otcMarket",
        "consolidatedShortInterest",
        {
            "limit": 5000,
            "domainFilters": [{"fieldName": "symbolCode", "values": SYMBOLS}],
            "dateRangeFilters": [
                {
                    "fieldName": "settlementDate",
                    "startDate": str(date.today() - timedelta(days=60)),
                    "endDate": str(date.today()),
                }
            ],
        },
    )
    mo.stop(_si.empty, mo.md("No short-interest rows in the window."))

    _settle = _si.settlementDate.max()
    _latest = _si[_si.settlementDate == _settle].sort_values(
        "currentShortPositionQuantity", ascending=False
    )

    # Per-symbol settlement figures, handed to the hub assistant's run record below.
    _record_rows = [{
        "symbol": str(_row.symbolCode),
        "short_position": float(_row.currentShortPositionQuantity),
        "change_pct": float(_row.changePercent),
        "days_to_cover": float(_row.daysToCoverQuantity),
        "avg_daily_volume": float(_row.averageDailyVolumeQuantity),
    } for _row in _latest.itertuples()]

    _cards = []
    for _r in _latest.itertuples():
        _chg = _r.changePercent
        _c = "#D8683C" if _chg > 0 else "#1D9E75"
        _sign = "+" if _chg > 0 else ""
        _dtc = _r.daysToCoverQuantity
        _cards.append(f"""
      <div style="flex:1;min-width:220px;background:#121917;border:1px solid #26332C;border-radius:10px;padding:20px 22px;">
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#93A49B;letter-spacing:.16em;text-transform:uppercase;margin-bottom:10px;">{_r.symbolCode} · short position</div>
        <div style="display:flex;align-items:baseline;gap:10px;">
          <span style="font-family:'DM Serif Display',serif;font-style:italic;font-size:34px;color:#E9EEEB;">{fmt_qty(_r.currentShortPositionQuantity)}</span>
          <span style="font-family:'DM Mono',monospace;font-size:12px;color:{_c};">{_sign}{_chg:.1f}%</span>
        </div>
        <div style="height:.5px;background:#26332C;margin:14px 0;"></div>
        <div style="display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:11px;color:#93A49B;">
          <span>days to cover</span><span style="color:#E9EEEB;">{_dtc:.2f}</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:11px;color:#93A49B;margin-top:6px;">
          <span>avg daily volume</span><span style="color:#E9EEEB;">{fmt_qty(_r.averageDailyVolumeQuantity)}</span>
        </div>
      </div>""")

    _view = mo.vstack(
        [
            page_header(
                "Short Interest",
                "03",
                subtitle=f"consolidated · settlement {_settle} · biweekly cycle "
                        f"(FINRA publishes ~8 business days after settlement)",
            ),
            mo.Html(
                f'<div style="display:flex;gap:14px;flex-wrap:wrap;">{"".join(_cards)}</div>'
            ),
        ]
    )

    _agent_record.record_otc(
        "short_interest",
        {"settlement_date": str(_settle), "rows": int(len(_si)), "symbols": _record_rows},
        symbols=SYMBOLS, sources=[s.label for s in ARMED], finra_live=True,
    )
    _view
    return


if __name__ == "__main__":
    app.run()
