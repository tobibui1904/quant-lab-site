import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full", css_file="../../theme.css", html_head_file="../../theme_head.html")


@app.cell
def _():
    import warnings
    warnings.filterwarnings("ignore")
    import os
    import pathlib
    from dotenv import load_dotenv
    import marimo as mo
    import datetime
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import yfinance as yf
    from dataclasses import dataclass
    from typing import Dict, List, Tuple, Optional

    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    # Load Alpaca credentials from .env in the Quant root (two levels up)
    # (assign to _ so marimo doesn't render load_dotenv's bool return).
    _ = load_dotenv(mo.notebook_dir().parents[1] / ".env")
    return (
        Dict,
        List,
        MarketOrderRequest,
        Optional,
        OrderSide,
        TimeInForce,
        TradingClient,
        Tuple,
        dataclass,
        datetime,
        go,
        mo,
        np,
        os,
        pathlib,
        pd,
        yf,
    )


@app.cell
def _(pathlib):
    # OTC Track only trusts ETF_triple.txt while this flag stands, so drop it
    # the moment the notebook reloads: until a triple is actually selected in
    # this session, the file on disk is a leftover and vouches for nothing.
    pathlib.Path(".letf_ready").unlink(missing_ok=True)

    # Handed to the cell that raises the flag again. Without that edge the two
    # cells share no dependency, marimo may run them in either order, and the
    # clear can land *after* the raise — leaving OTC Track reading a
    # ".letf_ready" that no longer holds. Same guard as known_pair_trading.py.
    letf_flag_cleared = True
    return (letf_flag_cleared,)


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
      <h1 style="font-family: 'DM Serif Display', serif; font-size: 36px; font-weight: 400; font-style: italic; margin: 0 0 6px; letter-spacing: -0.01em; color: var(--color-text-primary);">Leveraged ETF Strategy</h1>
      <p style="font-family: 'DM Mono', monospace; font-size: 11px; color: var(--color-text-secondary); letter-spacing: 0.18em; text-transform: uppercase; margin: 0 0 1.25rem;">Made by Tobi Bui</p>
      <div style="display: inline-flex; align-items: center; gap: 6px; font-family: 'DM Mono', monospace; font-size: 11px; color: #0F6E56; background: #E1F5EE; padding: 4px 14px; border-radius: 999px;">
        <span style="width: 6px; height: 6px; border-radius: 50%; background: #1D9E75; display: inline-block;"></span>
        Live
      </div>
      <div style="margin-top: 1.5rem; width: 40px; height: 0.5px; background: var(--color-border-tertiary); margin-left: auto; margin-right: auto;"></div>
    </div>
    """)
    return


@app.cell
def _(page_header):
    page_header("Leveraged ETF Selection", "01")
    return


@app.cell
def _(mo, pd):
    etf_triples = pd.DataFrame({
        "Underlying": [
            "QQQ", "SPY", "IWM", "DIA",
            "SOXX", "XLK", "XLF", "XBI",
            "EEM", "FXI", "TLT", "IEF",
            "IYR", "GDX",
        ],
        "Bull_3x": [
            "TQQQ", "SPXL", "TNA", "UDOW",
            "SOXL", "TECL", "FAS", "LABU",
            "EDC", "YINN", "TMF", "TYD",
            "DRN", "NUGT",
        ],
        "Bear_3x": [
            "SQQQ", "SPXS", "TZA", "SDOW",
            "SOXS", "TECS", "FAZ", "LABD",
            "EDZ", "YANG", "TMV", "TYO",
            "DRV", "DUST",
        ],
        "Leverage": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
        "Sector": [
            "Nasdaq 100", "S&P 500", "Russell 2000 Small Cap", "Dow Jones 30",
            "Semiconductors", "Technology", "Financials", "Biotech",
            "Emerging Markets", "China Large Cap", "20yr US Treasuries", "7-10yr US Treasuries",
            "Real Estate", "Gold Miners",
        ],
    })

    mo.ui.dataframe(etf_triples.rename(columns={"Bull_3x": "Bull ETF", "Bear_3x": "Bear ETF"}))
    return (etf_triples,)


@app.cell
def _(etf_triples, mo):
    underlying_select = mo.ui.dropdown(
        options=etf_triples["Underlying"].tolist(),
        label="Select Underlying ETF"
    )

    underlying_select
    return (underlying_select,)


@app.cell
def _(etf_triples, letf_flag_cleared, mo, pathlib, underlying_select):
    mo.stop(not underlying_select.value, mo.md("*Configure filters above and select the ETF please.*"))


    selected_row = etf_triples[etf_triples["Underlying"] == underlying_select.value].iloc[0]

    # Persist the selected triple for other notebooks, pair.txt-style.
    with open("data/ETF_triple.txt", "w") as _f:
        _f.write(
            f"{selected_row['Underlying']} "
            f"{selected_row['Bull_3x']} "
            f"{selected_row['Bear_3x']}\n"
        )

    # Flag last, and only once the write above has landed: this is what
    # unlocks the OTC Track tab and lets it read the triple.
    # `letf_flag_cleared` is an ordering dependency only (see the
    # invalidation cell).
    if letf_flag_cleared:
        pathlib.Path(".letf_ready").touch()

    def ticker_card(label, ticker, accent):
        return mo.Html(f"""
        <div style="
            background: #1a1a1a;
            border: 1px solid {accent}33;
            border-radius: 12px;
            padding: 20px 28px;
            min-width: 140px;
            text-align: center;
            box-shadow: 0 0 0 1px {accent}22, 0 4px 12px rgba(0,0,0,0.4);
        ">
            <div style="
                color: #999999;
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 1.5px;
                text-transform: uppercase;
                margin-bottom: 8px;
            ">{label}</div>
            <div style="
                color: {accent};
                font-size: 28px;
                font-weight: 700;
                font-family: 'SF Mono', Monaco, monospace;
            ">{ticker}</div>
        </div>
        """)

    mo.vstack([
        mo.Html(f"""
        <h2 style="
            color: #00e676;
            font-weight: 700;
            margin-bottom: 4px;
            font-family: -apple-system, sans-serif;
        ">{underlying_select.value} Leveraged Pair</h2>
        <p style="color: #ffffff; opacity: 0.6; margin-top: 0; font-size: 14px;">
            Leveraged bull/bear tickers for the selected underlying
        </p>
        """),
        mo.hstack([
            ticker_card("Underlying", selected_row["Underlying"], "#ffffff"),
            ticker_card(f"Bull {selected_row['Leverage']}x", selected_row["Bull_3x"], "#00e676"),
            ticker_card(f"Bear {selected_row['Leverage']}x", selected_row["Bear_3x"], "#ff5252"),
        ], justify="start", gap=2)
    ])
    return (selected_row,)


@app.cell
def _(mo, selected_row, underlying_select, yf):
    def ticker_card_etf(info: dict) -> mo.Html:
        # Yahoo can 502 / rate-limit — show a clear placeholder instead of crashing.
        if info.get("_error"):
            return mo.Html(f"""
        <div style="background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);border-radius:var(--border-radius-lg);padding:1.25rem;font-size:13px;">
          <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:20px;font-weight:500;">{info.get('symbol','')}</span>
            <span style="font-size:11px;background:#3a1a0d;color:#f78166;padding:2px 8px;border-radius:999px;">quote unavailable</span>
          </div>
          <div style="margin-top:8px;color:var(--color-text-secondary);font-size:12px;">
            Yahoo Finance returned no data ({info['_error']}). This panel is cosmetic —
            the strategy uses Alpaca data and is unaffected. Re-run this cell to retry.
          </div>
        </div>
        """)

        price = info.get("navPrice") or info.get("regularMarketPrice") or info.get("previousClose", 0)
        prev  = info.get("regularMarketPreviousClose") or info.get("previousClose", price)
        chg_pct = (price - prev) / prev * 100 if prev else 0
        chg_color = "var(--color-text-success)" if chg_pct >= 0 else "var(--color-text-danger)"
        chg_icon  = "ti-trending-up" if chg_pct >= 0 else "ti-trending-down"

        aum      = info.get("totalAssets", 0)
        aum_str  = f"${aum/1e9:.2f}B" if aum >= 1e9 else f"${aum/1e6:.0f}M" if aum else "N/A"
        expense  = info.get("annualReportExpenseRatio", 0)
        ytd      = info.get("ytdReturn", 0) or 0
        beta     = info.get("beta3Year") or info.get("beta", 0) or 0
        category = info.get("category") or info.get("fundFamily", "")
        nav      = info.get("navPrice") or info.get("regularMarketPrice", 0)
        spread   = info.get("bid", 0) and info.get("ask", 0) and abs(info.get("ask", 0) - info.get("bid", 0))

        low52  = info.get("fiftyTwoWeekLow", 0)
        high52 = info.get("fiftyTwoWeekHigh", 0)
        pct_from_high = (price - high52) / high52 * 100 if high52 else 0
        pct_from_low  = (price - low52)  / low52  * 100 if low52  else 0

        vol     = info.get("volume", 0) or 0
        avg_vol = info.get("averageVolume", 0) or 0
        vol_ratio = vol / avg_vol if avg_vol else 0

        div_yield = (info.get("yield") or info.get("dividendYield") or 0) * 100

        return mo.Html(f"""
        <div style="background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);border-radius:var(--border-radius-lg);padding:1.25rem;font-size:13px;">

          <!-- Header -->
          <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:1rem;">
            <div>
              <div style="display:flex;align-items:center;gap:10px;">
                <span style="font-size:20px;font-weight:500;">{info.get('symbol','')}</span>
                <span style="color:var(--color-text-secondary);font-size:12px;">{info.get('shortName','')}</span>
                <span style="font-size:11px;background:#E1F5EE;color:#0F6E56;padding:2px 8px;border-radius:999px;">{info.get('fullExchangeName','')}</span>
              </div>
              <div style="margin-top:4px;color:var(--color-text-secondary);font-size:12px;">
                {category}
              </div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:22px;font-weight:500;">${price:.2f}</div>
              <div style="font-size:12px;color:{chg_color};">
                <i class="ti {chg_icon}"></i> {chg_pct:+.2f}% today
              </div>
            </div>
          </div>

          <!-- Key metrics grid -->
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:1rem;">
            {"".join(f'<div style="background:var(--color-background-secondary);border-radius:var(--border-radius-md);padding:10px 12px;"><div style="font-size:11px;color:var(--color-text-secondary);margin-bottom:4px;">{label}</div><div style="font-size:15px;font-weight:500;color:{color};">{value}</div></div>'
            for label, value, color in [
                ("AUM",          aum_str,                                    "var(--color-text-primary)"),
                ("Expense Ratio",f"{expense*100:.2f}%" if expense else "N/A","var(--color-text-primary)"),
                ("YTD Return",   f"{ytd*100:+.2f}%",                        "var(--color-text-success)" if ytd >= 0 else "var(--color-text-danger)"),
                ("Beta (3yr)",   f"{beta:.2f}" if beta else "N/A",          "var(--color-text-primary)"),
            ])}
          </div>

          <!-- Price / volume table -->
          <div style="display:grid;grid-template-columns:1fr 1fr;border-top:0.5px solid var(--color-border-tertiary);padding-top:8px;gap:0;">
            <table style="width:100%;border-collapse:collapse;font-size:12px;">
              {"".join(f'<tr><td style="color:var(--color-text-secondary);padding:4px 8px 4px 0;">{l}</td><td style="text-align:right;font-weight:500;">{v}</td></tr>'
              for l, v in [
                ("NAV",          f"${nav:.2f}"),
                ("52w range",    f"${low52:.2f} – ${high52:.2f}"),
                ("From 52w high",f"{pct_from_high:+.1f}%"),
                ("From 52w low", f"{pct_from_low:+.1f}%"),
                ("50d avg",      f"${info.get('fiftyDayAverage', 0):.2f}"),
                ("200d avg",     f"${info.get('twoHundredDayAverage', 0):.2f}"),
              ])}
            </table>
            <table style="width:100%;border-collapse:collapse;font-size:12px;border-left:0.5px solid var(--color-border-tertiary);">
              {"".join(f'<tr><td style="color:var(--color-text-secondary);padding:4px 8px 4px 1rem;">{l}</td><td style="text-align:right;font-weight:500;">{v}</td></tr>'
              for l, v in [
                ("Volume",       f"{vol/1e6:.2f}M" if vol >= 1e6 else f"{vol:,.0f}"),
                ("Avg volume",   f"{avg_vol/1e6:.2f}M" if avg_vol >= 1e6 else f"{avg_vol:,.0f}"),
                ("Vol ratio",    f"{vol_ratio:.2f}x"),
                ("Bid/Ask spread", f"${spread:.3f}" if spread else "N/A"),
                ("Div yield",    f"{div_yield:.2f}%"),
                ("Fund family",  info.get("fundFamily", "N/A")),
              ])}
            </table>
          </div>

        </div>
        """)


    def _safe_info(symbol: str) -> dict:
        """
        yfinance/Yahoo is flaky: it 502s and rate-limits, and on a bad response
        yfinance itself raises (e.g. TypeError: argument of type 'NoneType' is not
        iterable). Never let that take down the cell — return a marker dict so the
        card renders an "unavailable" state instead.
        """
        try:
            info = yf.Ticker(symbol).info or {}
        except Exception as e:
            return {"symbol": symbol, "_error": type(e).__name__}
        # Only a raised error or a genuinely empty payload counts as failure.
        # A partial quote still renders fine — every field below uses .get()
        # with a default, and key availability varies a lot by ticker.
        if not info:
            return {"symbol": symbol, "_error": "empty quote response"}
        info.setdefault("symbol", symbol)
        return info

    mo.stop(not underlying_select.value)

    info1 = _safe_info(selected_row["Underlying"])
    info2 = _safe_info(selected_row["Bull_3x"])
    info3 = _safe_info(selected_row["Bear_3x"])

    mo.hstack([ticker_card_etf(info1), ticker_card_etf(info2), ticker_card_etf(info3)], gap="1rem")
    return


@app.cell
def _(
    Dict, List, Optional, Tuple, dataclass, datetime, go, mo, np, os,
    page_header, pd, selected_row,
):
    import letf_engine as _engine

    bull, bear, underlying = (selected_row[k] for k in ("Bull_3x", "Bear_3x", "Underlying"))
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
    MARGIN_DIVISOR = 5.0


    def log(msg: str, kind: str = "info"):
        mo.output.append(mo.callout(mo.md(msg), kind=kind))


    def fetch_data(
        underlying: str = underlying,
        letf: str = bear,
        bull_letf: str = bull,
        start: str = "2020-01-01",
        end: str = None,
        risk_free_annual: float = 0.05,
        api_key: str = ALPACA_API_KEY,
        api_secret: str = ALPACA_API_SECRET,
    ) -> tuple:
        """
        Download daily bars for `underlying`, `letf` (bear LETF), and `bull_letf` (bull LETF).

        rets columns : underlying | letf | bull_letf
        raw  columns : underlying | letf | bull_letf  (original ticker columns)
        """
        if end is None:
            end = datetime.date.today().isoformat()

        mo.output.append(page_header("Collecting Data Process", "02"))
        mo.output.append(
            mo.callout(
                mo.md(f"📡 Fetching **{underlying}**, **{letf}**, **{bull_letf}** from `{start}` to `{end}`"),
                kind="info"
            ))

        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = StockHistoricalDataClient(
            api_key=api_key or None,
            secret_key=api_secret or None,
        )

        request = StockBarsRequest(
            symbol_or_symbols=[underlying, letf, bull_letf],
            timeframe=TimeFrame.Day,
            start=datetime.datetime.fromisoformat(start),
            end=datetime.datetime.fromisoformat(end),
            adjustment="all",
        )

        bars = client.get_stock_bars(request).df  # MultiIndex: (symbol, timestamp)

        raw = (
            bars["close"]
            .unstack(level="symbol")
            .rename_axis(None, axis=1)
            .rename_axis("date")
            .tz_convert("America/New_York").tz_localize(None)
        )

        raw.index = raw.index.normalize()
        raw = raw[[underlying, letf, bull_letf]].dropna().sort_index()
        opens = bars["open"].unstack(level="symbol")
        opens.index = opens.index.tz_convert("America/New_York").tz_localize(None).normalize()
        opens = opens.reindex(index=raw.index, columns=raw.columns)
        if len(raw) < 100 or opens.isna().any().any():
            raise ValueError("insufficient or incomplete aligned opening/closing bars")

        rets = raw.pct_change().dropna()
        rets.columns = ["underlying", "letf", "bull_letf"]

        rf_daily = (1 + risk_free_annual) ** (1 / 252) - 1
        log(f"[Data] {len(rets)} trading days loaded.\n")

        return rets, rf_daily, raw, opens


    def get_account_nav(
        api_key: str = ALPACA_API_KEY,
        api_secret: str = ALPACA_API_SECRET,
        max_position: float = 10_000.0,
    ) -> float:
        from alpaca.trading.client import TradingClient

        trading_client = TradingClient(api_key, api_secret, paper=True)
        account = trading_client.get_account()
        nav = float(account.portfolio_value)
        nav = min(nav, max_position)
        log(f"[Account] Current portfolio value: ${float(account.portfolio_value):,.2f}  →  Using: ${nav:,.2f}")
        return nav


    def max_drawdown(nav):
        return _engine.max_drawdown(nav)

    def nav_from_returns(daily_rets: pd.DataFrame, initial: float = 1.0) -> pd.DataFrame:
        return (1 + daily_rets).cumprod() * initial


    def cvar(rets: pd.Series, alpha: float = 0.05) -> float:
        threshold = rets.quantile(alpha)
        return float(rets[rets <= threshold].mean())


    def omega_ratio(rets: pd.Series, threshold: float = 0.0) -> float:
        gains  = (rets[rets > threshold] - threshold).sum()
        losses = (threshold - rets[rets < threshold]).sum()
        return float(gains / losses) if losses != 0 else np.inf


    def compute_risk_table(
        daily_rets: pd.DataFrame,
        rf_daily: float,
        ann: int = 252,
    ) -> pd.DataFrame:
        rows = []
        for col in daily_rets.columns:
            r      = daily_rets[col].dropna()
            excess = r - rf_daily
            nav    = (1 + r).cumprod()

            # Geometric (compounded) CAGR — the arithmetic (1+mean)**252 overstates
            # the true return, badly so for high-vol leveraged strategies.
            ann_ret = nav.iloc[-1] ** (ann / len(r)) - 1 if len(r) > 0 else np.nan
            ann_vol = r.std() * np.sqrt(ann)
            sharpe  = excess.mean() / r.std() * np.sqrt(ann) if r.std() > 0 else np.nan
            # Downside deviation below the risk-free MAR over ALL observations —
            # not the std of only the negative returns, which measures the wrong thing.
            downside = np.minimum(0, r - rf_daily)
            downdev  = np.sqrt((downside ** 2).mean()) * np.sqrt(ann)
            sortino  = (excess.mean() * ann) / downdev if downdev > 0 else np.nan
            mdd     = max_drawdown(nav)
            calmar  = ann_ret / abs(mdd) if mdd != 0 else np.nan
            es95    = cvar(r, 0.05)
            omega   = omega_ratio(r)
            win_rt  = (r > 0).mean()

            rows.append({
                "Strategy":        col,
                "Ann. Return (%)": round(ann_ret * 100, 2),
                "Ann. Vol (%)":    round(ann_vol * 100, 2),
                "Sharpe":          round(sharpe, 3),
                "Sortino":         round(sortino, 3),
                "Max DD (%)":      round(mdd * 100, 2),
                "Calmar":          round(calmar, 3),
                "CVaR 95% (%)":    round(es95 * 100, 3),
                "Omega":           round(omega, 3),
                "Win Rate (%)":    round(win_rt * 100, 1),
            })
        return pd.DataFrame(rows).set_index("Strategy")


    def plot_all(
        daily_rets: pd.DataFrame,
        regimes: pd.Series,
        risk_table: pd.DataFrame,
        underlying: str,
        bull: str,
        bear: str,
        rf_daily: float,
    ):
        PALETTE_ = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657", "#79c0ff"]
        nav  = nav_from_returns(daily_rets)
        cols = list(nav.columns)

        def _rgba(h, a):
            h = h.lstrip("#")
            return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"

        def _short(s):
            return str(s).split("-")[0]

        def _style(f, height, title, hover="x unified"):
            f.update_layout(
                title=dict(text=title, x=0.012, xanchor="left",
                           font=dict(size=13, color="#e6edf3")),
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(family="DM Mono, monospace", color="#8b949e", size=11),
                margin=dict(l=58, r=18, t=46, b=40), height=height,
                legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0,
                            bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
                hovermode=hover, colorway=PALETTE_,
            )
            f.update_xaxes(gridcolor="#21262d", zerolinecolor="#30363d", linecolor="#30363d")
            f.update_yaxes(gridcolor="#21262d", zerolinecolor="#30363d", linecolor="#30363d")
            return f

        # ── 1. Portfolio NAV — regime-shaded, log scale ──────────────────────
        regime_colors = {"Bull": "#3fb950", "Bear": "#f78166", "Sideways": "#8b949e", "Cash": "#30363d"}
        fig_nav = go.Figure()
        for _, run in regimes.groupby((regimes != regimes.shift()).cumsum()):
            fig_nav.add_vrect(x0=run.index[0], x1=run.index[-1],
                              fillcolor=regime_colors[run.iloc[0]], opacity=0.06,
                              line_width=0, layer="below")
        for i, col in enumerate(cols):
            fig_nav.add_trace(go.Scatter(
                x=nav.index, y=nav[col], mode="lines", name=_short(col), legendgroup=col,
                line=dict(width=1.6, color=PALETTE_[i % len(PALETTE_)]),
                hovertemplate="%{y:.3f}<extra>" + _short(col) + "</extra>"))
        _style(fig_nav, 380, "Portfolio NAV — All Strategies  (shaded: Bull / Bear / Sideways)")
        fig_nav.update_yaxes(type="log", title_text="NAV (start = 1.0)")

        # ── 2. Drawdown ──────────────────────────────────────────────────────
        fig_dd = go.Figure()
        for i, col in enumerate(cols):
            dd = (nav[col] - nav[col].cummax().clip(lower=1.0)) / nav[col].cummax().clip(lower=1.0) * 100
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd, mode="lines", name=_short(col), legendgroup=col,
                line=dict(width=1.2, color=PALETTE_[i % len(PALETTE_)]),
                fill="tozeroy", fillcolor=_rgba(PALETTE_[i % len(PALETTE_)], 0.10),
                hovertemplate="%{y:.2f}%<extra>" + _short(col) + "</extra>"))
        _style(fig_dd, 300, "Drawdown (%)")
        fig_dd.update_yaxes(title_text="Drawdown (%)")

        # ── 3. Rolling 63-day Sharpe ─────────────────────────────────────────
        fig_rs = go.Figure()
        for i, col in enumerate(cols):
            rs = ((daily_rets[col] - rf_daily).rolling(63).mean() /
                  daily_rets[col].rolling(63).std() * np.sqrt(252))
            fig_rs.add_trace(go.Scatter(
                x=rs.index, y=rs, mode="lines", name=_short(col), legendgroup=col,
                line=dict(width=1.2, color=PALETTE_[i % len(PALETTE_)])))
        fig_rs.add_hline(y=0, line=dict(color="#8b949e", width=0.8, dash="dash"))
        _style(fig_rs, 300, "Rolling 63-day Sharpe")

        # ── 4. Risk-adjusted metrics ─────────────────────────────────────────
        fig_risk = go.Figure()
        mcolor = {"Sharpe": PALETTE_[0], "Sortino": PALETTE_[1], "Calmar": PALETTE_[2]}
        xs = [_short(s) for s in risk_table.index]
        for m in ["Sharpe", "Sortino", "Calmar"]:
            fig_risk.add_trace(go.Bar(x=xs, y=risk_table[m].astype(float),
                                      name=m, marker_color=mcolor[m]))
        _style(fig_risk, 320, "Risk-Adjusted Return Metrics", hover="x")
        fig_risk.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)

        # ── 5. Maximum drawdown ──────────────────────────────────────────────
        fig_mdd = go.Figure(go.Bar(
            y=[_short(s) for s in risk_table.index],
            x=risk_table["Max DD (%)"].astype(float), orientation="h",
            marker_color=[PALETTE_[i % len(PALETTE_)] for i in range(len(risk_table))],
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
        _style(fig_mdd, 320, "Maximum Drawdown (%)", hover="closest")

        # ── 6. Hedge-ratio optimisation (single axis: Sharpe vs h) ───────────
        # Single y-axis (Sharpe is the objective being optimised); Max DD lives in
        # the hover tooltip rather than on a hard-to-read secondary axis.
        header = mo.Html(
            f"""
            <div style="display:flex;align-items:baseline;gap:16px;
                        border-bottom:1px solid #30363d;padding:0 0 12px;margin:8px 0 6px;">
              <span style="font-family:'DM Mono',monospace;font-size:22px;font-weight:600;color:#e6edf3;">Net Portfolio Backtest</span>
              <span style="font-family:'DM Mono',monospace;font-size:11px;color:#8b949e;">Backtesting · Risk Metrics · Sensitivity</span>
              <span style="margin-left:auto;font-family:'DM Mono',monospace;font-size:13px;font-weight:600;color:#58a6ff;">{underlying} · {bull} · {bear}</span>
            </div>
            """
        )
        # One chart per row, full width — side-by-side panels were too cramped
        # to read (especially the hedge-ratio and drawdown charts).
        return mo.vstack([
            header,
            mo.ui.plotly(fig_nav),
            mo.ui.plotly(fig_dd),
            mo.ui.plotly(fig_rs),
            mo.ui.plotly(fig_risk),
            mo.ui.plotly(fig_mdd),
        ])


    def print_risk_table(risk_table: pd.DataFrame):
        mo.output.append(
            mo.vstack([
                mo.md("### 📊 Risk Metrics Table"),
                mo.ui.table(risk_table.reset_index()),
            ])
        )


    def print_daily_recommendation(
        chosen: str,
        detail: dict,
        trade_log: pd.DataFrame,
    ):
            # ── Chosen strategy callout ───────────────────────────────────────────
        kind = "success" if chosen != "HOLD CASH" else "warn"
        mo.output.append(
            mo.callout(mo.md(f"## ► {chosen}"), kind=kind)
        )

        # ── Summary stats ─────────────────────────────────────────────────────
        mo.output.append(
            mo.vstack([
                mo.md("### 📅 Daily Strategy Recommendation"),
                mo.md(f"""
    | | |
    |---|---|
    | **Today's Regime**       | `{detail['today_regime']}` |
    | **Regime-Best Strategy** | {detail['regime_best_strategy']} |
    | **Rolling-Sharpe Best**  | {detail['rolling_best_strategy']} |
    | **Signal**               | {detail['signal']} |
    """),
            ])
        )

        # ── Rolling Sharpe ranking table ──────────────────────────────────────
        sharpe_rows = [
            {
                "Strategy": strat,
                "Rolling Sharpe (63d)": f"{sh:+.3f}",
                "": "◄ BEST" if strat == detail["rolling_best_strategy"] else "",
            }
            for strat, sh in detail["rolling_sharpes"].items()
        ]
        mo.output.append(
            mo.vstack([
                mo.md("#### Rolling Sharpe Ranking — last 63 active days"),
                mo.ui.table(pd.DataFrame(sharpe_rows)),
            ])
        )

        # ── Today's trades ────────────────────────────────────────────────────
        if chosen == "HOLD CASH" or trade_log.empty:
            mo.output.append(
                mo.callout(mo.md("**Cash target:** reconcile existing managed positions to zero."), kind="warn")
            )
            return

        today = trade_log["date"].max()
        today_trades = trade_log[
            (trade_log["strategy"] == chosen) &
            (trade_log["date"] == today)
        ][["ticker", "role", "action", "quantity", "delta_quantity", "price", "notional_$", "leg_pnl_$"]]

        if today_trades.empty:
            mo.output.append(
                mo.callout(mo.md(f"⚠️ No trade rows found for `{chosen}` on `{today.date()}`."), kind="warn")
            )
            return

        mo.output.append(
            mo.vstack([
                mo.md(f"#### Trade Actions for `{today.date()}`"),
                mo.ui.table(today_trades.reset_index(drop=True)),
            ])
        )


    def main(use_live_data=True, start="2019-12-31", end=None, initial_nav_cap=10000.0):
        if use_live_data:
            try:
                rets, rf_daily, raw, opens = fetch_data(start=start, end=end)
            except Exception as exc:
                mo.stop(True, mo.callout(mo.md(f"Market data unavailable: {exc}. No targets generated."), kind="danger"))
        else:
            raw = _engine.synthetic_prices(underlying, bear, bull,
                                           leverage=float(selected_row["Leverage"]))
            opens = raw.shift().fillna(raw.iloc[0])
            rf_daily = (1.05 ** (1/252))-1
        config = _engine.Config(initial_nav=initial_nav_cap, allocation=1/MARGIN_DIVISOR)
        analysis = _engine.analyze(raw, opens, underlying, bear, bull, config, rf_daily)
        daily_rets = analysis["daily_returns"]
        risk_table = compute_risk_table(daily_rets, rf_daily)
        mo.output.append(page_header("Causal Portfolio Backtest", "02"))
        log("Signals use completed closes; historical fills occur at the NEXT session's open. "
            "Returns include the 20% capital allocation, whole adjusted-price units and $5,000 leg caps. "
            "Cost assumptions: 5bp per traded dollar, 3% annual short borrow, 0% cash yield and "
            "8% financing. These are estimates, not broker quotes. No extra ETF fee is deducted.")
        log("No parameter optimization is applied: the old full-sample short-inverse/cash scan "
            "did not represent this strategy. Evaluate growth changes on unseen periods.")
        print_risk_table(risk_table)
        mo.output.append(plot_all(daily_rets, analysis["regimes"], risk_table,
                                  underlying, bull, bear, rf_daily))
        execution_allowed = use_live_data
        try:
            actual_nav = get_account_nav(max_position=initial_nav_cap) if use_live_data else initial_nav_cap
        except Exception as exc:
            actual_nav = None
            execution_allowed = False
            log(f"Current account unavailable: {exc}. Backtest only; orders disabled.", "warn")
        if actual_nav is None:
            trade_log = _engine.current_target_log(analysis, raw.iloc[-1], initial_nav_cap).iloc[:0]
        else:
            trade_log = _engine.current_target_log(analysis, raw.iloc[-1], actual_nav)
        chosen, detail = analysis["chosen_strategy"], analysis["recommendation_detail"]
        log("Displayed quantities are indicative targets from current NAV and the latest completed "
            "close. Submission refreshes account/quotes. Historical NAV is never added to account NAV.")
        print_daily_recommendation(chosen, detail, trade_log)
        return {**analysis, "risk_table": risk_table, "trade_log": trade_log,
                "execution_allowed": execution_allowed, "data_source": "alpaca" if use_live_data else "synthetic",
                "managed_tickers": [underlying, bear, bull], "rf_daily": rf_daily}

    results = main()
    return ALPACA_API_KEY, ALPACA_API_SECRET, log, results


@app.cell
def _(mo, results):
    trade_log = results["trade_log"]

    today_df = trade_log[
        (trade_log["date"] == results["signal_date"]) &
        (trade_log["strategy"] == results["chosen_strategy"])
    ]

    if not today_df.empty:
        mo.output.append(mo.vstack([
            mo.md(f"#### 🗓️ Trade Actions for `{today_df['date'].iloc[-1].date()}`"),
            mo.ui.table(today_df.reset_index(drop=True)),
        ]))
    return (today_df,)


@app.cell
def _(mo):
    # Explicit guard: LETF orders are submitted only when this button is clicked,
    # never on an incidental reactive re-run of the notebook.
    exec_button = mo.ui.run_button(label="⚡ Submit Today's LETF Orders")
    exec_button
    return (exec_button,)


@app.cell
def _(
    ALPACA_API_KEY, ALPACA_API_SECRET, MarketOrderRequest, OrderSide,
    TimeInForce, TradingClient, exec_button, log, mo, results, selected_row, today_df,
):
    mo.stop(not exec_button.value, mo.md("Press Submit to reconcile paper-account targets."))
    mo.stop(not results.get("execution_allowed") or results.get("data_source") != "alpaca",
            mo.md("Orders disabled: verified market data and current account state are required."))
    import hashlib as _hashlib
    import json as _json
    import pathlib as _pathlib
    import pandas as _pd
    from alpaca.data.historical import StockHistoricalDataClient as _QuoteClient
    from alpaca.data.requests import StockLatestQuoteRequest as _QuoteRequest
    from alpaca.trading.requests import GetCalendarRequest as _CalendarRequest
    from letf_execution import build_order_plan as _build_plan

    _client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)
    _clock = _client.get_clock()
    mo.stop(not _clock.is_open, mo.md("Market is closed. Refresh and submit during regular hours."))
    _now = _pd.Timestamp(_clock.timestamp).tz_convert("America/New_York")
    _calendar = _client.get_calendar(_CalendarRequest(
        start=(_now - _pd.Timedelta(days=14)).date(), end=_now.date()))
    _completed = []
    for _session in _calendar:
        _close = _pd.Timestamp(_session.close)
        _close = _close.tz_localize("America/New_York") if _close.tzinfo is None else _close.tz_convert("America/New_York")
        if _close < _now:
            _completed.append(_close.date())
    mo.stop(not _completed or _pd.Timestamp(results["signal_date"]).date() != max(_completed),
            mo.md("Stale signal: refresh the analysis through the last completed exchange session."))

    _state_path = _pathlib.Path(mo.notebook_dir()) / ".letf_managed.json"
    _previous = _json.loads(_state_path.read_text()) if _state_path.exists() else []
    if not isinstance(_previous, list) or not all(isinstance(x, str) for x in _previous):
        raise ValueError("Invalid managed-symbol state; refusing incomplete reconciliation")
    _managed = sorted(set(_previous) | set(results["managed_tickers"]))
    _quotes = _QuoteClient(ALPACA_API_KEY, ALPACA_API_SECRET).get_stock_latest_quote(
        _QuoteRequest(symbol_or_symbols=_managed))
    _prices = {}
    for _symbol in _managed:
        _quote = _quotes[_symbol]
        _age = (_now - _pd.Timestamp(_quote.timestamp)).total_seconds()
        if _age < -5 or _age > 120 or not 0 < _quote.bid_price <= _quote.ask_price:
            raise ValueError(f"{_symbol}: fresh, valid two-sided quote required")
        _prices[_symbol] = (_quote.bid_price + _quote.ask_price) / 2
    _weights = results["combined_decisions"].iloc[-1]
    _plan = _build_plan(_client, _weights, _pd.Series(_prices), results["config"], _managed)
    # Remember every managed family before submitting, including partial fills.
    _tmp = _state_path.with_suffix(".tmp")
    _tmp.write_text(_json.dumps(_managed), encoding="utf-8")
    _tmp.replace(_state_path)
    letf_exec = {"signal_date": str(results["signal_date"])[:10], "rows": []}
    log("Paper execution uses current market quotes. Manual fills can differ from the modeled next-open fill. "
        "Reductions/closures run before new risk; refresh and submit again after fills if entry remains.")
    for _row in _plan:
        _side = OrderSide.BUY if _row["delta"] > 0 else OrderSide.SELL
        _key = f"{letf_exec['signal_date']}|{_row['symbol']}|{_row['current']}|{_row['target']}|{_row['phase']}"
        _order_id = "letf-" + _hashlib.sha256(_key.encode()).hexdigest()[:32]
        _record = {**_row, "side": _side.value.upper(), "outcome": "unknown"}
        letf_exec["rows"].append(_record)
        try:
            _order = _client.submit_order(MarketOrderRequest(
                symbol=_row["symbol"], qty=abs(_row["delta"]), side=_side,
                time_in_force=TimeInForce.DAY, client_order_id=_order_id))
            _record.update(outcome="submitted", order_ref=str(_order.id))
            log(f"Submitted {_side.value} {abs(_row['delta'])} {_row['symbol']} ({_row['phase']}).")
        except Exception as _exc:
            _record["outcome"] = "failed"
            log(f"Order failed: {_exc}. Stopped remaining orders; inspect any partial fills before retrying.", "danger")
            break
    return (letf_exec,)


if __name__ == "__main__":
    app.run()
