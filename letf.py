import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full", css_file="theme.css", html_head_file="theme_head.html")


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

    # Load Alpaca credentials from .env in the notebook's own directory
    # (assign to _ so marimo doesn't render load_dotenv's bool return).
    _ = load_dotenv(mo.notebook_dir() / ".env")
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
        "Sector": [
            "Nasdaq 100", "S&P 500", "Russell 2000 Small Cap", "Dow Jones 30",
            "Semiconductors", "Technology", "Financials", "Biotech",
            "Emerging Markets", "China Large Cap", "20yr US Treasuries", "7-10yr US Treasuries",
            "Real Estate", "Gold Miners",
        ],
    })

    mo.ui.dataframe(etf_triples)
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
    with open("ETF_triple.txt", "w") as _f:
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
            3x leveraged bull/bear tickers for the selected underlying
        </p>
        """),
        mo.hstack([
            ticker_card("Underlying", selected_row["Underlying"], "#ffffff"),
            ticker_card("Bull 3x", selected_row["Bull_3x"], "#00e676"),
            ticker_card("Bear 3x", selected_row["Bear_3x"], "#ff5252"),
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
        expense  = info.get("annualReportExpenseRatio") or info.get("annualHoldingsTurnover", 0)
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
    Dict,
    List,
    Optional,
    Tuple,
    dataclass,
    datetime,
    go,
    mo,
    np,
    os,
    page_header,
    pd,
    selected_row,
):
    # ── Dynamic ticker context (from the dropdown selection upstream) ─────────
    bull       = selected_row["Bull_3x"]
    bear       = selected_row["Bear_3x"]
    underlying = selected_row["Underlying"]

    # ── LIVE position-sizing divisor (a risk cap) ─────────────────────────────
    # Deploy only 1/MARGIN_DIVISOR of NAV per leg when placing real orders (~5x
    # margin buffer). This scales ONLY the live order quantities in the trade log
    # — the backtest and risk metrics evaluate the strategy at FULL weight, so
    # realized dollars ≈ backtest performance × (1 / MARGIN_DIVISOR).
    MARGIN_DIVISOR = 5.0

    # ── Volatility targeting ──────────────────────────────────────────────────
    # Scale exposure by TARGET_ANN_VOL / realised vol so RISK (not notional) is
    # roughly constant: lean out when the underlying is volatile, lean in when
    # it's calm. Uses only trailing vol, so it's causal. Clipped so it can never
    # demand extreme leverage. Set VOL_TARGETING = False to reproduce the old
    # un-scaled behaviour.
    VOL_TARGETING  = True
    TARGET_ANN_VOL = 0.20     # annualised vol we size the underlying exposure to
    VOL_SCALAR_MIN = 0.50
    VOL_SCALAR_MAX = 1.50

    # Days a regime must persist before the combined system will trade it.
    MIN_REGIME_PERSIST = 3

    # ── Credentials from .env (loaded in the imports cell), never hardcoded ───
    ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
    ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        print("[Warning] ALPACA_API_KEY / ALPACA_API_SECRET not found in environment. "
              "Live data calls will fail and fall back to synthetic data.")

    def log(msg: str, kind: str = "info"):
        mo.output.append(mo.callout(mo.md(msg), kind=kind))

    # ─────────────────────────────────────────────
    # 1.  DATA LAYER
    # ─────────────────────────────────────────────
    def fetch_data(
        underlying: str = underlying,
        letf: str = bear,
        bull_letf: str = bull,
        start: str = "2020-01-01",
        end: str = None,
        risk_free_annual: float = 0.05,
        api_key: str = ALPACA_API_KEY,
        api_secret: str = ALPACA_API_SECRET,
    ) -> Tuple[pd.DataFrame, float, pd.DataFrame]:
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
            .tz_localize(None)
        )

        raw = raw[[underlying, letf, bull_letf]].dropna()

        rets = raw.pct_change().dropna()
        rets.columns = ["underlying", "letf", "bull_letf"]

        rf_daily = (1 + risk_free_annual) ** (1 / 252) - 1
        log(f"[Data] {len(rets)} trading days loaded.\n")

        return rets, rf_daily, raw


    def simulate_synthetic_data(
        n_days: int = 1260,
        annual_mu: float = 0.08,
        annual_vol: float = 0.18,
        leverage: float = -3.0,
        vol_decay_factor: float = 1.0,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, float, pd.DataFrame]:
        """
        GBM-based synthetic data with leveraged-ETF vol decay.
        Useful for controlled sensitivity experiments / fallback when live data fails.
        """
        rng = np.random.default_rng(seed)
        dt = 1 / 252
        mu_d  = annual_mu * dt
        sig_d = annual_vol * np.sqrt(dt)

        idx_rets = rng.normal(mu_d, sig_d, n_days)
        decay = 0.5 * leverage * (leverage - 1) * (annual_vol ** 2) * dt * vol_decay_factor
        # Vol decay is always a DRAG on the fund → subtract it (the bull leg below
        # subtracts its decay too). Adding it gave the inverse LETF a phantom tailwind.
        letf_rets = leverage * idx_rets - decay + rng.normal(0, 0.001, n_days)

        # Bull LETF: +3× with same decay formula
        bull_letf_rets = 3.0 * idx_rets - 0.5 * 3.0 * 2.0 * (annual_vol ** 2) * dt + rng.normal(0, 0.001, n_days)

        dates = pd.bdate_range("2019-01-02", periods=n_days)
        rets = pd.DataFrame(
            {"underlying": idx_rets, "letf": letf_rets, "bull_letf": bull_letf_rets},
            index=dates,
        )
        rf_daily = (1.05) ** dt - 1

        # raw is synthetic price levels (not real tickers) — kept distinct from rets
        raw = (1 + rets).cumprod() * 100.0
        raw.columns = [underlying, bear, bull]

        return rets, rf_daily, raw


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


    # ─────────────────────────────────────────────
    # 2.  MARKET-REGIME CLASSIFICATION
    # ─────────────────────────────────────────────
    def classify_regimes(
        rets: pd.DataFrame,
        vol_window: int = 21,
        trend_window: int = 63,
        base_threshold: float = 0.05,
    ) -> pd.Series:
        """
        Label each day as Bull / Bear / Sideways using rolling trend, with the
        trend threshold adaptively widened/narrowed by realised volatility.

        Previously `rvol` was computed but never used — now it scales the
        classification threshold so that high-vol periods need a stronger trend
        to be called Bull/Bear (avoiding regime flip-flopping on noise), and
        low-vol periods can be classified on a smaller trend move.
        """
        price = (1 + rets["underlying"]).cumprod()
        trend = price.pct_change(trend_window).fillna(0)
        rvol  = rets["underlying"].rolling(vol_window).std().bfill() * np.sqrt(252)

        # Scale threshold by relative vol (clipped so it can't blow up on regime extremes).
        # Use an EXPANDING (causal) median — the full-sample median would leak future
        # volatility into historical regime labels (look-ahead bias in the backtest).
        rvol_med   = rvol.expanding(min_periods=vol_window).median().bfill()
        rvol_ratio = (rvol / rvol_med).clip(0.5, 2.0)
        adaptive_threshold = base_threshold * rvol_ratio

        regimes = pd.Series("Sideways", index=rets.index)
        regimes[trend >  adaptive_threshold] = "Bull"
        regimes[trend < -adaptive_threshold] = "Bear"
        return regimes


    # ─────────────────────────────────────────────
    # 3.  STRATEGY DEFINITIONS
    # ─────────────────────────────────────────────
    @dataclass
    class Strategy:
        name: str
        w_short_letf:      float = 1.0   # primary leg (bear LETF: long in Bear, short in Sideways)
        w_inverse_etf:     float = 0.0   # underlying leg (short in Bear, long in Bull)
        w_short_bull_letf: float = 0.0   # bull LETF short leg (Sideways double-short only)
        w_cash:            float = 0.0   # cash / T-bill buffer

    # ══════════════════════════════════════════════════════════════════════
    # REGIME-SWITCHING STRATEGIES
    #
    # Bear     : Long {bear} (60%) + Short {underlying} (40%)
    #            Net exposure ≈ −2.2× underlying on bear days (0.6·−3× + 0.4·−1×), cash otherwise.
    #
    # Bull     : Long {bull} (60%) + Long {underlying} (40%)
    #            Net exposure ≈ +2.2× underlying on bull days (0.6·+3× + 0.4·+1×), cash otherwise.
    #
    # Sideways : Short {bear} (50%) + Short {bull} (50%)   ← delta-neutral decay harvest
    #            Shorting ONLY the bear LETF left the sleeve net LONG the underlying
    #            at ~3×, i.e. a directional bet wearing a "sideways" label. Shorting
    #            both 3× legs dollar-balanced cancels the directional exposure
    #            (0.5·−3× + 0.5·+3× ≈ 0) and captures the compounding drag on BOTH,
    #            which is the only structurally reliable edge here.
    #            ⚠ Short-gamma / negative skew: in a strong sustained trend the
    #            winning leg's loss is unbounded while the losing leg only falls to
    #            zero. Keep the notional cap and regime filter tight.
    # ══════════════════════════════════════════════════════════════════════

    SIDEWAYS_NAME = f"Sideways-Regime: Short {bear} + Short {bull}"

    STRATEGIES: Dict[str, Strategy] = {
        f"Bear-Regime: Long {bear} + Short {underlying}": Strategy(
            name=f"Bear-Regime: Long {bear} + Short {underlying}",
            w_short_letf=0.60,    # long bear LETF weight (bear days only)
            w_inverse_etf=0.40,   # short underlying weight (bear days only)
            w_cash=0.00,
        ),
        f"Bull-Regime: Long {bull} + Long {underlying}": Strategy(
            name=f"Bull-Regime: Long {bull} + Long {underlying}",
            w_short_letf=0.60,    # long bull LETF weight (bull days only)
            w_inverse_etf=0.40,   # long underlying weight (bull days only)
            w_cash=0.00,
        ),
        SIDEWAYS_NAME: Strategy(
            name=SIDEWAYS_NAME,
            w_short_letf=0.50,      # short bear LETF (sideways days only)
            w_short_bull_letf=0.50, # short bull LETF → dollar-balanced, delta-neutral
            w_inverse_etf=0.00,
            w_cash=0.00,
        ),
    }


    # ─────────────────────────────────────────────
    # 4.  BACKTESTING ENGINE
    # ─────────────────────────────────────────────
    def compute_strategy_returns(
        rets: pd.DataFrame,
        strategy: Strategy,
        rf_daily: float,
        regimes: Optional[pd.Series] = None,
        vol_scalar: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Compute daily P&L of a strategy as a fraction of NAV.

        Bear     : Long bear-LETF (60%) + Short underlying (40%) on bear days → cash otherwise
        Bull     : Long bull-LETF (60%) + Long underlying (40%)  on bull days → cash otherwise
        Sideways : Short bear-LETF (100%) on sideways days                   → cash otherwise

        `vol_scalar` (optional, causal) scales exposure for volatility targeting.
        """
        idx_ret       = rets["underlying"]
        letf_ret      = rets["letf"]                                          # bear LETF returns
        bull_letf_ret = rets["bull_letf"] if "bull_letf" in rets.columns else idx_ret * 3.0

        def _mask(label: str) -> pd.Series:
            if regimes is None:
                return pd.Series(False, index=rets.index)
            return regimes == label

        def _scaled(pnl: pd.Series) -> pd.Series:
            """
            Apply the volatility-target scalar s: s·pnl + (1-s)·rf.
            The un-deployed fraction sits in cash at rf; when s > 1 the extra
            exposure is financed at rf. s = 1 reproduces the unscaled P&L.
            """
            if vol_scalar is None:
                return pnl
            s = vol_scalar.reindex(pnl.index).fillna(1.0)
            return pnl * s + rf_daily * (1 - s)

        # ── Bear regime ───────────────────────────────────────────────────────
        if strategy.name == f"Bear-Regime: Long {bear} + Short {underlying}":
            is_active = _mask("Bear")
            pnl_active = (
                letf_ret     * strategy.w_short_letf    # long bear LETF → profits when underlying ↓
                + (-idx_ret) * strategy.w_inverse_etf    # short underlying → profits when underlying ↓
            )
            total = pd.Series(rf_daily, index=rets.index)
            total[is_active] = _scaled(pnl_active)[is_active]
            total.name = strategy.name
            return total

        # ── Bull regime ───────────────────────────────────────────────────────
        if strategy.name == f"Bull-Regime: Long {bull} + Long {underlying}":
            is_active = _mask("Bull")
            pnl_active = (
                bull_letf_ret * strategy.w_short_letf    # long bull LETF → profits when underlying ↑
                + idx_ret     * strategy.w_inverse_etf    # long underlying → profits when underlying ↑
            )
            total = pd.Series(rf_daily, index=rets.index)
            total[is_active] = _scaled(pnl_active)[is_active]
            total.name = strategy.name
            return total

        # ── Sideways regime ───────────────────────────────────────────────────
        if strategy.name == SIDEWAYS_NAME:
            is_active = _mask("Sideways")
            # Short BOTH 3× legs, dollar-balanced: the ±3× directional exposures
            # cancel (0.5·−3× + 0.5·+3× ≈ 0) and what remains is the compounding
            # drag harvested off both funds.
            pnl_active = (
                -letf_ret       * strategy.w_short_letf        # short bear LETF
                - bull_letf_ret * strategy.w_short_bull_letf   # short bull LETF
            )
            total = pd.Series(rf_daily, index=rets.index)
            total[is_active] = _scaled(pnl_active)[is_active]
            total.name = strategy.name
            return total

        # ── Fallback for any ad-hoc static strategies ─────────────────────────
        pnl_short = -letf_ret  * strategy.w_short_letf
        pnl_inv   = (-idx_ret) * strategy.w_inverse_etf
        pnl_cash  = rf_daily   * strategy.w_cash
        total = pnl_short + pnl_inv + pnl_cash
        total.name = strategy.name
        return total


    def run_backtest(
        rets: pd.DataFrame,
        strategies: Dict[str, Strategy],
        rf_daily: float,
        regimes: Optional[pd.Series] = None,
        vol_scalar: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Run all strategies; return DataFrame of daily returns (one col per strategy)."""
        result = {}
        for name, strat in strategies.items():
            result[name] = compute_strategy_returns(
                rets, strat, rf_daily, regimes=regimes, vol_scalar=vol_scalar)
        return pd.DataFrame(result)


    def build_combined_returns(
        daily_rets: pd.DataFrame,
        regimes: pd.Series,
        rf_daily: float,
        min_persist: int = 3,
        use_agreement_gate: bool = True,
        rolling_window: int = 63,
        min_active_days: int = 15,
        sharpe_agreement_threshold: float = -0.5,
    ) -> pd.Series:
        """
        The strategy you ACTUALLY trade, as a single return series.

        Each sleeve above sits in cash whenever its own regime is inactive, so no
        individual curve represents the live system. Here we hold whichever
        regime's sleeve is active on each day — but only once that regime has
        persisted `min_persist` consecutive days; otherwise cash at rf.

        Fully causal: the run length is measured from data up to that day only,
        so this series is safe to judge profitability on.
        """
        regime_to_strategy = {
            "Bear":     f"Bear-Regime: Long {bear} + Short {underlying}",
            "Bull":     f"Bull-Regime: Long {bull} + Long {underlying}",
            "Sideways": SIDEWAYS_NAME,
        }
        run_id  = (regimes != regimes.shift()).cumsum()
        run_len = regimes.groupby(run_id).cumcount() + 1     # causal run length

        regime_arr = regimes.to_numpy()
        run_arr    = run_len.to_numpy()
        sleeve_arr = {reg: daily_rets[s].to_numpy()
                      for reg, s in regime_to_strategy.items()
                      if s in daily_rets.columns}

        out_arr = np.full(len(daily_rets), rf_daily, dtype=float)

        for t in range(len(out_arr)):
            reg = regime_arr[t]
            if reg not in sleeve_arr or run_arr[t] < min_persist:
                continue                                  # regime too fresh → cash

            if not use_agreement_gate:
                out_arr[t] = sleeve_arr[reg][t]
                continue

            # Causal agreement gate: rank each sleeve by its trailing Sharpe on its
            # OWN active days within the lookback, using data up to t only. Trade
            # today's regime sleeve unless it lags the best sleeve by more than the
            # threshold (mirrors pick_best_strategy_today, without look-ahead).
            lo      = max(0, t - rolling_window + 1)
            win_reg = regime_arr[lo:t + 1]
            sharpes = {}
            for reg2, arr in sleeve_arr.items():
                vals = arr[lo:t + 1][win_reg == reg2]
                if len(vals) < min_active_days:
                    continue
                sd = vals.std(ddof=1)
                if sd == 0:
                    continue
                ann_ret = np.prod(1 + vals) ** (252 / len(vals)) - 1
                sharpes[reg2] = ann_ret / (sd * np.sqrt(252))

            own = sharpes.get(reg)
            if own is None:                               # too little history to judge
                out_arr[t] = sleeve_arr[reg][t]           # → trust the regime pick
            elif own - max(sharpes.values()) > sharpe_agreement_threshold:
                out_arr[t] = sleeve_arr[reg][t]
            # else: sleeve is lagging badly → stay in cash at rf

        return pd.Series(out_arr, index=daily_rets.index,
                         name="Combined-Switched System")


    def nav_from_returns(daily_rets: pd.DataFrame, initial: float = 1.0) -> pd.DataFrame:
        return (1 + daily_rets).cumprod() * initial


    # ─────────────────────────────────────────────
    # 4b.  TRADE ACTION LOG
    # ─────────────────────────────────────────────
    def generate_trade_log(
        rets: pd.DataFrame,
        raw: pd.DataFrame,
        daily_rets: pd.DataFrame,
        rf_daily: float,
        underlying: str,
        bear: str,
        bull: str,
        initial_nav: float = 100_000.0,
        margin_divisor: float = MARGIN_DIVISOR,
        max_notional_per_leg: float = 5_000.0,   # ← add this
        vol_scalar: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Build a per-leg, per-day trade action log.

        FIX: `prev_nav` previously never updated inside the loop, so every day's
        position size was computed off `initial_nav` regardless of compounding.
        It now correctly uses the prior day's actual NAV per strategy.
        """
        spot_letf       = raw[bear]
        spot_underlying = raw[underlying]
        rvol            = rets["underlying"].rolling(21).std().bfill() * np.sqrt(252)
        regimes         = classify_regimes(rets)

        STRATEGY_LEGS = {
            f"Bear-Regime: Long {bear} + Short {underlying}": [
                {"ticker": bear,       "role": f"long_{bear.lower()}_bear",        "weight": 0.60},
                {"ticker": underlying, "role": f"short_{underlying.lower()}_bear", "weight": 0.40},
            ],
            f"Bull-Regime: Long {bull} + Long {underlying}": [
                {"ticker": bull,       "role": f"long_{bull.lower()}_bull",        "weight": 0.60},
                {"ticker": underlying, "role": f"long_{underlying.lower()}_bull",  "weight": 0.40},
            ],
            SIDEWAYS_NAME: [
                {"ticker": bear,       "role": f"short_{bear.lower()}_side",       "weight": 0.50},
                {"ticker": bull,       "role": f"short_{bull.lower()}_side",       "weight": 0.50},
            ],
        }

        rows        = []
        prev_shares = {}

        for strategy_name, legs in STRATEGY_LEGS.items():
            if strategy_name not in daily_rets.columns:
                log(f"[TradeLog] Skipping '{strategy_name}' — not found in daily_rets columns.")
                continue

            nav_series = (1 + daily_rets[strategy_name]).cumprod() * initial_nav
            daily_pnl_ = daily_rets[strategy_name] * nav_series.shift(1).fillna(initial_nav)

            for i, date in enumerate(rets.index):
                current_nav = float(nav_series.iloc[i])
                pnl_today   = float(daily_pnl_.iloc[i])
                cum_ret_pct = (current_nav / initial_nav - 1) * 100
                regime      = regimes.iloc[i]
                vol_today   = float(rvol.iloc[i])
                s_letf      = float(spot_letf.iloc[i])
                s_und       = float(spot_underlying.iloc[i])

                # ── FIX: size off the previous day's actual NAV, not a constant ──
                prev_nav = float(nav_series.iloc[i - 1]) if i > 0 else initial_nav

                is_bear     = (regime == "Bear")
                is_bull     = (regime == "Bull")
                is_sideways = (regime == "Sideways")

                s_bull_letf = float(raw[bull].iloc[i]) if bull in raw.columns else s_letf / 3.0

                # Volatility-target scalar for this day (1.0 when targeting is off),
                # so live order sizes match the vol-targeted backtest.
                vs = 1.0 if vol_scalar is None else float(vol_scalar.iloc[i])

                for leg in legs:
                    ticker = leg["ticker"]
                    role   = leg["role"]
                    weight = leg["weight"]

                    if role == f"long_{bear.lower()}_bear":
                        price      = s_letf
                        target_qty = (prev_nav * weight / price / margin_divisor) if is_bear else 0.0
                        action     = f"BUY LONG {bear} (BEAR)" if is_bear else "FLAT — CASH"
                        leg_pnl    = (rets["letf"].iloc[i] * weight * prev_nav
                                      if is_bear else rf_daily * weight * prev_nav)

                    elif role == f"short_{underlying.lower()}_bear":
                        price      = s_und
                        target_qty = -(prev_nav * weight / price / margin_divisor) if is_bear else 0.0
                        action     = f"SELL SHORT {underlying} (BEAR)" if is_bear else "FLAT — CASH"
                        leg_pnl    = (-rets["underlying"].iloc[i] * weight * prev_nav
                                      if is_bear else rf_daily * weight * prev_nav)

                    elif role == f"long_{bull.lower()}_bull":
                        price      = s_bull_letf
                        target_qty = (prev_nav * weight / price / margin_divisor) if is_bull else 0.0
                        action     = f"BUY LONG {bull} (BULL)" if is_bull else "FLAT — CASH"
                        bull_ret   = (rets["bull_letf"].iloc[i] if "bull_letf" in rets.columns
                                      else rets["underlying"].iloc[i] * 3.0)
                        leg_pnl    = (bull_ret * weight * prev_nav
                                      if is_bull else rf_daily * weight * prev_nav)

                    elif role == f"long_{underlying.lower()}_bull":
                        price      = s_und
                        target_qty = (prev_nav * weight / price / margin_divisor) if is_bull else 0.0
                        action     = f"BUY LONG {underlying} (BULL)" if is_bull else "FLAT — CASH"
                        leg_pnl    = (rets["underlying"].iloc[i] * weight * prev_nav
                                      if is_bull else rf_daily * weight * prev_nav)

                    elif role == f"short_{bear.lower()}_side":
                        price      = s_letf
                        target_qty = -(prev_nav * weight / price / margin_divisor) if is_sideways else 0.0
                        action     = f"SELL SHORT {bear} (SIDEWAYS)" if is_sideways else "FLAT — CASH"
                        leg_pnl    = (-rets["letf"].iloc[i] * weight * prev_nav
                                      if is_sideways else rf_daily * weight * prev_nav)

                    elif role == f"short_{bull.lower()}_side":
                        # Second leg of the delta-neutral decay harvest.
                        price      = s_bull_letf
                        target_qty = -(prev_nav * weight / price / margin_divisor) if is_sideways else 0.0
                        action     = f"SELL SHORT {bull} (SIDEWAYS)" if is_sideways else "FLAT — CASH"
                        bull_ret_s = (rets["bull_letf"].iloc[i] if "bull_letf" in rets.columns
                                      else rets["underlying"].iloc[i] * 3.0)
                        leg_pnl    = (-bull_ret_s * weight * prev_nav
                                      if is_sideways else rf_daily * weight * prev_nav)

                    elif role == "cash":
                        price      = 1.0
                        target_qty = prev_nav * weight / margin_divisor
                        action     = "HOLD CASH"
                        leg_pnl    = rf_daily * weight * prev_nav

                    else:
                        continue

                    # Volatility targeting: scale size and the leg's P&L together,
                    # applied before the hard notional cap below.
                    target_qty = target_qty * vs
                    leg_pnl    = leg_pnl * vs

                    if price > 0:
                        max_qty = max_notional_per_leg / price
                        target_qty = np.sign(target_qty) * min(abs(target_qty), max_qty)

                    key       = f"{strategy_name}|{ticker}"
                    prev_qty  = prev_shares.get(key, 0.0)
                    delta_qty = target_qty - prev_qty
                    prev_shares[key] = target_qty

                    rows.append({
                        "date":           date,
                        "strategy":       strategy_name,
                        "ticker":         ticker,
                        "role":           role,
                        "action":         action,
                        "quantity":       round(target_qty, 4),
                        "delta_quantity": round(delta_qty, 4),
                        "price":          round(price, 4),
                        "notional_$":     round(abs(target_qty * price), 2),
                        "leg_pnl_$":      round(leg_pnl, 2),
                        "total_pnl_$":    round(pnl_today, 2),
                        "nav_$":          round(current_nav, 2),
                        "cum_return_%":   round(cum_ret_pct, 3),
                        "regime":         regime,
                        "rvol_ann_%":     round(vol_today * 100, 2),
                    })

        df = (pd.DataFrame(rows)
                .assign(date=lambda x: pd.to_datetime(x["date"]))
                .sort_values(["date", "strategy", "ticker"])
                .reset_index(drop=True))
        return df


    def print_trade_log_summary(trade_log: pd.DataFrame, tail_days: int = 1):
        if trade_log.empty:
            log("[TradeLog] No rows to display.")
            # print("[TradeLog] No rows to display.")
            return
        last_dates = trade_log["date"].drop_duplicates().nlargest(tail_days)
        subset     = trade_log[trade_log["date"].isin(last_dates)]
        mo.output.append(page_header("Trading Operation", "04"))
        mo.output.append(
            mo.vstack([
                mo.md(f"### 📋 Trade Action Log — last {tail_days} trading day(s)"),
                mo.md(f"*P&L / NAV columns show full-weight strategy performance. "
                      f"Live `quantity` is risk-capped to 1/{MARGIN_DIVISOR:g} of that, "
                      f"so realized dollars ≈ shown P&L ÷ {MARGIN_DIVISOR:g}.*"),
                mo.ui.table(subset.reset_index(drop=True)),
            ])
        )


    # ─────────────────────────────────────────────
    # 5.  RISK METRICS
    # ─────────────────────────────────────────────
    def max_drawdown(nav: pd.Series) -> float:
        roll_max = nav.cummax()
        dd = (nav - roll_max) / roll_max
        return float(dd.min())


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


    # ─────────────────────────────────────────────
    # 6.  HEDGE-RATIO OPTIMISATION
    # ─────────────────────────────────────────────
    def optimize_hedge_ratio(
        rets: pd.DataFrame,
        rf_daily: float,
        objective: str = "sharpe",    # "sharpe" | "min_mdd"
        resolution: int = 50,
    ) -> Tuple[float, pd.DataFrame]:
        h_grid  = np.linspace(0, 1, resolution)
        records = []
        for h in h_grid:
            strat = Strategy(name=f"h={h:.2f}", w_short_letf=1 - h, w_cash=h)
            r     = compute_strategy_returns(rets, strat, rf_daily)
            nav   = (1 + r).cumprod()
            ann_ret = nav.iloc[-1] ** (252 / len(r)) - 1 if len(r) > 0 else np.nan
            ann_vol = r.std() * np.sqrt(252)
            sharpe  = (ann_ret - rf_daily * 252) / ann_vol if ann_vol > 0 else np.nan
            mdd     = max_drawdown(nav)
            records.append({"h": h, "Sharpe": sharpe, "MDD": mdd, "Ann_Ret": ann_ret})

        df = pd.DataFrame(records)
        if objective == "sharpe":
            best_h = float(df.loc[df["Sharpe"].idxmax(), "h"])
        else:
            # FIX: minimizing drawdown means MDD closest to zero, not idxmax
            best_h = float(df.loc[df["MDD"].idxmax(), "h"])  # MDD is negative; idxmax = least negative

        return best_h, df


    # ─────────────────────────────────────────────
    # 7.  VOLATILITY-DECAY QUANTIFICATION
    # ─────────────────────────────────────────────
    def quantify_vol_decay(
        leverage: float = 3.0,
        annual_vols: np.ndarray = np.linspace(0.05, 0.60, 200),
        annual_mu: float = 0.00,
        n_days: int = 252,
    ) -> pd.DataFrame:
        records = []
        for vol in annual_vols:
            decay    = 0.5 * leverage * (leverage - 1) * (vol ** 2) / 252
            terminal = np.exp((leverage * annual_mu - 0.5 * leverage * (leverage - 1) * vol ** 2) * 1.0)
            records.append({
                "Annual Vol (%)":    vol * 100,
                "3× LETF Terminal":  terminal,
                "Daily Decay (bps)": decay * 1e4,
            })
        return pd.DataFrame(records)


    # ─────────────────────────────────────────────
    # 8.  REGIME-CONDITIONAL PERFORMANCE
    # ─────────────────────────────────────────────
    def regime_performance(
        daily_rets: pd.DataFrame,
        regimes: pd.Series,
        rf_daily: float,
    ) -> pd.DataFrame:
        rows = []
        for regime in ["Bull", "Bear", "Sideways"]:
            mask = regimes == regime
            sub  = daily_rets[mask]
            if sub.empty:
                continue
            for col in daily_rets.columns:
                r = sub[col]
                ann_ret = (1 + r).prod() ** (252 / len(r)) - 1 if len(r) > 0 else np.nan
                ann_vol = r.std() * np.sqrt(252)
                sharpe  = (ann_ret - rf_daily * 252) / ann_vol if ann_vol > 0 else np.nan
                rows.append({
                    "Regime":       regime,
                    "Strategy":     col,
                    "Ann. Ret (%)": round(ann_ret * 100, 2),
                    "Ann. Vol (%)": round(ann_vol * 100, 2),
                    "Sharpe":       round(sharpe, 3),
                    "Days":         int(mask.sum()),
                })
        return pd.DataFrame(rows)


    # ─────────────────────────────────────────────
    # 9.  SENSITIVITY ANALYSIS — HEAT MAPS
    # ─────────────────────────────────────────────
    def sensitivity_sharpe_heatmap(
        rets: pd.DataFrame,
        rf_daily: float,
        h_vals: np.ndarray = np.linspace(0, 0.5, 11),
        vol_quantiles: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
        vol_window: int = 21,
    ) -> pd.DataFrame:
        rvol = rets["underlying"].rolling(vol_window).std().bfill() * np.sqrt(252)
        breakpoints = rvol.quantile(vol_quantiles).values
        labels = [f"Q{i+1}" for i in range(len(vol_quantiles) - 1)]

        matrix = pd.DataFrame(
            index=[f"h={h:.2f}" for h in h_vals],
            columns=labels,
            dtype=float,
        )

        for h in h_vals:
            strat = Strategy(name=f"h={h:.2f}", w_short_letf=1 - h, w_cash=h)
            r_all = compute_strategy_returns(rets, strat, rf_daily)

            for i, label in enumerate(labels):
                lo, hi = breakpoints[i], breakpoints[i + 1]
                mask   = (rvol >= lo) & (rvol <= hi)
                r_sub  = r_all[mask]
                if r_sub.empty or r_sub.std() == 0:
                    continue
                ann_ret = (1 + r_sub).prod() ** (252 / len(r_sub)) - 1
                sharpe  = (ann_ret - rf_daily * 252) / (r_sub.std() * np.sqrt(252))
                matrix.loc[f"h={h:.2f}", label] = round(sharpe, 3)

        return matrix.astype(float)


    # ─────────────────────────────────────────────
    # 10.  VISUALISATION
    # ─────────────────────────────────────────────
    def plot_all(
        daily_rets: pd.DataFrame,
        regimes: pd.Series,
        risk_table: pd.DataFrame,
        hedge_grid: pd.DataFrame,
        best_h: float,
        vol_decay_df: pd.DataFrame,
        sens_matrix: pd.DataFrame,
        regime_perf: pd.DataFrame,
        underlying: str,
        bull: str,
        bear: str,
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
        regime_colors = {"Bull": "#3fb950", "Bear": "#f78166", "Sideways": "#8b949e"}
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
            dd = (nav[col] - nav[col].cummax()) / nav[col].cummax() * 100
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
            rs = (daily_rets[col].rolling(63).mean() /
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
        fig_hedge = go.Figure()
        fig_hedge.add_trace(go.Scatter(
            x=hedge_grid["h"], y=hedge_grid["Sharpe"], mode="lines", name="Sharpe",
            line=dict(color=PALETTE_[0], width=2), showlegend=False,
            customdata=(hedge_grid["MDD"] * 100),
            hovertemplate="h %{x:.2f}<br>Sharpe %{y:.2f}<br>Max DD %{customdata:.1f}%<extra></extra>"))
        fig_hedge.add_vline(x=best_h, line=dict(color="#d2a8ff", width=1.5, dash="dot"),
                            annotation_text=f"opt h = {best_h:.2f}", annotation_position="top")
        _style(fig_hedge, 340, "Hedge-Ratio Optimisation (Sharpe)")
        fig_hedge.update_xaxes(title_text="Cash Hedge Ratio (h)")
        fig_hedge.update_yaxes(title_text="Sharpe")

        # ── 7. Volatility decay ──────────────────────────────────────────────
        fig_vd = go.Figure(go.Scatter(
            x=vol_decay_df["Annual Vol (%)"], y=vol_decay_df["3× LETF Terminal"],
            mode="lines", line=dict(color=PALETTE_[3], width=2), showlegend=False,
            fill="tozeroy", fillcolor=_rgba(PALETTE_[3], 0.08),
            hovertemplate="vol %{x:.0f}%<br>terminal %{y:.2f}<extra></extra>"))
        fig_vd.add_hline(y=1.0, line=dict(color="#8b949e", width=0.8, dash="dash"),
                         annotation_text="Break-even", annotation_position="bottom right")
        _style(fig_vd, 340, "Volatility Decay: 3× LETF (μ=0)", hover="x")
        fig_vd.update_xaxes(title_text="Annual Vol (%)")
        fig_vd.update_yaxes(title_text="Terminal NAV (1yr)")

        # ── 8. Sensitivity heatmap ───────────────────────────────────────────
        zvals = sens_matrix.values.astype(float)
        fig_heat = go.Figure(go.Heatmap(
            z=zvals, x=list(sens_matrix.columns), y=list(sens_matrix.index),
            colorscale="RdYlGn", zmid=0, text=zvals, texttemplate="%{text:.2f}",
            textfont=dict(size=9, color="#0d1117"),
            colorbar=dict(title=dict(text="Sharpe", font=dict(size=9)),
                          thickness=12, len=0.9, tickfont=dict(size=8)),
            hovertemplate="%{y} · %{x}<br>Sharpe %{z:.2f}<extra></extra>"))
        # Taller: 11 hedge-ratio rows need vertical room for the cell labels.
        _style(fig_heat, 460, "Sharpe: Hedge Ratio × Vol Quartile", hover="closest")
        fig_heat.update_yaxes(title_text="Hedge Ratio (h)", autorange="reversed")
        fig_heat.update_xaxes(title_text="Vol Quartile")

        # ── 9. Regime-conditional Sharpe ─────────────────────────────────────
        fig_reg = go.Figure()
        pivot = regime_perf.pivot(index="Strategy", columns="Regime", values="Sharpe").reindex(
            columns=["Bull", "Bear", "Sideways"])
        rc = {"Bull": PALETTE_[1], "Bear": PALETTE_[2], "Sideways": PALETTE_[5]}
        xs2 = [_short(s) for s in pivot.index]
        for reg in ["Bull", "Bear", "Sideways"]:
            if reg in pivot.columns:
                fig_reg.add_trace(go.Bar(x=xs2, y=pivot[reg].astype(float),
                                         name=reg, marker_color=rc[reg]))
        _style(fig_reg, 340, "Regime-Conditional Sharpe by Strategy", hover="x")
        fig_reg.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)

        # ── Header + interactive layout (mo.ui.plotly = pan/zoom/hover) ──────
        header = mo.Html(
            f"""
            <div style="display:flex;align-items:baseline;gap:16px;
                        border-bottom:1px solid #30363d;padding:0 0 12px;margin:8px 0 6px;">
              <span style="font-family:'DM Mono',monospace;font-size:22px;font-weight:600;color:#e6edf3;">Optimal Hedging Strategy</span>
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
            mo.ui.plotly(fig_hedge),
            mo.ui.plotly(fig_vd),
            mo.ui.plotly(fig_heat),
            mo.ui.plotly(fig_reg),
        ])


    # ─────────────────────────────────────────────
    # 11.  REPORT PRINTER
    # ─────────────────────────────────────────────
    def print_risk_table(risk_table: pd.DataFrame):
        mo.output.append(
            mo.vstack([
                mo.md("### 📊 Risk Metrics Table"),
                mo.ui.table(risk_table.reset_index()),
            ])
        )
        # pd.set_option("display.max_columns", None)
        # pd.set_option("display.width", 160)
        # pd.set_option("display.float_format", "{:.3f}".format)
        # print("\n" + "═" * 120)
        # print("  RISK METRICS TABLE")
        # print("═" * 120)
        # print(risk_table.to_string())
        # print("═" * 120)


    # ─────────────────────────────────────────────
    # 12.  DAILY STRATEGY PICKER
    # ─────────────────────────────────────────────
    def pick_best_strategy_today(
        daily_rets: pd.DataFrame,
        regimes: pd.Series,
        regime_perf: pd.DataFrame,
        rolling_window: int = 63,
        min_active_days: int = 15,        # need at least this many active obs to trust the signal
        require_agreement: bool = True,
        sharpe_agreement_threshold: float = -0.5,
    ) -> Tuple[str, dict]:
        """
        FIX: rolling Sharpe is now computed only on each strategy's own ACTIVE
        days within the lookback window (not the flat rf_daily cash-fill days),
        so strategies that have been mostly idle recently don't get an
        artificially exploded/collapsed Sharpe from near-zero variance.
        """
        strategy_regime_map = {
            f"Bear-Regime: Long {bear} + Short {underlying}": "Bear",
            f"Bull-Regime: Long {bull} + Long {underlying}":  "Bull",
            SIDEWAYS_NAME:                                    "Sideways",
        }

        today_regime = regimes.iloc[-1]
        # Only real, tradeable sleeves are candidates. `daily_rets` also carries
        # the aggregate "Combined-Switched System" column, which has no legs in
        # the trade log — picking it would yield a recommendation we can't trade.
        regime_subset = regime_perf[
            (regime_perf["Regime"] == today_regime)
            & (regime_perf["Strategy"].isin(strategy_regime_map))
        ]

        if regime_subset.empty:
            regime_best = "HOLD CASH"
        else:
            regime_best = (regime_subset
                           .sort_values("Sharpe", ascending=False)
                           .iloc[0]["Strategy"])

        recent_window = daily_rets.tail(rolling_window)
        recent_regimes = regimes.tail(rolling_window)

        rolling_sharpe = {}
        active_counts = {}
        for col in daily_rets.columns:
            label = strategy_regime_map.get(col)
            if label is None:
                continue
            active_mask = recent_regimes == label
            active_rets = recent_window[col][active_mask]
            active_counts[col] = int(active_mask.sum())

            if len(active_rets) < min_active_days or active_rets.std() == 0:
                rolling_sharpe[col] = np.nan   # not enough signal to trust
                continue

            ann_ret = (1 + active_rets).prod() ** (252 / len(active_rets)) - 1
            ann_vol = active_rets.std() * np.sqrt(252)
            rolling_sharpe[col] = (ann_ret - 0) / ann_vol  # already excess vs rf since active-only

        rolling_sharpe = pd.Series(rolling_sharpe).dropna()

        if rolling_sharpe.empty:
            # No strategy has enough recent active days to judge — trust regime pick alone
            chosen = regime_best
            signal = "INSUFFICIENT ACTIVE-DAY DATA — defaulting to regime-best"
            rolling_best = None
        else:
            rolling_best = rolling_sharpe.idxmax()
            if regime_best == rolling_best:
                chosen = regime_best
                signal = "AGREE — exact match"
            elif regime_best in rolling_sharpe.index:
                sharpe_gap = rolling_sharpe[regime_best] - rolling_sharpe.max()
                if sharpe_gap > sharpe_agreement_threshold:
                    chosen = regime_best
                    signal = f"AGREE (within threshold) — gap={sharpe_gap:+.3f} Sharpe"
                else:
                    chosen = "HOLD CASH"
                    signal = f"DISAGREE — gap={sharpe_gap:+.3f} Sharpe exceeds threshold"
            else:
                # regime_best had too few active days this window to even be scored
                chosen = regime_best
                signal = "Regime pick has insufficient recent active days to compare — trusting regime"

        detail = {
            "today_regime":          today_regime,
            "regime_best_strategy":  regime_best,
            "rolling_best_strategy": rolling_best,
            "rolling_sharpes":       rolling_sharpe.sort_values(ascending=False).round(3).to_dict(),
            "active_day_counts":     active_counts,
            "signal":                signal,
            "chosen_strategy":       chosen,
        }
        return chosen, detail


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
                mo.callout(mo.md("💵 **No trades today** — signals disagree or holding cash."), kind="warn")
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

    def regime_has_persisted(regimes: pd.Series, min_days: int = 3) -> bool:
        """True if the current regime has held for at least min_days in a row."""
        current = regimes.iloc[-1]
        run_length = 0
        for r in reversed(regimes.values):
            if r != current:
                break
            run_length += 1
        return run_length >= min_days

    # ─────────────────────────────────────────────
    # 13.  MAIN ORCHESTRATOR
    # ─────────────────────────────────────────────
    def main(
        use_live_data: bool = True,
        underlying: str = underlying,
        letf_ticker: str = bear,
        bull_letf_ticker: str = bull,
        start: str = "2020-01-01",
        end: str = None,                 # FIX: was hardcoded "2024-12-31"; now defaults to today
        initial_nav_cap: float = 10_000.0,
    ):
        if end is None:
            end = datetime.date.today().isoformat()

        # ── Load data ──────────────────────────────────────────────────────────
        if use_live_data:
            try:
                rets, rf_daily, raw = fetch_data(underlying, letf_ticker, bull_letf_ticker, start, end)
            except Exception as e:
                log(f"[Warning] Live data failed ({e}). Falling back to synthetic data.")
                rets, rf_daily, raw = simulate_synthetic_data()
        else:
            rets, rf_daily, raw = simulate_synthetic_data()

        # ── Regimes ────────────────────────────────────────────────────────────
        regimes = classify_regimes(rets)
        mo.output.append(page_header("EDA", "03"))
        mo.output.append(
        mo.vstack([
            mo.md("#### 📊 Regime Distribution"),
            mo.ui.table(
                regimes.value_counts()
                       .reset_index()
                       .rename(columns={"regime": "Regime", "count": "Days"})
            ),
        ])
        )
        # print(regimes.value_counts().to_string(), "\n")

        # ── Volatility-target scalar (causal: trailing realised vol only) ──────
        if VOL_TARGETING:
            _rvol_ann  = rets["underlying"].rolling(21).std().bfill() * np.sqrt(252)
            vol_scalar = (TARGET_ANN_VOL / _rvol_ann.replace(0, np.nan)) \
                            .clip(VOL_SCALAR_MIN, VOL_SCALAR_MAX).fillna(1.0)
            log(f"[Sizing] Vol targeting ON — target {TARGET_ANN_VOL:.0%} ann. vol, "
                f"scalar range [{vol_scalar.min():.2f}, {vol_scalar.max():.2f}], "
                f"mean {vol_scalar.mean():.2f}")
        else:
            vol_scalar = None

        # ── Backtest ───────────────────────────────────────────────────────────
        log("[Backtest] Running strategies …")
        daily_rets = run_backtest(rets, STRATEGIES, rf_daily, regimes=regimes,
                                  vol_scalar=vol_scalar)

        # The switched system you'd actually trade — measured alongside the
        # individual sleeves (each of which idles in cash outside its regime).
        daily_rets["Combined-Switched System"] = build_combined_returns(
            daily_rets, regimes, rf_daily,
            min_persist=MIN_REGIME_PERSIST,
            use_agreement_gate=True,          # same rule the live picker applies
            rolling_window=63,
            min_active_days=15,
            sharpe_agreement_threshold=-0.5,
        )

        # ── Risk table ─────────────────────────────────────────────────────────
        risk_table = compute_risk_table(daily_rets, rf_daily)
        print_risk_table(risk_table)

        # ── Hedge-ratio optimisation ─────────────────────────────────────────
        log("\n[Optimisation] Scanning hedge ratios …")
        # print("\n[Optimisation] Scanning hedge ratios …")
        best_h, hedge_grid = optimize_hedge_ratio(rets, rf_daily, objective="sharpe")
        log(f"  → Optimal cash-hedge ratio (max Sharpe): h = {best_h:.3f}  "
              f"({best_h*100:.1f}% cash, {(1-best_h)*100:.1f}% short LETF)")

        # ── Volatility decay ─────────────────────────────────────────────────
        vol_decay_df = quantify_vol_decay(leverage=3.0)

        # ── Sensitivity heatmap ──────────────────────────────────────────────
        sens_matrix = sensitivity_sharpe_heatmap(rets, rf_daily)

        # ── Regime-conditional performance ───────────────────────────────────
        log("[Analysis] Regime-conditional performance …")
        regime_perf = regime_performance(daily_rets, regimes, rf_daily)
        mo.output.append(
        mo.vstack([
            mo.md("#### 📈 Regime-Conditional Performance"),
            mo.ui.table(regime_perf),
        ])
        )

        log("\n[Analysis] Volatility decay curve …")
        log("[Analysis] Sensitivity heatmap …")
        # ── Plot ───────────────────────────────────────────────────────────────
        fig = plot_all(
            daily_rets=daily_rets,
            regimes=regimes,
            risk_table=risk_table,
            hedge_grid=hedge_grid,
            best_h=best_h,
            vol_decay_df=vol_decay_df,
            sens_matrix=sens_matrix,
            regime_perf=regime_perf,
            underlying=underlying,
            bull=bull_letf_ticker,
            bear=letf_ticker,
        )
        mo.output.append(fig)

        log("\n[Done] All analyses complete.")

        # ── Trade log ──────────────────────────────────────────────────────────
        log("\n[TradeLog] Generating daily action log …")

        try:
            actual_nav = get_account_nav(max_position=initial_nav_cap)
        except Exception as e:
            log(f"[Warning] Could not fetch live account NAV ({e}). Using fallback ${initial_nav_cap:,.2f}.")
            actual_nav = initial_nav_cap

        trade_log = generate_trade_log(
            rets=rets,
            raw=raw,
            daily_rets=daily_rets,
            rf_daily=rf_daily,
            underlying=underlying,
            bear=letf_ticker,
            bull=bull_letf_ticker,
            initial_nav=actual_nav,
            max_notional_per_leg=5_000.0,   # ← ~$5k cap per leg
            vol_scalar=vol_scalar,
        )

        print_trade_log_summary(trade_log, tail_days=1)

        # ── Daily recommendation ─────────────────────────────────────────────
        log("\n[Recommendation] Picking best strategy for today …")

        if not regime_has_persisted(regimes, min_days=3):
            log("[Recommendation] Current regime hasn't persisted long enough — overriding to HOLD CASH.")
            chosen = "HOLD CASH"
            # FIX: `detail` must be defined on this branch too, or the calls below
            # (print_daily_recommendation / the return dict) crash with NameError.
            detail = {
                "today_regime":          regimes.iloc[-1],
                "regime_best_strategy":  "HOLD CASH",
                "rolling_best_strategy": None,
                "rolling_sharpes":       {},
                "active_day_counts":     {},
                "signal":                "Regime has not persisted ≥3 days — holding cash.",
                "chosen_strategy":       "HOLD CASH",
            }
        else:
            chosen, detail = pick_best_strategy_today(
                daily_rets=daily_rets,
                regimes=regimes,
                regime_perf=regime_perf,
                rolling_window=63,
                require_agreement=True,
                sharpe_agreement_threshold=-0.5,
            )

        print_daily_recommendation(chosen, detail, trade_log)

        return {
            "daily_returns":         daily_rets,
            "risk_table":            risk_table,
            "hedge_grid":            hedge_grid,
            "best_h":                best_h,
            "vol_decay":             vol_decay_df,
            "sensitivity":           sens_matrix,
            "regime_perf":           regime_perf,
            "trade_log":             trade_log,
            "chosen_strategy":       chosen,
            "recommendation_detail": detail,
        }


    # ─────────────────────────────────────────────
    # Marimo cells execute top-to-bottom on reactivity; the __name__ == "__main__"
    # guard is a script-only pattern and won't reliably gate execution in a
    # notebook cell, so call main() directly.
    # ─────────────────────────────────────────────
    results = main(
        use_live_data=True,
        underlying=selected_row["Underlying"],
        letf_ticker=selected_row["Bear_3x"],
        bull_letf_ticker=selected_row["Bull_3x"],
        start="2019-12-31",
        end=datetime.date.today().isoformat(),
    )
    return ALPACA_API_KEY, ALPACA_API_SECRET, log, results


@app.cell
def _(mo, results):
    trade_log = results["trade_log"]

    today_df = trade_log[
        (trade_log["date"] == trade_log["date"].iloc[-1]) &
        (trade_log["strategy"] == results["chosen_strategy"])
    ]

    if not today_df.empty:
        mo.vstack([
            mo.md(f"#### 🗓️ Trade Actions for `{today_df['date'].iloc[-1].date()}`"),
            mo.ui.table(today_df.reset_index(drop=True)),
        ])
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
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    MarketOrderRequest,
    OrderSide,
    TimeInForce,
    TradingClient,
    exec_button,
    log,
    mo,
    results,
    today_df,
):
    mo.stop(not exec_button.value,
            mo.md("*Press **⚡ Submit Today's LETF Orders** to place orders.*"))

    trade_client = TradingClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_API_SECRET,
        paper=True
    )

    def _current_position_qty(symbol: str) -> float:
        """Signed shares currently held (negative = short); 0 if no position."""
        try:
            return float(trade_client.get_open_position(symbol).qty)
        except Exception:
            return 0.0

    # Every ticker this notebook's strategies can trade. Reconciling over the
    # whole universe (not just today's legs) means a position left over from a
    # switched-away strategy is driven back to target 0 — i.e. closed out.
    managed_tickers = list(results["trade_log"]["ticker"].unique())

    # Today's target position per ticker (chosen strategy only). Anything absent
    # defaults to target 0 below and gets flattened.
    target_by_ticker = {}
    for _sym, _tq in zip(today_df["ticker"], today_df["quantity"]):
        target_by_ticker[_sym] = target_by_ticker.get(_sym, 0.0) + float(_tq)

    # Reconcile each managed ticker against the actual broker position, ordering
    # only the difference. Submitting the absolute target (the previous behavior)
    # re-opened the full position every run and fired the wrong side when a
    # position needed reducing. Whole shares only — leveraged ETFs and short
    # sales don't allow fractional.
    for symbol in managed_tickers:
        target  = int(round(target_by_ticker.get(symbol, 0.0)))
        current = int(round(_current_position_qty(symbol)))
        delta   = target - current

        if delta == 0:
            continue   # already at target (includes flat tickers we don't hold)

        side    = OrderSide.BUY if delta > 0 else OrderSide.SELL
        abs_qty = abs(delta)

        req = MarketOrderRequest(
            symbol=symbol,
            qty=abs_qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )

        try:
            order = trade_client.submit_order(req)
            log(
                f"✅ `{side.value.upper()} {symbol}` — "
                f"qty={abs_qty} (target {target}, was {current}) | order_id=`{order.id}`",
                kind="success"
            )
        except Exception as e:
            log(f"❌ Order failed for `{symbol}`: {e}", kind="danger")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
