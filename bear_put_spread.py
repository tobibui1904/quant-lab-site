import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", css_file="theme.css", html_head_file="theme_head.html")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from scipy.stats import norm
    from scipy.optimize import brentq
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from dotenv import load_dotenv
    import os
    import requests
    from typing import Any, Dict, List, Optional, Tuple
    import marimo as mo

    # Load .env from the notebook's own directory. Bare load_dotenv() searches
    # upward from the *working* directory, so it silently finds nothing when
    # marimo is launched from elsewhere.
    load_dotenv(mo.notebook_dir() / ".env")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import alpaca
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient, StockLatestTradeRequest
    from alpaca.data.requests import StockBarsRequest, OptionLatestQuoteRequest, OptionSnapshotRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        GetOptionContractsRequest,
        LimitOrderRequest,
        OptionLegRequest,
        ClosePositionRequest,
    )
    from alpaca.trading.enums import (
        AssetStatus,
        OrderSide,
        OrderClass,
        TimeInForce,
        ContractType,
    )
    from alpaca.trading import OrderStatus

    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return (
        Any,
        AssetStatus,
        ContractType,
        Dict,
        GetOptionContractsRequest,
        LimitOrderRequest,
        List,
        OptionHistoricalDataClient,
        OptionLatestQuoteRequest,
        OptionLegRequest,
        OrderClass,
        OrderSide,
        StockBarsRequest,
        StockHistoricalDataClient,
        StockLatestTradeRequest,
        TimeFrame,
        TimeFrameUnit,
        TimeInForce,
        TradingClient,
        ZoneInfo,
        brentq,
        datetime,
        logger,
        mo,
        norm,
        np,
        os,
        pd,
        requests,
        timedelta,
    )


@app.cell
def _():
    with open('pair.txt', 'r') as f:
        content = f.read()
        lines = content.splitlines()
        pair = lines[0].split(" ")
    return (pair,)


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
          <div style="width: 2px; height: 52px; background: #C0492A;"></div>
        </div>
        <div>
          <div style="font-family: 'DM Mono', monospace; font-size: 11px; color: #C0492A; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 6px;">Section {section_number}</div>
          <div style="font-family: 'DM Serif Display', serif; font-size: 42px; font-style: italic; font-weight: 400; line-height: 1.05; color: var(--color-text-primary);">{title}</div>
          {subtitle_html}
        </div>
      </div>
    </div>
    """)

    return (page_header,)


@app.cell
def _(mo):
    def section_header(title, number):
        return mo.Html(f"""
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <div style="padding: 1.5rem 0 1rem; display: flex; align-items: center; gap: 16px;">
      <div style="font-family: 'DM Serif Display', serif; font-size: 26px; font-style: italic; font-weight: 400; color: var(--color-text-primary); white-space: nowrap;">{title}</div>
      <div style="flex: 1; height: 0.5px; background: var(--color-border-tertiary);"></div>
      <div style="font-family: 'DM Mono', monospace; font-size: 10px; color: #C0492A; letter-spacing: 0.18em; text-transform: uppercase; white-space: nowrap;">{number}</div>
    </div>
    """)

    return (section_header,)


@app.cell
def _(mo):
    mo.Html("""
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

    <div style="padding: 2.5rem 0 2rem; text-align: center;">
      <div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; border-radius: 50%; border: 0.5px solid #C0492A; margin-bottom: 1rem;">
        <i class="ti ti-chart-candle" style="font-size: 22px; color: #C0492A;"></i>
      </div>
      <h1 style="font-family: 'DM Serif Display', serif; font-size: 36px; font-weight: 400; font-style: italic; margin: 0 0 6px; letter-spacing: -0.01em; color: var(--color-text-primary);">Bear Put Spread Trading Strategy</h1>
      <p style="font-family: 'DM Mono', monospace; font-size: 11px; color: var(--color-text-secondary); letter-spacing: 0.18em; text-transform: uppercase; margin: 0 0 1.25rem;">Recommended by AI</p>
      <div style="display: inline-flex; align-items: center; gap: 6px; font-family: 'DM Mono', monospace; font-size: 11px; color: #993C1D; background: #FAECE7; padding: 4px 14px; border-radius: 999px;">
        <span style="width: 6px; height: 6px; border-radius: 50%; background: #C0492A; display: inline-block;"></span>
        Live
      </div>
      <div style="margin-top: 1.5rem; width: 40px; height: 0.5px; background: var(--color-border-tertiary); margin-left: auto; margin-right: auto;"></div>
    </div>
    """)
    return


@app.cell
def _(page_header):
    page_header("Available pair and Selection", "01")
    return


@app.cell
def _(logger, os, requests):
    def get_risk_free_rate() -> float:
        """Fetch latest 3-month T-bill rate from FRED (DGS3MO series)."""
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": "DGS3MO",
                # From .env (see .env.example). If unset, the request fails and
                # the except-branch below falls back to a default rate.
                "api_key": os.getenv("FRED_API_KEY"),
                "sort_order": "desc",
                "limit": 1,
                "file_type": "json",
            }
            r = requests.get(url, params=params, timeout=5)
            r.raise_for_status()
            value = r.json()["observations"][0]["value"]
            return float(value) / 100  # FRED returns percent, e.g. "3.75"
        except Exception as e:
            logger.warning(f"Failed to fetch risk-free rate from FRED, defaulting to 0.0375: {e}")
            return 0.0375  # fallback to current approximate rate

    risk_free_rate_fred = get_risk_free_rate()
    return (risk_free_rate_fred,)


@app.cell
def _(
    OptionHistoricalDataClient,
    StockHistoricalDataClient,
    TradingClient,
    os,
):
    # API credentials for Alpaca — loaded from .env (see .env.example).
    # No hardcoded fallback: secrets must never live in source that git can see.
    API_KEY = os.getenv('ALPACA_API_KEY')
    API_SECRET = os.getenv('ALPACA_API_SECRET')
    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            "Missing Alpaca credentials. Copy .env.example to .env and set "
            "ALPACA_API_KEY and ALPACA_API_SECRET."
        )
    BASE_URL = None
    ## We use paper environment for this example (Please do not modify this. This example is for paper trading only)
    PAPER = True

    # Initialize Alpaca clients
    trade_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)
    option_historical_data_client = OptionHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET, url_override=BASE_URL)
    stock_data_client = StockHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)

    # Below are the variables for development this documents
    # Please do not change these variables
    trade_api_url = None
    trade_api_wss = None
    data_api_url = None
    option_stream_data_wss = None
    return option_historical_data_client, stock_data_client, trade_client


@app.cell
def _(brentq, logger, norm, np):
    def calculate_implied_volatility(option_price, S, K, T, r, option_type):
        intrinsic_value = max(0, (S - K) if option_type == 'call' else (K - S))

        # Option price at or below intrinsic — IV is essentially 0
        if option_price <= intrinsic_value + 1e-6:
            return 0.0

        def option_price_diff(sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return price - option_price

        # Dynamically find a valid bracket instead of assuming [1e-6, 5.0] works
        sigma_lower = 1e-6
        sigma_upper = 5.0
        try:
            f_lower = option_price_diff(sigma_lower)
            f_upper = option_price_diff(sigma_upper)

            # If signs are the same, try to widen the upper bound
            if f_lower * f_upper > 0:
                for upper in [8.0, 15.0, 30.0]:
                    if option_price_diff(upper) * f_lower < 0:
                        sigma_upper = upper
                        break
                else:
                    logger.debug(f"IV bracket failed for price={option_price:.4f} S={S:.2f} K={K:.2f} T={T:.4f}")
                    return None

            return brentq(option_price_diff, sigma_lower, sigma_upper)
        except ValueError as e:
            logger.debug(f"IV solver error: {e} | price={option_price:.4f} S={S:.2f} K={K:.2f}")
            return None

    return (calculate_implied_volatility,)


@app.cell
def _(calculate_implied_volatility, norm, np, pd):
    def years_to_expiry(expiration) -> float:
        """Time to expiry in years, to sub-day precision.

        Using `.days` truncates to whole days, which collapses anything expiring
        today to ~0 and destroys IV/Greeks for short-dated contracts.
        """
        seconds = (expiration - pd.Timestamp.now()).total_seconds()
        return max(seconds / (365.0 * 24 * 3600), 0.0)

    def calculate_greeks(option_price, strike_price, expiration, underlying_price,
                         risk_free_rate, option_type, T=None):
        """Return (delta, gamma, theta, vega, IV).

        IV is returned so callers don't have to run a second brentq solve.
        """
        if T is None:
            T = years_to_expiry(expiration)

        if T <= 1e-6:
            print('Option has expired or is expiring now; setting Greeks based on intrinsic value.')
            if option_type == 'put':
                delta = -1.0 if underlying_price < strike_price else 0.0
            else:
                delta = 1.0 if underlying_price > strike_price else 0.0
            return delta, 0.0, 0.0, 0.0, 0.0

        # Calculate IV
        IV = calculate_implied_volatility(option_price, underlying_price, strike_price, T, risk_free_rate, option_type)

        if IV is None or IV == 0.0:
            print('Implied volatility could not be determined, skipping Greek calculations.')
            return None

        d1 = (np.log(underlying_price / strike_price) + (risk_free_rate + 0.5 * IV ** 2) * T) / (IV * np.sqrt(T))
        d2 = d1 - IV * np.sqrt(T) # d2 for Theta calculation
        # Calculate Delta
        delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
        # Calculate Gamma
        gamma = norm.pdf(d1) / (underlying_price * IV * np.sqrt(T))
        # Calculate Vega
        vega = underlying_price * np.sqrt(T) * norm.pdf(d1) / 100
        # Calculate Theta
        if option_type == 'call':
            theta = (
                - (underlying_price * norm.pdf(d1) * IV) / (2 * np.sqrt(T))
                - (risk_free_rate * strike_price * np.exp(-risk_free_rate * T) * norm.cdf(d2))
            )
        else:
            theta = (
                - (underlying_price * norm.pdf(d1) * IV) / (2 * np.sqrt(T))
                + (risk_free_rate * strike_price * np.exp(-risk_free_rate * T) * norm.cdf(-d2))
            )
        # Convert annualized theta to daily theta
        theta /= 365

        # IV is part of the contract: callers unpack 5 values.
        return delta, gamma, theta, vega, IV

    return (calculate_greeks,)


@app.cell
def _(mo, pair):
    symbol_select = mo.ui.radio(
        options={pair[0]: pair[0], pair[1]: pair[1]},
        value=None,
        label="Select underlying symbol",
    )
    symbol_select
    return (symbol_select,)


@app.cell
def _(
    StockBarsRequest,
    StockLatestTradeRequest,
    TimeFrame,
    TimeFrameUnit,
    ZoneInfo,
    datetime,
    mo,
    np,
    risk_free_rate_fred,
    stock_data_client,
    symbol_select,
    timedelta,
    trade_client,
):
    mo.stop(not symbol_select.value, output = "Please select either of the pair for the Option Strategy")

    # Select the underlying stock
    underlying_symbol = symbol_select.value

    # Set the timezone
    timezone = ZoneInfo('America/New_York')

    # Get current date in US/Eastern timezone
    today = datetime.now(timezone).date()

    # Define a 6% range around the underlying price
    STRIKE_RANGE = 0.06

    # Buying power percentage to use for the trade
    BUY_POWER_LIMIT = 0.05

    # Risk free rate for the options greeks and IV calculations
    risk_free_rate = risk_free_rate_fred

    # Check account buying power
    buying_power = float(trade_client.get_account().buying_power)

    # Set the open interest volume threshold
    OI_THRESHOLD = 25

    # Calculate the limit amount of buying power to use for the trade
    buying_power_limit = buying_power * BUY_POWER_LIMIT

    # Set the expiration date range for the options
    min_expiration = today + timedelta(days=21)
    max_expiration = today + timedelta(days=60)

    # Get the latest price of the underlying stock
    def get_underlying_price(symbol):
        # Get the latest trade for the underlying stock
        underlying_trade_request = StockLatestTradeRequest(symbol_or_symbols=symbol)
        underlying_trade_response = stock_data_client.get_stock_latest_trade(underlying_trade_request)
        return underlying_trade_response[symbol].price

    # Get the latest price of the underlying stock
    underlying_price = get_underlying_price(underlying_symbol)

    # Set the minimum and maximum strike prices based on the underlying price
    min_strike = str(underlying_price * (1 - STRIKE_RANGE))
    max_strike = str(underlying_price * (1 + STRIKE_RANGE))

    # Get the historical data for the underlying stock by symbol and timeframe
    # ref. https://alpaca.markets/sdks/python/api_reference/data/option/historical.html
    def get_stock_data(underlying_symbol, days):
        today = datetime.now(timezone).date()
        req = StockBarsRequest(
            symbol_or_symbols=[underlying_symbol],
            timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Day),     # specify timeframe
            start=today - timedelta(days=days),                          # specify start datetime, default=the beginning of the current day.
        )
        return stock_data_client.get_stock_bars(req).df

    # List of stock agg objects while dropping the symbol column
    priceData = get_stock_data(underlying_symbol, days=180).reset_index(level='symbol', drop=True)

    def compute_realized_vol(price_data, short_window=21, long_window=63):
        log_returns = np.log(price_data['close'] / price_data['close'].shift(1)).dropna()
        short_vol = log_returns.rolling(short_window).std().iloc[-1] * np.sqrt(252)
        long_vol = log_returns.rolling(long_window).std().iloc[-1] * np.sqrt(252)
        # Weight toward recent vol but temper with longer-term context
        return 0.6 * short_vol + 0.4 * long_vol

    realized_vol = compute_realized_vol(priceData)

    # Put deltas are NEGATIVE, so the bands are the mirror image of the bull call
    # spread's: the long (higher-strike, closer to ITM) leg carries the larger
    # magnitude delta, the short (lower-strike, further OTM) leg the smaller one.
    criteria = {
        'short_put': ((21, 90), (realized_vol * 0.8, realized_vol * 2.5), (-0.45, -0.20), (0.05, 0.60)),
        'long_put':  ((21, 90), (realized_vol * 0.8, realized_vol * 2.5), (-0.80, -0.45), (0.05, 0.60)),
    }

    # Set target profit levels
    TARGET_PROFIT_PERCENTAGE = 0.4
    DELTA_STOP_LOSS = 0.80
    VEGA_STOP_LOSS = 0.40


    ui = mo.vstack([
        # ── Header ────────────────────────────────────────────────────────
        mo.Html(f"""
        <style>
          .scanner-wrap {{
            font-family: 'Inter', system-ui, sans-serif;
            background: #0d1117;
            border: 1px solid #21262d;
            border-radius: 10px;
            padding: 24px 28px;
            color: #e6edf3;
            max-width: 860px;
          }}

          /* ── Header ── */
          .scanner-header {{
            display: flex;
            align-items: baseline;
            gap: 12px;
            margin-bottom: 28px;
            border-bottom: 1px solid #21262d;
            padding-bottom: 16px;
          }}
          .scanner-header h2 {{
            font-size: 15px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #58a6ff;
            margin: 0;
          }}
          .scanner-badge {{
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 11px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 2px 8px;
            color: #8b949e;
            letter-spacing: 0.04em;
          }}

          /* ── Sections ── */
          .scanner-section {{
            margin-bottom: 22px;
          }}
          .scanner-section-label {{
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #6e7681;
            margin-bottom: 10px;
          }}

          /* ── Stat grid ── */
          .stat-grid {{
            display: grid;
            gap: 8px;
          }}
          .stat-grid-4 {{ grid-template-columns: repeat(4, 1fr); }}
          .stat-grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
          .stat-grid-2 {{ grid-template-columns: repeat(2, 1fr); }}

          .stat-card {{
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 6px;
            padding: 12px 14px;
            transition: border-color 0.15s;
          }}
          .stat-card:hover {{ border-color: #388bfd; }}

          .stat-label {{
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: #6e7681;
            margin-bottom: 6px;
          }}
          .stat-value {{
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 18px;
            font-weight: 700;
            color: #e6edf3;
            line-height: 1;
          }}
          .stat-value.accent   {{ color: #58a6ff; }}
          .stat-value.positive {{ color: #3fb950; }}
          .stat-value.muted    {{ color: #8b949e; }}

          /* ── Divider ── */
          .scanner-divider {{
            border: none;
            border-top: 1px solid #21262d;
            margin: 20px 0;
          }}
        </style>

        <div class="scanner-wrap">

          <!-- Header -->
          <div class="scanner-header">
            <h2>⬡ Options Scanner</h2>
            <span class="scanner-badge">CONFIGURATION</span>
          </div>

          <!-- Market snapshot -->
          <div class="scanner-section">
            <div class="scanner-section-label">Market Snapshot</div>
            <div class="stat-grid stat-grid-4">
              <div class="stat-card">
                <div class="stat-label">Underlying</div>
                <div class="stat-value accent">{symbol_select.value}</div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Price</div>
                <div class="stat-value">{underlying_price:,.2f}</div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Buying Power</div>
                <div class="stat-value positive">${buying_power:,.0f}</div>
              </div>
              <div class="stat-card">
                <div class="stat-label">BP Limit</div>
                <div class="stat-value muted">${buying_power_limit:,.0f}</div>
              </div>
            </div>
          </div>

          <hr class="scanner-divider" />

          <!-- Filter params -->
          <div class="scanner-section">
            <div class="scanner-section-label">Filter Parameters</div>
            <div class="stat-grid stat-grid-3">
              <div class="stat-card">
                <div class="stat-label">Strike Range</div>
                <div class="stat-value">±{STRIKE_RANGE:.0%}</div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Risk-Free Rate</div>
                <div class="stat-value">{risk_free_rate:.2%}</div>
              </div>
              <div class="stat-card">
                <div class="stat-label">OI Threshold</div>
                <div class="stat-value">{OI_THRESHOLD}</div>
              </div>
            </div>
          </div>

          <hr class="scanner-divider" />

          <!-- Expiry + Strike side by side -->
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:24px;">

            <div class="scanner-section" style="margin-bottom:0">
              <div class="scanner-section-label">Expiration Window</div>
              <div class="stat-grid stat-grid-2">
                <div class="stat-card">
                  <div class="stat-label">Min Expiry</div>
                  <div class="stat-value" style="font-size:14px;">{min_expiration}</div>
                </div>
                <div class="stat-card">
                  <div class="stat-label">Max Expiry</div>
                  <div class="stat-value" style="font-size:14px;">{max_expiration}</div>
                </div>
              </div>
            </div>

            <div class="scanner-section" style="margin-bottom:0">
              <div class="scanner-section-label">Strike Constraints</div>
              <div class="stat-grid stat-grid-2">
                <div class="stat-card">
                  <div class="stat-label">Min Strike</div>
                  <div class="stat-value">${float(min_strike):,.0f}</div>
                </div>
                <div class="stat-card">
                  <div class="stat-label">Max Strike</div>
                  <div class="stat-value">${float(max_strike):,.0f}</div>
                </div>
              </div>
            </div>

          </div>

        </div>
        """),
    ])

    ui
    return (
        OI_THRESHOLD,
        buying_power_limit,
        criteria,
        max_expiration,
        max_strike,
        min_expiration,
        min_strike,
        priceData,
        risk_free_rate,
        underlying_price,
        underlying_symbol,
    )


@app.cell
def _(page_header):
    page_header("Price Graph with Boilinger Bands", "02")
    return


@app.cell
def _(mo, priceData):
    import plotly.graph_objects as go

    # Bar chart for the stock price
    fig = go.Figure(data=[go.Candlestick(x=priceData.index,
                    open=priceData['open'],
                    high=priceData['high'],
                    low=priceData['low'],
                    close=priceData['close'])])

    mo.ui.plotly(fig)
    return (go,)


@app.cell
def _(priceData, underlying_price, underlying_symbol):
    # setup bollinger band calculations
    def check_bb(df, period=14, multiplier=2):
        bollinger_bands = []
        # Calculate the Simple Moving Average (SMA)
        df['SMA'] = df['close'].rolling(window=period).mean()
        # Calculate the rolling standard deviation
        df['StdDev'] = df['close'].rolling(window=period).std()
        # Calculate the Upper Bollinger Band (two standard deviation)
        df['Upper Band'] = df['SMA'] + (multiplier * df['StdDev'])
        # Calculate the Lower Bollinger Band (two standard deviation)
        df['Lower Band'] = df['SMA'] - (multiplier * df['StdDev'])
        # Get the most recent Upper Band value
        upper_bollinger_band = df['Upper Band'].iloc[-1]
        lower_bollinger_band = df['Lower Band'].iloc[-1]

        bollinger_bands = [upper_bollinger_band, lower_bollinger_band]
        return bollinger_bands

    bollinger_bands = check_bb(priceData, 14, 2)

    # For a bearish setup we care about proximity to the UPPER band: price pinned
    # near it suggests stretched upside and room to mean-revert downward.
    print(f"Latest Upper Bollinger Band is: {bollinger_bands[0]}. Latest Lower Bollinger Band is {bollinger_bands[1]}; while underlying stock '{underlying_symbol}' price is {underlying_price}.")
    return


@app.cell
def _(AssetStatus, GetOptionContractsRequest, trade_client):
    # option_type: ContractType.CALL or ContractType.PUT.
    def get_options(underlying_symbol, min_strike, max_strike, min_expiration, max_expiration, option_type):
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying_symbol],
            status=AssetStatus.ACTIVE,
            type=option_type,
            strike_price_gte=min_strike,
            strike_price_lte=max_strike,
            expiration_date_gte=min_expiration,
            expiration_date_lte=max_expiration,
        )
        return trade_client.get_option_contracts(req).option_contracts

    return (get_options,)


@app.function
def validate_sufficient_OI(option_data, OI_THRESHOLD):
    '''Ensure that the option has the required fields and sufficient open interest.'''
    if option_data.open_interest is None or option_data.open_interest_date is None:
        return False
    if float(option_data.open_interest) <= OI_THRESHOLD:
        return False
    return True


@app.cell
def _(
    OptionLatestQuoteRequest,
    calculate_greeks,
    option_historical_data_client,
    pd,
):
    # Reject a contract whose bid/ask spread exceeds this fraction of its mid.
    # Illiquid options (e.g. 0.05 x 4.00) otherwise yield a nonsense mid price.
    MAX_RELATIVE_SPREAD = 0.50

    def calculate_option_metrics(option_data, underlying_price, risk_free_rate):
        """
        Calculate key option metrics including option price, implied volatility (IV), and option Greeks.
        """
        # Retrieve the latest quote for the option
        option_quote_req = OptionLatestQuoteRequest(symbol_or_symbols=option_data['symbol'])
        option_quote = option_historical_data_client.get_option_latest_quote(option_quote_req)[option_data['symbol']]

        bid = getattr(option_quote, 'bid_price', None) or 0.0
        ask = getattr(option_quote, 'ask_price', None) or 0.0

        if bid > 0 and ask > 0:
            if ask < bid:
                # Crossed market — the quote is stale/bad, don't price off it.
                return None
            option_price = (bid + ask) / 2
            # Reject unusably wide markets: a 0.05 x 4.00 quote produces a
            # meaningless mid that would poison IV and every Greek downstream.
            if option_price > 0 and (ask - bid) / option_price > MAX_RELATIVE_SPREAD:
                return None
        else:
            # `last_price` is a trade field and may be absent on a quote object.
            option_price = getattr(option_quote, 'last_price', None)

        if not option_price or option_price <= 0:
            return None

        # Calculate expiration and remaining days
        expiration_date = pd.Timestamp(option_data['expiration_date'])
        remaining_days = (expiration_date - pd.Timestamp.now()).total_seconds() / 86400.0

        # Greeks also return IV, so it is solved exactly once per contract
        # (previously brentq ran twice for every option in the chain).
        greeks = calculate_greeks(
            option_price=option_price,
            strike_price=float(option_data['strike_price']),
            expiration=expiration_date,
            underlying_price=underlying_price,
            risk_free_rate=risk_free_rate,
            option_type=option_data['type'].value
        )

        if greeks is None:
            return None

        delta, gamma, theta, vega, iv = greeks
        return {
            'option_price': option_price,
            'expiration_date': expiration_date,
            'remaining_days': remaining_days,
            'iv': iv,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }

    return (calculate_option_metrics,)


@app.function
def ensure_dict(option_data):
    """
    Convert option_data to a dict using model_dump() if available (for Pydantic models),
    otherwise return the data as-is.
    """
    if hasattr(option_data, "model_dump"):
        return option_data.model_dump()
    return option_data


@app.cell
def _(calculate_option_metrics):
    def build_option_dict(option_data, underlying_price, risk_free_rate):
        """
        Build an option dictionary by merging raw option data with calculated metrics.
        """
        option_data = ensure_dict(option_data)  # Convert to dict if necessary
        metrics = calculate_option_metrics(option_data, underlying_price, risk_free_rate)
        if metrics is None:
            return None

        candidate = {
            'id': option_data['id'],
            'name': option_data['name'],
            'symbol': option_data['symbol'],
            # Normalize to float: the API may hand back a string, and downstream
            # strike comparisons against a float price would then TypeError (or,
            # worse, compare lexicographically so "95" > "100").
            'strike_price': float(option_data['strike_price']),
            'root_symbol': option_data['root_symbol'],
            'underlying_symbol': option_data['underlying_symbol'],
            'underlying_asset_id': option_data['underlying_asset_id'],
            'close_price': option_data['close_price'],
            'close_price_date': option_data['close_price_date'],
            'expiration_date': metrics['expiration_date'],
            'remaining_days': metrics['remaining_days'],
            'open_interest': option_data['open_interest'],
            'open_interest_date': option_data['open_interest_date'],
            'size': option_data['size'],
            'status': option_data['status'],
            'style': option_data['style'],
            'tradable': option_data['tradable'],
            'type': option_data['type'],
            'initial_IV': metrics['iv'],
            'initial_delta': metrics['delta'],
            'initial_gamma': metrics['gamma'],
            'initial_theta': metrics['theta'],
            'initial_vega': metrics['vega'],
            'initial_option_price': metrics['option_price'],
        }

        return candidate

    return (build_option_dict,)


@app.cell
def _(logger):
    def check_candidate_option_conditions(candidate, criteria, label):
        """
        Check whether a candidate option meets the filtering criteria.
        The criteria is a tuple of (expiration_range, iv_range, delta_range, vega_range).
        Logs detailed information if a candidate fails a criterion.
        """
        expiration_range, iv_range, delta_range, vega_range = criteria

        # Guard for missing values FIRST — comparing None to a float raises
        # TypeError in Python 3, so these checks must precede every range test.
        for field in ('remaining_days', 'initial_IV', 'initial_delta', 'initial_vega'):
            if candidate.get(field) is None:
                logger.debug(f"{candidate['symbol']} skipped for {label}: missing {field}.")
                return False

        if not (expiration_range[0] <= candidate['remaining_days'] <= expiration_range[1]):
            logger.debug(f"{candidate['symbol']} fails expiration condition for {label}: remaining_days {candidate['remaining_days']} not in {expiration_range}.")
            return False
        if not (iv_range[0] <= candidate['initial_IV'] <= iv_range[1]):
            logger.debug(f"{candidate['symbol']} fails IV condition for {label}: initial_IV {candidate['initial_IV']} not in {iv_range}.")
            return False
        # Put deltas are negative; delta_range is already expressed as a signed
        # (low, high) band, so the same inclusive comparison works unchanged.
        if not (delta_range[0] <= candidate['initial_delta'] <= delta_range[1]):
            logger.debug(f"{candidate['symbol']} fails delta condition for {label}: initial_delta {candidate['initial_delta']} not in {delta_range}.")
            return False
        if not (vega_range[0] <= candidate['initial_vega'] <= vega_range[1]):
            logger.debug(f"{candidate['symbol']} fails vega condition for {label}: initial_vega {candidate['initial_vega']} not in {vega_range}.")
            return False

        return True

    return (check_candidate_option_conditions,)


@app.cell
def _(logger):
    def pair_put_candidates(short_puts, long_puts, underlying_price):
        """
        Build the BEST bear put spread from the candidate legs.

        Mirror image of the bull call spread: a bear put spread requires
        short_strike < long_strike (buy the higher-strike put, sell the
        lower-strike one). Moneyness is already governed by the delta bands
        in `criteria`, so no additional spot-between-strikes constraint.

        Every valid pair is scored on reward:risk = (width - debit) / debit and
        the best one is returned, so the choice does not depend on arbitrary
        API ordering.
        """
        best, best_score = None, None

        for sp in short_puts:
            for lp in long_puts:
                if sp['expiration_date'] != lp['expiration_date']:
                    continue
                if sp['symbol'] == lp['symbol']:
                    continue
                # Long leg is the HIGHER strike for a bear put spread.
                if not (sp['strike_price'] < lp['strike_price']):
                    continue

                debit = lp['initial_option_price'] - sp['initial_option_price']
                width = lp['strike_price'] - sp['strike_price']
                if debit <= 0 or width <= debit:
                    # Non-debit, or no profit potential (debit >= width).
                    continue

                score = (width - debit) / debit          # reward : risk
                if best_score is None or score > best_score:
                    best, best_score = (sp, lp), score

        if best is not None:
            sp, lp = best
            logger.info(
                f"Selected Bear put spread: short_put {sp['symbol']} and long_put "
                f"{lp['symbol']} with expiration {sp['expiration_date']} "
                f"(reward:risk {best_score:.2f})."
            )
            return sp, lp

        # If no valid pair is found, log the expiration date (if available) from the candidate lists.
        expiration_info = None
        if short_puts:
            expiration_info = short_puts[0]['expiration_date']
        elif long_puts:
            expiration_info = long_puts[0]['expiration_date']

        if expiration_info:
            logger.info(f"No valid bear put spread pair found for expiration {expiration_info} with the given candidates and underlying price conditions.")
        else:
            logger.info("No valid bear put spread pair found: no candidate data available.")

        return None, None

    return (pair_put_candidates,)


@app.cell
def _(Any, Dict, logger):
    def check_buying_power(short_put: Dict[str, Any], long_put: Dict[str, Any], buying_power_limit: float) -> None:
        """
        Calculates the total premium paid (risk) for a bear put spread and checks it against the buying power limit.
        If the buying power requirement is not met, the exception is thrown and the rest of the code is never executed.
        """
        option_size = float(short_put['size'])
        debit = long_put['initial_option_price'] - short_put['initial_option_price']

        # A bear put spread MUST be a net debit (the long leg is the higher strike
        # and therefore costs more). Never abs() this — that would hide an
        # inverted/mispriced pair by turning a credit into a plausible risk number.
        if debit <= 0:
            raise Exception(
                f'Not a debit spread: long {long_put["symbol"]} @ '
                f'{long_put["initial_option_price"]} is not priced above short '
                f'{short_put["symbol"]} @ {short_put["initial_option_price"]}.'
            )

        risk = debit * option_size
        logger.info(f"Calculated bear put spread risk: {risk}.")

        if risk > buying_power_limit:
            raise Exception('Buying power limit exceeded for a bear put spread risk.')

    return (check_buying_power,)


@app.cell
def _(
    Any,
    Dict,
    List,
    build_option_dict,
    check_buying_power,
    check_candidate_option_conditions,
    logger,
    pair_put_candidates,
    pd,
):
    def find_options_for_bear_put_spread(put_options, underlying_price, risk_free_rate, buying_power_limit, criteria, OI_THRESHOLD):
        """
        Orchestrates the workflow to build a bear put spread.
        Groups options by expiration, filters them with criteria, pairs candidates using helper functions,
        and checks buying power.

        Returns:
            A list of legs [short_put, long_put] if a valid pair is found, or an empty list otherwise.
        """
        short_put_candidates_by_exp: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
        long_put_candidates_by_exp: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}

        # Process each option candidate
        for option_data in put_options:
            if not validate_sufficient_OI(option_data, OI_THRESHOLD):
                logger.warning(f"Insufficient open interest for option {getattr(option_data, 'symbol', 'unknown')} (threshold: {OI_THRESHOLD}). Skipping candidate.")
                continue

            candidate = build_option_dict(option_data, underlying_price, risk_free_rate)
            if candidate is None:
                # SKIP this contract, don't abandon the search. Options routinely
                # fail to price (deep ITM at intrinsic, stale/wide quotes, near
                # expiry); returning here would abort the entire scan on the
                # first such contract, so a spread would almost never be found.
                logger.debug(
                    f"Skipping {getattr(option_data, 'symbol', 'unknown')}: "
                    f"could not compute price/IV/Greeks."
                )
                continue
            expiration_date = candidate['expiration_date']
            short_put_candidates_by_exp.setdefault(expiration_date, [])
            long_put_candidates_by_exp.setdefault(expiration_date, [])

            # Check each candidate for both put criteria
            if check_candidate_option_conditions(candidate, criteria['short_put'], 'short_put'):
                short_put_candidates_by_exp[expiration_date].append(candidate)
                logger.info(f"Added {candidate['symbol']} as a short put candidate for expiration {expiration_date}.")
            if check_candidate_option_conditions(candidate, criteria['long_put'], 'long_put'):
                long_put_candidates_by_exp[expiration_date].append(candidate)
                logger.info(f"Added {candidate['symbol']} as a long put candidate for expiration {expiration_date}.")

        # Process only expiration dates common to both candidate groups.
        # SORTED (nearest expiry first) — iterating a raw set gives an arbitrary,
        # run-to-run-unstable choice of expiration.
        common_expirations = sorted(
            set(short_put_candidates_by_exp) & set(long_put_candidates_by_exp)
        )
        for expiration_date in common_expirations:
            sp, lp = pair_put_candidates(short_put_candidates_by_exp[expiration_date],
                                         long_put_candidates_by_exp[expiration_date],
                                         underlying_price)
            if sp and lp:
                try:
                    check_buying_power(sp, lp, buying_power_limit)
                except Exception as e:
                    logger.error(f"Pair for expiration {expiration_date} failed buying power check: {e}")
                    continue
                logger.info(f"Selected bear put spread for expiration {expiration_date}: short {sp['symbol']}, long {lp['symbol']}.")
                return [sp, lp]

        logger.info("No valid bear put spread found.")
        return [None, None]

    return (find_options_for_bear_put_spread,)


@app.cell
def _(
    ContractType,
    OI_THRESHOLD,
    buying_power_limit,
    criteria,
    find_options_for_bear_put_spread,
    get_options,
    max_expiration,
    max_strike,
    min_expiration,
    min_strike,
    risk_free_rate,
    underlying_price,
    underlying_symbol,
):
    put_options = get_options(underlying_symbol, min_strike, max_strike, min_expiration, max_expiration, ContractType.PUT)
    sp, lp = find_options_for_bear_put_spread(put_options, underlying_price, risk_free_rate, buying_power_limit, criteria, OI_THRESHOLD)
    return lp, sp


@app.cell
def _(
    LimitOrderRequest,
    OptionLegRequest,
    OrderClass,
    OrderSide,
    TimeInForce,
    logger,
    trade_client,
):
    def place_bear_put_spread_order(short_put, long_put):
        """
        Place a bear put spread order if both short_put and long_put data are provided.
        """
        if not (short_put and long_put):
            logger.info("No valid bear put spread found.")
            return None
        try:
            # Build order legs: sell the lower-strike put and buy the higher-strike put.
            order_legs = [
                OptionLegRequest(
                    symbol=short_put['symbol'],
                    side=OrderSide.SELL,
                    ratio_qty=1
                ),
                OptionLegRequest(
                    symbol=long_put['symbol'],
                    side=OrderSide.BUY,
                    ratio_qty=1
                )]
            # Create a limit order for a multi-leg (spread) order.
            net_debit = (long_put['initial_option_price'] - short_put['initial_option_price'])
            req = LimitOrderRequest(
                qty=1,
                order_class=OrderClass.MLEG,
                time_in_force=TimeInForce.DAY,
                limit_price=round(net_debit, 2),
                legs=order_legs
            )
            res = trade_client.submit_order(req)
            logger.info("A bear put spread order placed successfully.")
            return res
        except Exception as e:
            logger.error(f"Failed to place a bear put spread order: {e}")
            return None

    return (place_bear_put_spread_order,)


@app.cell
def _(lp, place_bear_put_spread_order, sp):
    res = place_bear_put_spread_order(sp, lp)
    return (res,)


@app.cell
def _(page_header):
    page_header("Trading Operation", "03")
    return


@app.cell
def _(go, mo, np, symbol_select):
    def _parse_strike(symbol: str) -> float | None:
        """Parse strike price from OCC symbol: last 8 digits / 1000."""
        try:
            return int(symbol[-8:]) / 1000
        except Exception:
            return None


    def _fmt_dt(dt) -> str:
        """Format a datetime (aware or naive) as HH:MM:SS."""
        if dt is None:
            return "—"
        return dt.strftime("%H:%M:%S")


    def _extract_order_fields(res) -> dict:
        """
        Pull display-relevant fields out of an Alpaca Order object (or plain dict).
        Returns a flat dict safe to render.
        """
        def get(obj, key, default=None):
            try:
                return getattr(obj, key, None) or (obj.get(key) if isinstance(obj, dict) else None) or default
            except Exception:
                return default

        def enum_val(v):
            return v.value if hasattr(v, "value") else str(v) if v else "—"

        order_id   = get(res, "id")
        status     = get(res, "status")
        limit_px   = get(res, "limit_price")
        submitted  = get(res, "submitted_at")
        expires    = get(res, "expires_at")
        updated    = get(res, "updated_at")
        client_oid = get(res, "client_order_id")
        filled_qty = get(res, "filled_qty", "0")
        filled_avg = get(res, "filled_avg_price")
        legs_raw   = get(res, "legs") or []

        legs = []
        for leg in legs_raw:
            legs.append({
                "symbol":          get(leg, "symbol", "—"),
                "side":            enum_val(get(leg, "side")),
                "position_intent": enum_val(get(leg, "position_intent")),
                "status":          enum_val(get(leg, "status")),
                "qty":             get(leg, "qty", "1"),
                "ratio_qty":       get(leg, "ratio_qty", "1"),
                "asset_id":        str(get(leg, "asset_id", "")),
                "order_id":        str(get(leg, "id", "")),
                "submitted_at":    _fmt_dt(get(leg, "submitted_at")),
                "updated_at":      _fmt_dt(get(leg, "updated_at")),
            })

        return {
            "order_id":    str(order_id) if order_id else "—",
            "client_oid":  str(client_oid) if client_oid else "—",
            "status":      enum_val(status),
            "limit_price": float(limit_px) if limit_px else None,
            "submitted_at": _fmt_dt(submitted),
            "expires_at":   _fmt_dt(expires),
            "updated_at":   _fmt_dt(updated),
            "filled_qty":   filled_qty,
            "filled_avg":   float(filled_avg) if filled_avg else None,
            "legs":         legs,
        }


    def render_bear_put_spread_ui(
        long_put: dict,
        short_put: dict,
        order_result=None,
        error: str = None,
    ):
        """
        Render a styled Marimo UI card for a bear put spread order.

        Args:
            short_put:     dict with 'symbol' and 'initial_option_price' (lower strike)
            long_put:      dict with 'symbol' and 'initial_option_price' (higher strike)
            order_result:  Alpaca Order object returned by trade_client.submit_order()
            error:         error string from the except block (or None)
        """
        if not (short_put and long_put):
            return mo.callout(
                mo.md("**No valid bear put spread found.** Provide both `short_put` and `long_put`."),
                kind="warn",
            )

        # higher strike put = long put
        # lower strike put  = short put

        long_price  = long_put["initial_option_price"]
        short_price = short_put["initial_option_price"]

        net_debit = round(long_price - short_price, 2)

        long_strike  = _parse_strike(long_put["symbol"])
        short_strike = _parse_strike(short_put["symbol"])

        spread_width = (
            round(long_strike - short_strike, 2)
            if (long_strike is not None and short_strike is not None)
            else None
        )

        max_loss = round(net_debit * 100, 2)

        max_gain = (
            round((spread_width - net_debit) * 100, 2)
            if spread_width is not None
            else None
        )

        # Bear put spread breaks even BELOW the long strike.
        breakeven = (
            round(long_strike - net_debit, 2)
            if long_strike is not None
            else None
        )

        # ── Parse real order result ───────────────────────────────────────────────
        ord_fields = _extract_order_fields(order_result) if order_result else None

        # Prefer data from the real result's legs if available
        if ord_fields and ord_fields["legs"]:
            real_legs = ord_fields["legs"]
            sell_leg  = next((l for l in real_legs if l["side"] == "sell"), None)
            buy_leg   = next((l for l in real_legs if l["side"] == "buy"),  None)
        else:
            sell_leg = buy_leg = None

        # ── Status ────────────────────────────────────────────────────────────────
        if error:
            dot_color  = "#E24B4A"
            status_txt = f"Order failed: {error}"
        elif ord_fields:
            status_raw = ord_fields["status"]
            if "pending" in status_raw:
                dot_color  = "#BA7517"
                status_txt = f"Order pending — {status_raw.replace('_', ' ')}"
            elif status_raw in ("filled", "partially_filled"):
                dot_color  = "#1D9E75"
                status_txt = f"Order {status_raw.replace('_', ' ')}."
            elif status_raw in ("canceled", "expired", "rejected"):
                dot_color  = "#E24B4A"
                status_txt = f"Order {status_raw}."
            else:
                dot_color  = "#1D9E75"
                status_txt = f"Order accepted — {status_raw.replace('_', ' ')}"
        else:
            dot_color  = "#888780"
            status_txt = "Order not yet submitted."

        # ── Helpers ───────────────────────────────────────────────────────────────
        def section(label):
            return (f'<p style="font-size:11px;font-weight:500;letter-spacing:.1em;'
                    f'text-transform:uppercase;color:var(--color-text-secondary);margin:0 0 8px">{label}</p>')

        def divider():
            return '<hr style="border:none;border-top:0.5px solid var(--color-border-tertiary);margin:14px 0">'

        def field_row(key, val, val_color=None):
            vc = f"color:{val_color};" if val_color else ""
            return (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:5px 0;font-size:13px;">'
                f'<span style="color:var(--color-text-secondary)">{key}</span>'
                f'<span style="font-weight:500;font-family:\'DM Mono\',monospace;{vc}">{val}</span></div>'
            )

        def metric(label, value, color=None):
            vc = f"color:{color};" if color else ""
            return (
                f'<div style="background:var(--color-background-secondary);border-radius:8px;padding:12px 14px;">'
                f'<p style="font-size:11px;color:var(--color-text-secondary);letter-spacing:.06em;'
                f'text-transform:uppercase;margin:0 0 4px">{label}</p>'
                f'<p style="font-size:20px;font-weight:500;margin:0;{vc}">{value}</p></div>'
            )

        def leg_card(leg_data, input_put, badge_bg, badge_color, arrow, action_label):
            """Render one leg card, merging live order data with input prices."""
            symbol       = leg_data["symbol"]   if leg_data else input_put["symbol"]
            status_badge = leg_data["status"]   if leg_data else "—"
            pi           = leg_data["position_intent"] if leg_data else "—"
            sub_at       = leg_data["submitted_at"]    if leg_data else "—"
            upd_at       = leg_data["updated_at"]      if leg_data else "—"
            oid_short    = (leg_data["order_id"][:8] + "…") if (leg_data and leg_data["order_id"]) else "—"

            status_color = "#BA7517" if "pending" in (status_badge or "") else "#0F6E56"

            return f"""
            <div style="background:var(--color-background-secondary);border-radius:8px;padding:14px 16px;">
              <span style="display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:500;
                    letter-spacing:.06em;text-transform:uppercase;padding:3px 9px;border-radius:4px;
                    margin-bottom:10px;background:{badge_bg};color:{badge_color}">
                {arrow} {action_label}
              </span>
              <p style="font-size:14px;font-weight:500;color:var(--color-text-primary);margin:0 0 6px;
                        word-break:break-all">{symbol}</p>
              <p style="font-size:12px;color:var(--color-text-secondary);margin:0 0 2px">
                Premium: <strong>${input_put['initial_option_price']:.2f}</strong>
              </p>
              <p style="font-size:11px;color:var(--color-text-tertiary);margin:2px 0 0">ratio qty: 1</p>
              <div style="border-top:0.5px solid var(--color-border-tertiary);margin:10px 0 8px"></div>
              <p style="font-size:11px;color:{status_color};margin:0 0 3px">● {status_badge}</p>
              <p style="font-size:11px;color:var(--color-text-tertiary);margin:0 0 2px">intent: {pi}</p>
              <p style="font-size:11px;color:var(--color-text-tertiary);margin:0 0 2px">submitted: {sub_at}</p>
              <p style="font-size:11px;color:var(--color-text-tertiary);margin:0 0 2px">updated: {upd_at}</p>
              <p style="font-size:11px;color:var(--color-text-tertiary);margin:0">id: {oid_short}</p>
            </div>"""

        # ── Order meta rows ───────────────────────────────────────────────────────
        lp_display = f"${ord_fields['limit_price']:.2f}" if (ord_fields and ord_fields["limit_price"]) else f"${net_debit:.2f} (computed)"
        oid_display = (ord_fields["order_id"][:18] + "…") if ord_fields else "—"
        filled_display = ord_fields["filled_qty"] if ord_fields else "0"
        filled_avg_display = f"${ord_fields['filled_avg']:.2f}" if (ord_fields and ord_fields["filled_avg"]) else "—"
        sub_display  = ord_fields["submitted_at"] if ord_fields else "—"
        exp_display  = ord_fields["expires_at"]   if ord_fields else "—"
        upd_display  = ord_fields["updated_at"]   if ord_fields else "—"

        html = f"""
        <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
        <div style="font-family:'DM Mono',monospace;padding:1.5rem 0">

          <p style="font-family:'DM Serif Display',serif;font-size:22px;
                    color:var(--color-text-primary);margin:0 0 4px">Bear Put Spread</p>
          <p style="font-size:12px;color:var(--color-text-secondary);letter-spacing:.08em;
                    text-transform:uppercase;margin:0 0 1.5rem">{symbol_select.value} · Options · MLEG · Day order</p>

          <div style="background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);
                      border-radius:12px;padding:1rem 1.25rem">

            {section("Order legs")}
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:1rem">
              {leg_card(sell_leg, short_put, "#FAECE7", "#993C1D", "↓", "Sell to open")}
              {leg_card(buy_leg,  long_put,  "#E1F5EE", "#0F6E56", "↑", "Buy to open")}
            </div>

            {divider()}

            {section("Order parameters")}
            {field_row("order class",          "MLEG")}
            {field_row("time in force",        "DAY")}
            {field_row("qty",                  "1 contract")}
            {field_row("limit price",          lp_display,       "#185FA5")}
            {field_row("filled qty",           filled_display)}
            {field_row("filled avg price",     filled_avg_display)}
            {field_row("order id",             oid_display)}
            {field_row("submitted at",         sub_display)}
            {field_row("expires at",           exp_display)}
            {field_row("last updated",         upd_display)}

            {divider()}

            {section("P&L profile")}
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:1rem">
              {metric("Max loss",  f"-${max_loss:,.2f}",                        "#993C1D")}
              {metric("Max gain",  f"+${max_gain:,.2f}" if max_gain else "—",   "#0F6E56")}
              {metric("Breakeven", f"${breakeven:,.2f}" if breakeven else "—")}
            </div>

            <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;border-radius:8px;
                        border:0.5px solid var(--color-border-tertiary);margin-bottom:1rem;font-size:13px">
              <div style="width:8px;height:8px;border-radius:50%;background:{dot_color};flex-shrink:0"></div>
              <span>{status_txt}</span>
            </div>

          </div>
        </div>
        """

        return mo.Html(html)

    def bear_put_payoff_chart(long_put, short_put):

        long_price  = long_put["initial_option_price"]
        short_price = short_put["initial_option_price"]

        K_long  = _parse_strike(long_put["symbol"])    # higher strike
        K_short = _parse_strike(short_put["symbol"])   # lower strike

        debit = long_price - short_price

        # Profit accrues as the underlying falls: breakeven sits BELOW K_long.
        breakeven = K_long - debit
        max_loss = debit * 100
        max_gain = (K_long - K_short - debit) * 100

        S = np.linspace(K_short * 0.75, K_long * 1.2, 500)

        payoff = (
            np.maximum(K_long - S, 0)
            - np.maximum(K_short - S, 0)
            - debit
        ) * 100

        fig = go.Figure()

        # payoff line
        fig.add_trace(
            go.Scatter(
                x=S,
                y=payoff,
                mode="lines",
                name="Payoff",
                line=dict(width=4),
                hovertemplate=(
                    f"{symbol_select.value} Price: $%{{x:.2f}}<br>"
                    "P/L: $%{y:.2f}<extra></extra>"
                )
            )
        )

        # profit region
        fig.add_trace(
            go.Scatter(
                x=S,
                y=np.maximum(payoff, 0),
                fill="tozeroy",
                mode="none",
                name="Profit",
                hoverinfo="skip"
            )
        )

        # loss region
        fig.add_trace(
            go.Scatter(
                x=S,
                y=np.minimum(payoff, 0),
                fill="tozeroy",
                mode="none",
                name="Loss",
                hoverinfo="skip"
            )
        )

        # zero line
        fig.add_hline(
            y=0,
            line_dash="dash"
        )

        # strikes
        fig.add_vline(
            x=K_short,
            line_dash="dot",
            annotation_text=f"Short {K_short:.0f}"
        )

        fig.add_vline(
            x=K_long,
            line_dash="dot",
            annotation_text=f"Long {K_long:.0f}"
        )

        fig.add_vline(
            x=breakeven,
            line_dash="dash",
            annotation_text=f"BE {breakeven:.2f}"
        )

        # max loss occurs at/above the long (higher) strike
        fig.add_annotation(
            x=K_long,
            y=-max_loss,
            text=f"Max Loss<br>${max_loss:.0f}",
            showarrow=True
        )

        # max gain occurs at/below the short (lower) strike
        fig.add_annotation(
            x=K_short,
            y=max_gain - 75,   # place text below payoff line
            text=f"Max Gain<br>${max_gain:.0f}",
            showarrow=True,
            ax=0,
            ay=40             # arrow points upward to payoff
        )

        fig.update_layout(
            template="plotly_dark",
            title="Bear Put Spread Payoff",
            height=450,
            margin=dict(
                l=30,
                r=30,
                t=50,
                b=30
            ),
            xaxis_title=f"{symbol_select.value} Price at Expiration",
            yaxis_title="Profit / Loss ($)",
            hovermode="x unified",
            showlegend=False
        )

        return fig

    return bear_put_payoff_chart, render_bear_put_spread_ui


@app.cell
def _(
    bear_put_payoff_chart,
    lp,
    mo,
    render_bear_put_spread_ui,
    res,
    section_header,
    sp,
):
    if res is None:
        mo.stop(
            True,
            mo.callout(
                mo.md("**No valid bear put spread found.**"),
                kind="warn"
            )
        )

    mo.vstack([
        section_header("Order Logistics", "3.1"),
        render_bear_put_spread_ui(lp, sp, order_result=res),
        section_header("Payoff Chart", "3.2"),
        mo.ui.plotly(
            bear_put_payoff_chart(
                lp,
                sp
            )
        )
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
