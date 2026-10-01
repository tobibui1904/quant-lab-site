import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", css_file="../../theme.css", html_head_file="../../theme_head.html")


@app.cell
def _():
    import logging
    import os
    import sys
    import requests
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from dotenv import load_dotenv
    import marimo as mo

    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    # Load .env from the Quant root (two levels up). Bare load_dotenv() searches
    # upward from the *working* directory, so it silently finds nothing when
    # marimo is launched from elsewhere.
    _nb_dir = str(mo.notebook_dir())
    load_dotenv(mo.notebook_dir().parents[1] / ".env")

    # Same reasoning for the engine import: the hub launches this notebook from
    # its own working directory, so put the notebook's directory on the path
    # rather than relying on cwd.
    if _nb_dir not in sys.path:
        sys.path.insert(0, _nb_dir)

    import spread_engine as eng

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    eng.logger.setLevel(logging.DEBUG)

    # Call/put polarity, accent colour and delta bands all come from here.
    SPEC = eng.SPECS["bull_call"]
    return (
        OptionHistoricalDataClient,
        StockHistoricalDataClient,
        SPEC,
        TradingClient,
        ZoneInfo,
        datetime,
        eng,
        mo,
        os,
        requests,
        timedelta,
    )


@app.cell
def _(mo):
    with open(mo.notebook_dir().parents[1] / "data" / "pair.txt", "r") as f:
        pair = f.read().splitlines()[0].split(" ")
    return (pair,)


@app.cell
def _(SPEC, eng):
    eng.hero(SPEC)
    return


@app.cell
def _(SPEC, eng):
    eng.page_header(SPEC, "Available pair and Selection", "01")
    return


@app.cell
def _(eng, os, requests):
    risk_free_rate_fred = eng.get_risk_free_rate(requests, os)
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
    return option_historical_data_client, stock_data_client, trade_client


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
    SPEC,
    ZoneInfo,
    datetime,
    eng,
    mo,
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
    underlying_price = eng.get_underlying_price(stock_data_client, underlying_symbol)

    # Set the minimum and maximum strike prices based on the underlying price
    min_strike = str(underlying_price * (1 - STRIKE_RANGE))
    max_strike = str(underlying_price * (1 + STRIKE_RANGE))

    # Multi-year adjusted history: the GARCH fit needs it, the charts below use
    # the recent slice only (see eng.display_window).
    priceHistory = eng.get_stock_data(
        stock_data_client, underlying_symbol, timezone
    ).reset_index(level='symbol', drop=True)
    priceData = eng.display_window(priceHistory)

    # Term-matched GARCH forecast, BIC-selected between GARCH(1,1) and
    # GJR(1,1,1) with skew-t errors; falls back to the old rolling blend when
    # the sample is short or no model converges.
    realized_vol = eng.compute_realized_vol(priceHistory)

    criteria = eng.build_criteria(SPEC, realized_vol)

    eng.scanner_ui(
        symbol_select.value,
        underlying_price,
        buying_power,
        buying_power_limit,
        STRIKE_RANGE,
        risk_free_rate,
        OI_THRESHOLD,
        min_expiration,
        max_expiration,
        min_strike,
        max_strike,
        realized_vol,
    )
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
def _(SPEC, eng):
    eng.page_header(SPEC, "Price Graph with Boilinger Bands", "02")
    return


@app.cell
def _(eng, mo, priceData):
    mo.ui.plotly(eng.candlestick(priceData))
    return


@app.cell
def _(SPEC, eng, priceData, underlying_price, underlying_symbol):
    bollinger_bands = eng.check_bb(priceData, 14, 2)

    print(
        f"Latest Upper Bollinger Band is: {bollinger_bands[0]}. "
        f"Latest Lower Bollinger Band is {bollinger_bands[1]}; while underlying "
        f"stock '{underlying_symbol}' price is {underlying_price}. "
        f"{SPEC.bollinger_note}"
    )
    return


@app.cell
def _(
    OI_THRESHOLD,
    SPEC,
    buying_power_limit,
    criteria,
    eng,
    max_expiration,
    max_strike,
    min_expiration,
    min_strike,
    option_historical_data_client,
    risk_free_rate,
    trade_client,
    underlying_price,
    underlying_symbol,
):
    call_options = eng.get_options(
        trade_client,
        underlying_symbol,
        min_strike,
        max_strike,
        min_expiration,
        max_expiration,
        SPEC.contract_type,
    )
    sc, lc = eng.find_options_for_spread(
        SPEC,
        call_options,
        underlying_price,
        risk_free_rate,
        buying_power_limit,
        criteria,
        OI_THRESHOLD,
        option_historical_data_client,
    )
    return lc, sc


@app.cell
def _(SPEC, eng, lc, sc, trade_client):
    res = eng.place_spread_order(SPEC, sc, lc, trade_client)
    return (res,)


@app.cell
def _(SPEC, eng):
    eng.page_header(SPEC, "Trading Operation", "03")
    return


@app.cell
def _(SPEC, eng, lc, mo, res, sc, symbol_select):
    if res is None:
        mo.stop(
            True,
            mo.callout(
                mo.md(f"**No valid {SPEC.label.lower()} found.**"),
                kind="warn"
            )
        )

    import inspect

    # Marimo can reload this notebook while retaining an older imported engine.
    _payoff_kwargs = (
        {"order_result": res}
        if "order_result" in inspect.signature(eng.payoff_chart).parameters
        else {}
    )
    mo.vstack([
        eng.section_header(SPEC, "Order Logistics", "3.1"),
        eng.render_spread_ui(SPEC, lc, sc, symbol_select.value, order_result=res),
        eng.section_header(SPEC, "Payoff Chart", "3.2"),
        mo.ui.plotly(eng.payoff_chart(SPEC, lc, sc, symbol_select.value, **_payoff_kwargs)),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
