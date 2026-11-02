import marimo

__generated_with = "0.24.2"
app = marimo.App(
    width="medium",
    css_file="../../theme.css",
    html_head_file="../../theme_head.html",
)


@app.cell
def _():
    import asyncio
    import os
    import sys
    from datetime import timezone
    from pathlib import Path

    import aiohttp
    import feedparser
    import numpy as np
    import pandas as pd
    import requests
    import torch
    from aiohttp.resolver import ThreadedResolver
    from dotenv import load_dotenv
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    import marimo as mo

    # When stdout is a pipe rather than a console, Python encodes it with the
    # locale codec — cp1252 on this machine — and the → in the order log raises
    # UnicodeEncodeError. Force UTF-8 on the real streams; marimo's in-notebook
    # capture stream has no reconfigure().
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")

    # Load .env from the Quant root (two levels up), not the cwd marimo happens to
    # be launched from. Real environment variables still win over the file.
    # (Assign to _ so marimo doesn't render load_dotenv's True as cell output.)
    _ = load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    return (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Path,
        ThreadedResolver,
        aiohttp,
        asyncio,
        feedparser,
        hf_hub_download,
        mo,
        np,
        os,
        pd,
        requests,
        sys,
        timezone,
        torch,
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
    def section_header(title, number):
        return mo.Html(f"""
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <div style="padding: 1.5rem 0 1rem; display: flex; align-items: center; gap: 16px;">
      <div style="font-family: 'DM Serif Display', serif; font-size: 26px; font-style: italic; font-weight: 400; color: var(--color-text-primary); white-space: nowrap;">{title}</div>
      <div style="flex: 1; height: 0.5px; background: var(--color-border-tertiary);"></div>
      <div style="font-family: 'DM Mono', monospace; font-size: 10px; color: #1D9E75; letter-spacing: 0.18em; text-transform: uppercase; white-space: nowrap;">{number}</div>
    </div>
    """)

    return (section_header,)


@app.cell
def _(mo):
    mo.Html("""
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

    <div style="padding: 2.5rem 0 2rem; text-align: center;">
      <div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; border-radius: 50%; border: 0.5px solid #1D9E75; margin-bottom: 1rem;">
        <i class="ti ti-chart-candle" style="font-size: 22px; color: #1D9E75;"></i>
      </div>
      <h1 style="font-family: 'DM Serif Display', serif; font-size: 36px; font-weight: 400; font-style: italic; margin: 0 0 6px; letter-spacing: -0.01em; color: var(--color-text-primary);">Forex Trading Strategy</h1>
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
def _(os, pd, requests):
    # Practice vs live endpoint. Flip PRACTICE to False only when you mean it.
    PRACTICE = True
    OANDA_BASE = "https://api-fxpractice.oanda.com" if PRACTICE else "https://api-fxtrade.oanda.com"
    ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")

    token = os.environ.get("OANDA_TOKEN")
    if not token:
        raise RuntimeError(
            "OANDA_TOKEN is not set.\n"
            "  Add it to the .env file in the Quant root folder (it is gitignored):\n"
            "      OANDA_TOKEN=your-token-here\n"
            "  or export it in your shell. Note that everything downstream of this\n"
            "  cell — instruments, ARA predictions, sizing, execution — depends on\n"
            "  it, so those cells will report NameErrors until this is fixed."
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.get(
        f"{OANDA_BASE}/v3/accounts/{ACCOUNT_ID}/instruments", headers=headers, timeout=15
    )
    response.raise_for_status()
    raw = response.json()

    instruments = raw['instruments']

    # Base fields
    df = pd.DataFrame(instruments)

    # Flatten financing dict
    financing_df = pd.json_normalize([i['financing'] for i in instruments])
    financing_df.columns = [f"financing_{c}" for c in financing_df.columns]

    # Drop nested cols, join flattened ones
    df = df.drop(columns=['financing', 'tags']).join(financing_df)

    # Convert numeric string columns
    numeric_cols = [
        'minimumTradeSize', 'maximumTrailingStopDistance', 'minimumTrailingStopDistance',
        'maximumPositionSize', 'maximumOrderUnits', 'marginRate',
        'financing_longRate', 'financing_shortRate'
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Drop the nested days-of-week column (list of dicts, not useful flat)
    df = df.drop(columns=['financing_financingDaysOfWeek'])

    df = df.set_index('name')
    df['ticker'] = df['displayName'].str.replace('/', '') + '=X'
    return ACCOUNT_ID, OANDA_BASE, df, headers


@app.cell
def _(page_header):
    page_header("Tradable Assets", "01")
    return


@app.cell
def _():
    # Major forex pairs (Yahoo Finance format).
    #
    # Deliberately in its own cell with NO dependencies. This is a static literal,
    # and the entire news/sentiment branch keys off it — when it lived in the same
    # cell as df_filtered it inherited that cell's dependency on the OANDA call,
    # so a missing token took down sentiment analysis too, three sections away.
    forex_pairs = {
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "USDCHF=X",
    "AUDUSD=X",
    "USDCAD=X",
    "NZDUSD=X",
    # Cross pairs
    "EURJPY=X",
    "GBPJPY=X",
    "EURGBP=X",
    "EURAUD=X",
    "EURCHF=X",
    "AUDJPY=X",
    "GBPAUD=X",
    "GBPCAD=X",
    # Exotic pairs
    "USDMXN=X",
    "USDZAR=X",
    "USDTRY=X",
    "USDBRL=X",
    }
    return (forex_pairs,)


@app.cell
def _(df, forex_pairs, mo):
    df_filtered = df[df['ticker'].isin(forex_pairs)].copy()
    mo.ui.dataframe(df_filtered)
    return (df_filtered,)


@app.cell
def _(page_header):
    page_header("Training Result", "02")
    return


@app.cell
def _(section_header):
    section_header("Ara AI Prediction", "02.1")
    return


@app.cell
def _(Path, df_filtered, hf_hub_download, mo, np, os, pd, sys):
    # AraAI is its own git clone, kept beside this notebook in desks/forex/.
    ARA_PATH = os.environ.get("ARA_AI_PATH") or str(Path(__file__).resolve().parent / "AraAI")
    if ARA_PATH not in sys.path:
        sys.path.insert(0, ARA_PATH)

    from meridianalgo.forex_ml import ForexML
    model_path = hf_hub_download(
        repo_id="meridianal/ARA.AI",
        filename="models/Meridian.AI_Forex.pt",
    )

    ml = ForexML(model_path=model_path)

    # predict_forex returns {"error": ..., "pair": ...} with no "predictions" key
    # on failure — and on the exception path "pair" is the raw yfinance ticker
    # rather than "EUR/USD", which would silently create phantom rows in the
    # downstream merge. Drop those instead of carrying them through.
    records, skipped = [], []
    for ticker in df_filtered["ticker"]:
        result = ml.predict_forex(ticker, days=5)
        if result.get("error") or not result.get("predictions"):
            skipped.append((ticker, result.get("error", "no predictions returned")))
            continue
        records.append(result)

    for _ticker, _why in skipped:
        print(f"skipped {_ticker}: {_why}")

    if not records:
        raise RuntimeError(
            "No forex predictions succeeded — check the ARA model weights and yfinance data."
        )

    df_predict = pd.DataFrame(records)

    # Step 1: Explode list into individual dict rows
    df_long = df_predict.explode('predictions').reset_index(drop=True)

    # Step 2: Normalize the dict column into flat columns
    pred_cols = pd.json_normalize(df_long['predictions'])

    # Step 3: Merge back and drop the raw column
    df_final = pd.concat([df_long.drop(columns='predictions'), pred_cols], axis=1)

    # ForexML reports volatility as an ANNUALISED PERCENT (std * sqrt(252) * 100,
    # e.g. 8.2 for 8.2%). Everything downstream — including the 0.005 fallback in
    # the reconciliation cell — expects a DAILY sigma as a decimal fraction of
    # price. Without this conversion stop_loss_pips came out ~1000x too wide and
    # every Kelly-sized position collapsed to the 0.01-lot floor.
    df_final["volatility"] = (
        pd.to_numeric(df_final["volatility"], errors="coerce") / 100.0 / np.sqrt(252)
    )

    # model_accuracy is "N/A" when the model ships without metadata; coerce so the
    # Kelly maths gets a float or a NaN it can fall back on, never a string.
    df_final["model_accuracy"] = pd.to_numeric(df_final["model_accuracy"], errors="coerce")

    mo.ui.dataframe(df_final)
    return (df_final,)


@app.cell
def _(section_header):
    section_header("Trading signals based on ML", "02.2")
    return


@app.cell
def _(ACCOUNT_ID, OANDA_BASE, df_final, headers, mo, np, pd, requests):
    # Sizing constants. `volatility` is carried through this notebook as a DAILY
    # sigma expressed as a decimal fraction of price (0.005 = 0.5%/day).
    UNITS_PER_LOT = 100_000
    DEFAULT_PIP_SIZE = 0.0001   # 4-dp quotes
    JPY_PIP_SIZE = 0.01         # JPY-quoted pairs quote to 2 dp
    MIN_STOP_PIPS = 5.0         # floor: a near-zero vol estimate must not divide by ~0
    MIN_LOTS = 0.01
    # Hard per-pair cap (policy set 2026-10-12). It replaced a 10-lot ceiling that let
    # risk-based sizing stage 600k-1M unit tickets and tie up ~90% of account margin.
    MAX_UNITS = 5_000
    # Commit at most this share of marginAvailable so the whole batch fills even if
    # prices drift between sizing and submission.
    MARGIN_BUFFER = 0.90
    FALLBACK_MARGIN_RATE = 0.05  # OANDA's highest listed rate among this account's majors

    def quote_ccy(pair: str) -> str:
        pair = str(pair).upper()
        return pair.split("/")[-1] if "/" in pair else pair[-3:]

    def pip_size(pair: str) -> float:
        return JPY_PIP_SIZE if quote_ccy(pair) == "JPY" else DEFAULT_PIP_SIZE

    def usd_per_quote(pair: str, price_lookup: dict):
        """USD per one unit of the pair's quote currency, or None when no rate was seen this run."""
        quote = quote_ccy(pair)
        if quote == "USD":
            return 1.0
        if price_lookup.get(f"{quote}/USD"):
            return float(price_lookup[f"{quote}/USD"])
        if price_lookup.get(f"USD/{quote}"):
            return 1.0 / float(price_lookup[f"USD/{quote}"])
        return None

    def pip_value_usd(pair: str, price_lookup: dict) -> float:
        """USD value of one pip on one standard lot, converted via the quote currency.

        The old flat $10/pip only holds for USD-quoted pairs; it overstated size on
        USD/JPY by ~50% and was simply wrong on crosses like EUR/GBP.
        """
        rate = usd_per_quote(pair, price_lookup)
        if rate is None:
            return 10.0  # no conversion rate this run — fall back to the USD-quoted approximation
        value = pip_size(pair) * UNITS_PER_LOT * rate
        return float(value) if np.isfinite(value) and value > 0 else 10.0

    def generate_signals(df):
        df = df.copy()
        df['signal'] = 'HOLD'

        # Price movement from current to predicted
        df['price_delta'] = df['predicted_price'] - df['current_price']

        buy_mask = (
            (df['price_delta'] > 0) &               # Price expected to go UP
            (df['predicted_return'] > 0) &
            (df['pips'] > 0) &
            (df['confidence'] >= 0.70) &
            (df['trend'].str.upper() == 'BULLISH')
        )

        sell_mask = (
            (df['price_delta'] < 0) &               # Price expected to go DOWN
            (df['predicted_return'] < 0) &
            (df['pips'] < 0) &
            (df['confidence'] >= 0.70) &
            (df['trend'].str.upper() == 'BEARISH')
        )

        df.loc[buy_mask, 'signal'] = 'BUY'
        df.loc[sell_mask, 'signal'] = 'SELL'

        return df

    def build_trade_plan(df_signals):
        """
        For each pair, the signal is the same across all days (enter today).
        day=1..5 gives you progressive TP targets.
        """
        # Get the signal direction per pair (should be consistent across days)
        pair_signal = (
            df_signals.groupby('pair')['signal']
            .agg(lambda x: x[x != 'HOLD'].mode()[0] if (x != 'HOLD').any() else 'HOLD')
            .reset_index()
        )

        # Pivot predicted prices as TP levels per day
        tp_table = df_signals.pivot_table(
            index='pair',
            columns='day',
            values='predicted_price'
        ).rename(columns=lambda d: f'TP_day{int(d)}')

        # Avg confidence across the 5-day window
        meta = df_signals.groupby('pair').agg(
            current_price=('current_price', 'first'),
            avg_confidence=('confidence', 'mean'),
            trend=('trend', 'first'),
            forecast_timestamp=('timestamp', 'first'),
            volatility=('volatility', 'first'),
            model_accuracy=('model_accuracy', 'first'),
        ).reset_index()

        trade_plan = pair_signal.merge(meta, on='pair').merge(tp_table, on='pair')
        return trade_plan

    def apply_kelly(
        df_trade_plan,
        account_balance,
        kelly_fraction=0.25,
        max_risk_pct=0.02,
        price_lookup=None,
        margin_rates=None,
        margin_available=None,
        max_units=MAX_UNITS,
        margin_buffer=MARGIN_BUFFER,
    ):
        """Kelly risk budget -> stop-based size, then the per-pair unit cap, then margin fitting.

        Under a binding cap the growth-optimal size is the cap itself (Kelly growth is
        concave and still rising there), so pairs are not scaled relative to each other.
        If the whole batch would exceed `margin_buffer` x `margin_available`, every
        active order is shrunk by one common factor so all of them can be submitted.
        """
        df = df_trade_plan.copy()
        price_lookup = price_lookup or {}
        margin_rates = margin_rates or {}

        # Guard both Kelly inputs: model_accuracy can arrive as NaN (the ARA model
        # reports "N/A" without metadata), and an accuracy of exactly 1.0 sends the
        # payoff odds to infinity.
        p = pd.to_numeric(df['avg_confidence'], errors='coerce').clip(0.0, 1.0)
        acc = pd.to_numeric(df['model_accuracy'], errors='coerce').clip(0.5, 0.95)
        q = 1 - p
        b = acc / (1 - acc)

        raw_kelly = ((p * b - q) / b).clip(lower=0).fillna(0.0)
        df['kelly_f']         = (raw_kelly * kelly_fraction).round(4)
        df['risk_budget_usd'] = (account_balance * df['kelly_f']).clip(upper=account_balance * max_risk_pct).round(2)

        # Daily sigma is a fraction of price, so convert to price units first and
        # only then to pips using the pair's own pip size.
        if 'current_price' in df.columns:
            prices = pd.to_numeric(df['current_price'], errors='coerce')
        else:
            prices = pd.Series(np.nan, index=df.index, dtype='float64')
        # A pair with no price anywhere (not in the forecast, not in the broker lookup)
        # cannot be stopped, sized or margined; it stays NaN and is zeroed below rather
        # than priced at 1.0, which turned JPY and exotic pairs into max-size tickets.
        prices = prices.fillna(df['pair'].map(price_lookup)).astype('float64')

        vol = pd.to_numeric(df['volatility'], errors='coerce').abs()
        pip_sizes = df['pair'].map(pip_size).astype('float64')

        # A missing or zero volatility estimate gives us no basis for a stop, and
        # therefore none for a size. Leave it NaN rather than letting it fall
        # through to MIN_STOP_PIPS, which would turn a degenerate vol reading into
        # a maximum-size ticket.
        _stop = ((vol * prices) / pip_sizes).round(1)
        df['stop_loss_pips'] = _stop.where(_stop > 0).clip(lower=MIN_STOP_PIPS)
        df['pip_value_usd']  = df['pair'].map(lambda pr: pip_value_usd(pr, price_lookup))

        _raw_lots = (
            df['risk_budget_usd'] / (df['stop_loss_pips'] * df['pip_value_usd'])
        ).replace([np.inf, -np.inf], np.nan)

        # Quote->USD rate per row (NaN when this run saw no rate). It prices pips,
        # notional and margin, so a row without one is unsizable rather than guessed.
        _fx = df['pair'].map(lambda pr: usd_per_quote(pr, price_lookup)).astype('float64')

        # Only floor to the minimum ticket where there is real risk budget behind
        # it — otherwise a zero-conviction row still sends the broker 0.01 lots.
        _sizable = (df['risk_budget_usd'] > 0) & _raw_lots.notna() & prices.notna() & _fx.notna()
        _max_lots = max_units / UNITS_PER_LOT
        df['lots'] = (
            _raw_lots.clip(lower=MIN_LOTS, upper=_max_lots).where(_sizable, 0.0).round(4)
        )

        # OANDA uses units, not lots (1 standard lot = 100,000 units)
        df['units'] = (df['lots'] * UNITS_PER_LOT).round().astype(int).clip(upper=max_units)

        # Negative units = SELL, positive = BUY
        df.loc[df['signal'] == 'SELL', 'units'] = -df.loc[df['signal'] == 'SELL', 'units']

        # Zero out sizing for anything that isn't an actionable direction
        _inactive = ~df['signal'].isin(['BUY', 'SELL'])
        df.loc[_inactive, ['kelly_f', 'risk_budget_usd', 'lots', 'units']] = 0

        # Margin: USD notional of the base-currency units x OANDA's per-instrument rate.
        # base/USD = (base/quote) x (quote/USD). Unsizable rows already have 0 units.
        _fx = _fx.fillna(0.0)
        prices = prices.fillna(0.0)

        def _margin_rate(pr):
            rate = margin_rates.get(pr, margin_rates.get(str(pr).replace('/', '_')))
            try:
                rate = float(rate)
            except (TypeError, ValueError):
                return FALLBACK_MARGIN_RATE
            return rate if np.isfinite(rate) and rate > 0 else FALLBACK_MARGIN_RATE

        df['margin_rate'] = df['pair'].map(_margin_rate).astype('float64')

        def _refresh_exposure(frame):
            frame['lots'] = (frame['units'].abs() / UNITS_PER_LOT).round(4)
            frame['notional_usd'] = (frame['units'].abs() * prices * _fx).round(2)
            frame['margin_required_usd'] = (frame['notional_usd'] * frame['margin_rate']).round(2)
            # True dollar risk at the stop for the size actually sent, not the Kelly budget.
            frame['risk_amount_usd'] = (frame['lots'] * frame['stop_loss_pips'] * frame['pip_value_usd']).round(2)
            # Unsizable rows have 0 units but a NaN stop; report 0 exposure, not NaN.
            frame.loc[frame['units'] == 0, ['notional_usd', 'margin_required_usd', 'risk_amount_usd']] = 0.0

        _refresh_exposure(df)
        df['margin_scale'] = 1.0
        _active = ~_inactive & (df['units'] != 0)
        _needed = df.loc[_active, 'margin_required_usd'].sum()
        if margin_available is not None and _needed > 0:
            _cap = max(float(margin_available), 0.0) * margin_buffer
            if _needed > _cap:
                _scale = _cap / _needed
                df.loc[_active, 'margin_scale'] = round(_scale, 6)
                df.loc[_active, 'units'] = (
                    np.sign(df.loc[_active, 'units']) * np.floor(df.loc[_active, 'units'].abs() * _scale)
                ).astype(int)
                _refresh_exposure(df)
        df.loc[_inactive, ['notional_usd', 'margin_required_usd', 'risk_amount_usd']] = 0

        return df

    response_acc = requests.get(f"{OANDA_BASE}/v3/accounts/{ACCOUNT_ID}", headers=headers, timeout=15)
    response_acc.raise_for_status()
    raw_acc = response_acc.json()

    df_signals = generate_signals(df_final)
    df_trade_plan = build_trade_plan(df_signals)
    # df_final_signals = apply_kelly(df_trade_plan, account_balance= float(raw_acc['account']['balance']), kelly_fraction=0.25, max_risk_pct=0.02)
    mo.ui.dataframe(df_trade_plan)
    return apply_kelly, df_trade_plan, raw_acc


@app.cell
def _(page_header):
    page_header("News Reaction Validation", "03")
    return


@app.cell
def _(ThreadedResolver, aiohttp, asyncio, feedparser, pd, timezone):
    # NOTE: no event-loop-policy juggling here. Marimo already owns a running
    # loop, so set_event_loop_policy() would be a no-op; the ThreadedResolver on
    # the connector below is what actually makes DNS work on Windows.

    # --- RSS sources ---
    FX_RSS_SOURCES = {
        "forexlive": "https://www.forexlive.com/feed/news",
        "fxstreet": "https://www.fxstreet.com/rss/news",
        "investing_forex": "https://www.investing.com/rss/news_1.rss",
    }

    # Several of these feeds 403 aiohttp's default User-Agent.
    # Accept-Encoding is pinned to codecs aiohttp can always decode: a brotli
    # module on this machine makes aiohttp advertise "br", but it is not the
    # variant aiohttp's decoder needs, so fxstreet's br responses died with
    # "Can not decode content-encoding: br". Asking for gzip/deflate only
    # keeps the server off brotli entirely.
    RSS_HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; fx-news-poller/1.0)",
        "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
    }


    async def fetch_feed(session: aiohttp.ClientSession, name: str, url: str) -> list[dict]:
        try:
            async with session.get(
                url, headers=RSS_HEADERS, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                resp.raise_for_status()
                raw = await resp.text()

            # feedparser is CPU-bound and blocking; keep it off the event loop.
            parsed = await asyncio.to_thread(feedparser.parse, raw)

            return [
                {
                    "source": name,
                    "title": title,
                    "link": e.get("link", ""),
                    "published": e.get("published", ""),
                    "summary": e.get("summary", ""),
                }
                for e in parsed.entries
                if (title := (e.get("title") or "").strip())
            ]
        except Exception as e:
            print(f"[{name}] fetch failed: {e}")
            return []


    async def poll_all_feeds_df() -> pd.DataFrame:
        connector = aiohttp.TCPConnector(resolver=ThreadedResolver())

        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [fetch_feed(session, name, url) for name, url in FX_RSS_SOURCES.items()]
            results = await asyncio.gather(*tasks)
            flat = [item for sublist in results for item in sublist]

        df = pd.DataFrame(flat, columns=["source", "title", "link", "published"])

        if df.empty:
            return df

        # Parse publish times, fall back to fetch time when parsing fails
        df["published_dt"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
        df["published_dt"] = df["published_dt"].fillna(pd.Timestamp.now(tz=timezone.utc))

        # Dedupe wire stories that show up across multiple sources
        df = df.drop_duplicates(subset="title", keep="first")

        df = df.sort_values("published_dt", ascending=False).reset_index(drop=True)

        return df[["source", "title", "link", "published_dt"]]

    return (poll_all_feeds_df,)


@app.cell
def _(section_header):
    section_header("Fetching real time news", "03.1")
    return


@app.cell
def _(mo):
    refresh_btn = mo.ui.run_button(label="Fetch FX News")
    refresh_btn
    return (refresh_btn,)


@app.cell
def _(AutoModelForSequenceClassification, AutoTokenizer, np, pd, torch):
    # --- Load FinBERT (cache this in Marimo so it doesn't reload every re-run) ---
    FINBERT_MODEL = "ProsusAI/finbert"
    FINBERT_BATCH_SIZE = 32

    _tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    _model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    _model.eval()

    # Read the label order off the checkpoint rather than hardcoding it — FinBERT
    # happens to be 0=positive, 1=negative, 2=neutral, but that is a per-model
    # detail and getting it wrong silently inverts every score.
    _LABELS = [
        _model.config.id2label[i].lower() for i in range(_model.config.num_labels)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(device)


    def score_sentiments(texts: list[str], batch_size: int = FINBERT_BATCH_SIZE):
        """Score a list of headlines in batches. Returns an (n, n_labels) prob matrix.

        Batched rather than one forward pass per row — for a few hundred headlines
        that is the difference between a couple of seconds and the better part of a
        minute.
        """
        if not texts:
            return np.zeros((0, len(_LABELS)), dtype="float64")

        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = _tokenizer(
                batch, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(device)

            with torch.no_grad():
                logits = _model(**inputs).logits
                chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())

        return np.vstack(chunks)


    def apply_finbert_sentiment(df: pd.DataFrame, text_col: str = "title") -> pd.DataFrame:
        """
        Apply FinBERT sentiment scoring to a DataFrame of news items.
        Scores off `text_col` (default: title — swap to summary_clean if you
        want summary-based scoring instead of headline-based).
        """
        if df.empty:
            return df

        out = df.reset_index(drop=True).copy()
        texts = out[text_col].fillna("").astype(str).str.strip()

        probs = score_sentiments(texts.tolist())
        scores = pd.DataFrame(probs, columns=_LABELS)

        # Blank headlines get a hard neutral rather than whatever the model does
        # with an empty string.
        blank = (texts == "").to_numpy()
        if blank.any():
            for _lbl in _LABELS:
                scores.loc[blank, _lbl] = 1.0 if _lbl == "neutral" else 0.0

        scores["label"] = scores[_LABELS].idxmax(axis=1)
        scores["confidence"] = scores[_LABELS].max(axis=1)

        out = pd.concat([out, scores], axis=1)

        # Directional sentiment score: positive minus negative, range [-1, 1]
        out["sentiment_score"] = out["positive"] - out["negative"]

        return out

    return (apply_finbert_sentiment,)


@app.cell
def _(forex_pairs):
    from forex_sentiment import annotate_news, generate_pair_signals, tag_news

    def apply_currency_and_pair_tags(df, text_col="title"):
        return tag_news(df, forex_pairs, text_col=text_col)

    def apply_signal_gate(df):
        # Evidence annotation only: negative news remains in the aggregation.
        return annotate_news(df)

    return (
        apply_currency_and_pair_tags,
        apply_signal_gate,
        generate_pair_signals,
    )

@app.cell
async def _(
    apply_currency_and_pair_tags,
    apply_finbert_sentiment,
    apply_signal_gate,
    mo,
    poll_all_feeds_df,
    refresh_btn,
):
    mo.stop(not refresh_btn.value)

    news_df = await poll_all_feeds_df()
    scored_df = apply_finbert_sentiment(news_df, text_col="title")
    tagged_df = apply_currency_and_pair_tags(scored_df, text_col="title")
    gated_df = apply_signal_gate(tagged_df)

    mo.ui.table(
        gated_df[["source", "title", "link", "pairs", "currencies", "label", "sentiment_score", "confidence", "currency_scores", "attribution_method", "signal", "published_dt"]],
        page_size=20,
    )
    return (gated_df,)


@app.cell
def _(forex_pairs, gated_df, generate_pair_signals, mo):
    # Keep all news, including negative and ambiguous stories. The helper only
    # aggregates attributed currency scores and preserves every relevant headline.
    signals_df = generate_pair_signals(gated_df, forex_pairs)
    signals_df["pair"] = (
        signals_df["pair"]
        .str.replace("=X", "", regex=False)
        .str.replace(r"^([A-Z]{3})([A-Z]{3})$", r"\1/\2", regex=True)
    )
    mo.ui.table(
        signals_df[["pair", "signal", "pair_score", "base_sentiment", "quote_sentiment", "n_articles",
                    "n_directional_articles", "news_evidence"]],
        page_size=20,
    )
    return (signals_df,)


@app.cell
def _(section_header):
    section_header("Conflict Dataframe", "03.2")
    return


@app.cell
def _(df_trade_plan, mo, pd, signals_df):
    def merge_signals(df_trade_plan: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
        merged = df_trade_plan.merge(
            signals_df[["pair", "signal", "pair_score", "base_sentiment", "quote_sentiment", "n_articles",
                        "n_directional_articles", "news_evidence"]],
            on="pair",
            how="outer",
            suffixes=("_ml", "_sentiment"),
        )

        def classify(row):
            ml_sig = row.get("signal_ml")
            sent_sig = row.get("signal_sentiment")

            if pd.isna(ml_sig) or pd.isna(sent_sig):
                return "single_source"
            if ml_sig == sent_sig:
                return "agree"  # includes BUY==BUY, SELL==SELL, and HOLD==HOLD
            if "HOLD" in (ml_sig, sent_sig):
                return "soft_conflict"  # one wants HOLD, other wants BUY/SELL
            return "hard_conflict"  # BUY vs SELL outright

        merged["status"] = merged.apply(classify, axis=1)
        return merged

    combined_df = merge_signals(df_trade_plan, signals_df)
    mo.ui.dataframe(combined_df)
    return (combined_df,)


@app.cell
def _(combined_df, pd):
    agree_df = combined_df[combined_df["status"] == "agree"].copy()
    agree_df["final_action"] = agree_df["signal_ml"]
    agree_df["decision_source"] = "agreement"

    single_df = combined_df[combined_df["status"] == "single_source"].copy()
    single_df["final_action"] = single_df["signal_ml"].fillna(single_df["signal_sentiment"])
    single_df["decision_source"] = single_df.apply(
        lambda r: "ml_only" if pd.notna(r["signal_ml"]) else "sentiment_only", axis=1
    )
    return agree_df, single_df


@app.cell
def _(mo):
    get_resolve_clicked, set_resolve_clicked = mo.state(False)
    return get_resolve_clicked, set_resolve_clicked


@app.cell
def _(agree_df, combined_df, mo, single_df):
    needs_resolution_df = combined_df[combined_df["status"].isin(["soft_conflict", "hard_conflict"])].copy()

    mo.md(f"""
    **Signal reconciliation summary**
    - Agreement: {len(agree_df)} pairs
    - Single-source: {len(single_df)} pairs
    - Soft conflicts (HOLD vs directional): {(needs_resolution_df["status"] == "soft_conflict").sum()} pairs
    - Hard conflicts (BUY vs SELL): {(needs_resolution_df["status"] == "hard_conflict").sum()} pairs
    """)

    resolve_button = mo.ui.run_button(label=f"Resolve {len(needs_resolution_df)} conflicts with Clef-omni")
    return needs_resolution_df, resolve_button


@app.cell
def _(section_header):
    section_header("Clef-omni Conflict Resolution", "03.3")
    return


@app.cell
def _(resolve_button, set_resolve_clicked):
    if resolve_button.value:
        set_resolve_clicked(True)

    resolve_button
    return


@app.cell
def _(get_resolve_clicked, mo, needs_resolution_df, os, pd):
    import sys as _sys
    from pathlib import Path as _Path
    _shared = str(_Path(__file__).resolve().parents[2] / "shared")
    if _shared not in _sys.path:
        _sys.path.insert(0, _shared)
    from forex_resolution import COLUMNS, ClefFlashResolver, R2Ledger
    import market_cache

    # The Neuron ledger and decision cache live on R2 (one JSON object per UTC day); nothing is written locally.
    _account = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
    _resolver = ClefFlashResolver(
        account_id=_account,
        token=os.getenv("CLOUDFLARE_AUTH_TOKEN", ""),
        workers_plan=os.getenv("CLOUDFLARE_WORKERS_PLAN", "free").strip().lower(),
        daily_neurons=os.getenv("CLEF_DAILY_NEURON_LIMIT", "9000"),
        ledger=R2Ledger(market_cache.R2Store.from_env(), account=_account or "unconfigured"),
    )
    if get_resolve_clicked() and len(needs_resolution_df):
        _results = [_resolver.resolve(_row) for _, _row in needs_resolution_df.iterrows()]
        needs_resolution_df_ai = pd.concat([
            needs_resolution_df.reset_index(drop=True), pd.DataFrame(_results)
        ], axis=1)
        _usage = _resolver.usage()
        _output = mo.vstack([
            mo.md(f"Clef-omni local budget: **{_usage['reserved_neurons']} / {_usage['daily_neurons']} Neurons** "
                  f"used or reserved on {_usage['day']} (UTC). Cloudflare also counts other apps against its free allowance."),
            mo.ui.table(needs_resolution_df_ai[
                ["pair", "final_action", "confidence", "requires_manual", "rationale"]
            ]),
        ])
    else:
        needs_resolution_df_ai = needs_resolution_df.reindex(
            columns=list(needs_resolution_df.columns) + COLUMNS
        ).iloc[0:0]
        _output = mo.md("_Click Resolve to run Clef-omni; unavailable requests will require manual selection._"
                       if len(needs_resolution_df) else "_No conflicts to resolve._")
    _output
    return (needs_resolution_df_ai,)


@app.cell
def _(mo, needs_resolution_df_ai):
    # Pairs still awaiting a human decision come first; NaN (Resolve not clicked yet) counts as pending.
    _pending_first = (needs_resolution_df_ai["requires_manual"].fillna(True).astype(bool)
                      .sort_values(ascending=False, kind="stable").index)
    override_widgets = mo.ui.dictionary({
        row["pair"]: mo.ui.radio(
            options=["BUY", "SELL", "HOLD"] if row["requires_manual"] else
                    ["Keep Clef decision", "BUY", "SELL", "HOLD"],
            value=None if row["requires_manual"] else "Keep Clef decision",
            label=f"{row['pair']}: manual decision required" if row["requires_manual"] else
                  f"{row['pair']}: Clef-omni {row['final_action']} — optional override",
        )
        for _, row in needs_resolution_df_ai.loc[_pending_first].iterrows()
    })
    return (override_widgets,)


@app.cell
def _(mo, needs_resolution_df_ai, override_widgets):
    from forex_resolution import apply_manual_decisions, brief_markdown, decision_brief

    mo.stop(len(needs_resolution_df_ai) == 0, mo.md("_No conflicts to review._"))
    _reviewed, _pending = apply_manual_decisions(needs_resolution_df_ai, override_widgets.value)
    _total = len(_reviewed)
    # One evidence brief per pair, rendered under its radio so the manual call is made on the numbers.
    _briefs = {_row["pair"]: brief_markdown(decision_brief(_row))
               for _row in needs_resolution_df_ai.to_dict("records")}
    mo.vstack([
        mo.md(f"**{_total - _pending}/{_total} conflicts resolved.** "
              + (f"{_pending} require your selection." if _pending else "All decisions are ready.")),
        mo.ui.table(_reviewed[["pair", "signal_ml", "signal_sentiment", "final_action",
                               "decision_source", "confidence", "rationale"]]),
        mo.md("Pending pairs are listed first. Each shows AraAI's forecast targets, FinBERT's score and "
              "headlines, and Clef's probabilities. None of these numbers is a win probability."),
        *[mo.vstack([mo.md("---"), override_widgets[_pair], mo.md(_briefs[_pair])], gap=0.25)
          for _pair in override_widgets.elements],
    ])
    return (apply_manual_decisions,)


@app.cell
def _(
    agree_df,
    get_resolve_clicked,
    mo,
    needs_resolution_df_ai,
    override_widgets,
    pd,
    single_df,
):
    from forex_resolution import apply_manual_decisions as _apply_manual_decisions

    _clicked = get_resolve_clicked()
    _overridden, _n_pending = _apply_manual_decisions(needs_resolution_df_ai, override_widgets.value)
    _all_resolved = _n_pending == 0
    mo.stop(
        _clicked and not _all_resolved,
        mo.callout(
            mo.md(f"⚠️ **{_n_pending} conflict(s) require a manual decision.** "
                  "Choose BUY/SELL/HOLD for each pending pair to finish the trade plan."),
            kind="warn",
        ),
    )

    final_columns = [
        "pair", "final_action", "decision_source", "status", "pair_score", "n_articles",
        "signal_ml", "signal_sentiment",
        "avg_confidence", "model_accuracy", "volatility", "current_price",
    ]

    final_decisions = pd.concat(
        [
            agree_df.reindex(columns=final_columns),
            single_df.reindex(columns=final_columns),
            _overridden.reindex(columns=final_columns) if (_clicked and _all_resolved) else pd.DataFrame(columns=final_columns),
        ],
        ignore_index=True,
    ).dropna(subset=["final_action"])

    final_decisions = final_decisions.rename(columns={"final_action": "signal"})

    # --- Fallback sizing basis for rows with no ML-side data (sentiment_only,
    # or any override that lands on a sentiment-only pair) ---
    final_decisions["model_accuracy"] = pd.to_numeric(
        final_decisions["model_accuracy"], errors="coerce"
    )
    _has_ml_basis = final_decisions["model_accuracy"].notna()

    # avg_confidence: use |pair_score| as a strength proxy, mapped into a
    # conservative confidence-like range (0.50-0.80). pair_score of 0 -> pure
    # coin-flip confidence; a strong ±1.0 score -> 0.80, still below what a
    # real validated ML signal could hit.
    _sentiment_confidence_proxy = (0.50 + 0.30 * final_decisions["pair_score"].abs()).clip(upper=0.80)
    final_decisions["avg_confidence"] = final_decisions["avg_confidence"].where(
        _has_ml_basis, _sentiment_confidence_proxy
    )

    # model_accuracy: fixed conservative prior, just above coin-flip, since
    # this signal was never backtested/validated like the ML model was.
    final_decisions["model_accuracy"] = final_decisions["model_accuracy"].where(
        _has_ml_basis, 0.55
    )

    # volatility: no sentiment-side vol estimate exists, so fall back to the
    # median volatility across pairs that DO have ML data this run. If literally
    # none do, fall back to a fixed 0.5% placeholder so stop_loss_pips isn't 0.
    _vol_fallback = final_decisions.loc[_has_ml_basis, "volatility"].median()
    if pd.isna(_vol_fallback):
        _vol_fallback = 0.005
    final_decisions["volatility"] = final_decisions["volatility"].where(
        _has_ml_basis, _vol_fallback
    )

    final_decisions["sizing_basis"] = _has_ml_basis.map({True: "ml", False: "sentiment_fallback"})
    return (final_decisions,)


@app.cell
def _(section_header):
    section_header("Agreement with Kelly criterion allocation", "03.4")
    return


@app.cell
def _(
    ACCOUNT_ID,
    OANDA_BASE,
    apply_kelly,
    df,
    df_trade_plan,
    final_decisions,
    headers,
    mo,
    raw_acc,
    requests,
):
    # Prices for sizing and for quote-currency conversion (USD/JPY prices a JPY pip
    # and a JPY notional in USD). The ML plan only covers forecast pairs, so live
    # closeout mids for EVERY tradable instrument are fetched from OANDA; without
    # them a sentiment-only pair has no price and a cross may have no conversion.
    price_lookup = (
        df_trade_plan.dropna(subset=["current_price"])
        .set_index("pair")["current_price"]
        .to_dict()
    )
    _pricing_note = ""
    try:
        _resp = requests.get(f"{OANDA_BASE}/v3/accounts/{ACCOUNT_ID}/pricing", headers=headers,
                             params={"instruments": ",".join(str(i) for i in df.index)}, timeout=15)
        _resp.raise_for_status()
        for _quote in _resp.json().get("prices", []):
            _bid = _quote.get("closeoutBid") or (_quote.get("bids") or [{}])[0].get("price")
            _ask = _quote.get("closeoutAsk") or (_quote.get("asks") or [{}])[0].get("price")
            if _bid and _ask:
                price_lookup[str(_quote["instrument"]).replace("_", "/")] = (float(_bid) + float(_ask)) / 2
    except Exception as _exc:  # noqa: BLE001 - sizing must degrade, not crash the desk
        _pricing_note = (" OANDA pricing unavailable; only forecast pairs have prices, so pairs "
                         "without one are sized to zero.")
    # OANDA's per-instrument margin rates ('EUR_USD' -> 'EUR/USD') and the account's
    # free margin at load time, so the whole batch is sized to be submittable.
    margin_rates = {str(name).replace("_", "/"): float(rate) for name, rate in df["marginRate"].items()}
    margin_available = float(raw_acc["account"].get("marginAvailable") or 0.0)

    df_final_signals = apply_kelly(
        final_decisions,
        account_balance=float(raw_acc['account']['balance']),
        kelly_fraction=0.25,
        max_risk_pct=0.02,
        price_lookup=price_lookup,
        margin_rates=margin_rates,
        margin_available=margin_available,
    )
    _active = df_final_signals[df_final_signals["signal"].isin(["BUY", "SELL"]) & (df_final_signals["units"] != 0)]
    _needed = float(_active["margin_required_usd"].sum()) if len(_active) else 0.0
    _scale = float(_active["margin_scale"].min()) if len(_active) else 1.0
    _unsized = df_final_signals[df_final_signals["signal"].isin(["BUY", "SELL"]) & (df_final_signals["units"] == 0)]
    mo.vstack([
        mo.md(f"**Margin plan:** {len(_active)} order(s) need **${_needed:,.0f}** of **${margin_available:,.0f}** "
              f"available margin. Cap 5,000 units per pair"
              + (f"; every order scaled to {_scale:.0%} of its size so all of them fit." if _scale < 1 else ".")
              + (f" Not sized (no price or conversion rate): {', '.join(_unsized['pair'])}." if len(_unsized) else "")
              + _pricing_note),
        mo.ui.dataframe(df_final_signals),
    ])
    return (df_final_signals,)


@app.cell
def _(page_header):
    page_header("Trading Transaction", "04")
    return


@app.cell
def _(df, df_final_signals, mo):
    def to_oanda_instrument(pair: str) -> str:
        """'EUR/USD' -> 'EUR_USD'. OANDA rejects the slash form outright."""
        return str(pair).replace("/", "_").upper()

    # Only signed, directional rows are orders. HOLD rows are sized to zero by
    # apply_kelly and OANDA rejects a zero-unit order, so filter them out here.
    orders_to_place = df_final_signals.loc[
        df_final_signals["signal"].isin(["BUY", "SELL"])
        & (df_final_signals["units"].fillna(0) != 0)
    ].copy()
    orders_to_place["instrument"] = orders_to_place["pair"].map(to_oanda_instrument)

    # Anything OANDA doesn't list as tradable would 400 at submit time.
    _tradable = set(df.index)
    untradable = orders_to_place.loc[~orders_to_place["instrument"].isin(_tradable)]
    orders_to_place = orders_to_place.loc[orders_to_place["instrument"].isin(_tradable)]

    execute_btn = mo.ui.run_button(
        label=f"⚠ Submit {len(orders_to_place)} live order(s) to OANDA", kind="danger"
    )

    _view = mo.vstack([
        mo.md(f"**{len(orders_to_place)} order(s) staged, "
              f"${float(orders_to_place['margin_required_usd'].sum()) if len(orders_to_place) else 0:,.0f} margin in total.**"
              + (f" Skipping {len(untradable)} untradable instrument(s): "
                 f"{', '.join(untradable['instrument'])}." if len(untradable) else "")),
        mo.ui.table(
            orders_to_place[["instrument", "signal", "units", "lots", "stop_loss_pips",
                             "pip_value_usd", "risk_amount_usd", "notional_usd", "margin_rate",
                             "margin_required_usd", "margin_scale", "sizing_basis"]],
            page_size=25,
        ),
        execute_btn,
    ])

    _view
    return execute_btn, orders_to_place


@app.cell
def _(
    ACCOUNT_ID,
    OANDA_BASE,
    execute_btn,
    headers,
    mo,
    orders_to_place,
    pd,
    requests,
):
    # HARD GATE. This cell posts real market orders. Without it, every reactive
    # re-run of the notebook — refreshing news, toggling an override — would
    # silently fire the whole order book again.
    mo.stop(
        not execute_btn.value,
        mo.md("_No orders submitted. Review the staged orders above, then click the submit button._"),
    )
    mo.stop(orders_to_place.empty, mo.md("_Nothing to submit._"))

    def place_order(instrument, units):
        body = {
            "order": {
                "units": str(int(units)),      # Positive = BUY, Negative = SELL
                "instrument": instrument,
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT",
            }
        }
        resp = requests.post(
            f"{OANDA_BASE}/v3/accounts/{ACCOUNT_ID}/orders",
            headers=headers,
            json=body,
            timeout=15,
        )
        try:
            payload = resp.json()
        except ValueError:
            payload = {"raw": resp.text}
        return resp.status_code, payload

    results = []
    for _, row_ord in orders_to_place.iterrows():
        status_code, payload = place_order(row_ord["instrument"], row_ord["units"])
        filled = "orderFillTransaction" in payload

        results.append({
            "instrument": row_ord["instrument"],
            "signal": row_ord["signal"],
            "units": row_ord["units"],
            "http_status": status_code,
            "filled": filled,
            "response": payload,
        })
        print(f"  → {row_ord['instrument']} {row_ord['units']:+d} [{status_code}] filled={filled}")

    df_results = pd.DataFrame(results)

    mo.ui.dataframe(df_results)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
