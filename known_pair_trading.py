import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", css_file="theme.css", html_head_file="theme_head.html")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    def page_header(title, section_number, subtitle=None):
        subtitle_html = f"""
          <div style="font-family: 'DM Mono', monospace; font-size: 11px; color: var(--color-text-secondary); letter-spacing: 0.1em; margin-top: 8px;">{subtitle}</div>
        """ if subtitle else ""

        return mo.Html(f"""
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
    <div style="padding: 1.5rem 0 1rem; display: flex; align-items: center; gap: 16px;">
      <div style="font-family: 'DM Serif Display', serif; font-size: 26px; font-style: italic; font-weight: 400; color: var(--color-text-primary); white-space: nowrap;">{title}</div>
      <div style="flex: 1; height: 0.5px; background: var(--color-border-tertiary);"></div>
      <div style="font-family: 'DM Mono', monospace; font-size: 10px; color: #1D9E75; letter-spacing: 0.18em; text-transform: uppercase; white-space: nowrap;">{number}</div>
    </div>
    """)

    return (section_header,)


@app.cell
def _(mo):
    # Single load point for the notebook's webfonts. page_header and
    # section_header used to each re-emit this <link>, putting ~30 duplicate
    # stylesheet tags in the document.
    mo.Html("""
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

    <div style="padding: 2.5rem 0 2rem; text-align: center;">
      <div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; border-radius: 50%; border: 0.5px solid #1D9E75; margin-bottom: 1rem;">
        <i class="ti ti-chart-candle" style="font-size: 22px; color: #1D9E75;"></i>
      </div>
      <h1 style="font-family: 'DM Serif Display', serif; font-size: 36px; font-weight: 400; font-style: italic; margin: 0 0 6px; letter-spacing: -0.01em; color: var(--color-text-primary);">Pair Trading Strategy</h1>
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
def _():
    from tradingview_screener import Query, col
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # Figures are built via Figure() rather than plt.subplots() so they never
    # enter pyplot's global registry. marimo builds its own figure manager from
    # whatever Figure it is handed, and in a reactive notebook every cell re-run
    # would otherwise leak another pyplot figure that nothing ever closes.
    from matplotlib.figure import Figure
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.tools.tools import add_constant
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.multitest import multipletests
    from joblib import Parallel, delayed
    import yfinance as yf
    import pathlib
    from skfolio import RatioMeasure

    # ML libraries
    import itertools
    from prophet import Prophet
    from prophet.diagnostics import cross_validation
    from prophet.diagnostics import performance_metrics

    # DuckDB for SQL-like queries
    import duckdb

    # Alpaca API
    import os
    from dotenv import load_dotenv
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca_trade_api.rest import REST, TimeFrame

    return (
        MarketOrderRequest,
        Figure,
        OLS,
        OrderSide,
        Parallel,
        Prophet,
        Query,
        REST,
        RatioMeasure,
        TimeFrame,
        TimeInForce,
        TradingClient,
        add_constant,
        adfuller,
        coint,
        col,
        cross_validation,
        delayed,
        duckdb,
        itertools,
        load_dotenv,
        multipletests,
        np,
        os,
        pathlib,
        pd,
        performance_metrics,
        plt,
        yf,
    )


@app.cell
def _(page_header):
    page_header("Stock selection", "01")
    return


@app.cell
def _(section_header):
    section_header("Sector Screener", "01.1") # subsection
    return


@app.cell
def _(mo):
    sector_dropdown = mo.ui.dropdown(
        options=['Technology Services', 'Electronic Technology', 'Finance', 'Health Technology', 'Retail Trade', 'Producer Manufacturing', 'Consumer Non-Durables', 'Consumer Durables', 'Energy Minerals', 'Consumer Services', 'Utilities', 'Non-Energy Minerals', 'Industrial Services', 'Transportation', 'Commercial Services', 'Communications', 'Process Industries', 'Health Services', 'Distribution Services', 'Miscellaneous'],
        label="Sector"
    )

    market_cap_thresholds = {
        "Mega Cap": (200_000_000_000, 5_000_000_000_000),
        "Large Cap": (10_000_000_000, 200_000_000_000),
        "Mid Cap": (2_000_000_000, 10_000_000_000),
        "Small Cap": (300_000_000, 2_000_000_000),
        "Micro Cap": (50_000_000, 300_000_000),
    }

    market_cap_dropdown = mo.ui.dropdown(
        options=list(market_cap_thresholds.keys()),
        label="Market Cap"
    )

    run_button = mo.ui.run_button(label="Screen Stocks")

    mo.vstack([sector_dropdown, market_cap_dropdown, run_button])
    return (
        market_cap_dropdown,
        market_cap_thresholds,
        run_button,
        sector_dropdown,
    )


@app.cell
def _(
    Query,
    col,
    market_cap_dropdown,
    market_cap_thresholds,
    mo,
    pd,
    run_button,
    sector_dropdown,
):
    mo.stop(not run_button.value, mo.md("*Configure filters above and click Screen Stocks.*"))

    sector_input = sector_dropdown.value
    market_cap_input = market_cap_dropdown.value

    low, high = market_cap_thresholds[market_cap_input]

    _, raw = (Query()
        .select('name')
        .where(
            col('market_cap_basic').between(low, high),
            col('sector') == sector_input
        )
        .get_scanner_data())

    sector_result = pd.DataFrame(raw)
    # Tickers arrive as "EXCHANGE:SYMBOL"; drop OTC venues by matching the
    # exchange prefix, not a bare substring (which would also eat a symbol
    # that merely contains the letters OTC).
    sector_result = sector_result[~sector_result['ticker'].str.startswith('OTC:')]

    mo.ui.dataframe(sector_result)
    return market_cap_input, sector_input, sector_result


@app.cell
def _(section_header):
    section_header("Stock OHLCV", "01.2") # subsection
    return


@app.cell
def _(REST, TimeFrame, load_dotenv, mo, os, pd, sector_result):
    # API credentials for Alpaca — loaded from .env (see .env.example).
    # No hardcoded fallback: secrets must never live in source that git can see.
    load_dotenv(mo.notebook_dir() / ".env")
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')
    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Copy .env.example to .env and set "
            "ALPACA_API_KEY and ALPACA_API_SECRET."
        )
    base_url = 'https://paper-api.alpaca.markets/v2'
    api = REST(api_key, api_secret, base_url)
    tickers = sector_result['ticker'].tolist()
    tickers = [t.split(':')[-1] for t in tickers]
    start_date = '2022-01-01'

    # TradingView lists share classes and preferred lines that Alpaca does not
    # trade (e.g. "BEP/PA"). Alpaca rejects the WHOLE request if any symbol in
    # it is invalid, so drop the obviously-unsupported ones before asking.
    # Dotted class tickers like BRK.B are fine and are kept.
    _skipped = [t for t in tickers if '/' in t]
    tickers = [t for t in tickers if '/' not in t]

    # Alpaca's bars endpoint is multi-symbol: one request per chunk instead of
    # one request per ticker. Chunked because the symbol list travels in the
    # query string and a whole sector can be hundreds of names.
    _CHUNK = 100
    _frames = []
    _failed = []

    def _fetch(symbols):
        """Fetch a batch, bisecting on failure to isolate bad symbols.

        A single unknown symbol fails the entire multi-symbol request, so a
        plain try/except around the batch would discard up to 100 good tickers
        along with the one bad one — which is exactly what happened when a
        preferred line slipped into the screen. Halving on failure narrows the
        blame to the individual symbols in log2(n) extra requests instead of
        giving up on the batch or falling back to one request per ticker.
        """
        if not symbols:
            return
        try:
            _bars = api.get_bars(symbols, TimeFrame.Day, start=start_date).df
            if not _bars.empty:
                _frames.append(_bars[['symbol', 'close']])
            return
        except Exception as e:
            if len(symbols) == 1:
                _failed.append((symbols[0], str(e)))
                return
            _mid = len(symbols) // 2
            _fetch(symbols[:_mid])
            _fetch(symbols[_mid:])

    for _start in range(0, len(tickers), _CHUNK):
        _fetch(tickers[_start:_start + _CHUNK])

    if _skipped or _failed:
        print(
            f'Skipped {len(_skipped)} unsupported symbol(s): {_skipped[:10]}\n'
            f'Dropped {len(_failed)} symbol(s) Alpaca rejected: '
            f'{[s for s, _ in _failed][:10]}'
        )

    if not _frames:
        raise RuntimeError(
            f'No price data returned for any of {len(tickers)} tickers. '
            f'{len(_failed)} were rejected by Alpaca'
            + (f' (e.g. {_failed[0][0]}: {_failed[0][1]})' if _failed else '')
            + '. Check that the screen produced US-listed common stock.'
        )

    _all_bars = pd.concat(_frames)
    _all_bars.index = _all_bars.index.tz_localize(None)
    # Long -> wide. pivot_table with 'last' tolerates the duplicate
    # (timestamp, symbol) rows that pagination can hand back; plain pivot raises.
    close_prices = _all_bars.pivot_table(
        index=_all_bars.index, columns='symbol', values='close', aggfunc='last'
    )
    close_prices.index.name = 'timestamp'

    # Two different things produce a NaN here, and they need opposite fixes:
    #   * a stray timestamp only one or two symbols reported (a bad *date*)
    #   * a symbol with a short history — late IPO, delisting (a bad *ticker*)
    # Coverage separates them. A real trading day is reported by most of the
    # sector even when some names are too young; a stray bar is reported by
    # almost nobody. So drop the sparse rows first, then the columns that are
    # still incomplete. Skipping the row pass (the original code went straight
    # to dropna(axis=1)) let a single stray timestamp put a NaN in every other
    # column and wipe the entire frame.
    MIN_ROW_COVERAGE = 0.5
    _coverage = close_prices.notna().sum(axis=1)
    close_prices = close_prices[_coverage >= MIN_ROW_COVERAGE * close_prices.shape[1]]

    adj_close_data = close_prices.dropna(axis=1)
    mo.ui.dataframe(adj_close_data)
    return adj_close_data, api_key, api_secret, base_url


@app.cell
def _(section_header):
    section_header("Train / Test Split", "01.3") # subsection
    return


@app.cell
def _(adj_close_data, mo):
    # Pair selection and performance measurement must not share data. Every
    # selection step below — cointegration, the ADF filter, the OLS hedge ratio
    # — sees ONLY the train window; the backtest and its reported P&L run only
    # on the held-out test window. Fitting and scoring on the same bars means a
    # pair is chosen partly *because* it mean-reverted in the very period used
    # to judge it, which flatters every number downstream.
    SPLIT_FRAC = 0.70

    _n_train = int(len(adj_close_data) * SPLIT_FRAC)
    mo.stop(
        _n_train < 120 or len(adj_close_data) - _n_train < 60,
        mo.md(
            f"Not enough history to split: {len(adj_close_data)} bars gives "
            f"{_n_train} train / {len(adj_close_data) - _n_train} test. "
            "Pull a longer `start_date`."
        ),
    )

    adj_close_train = adj_close_data.iloc[:_n_train]
    adj_close_test = adj_close_data.iloc[_n_train:]
    split_date = adj_close_test.index[0]

    mo.callout(
        mo.md(
            f"**Train** `{adj_close_train.index[0].date()}` → "
            f"`{adj_close_train.index[-1].date()}` ({len(adj_close_train)} bars) — "
            "used to *select* the pair.  \n"
            f"**Test** `{adj_close_test.index[0].date()}` → "
            f"`{adj_close_test.index[-1].date()}` ({len(adj_close_test)} bars) — "
            "used to *evaluate* it. All reported P&L is out-of-sample."
        ),
        kind="info",
    )
    return adj_close_test, adj_close_train, split_date


@app.cell
def _(section_header):
    section_header("Cointegration Test", "01.4") # subsection
    return


@app.cell
def _(
    Figure,
    Parallel,
    adj_close_train,
    coint,
    delayed,
    mo,
    multipletests,
    np,
    plt,
    sector_input,
):
    # Cointegration Test to find the best pairs — TRAIN WINDOW ONLY.
    Ticker_list = adj_close_train.columns.tolist()

    FDR_Q = 0.05

    def cointegration_checker(stock_dataframe, corr_floor=0.7, fdr_q=FDR_Q):
        """Engle-Granger cointegration over every pair.

        Beyond a plain double loop this does four things:

        1. Works on a plain float array. `coint` converts its inputs to arrays
           anyway, so handing it columns of a DataFrame paid for a label lookup
           plus a Series->array conversion on every call.
        2. Pre-screens on price correlation. The regression underlying the test
           only finds a stationary residual when the two series track each
           other, so pairs below `corr_floor` are near-certain rejects.
        3. Tests BOTH regression directions and keeps the stronger. Engle-Granger
           is not symmetric — regressing A on B and B on A give different
           residuals and different p-values, and only one may be stationary.
           Previously whichever direction fell out of the column ordering won,
           which is arbitrary. Taking the better of two looks is itself a
           multiple comparison, so the winner pays a factor-of-2 Bonferroni
           penalty before going forward.
        4. Controls the FALSE DISCOVERY RATE across pairs. This is the one that
           matters. Screening m pairs at a flat p < 0.02 produces ~0.02*m
           spurious "cointegrated" pairs by construction: on 100 independent
           random walks — where the true count is zero — the flat threshold
           flags ~50 of the 4950 pairs. Benjamini-Hochberg at q=`fdr_q` bounds
           the expected proportion of false discoveries among those reported.

        Returns pairs as (dependent, independent, p_raw, q_value).
        """
        keys = list(stock_dataframe.columns)
        k = len(keys)
        values = stock_dataframe.to_numpy(dtype=float)
        p_values = np.ones((k, k))

        # Constant series break both the ADF regression and the correlation
        # matrix (zero stddev -> divide-by-zero -> NaN), so drop them up front
        # and correlate only what is left.
        varying = np.flatnonzero(values.min(axis=0) != values.max(axis=0))
        if len(varying) < 2:
            return p_values, [], (0, 0)

        corr = np.abs(np.corrcoef(values[:, varying], rowvar=False))

        # Indices here are positions within `varying`, mapped back to real
        # column indices when the pair is recorded.
        candidates = [
            (varying[a], varying[b])
            for a in range(len(varying) - 1)
            for b in range(a + 1, len(varying))
            if corr[a, b] >= corr_floor
        ]
        if not candidates:
            return p_values, [], (0, 0)

        # Both orientations of every candidate, in one fan-out. statsmodels'
        # ADF work sits in NumPy/LAPACK, which drops the GIL, so threads give
        # real parallelism without pickling the price matrix to workers.
        jobs = [(i, j) for i, j in candidates] + [(j, i) for i, j in candidates]
        results = Parallel(n_jobs=-1, prefer='threads')(
            delayed(coint)(values[:, a], values[:, b]) for a, b in jobs
        )

        # Winner per unordered pair, with the 2-look Bonferroni penalty.
        best = {}
        for (a, b), coint_test in zip(jobs, results):
            slot = (min(a, b), max(a, b))
            p = coint_test[1]
            if slot not in best or p < best[slot][0]:
                best[slot] = (p, a, b)          # a = dependent, b = independent

        slots = list(best.keys())
        p_raw = np.array([min(2.0 * best[s][0], 1.0) for s in slots])

        # NOTE: `corr_floor` is data-dependent, so the pairs entering this
        # correction were themselves chosen by looking at the data. BH over the
        # tests actually run is therefore an approximation — an optimistic one.
        reject, q_values, _, _ = multipletests(p_raw, alpha=fdr_q, method='fdr_bh')

        cointegrated_pairs = []
        for slot, p, rej, q in zip(slots, p_raw, reject, q_values):
            p_values[slot] = p
            if rej:
                _, dep, indep = best[slot]
                cointegrated_pairs.append((keys[dep], keys[indep], float(p), float(q)))

        return p_values, cointegrated_pairs, (len(slots), int(reject.sum()))

    pvalues, raw_pairs, (n_tested, n_signif) = cointegration_checker(adj_close_train)
    fig = Figure(figsize=(25, 10))
    ax = fig.subplots()
    # Shade the pairs that actually survived the FDR correction, not a flat
    # p-cut — the flat cut is what produced the spurious matches.
    _surviving = np.zeros_like(pvalues, dtype=bool)
    for _d, _i, _p, _q in raw_pairs:
        _a, _b = Ticker_list.index(_d), Ticker_list.index(_i)
        _surviving[min(_a, _b), max(_a, _b)] = True

    im = ax.imshow(_surviving)
    ax.set_xticks(np.arange(len(Ticker_list)))
    ax.set_yticks(np.arange(len(Ticker_list)))
    ax.set_xticklabels(Ticker_list)  # statsmodels coint returns p-values (our primary concern) in the 1th index slot
    ax.set_yticklabels(Ticker_list)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')  #p value matrix where the output of the coint test is the ith, jth index
    ax.set_title(
        f'Cointegration Matrix Of {sector_input} Stocks (train window) — '
        f'{n_signif} of {n_tested} pairs significant at FDR q<{FDR_Q}'
    )

    mo.vstack([
        mo.md(
            f"Tested **{n_tested}** correlated pairs in both directions. "
            f"**{n_signif}** survived Benjamini-Hochberg at q<{FDR_Q}. "
            f"*A flat p<0.02 cut would have passed "
            f"{int((pvalues[pvalues < 1.0] < 0.02).sum())} — most of the extra are noise.*"
        ),
        mo.center(mo.mpl.interactive(fig)),
    ])
    return (raw_pairs,)


@app.cell
def _(section_header):
    section_header("Stock Pair Selection", "01.5") # subsection
    return


@app.cell
def _(OLS, add_constant, adfuller, adj_close_train, mo, np, pd, raw_pairs):
    def stock_pair(candidate_pairs):
        """Keep pairs whose price ratio is ADF-stationary, best p-value first.

        Runs on the TRAIN window only — this is a selection step, and letting it
        see the test bars is what makes an in-sample backtest look good.

        The result is what the rest of the notebook trades: previously this
        filter was computed into a local and thrown away, so the selector
        downstream still offered every cointegrated pair regardless of the
        ADF outcome.
        """
        kept, diagnostics = [], []

        # Ascending: the strongest cointegration (smallest p-value) leads.
        # Sorting descending put the weakest pair at [0] in the selector.
        for pair in sorted(candidate_pairs, key=lambda p: p[2]):
            ticker_1, ticker_2 = pair[1], pair[0]
            Asset_1 = adj_close_train[ticker_1]
            Asset_2 = adj_close_train[ticker_2]
            price_ratio = Asset_1 / Asset_2

            if price_ratio.isin([np.inf, -np.inf]).any():
                continue
            if price_ratio.isna().any():
                continue
            if price_ratio.max() == price_ratio.min():
                continue

            result_price_ratio = adfuller(price_ratio, autolag='AIC')

            const = add_constant(Asset_1)
            results = OLS(Asset_2, const).fit()
            coef = results.params[ticker_1]
            spread = Asset_2 - coef * Asset_1
            result_spread = adfuller(spread, autolag='AIC')  # ← was incorrectly using price_ratio

            ratio_stationary = result_price_ratio[1] <= 0.1
            spread_stationary = result_spread[1] <= 0.1

            # Selection still turns on the ratio, which is what the rolling
            # z-score path trades. The hedge-ratio spread ADF is reported
            # alongside it so a pair that mean-reverts on one construction but
            # not the other is visible rather than silently accepted.
            if ratio_stationary:
                kept.append(pair)
                diagnostics.append({
                    'Dependent': ticker_2,
                    'Independent': ticker_1,
                    'Coint p': f'{pair[2]:.4g}',
                    'FDR q': f'{pair[3]:.4g}',
                    'Ratio ADF p': f'{result_price_ratio[1]:.4f}',
                    'Spread ADF p': f'{result_spread[1]:.4f}',
                    'Spread stationary': '✓' if spread_stationary else '—',
                })

        return kept, pd.DataFrame(diagnostics)


    pairs, pair_diagnostics = stock_pair(raw_pairs)
    mo.stop(
        len(pairs) == 0,
        mo.md(
            f"No eligible pairs found — {len(raw_pairs)} pair(s) cleared the FDR "
            "correction but none had a stationary price ratio (ADF p ≤ 0.1) on "
            "the train window."
        ),
    )

    mo.vstack([
        mo.md(f"**{len(pairs)}** of {len(raw_pairs)} FDR-significant pairs passed the ADF filter."),
        mo.ui.table(pair_diagnostics),
    ])
    # pair_diagnostics is handed downstream for the hub assistant's run record.
    return pair_diagnostics, pairs


@app.cell
def _(duckdb):
    # Cell 1 — load existing pairs from DB
    with duckdb.connect('quant_trading.db') as _con:
        _items_df = _con.execute('SELECT * FROM Assets').df()

    existing_pairs = set(zip(_items_df['Asset1'], _items_df['Asset2'])) if not _items_df.empty else set()
    return (existing_pairs,)


@app.cell
def _(existing_pairs, mo, pairs):
    # Cell 2
    available_pairs = [
        (i, p) for i, p in enumerate(pairs)
        if (p[0], p[1]) not in existing_pairs
    ]

    pair_labels = {f"[{i}] {p[0]} / {p[1]}": (i, p) for i, p in available_pairs}

    pair_selector = mo.ui.dropdown(
        options=list(pair_labels.keys()),
        label="Select a pair",
    )
    pair_selector
    return pair_labels, pair_selector


@app.cell
def _(mo, pair_labels, pair_selector):
    # Cell 3 — confirm selection
    mo.stop(pair_selector.value is None, mo.md("👆 Select a pair above first."))

    index, chosen = pair_labels[pair_selector.value]

    confirm_btn = mo.ui.run_button(label=f"Confirm: {chosen[0]} / {chosen[1]}")

    mo.vstack([
        mo.md(f"Selected **{chosen[0]} / {chosen[1]}**"),
        confirm_btn,
    ])
    return chosen, confirm_btn, index


@app.cell
def _(pathlib):
    # Signal: no pair is ready yet, dependents should wait. Deliberately
    # ungated so a flag left behind by a previous session is cleared the moment
    # this notebook loads.
    pathlib.Path(".pair_ready").unlink(missing_ok=True)
    pathlib.Path(".fundamental_done").unlink(missing_ok=True)

    # Handed to the cell that re-raises the flag at the very end. Without that
    # edge the two cells share no dependency, marimo is free to run them in
    # either order, and the clear can land *after* the raise — leaving
    # downstream notebooks reading a ".pair_ready" that no longer holds.
    pair_flag_cleared = True
    return (pair_flag_cleared,)


@app.cell
def _(chosen, confirm_btn, mo):
    # Cell 4 — save to file
    mo.stop(not confirm_btn.value, mo.md("Press **Confirm** to save the selected pair."))

    with open('pair.txt', 'w') as f:
        print(chosen[0], chosen[1], file=f)

    mo.callout(mo.md(f"✅ Saved **{chosen[0]} / {chosen[1]}** to `pair.txt`"), kind="success")
    return


@app.cell
def _(section_header):
    section_header("Chosen Pair Basic Information", "01.5") # subsection
    return


@app.cell
def _(chosen, confirm_btn, mo, yf):
    def ticker_card(info: dict) -> mo.Html:
        price = info.get("currentPrice", 0)
        prev  = info.get("previousClose", price)
        chg_pct = (price - prev) / prev * 100 if prev else 0
        chg_color = "var(--color-text-success)" if chg_pct >= 0 else "var(--color-text-danger)"
        chg_icon  = "ti-trending-up" if chg_pct >= 0 else "ti-trending-down"

        mktcap = info.get("marketCap", 0)
        mktcap_str = f"${mktcap/1e12:.2f}T" if mktcap >= 1e12 else f"${mktcap/1e9:.1f}B"

        target   = info.get("targetMedianPrice", 0)
        upside   = (target - price) / price * 100 if price else 0
        rev      = info.get("totalRevenue", 0)
        rev_str  = f"${rev/1e9:.1f}B" if rev >= 1e9 else f"${rev/1e6:.0f}M"
        fcf      = info.get("freeCashflow", 0)
        fcf_str  = f"${fcf/1e9:.1f}B" if fcf >= 1e9 else f"${fcf/1e6:.0f}M"

        return mo.Html(f"""
        <div style="background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);border-radius:var(--border-radius-lg);padding:1.25rem;font-size:13px;">
          <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:1rem;">
            <div>
              <div style="display:flex;align-items:center;gap:10px;">
                <span style="font-size:20px;font-weight:500;">{info.get('symbol','')}</span>
                <span style="color:var(--color-text-secondary);">{info.get('shortName','')}</span>
                <span style="font-size:11px;background:#E1F5EE;color:#0F6E56;padding:2px 8px;border-radius:999px;">{info.get('fullExchangeName','')}</span>
              </div>
              <div style="margin-top:4px;color:var(--color-text-secondary);font-size:12px;">
                {info.get('sector','')} · {info.get('industry','')}
              </div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:22px;font-weight:500;">${price:.2f}</div>
              <div style="font-size:12px;color:{chg_color};">
                <i class="ti {chg_icon}"></i> {chg_pct:+.2f}% today
              </div>
            </div>
          </div>

          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-bottom:1rem;">
            {"".join(f'<div style="background:var(--color-background-secondary);border-radius:var(--border-radius-md);padding:10px 12px;"><div style="font-size:11px;color:var(--color-text-secondary);margin-bottom:4px;">{label}</div><div style="font-size:16px;font-weight:500;">{value}</div></div>'
            for label, value in [
                ("Market cap",  mktcap_str),
                ("Trailing P/E", f"{info.get('trailingPE',0):.2f}"),
                ("Forward P/E",  f"{info.get('forwardPE',0):.2f}"),
                ("EPS (TTM)",    f"${info.get('trailingEps',0):.2f}"),
            ])}
          </div>

          <div style="display:grid;grid-template-columns:1fr 1fr;border-top:0.5px solid var(--color-border-tertiary);padding-top:8px;gap:0;">
            <table style="width:100%;border-collapse:collapse;font-size:12px;">
              {"".join(f'<tr><td style="color:var(--color-text-secondary);padding:4px 8px 4px 0;">{l}</td><td style="text-align:right;font-weight:500;">{v}</td></tr>'
              for l,v in [
                ("52w range",  f"${info.get('fiftyTwoWeekLow',0):.2f} – ${info.get('fiftyTwoWeekHigh',0):.2f}"),
                ("50d avg",    f"${info.get('fiftyDayAverage',0):.2f}"),
                ("200d avg",   f"${info.get('twoHundredDayAverage',0):.2f}"),
                ("Beta",       f"{info.get('beta',0):.2f}"),
                ("Volume",     f"{info.get('volume',0)/1e6:.1f}M"),
                ("Avg volume", f"{info.get('averageVolume',0)/1e6:.1f}M"),
              ])}
            </table>
            <table style="width:100%;border-collapse:collapse;font-size:12px;padding-left:1rem;border-left:0.5px solid var(--color-border-tertiary);">
              {"".join(f'<tr><td style="color:var(--color-text-secondary);padding:4px 8px 4px 1rem;">{l}</td><td style="text-align:right;font-weight:500;">{v}</td></tr>'
              for l,v in [
                ("Revenue",      rev_str),
                ("Gross margin", f"{info.get('grossMargins',0)*100:.2f}%"),
                ("Net margin",   f"{info.get('profitMargins',0)*100:.2f}%"),
                ("ROE",          f"{info.get('returnOnEquity',0)*100:.2f}%"),
                ("Debt/equity",  f"{info.get('debtToEquity',0):.2f}"),
                ("Free cashflow",fcf_str),
              ])}
            </table>
          </div>

          <div style="border-top:0.5px solid var(--color-border-tertiary);margin-top:8px;padding-top:10px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
            <div style="display:flex;gap:16px;font-size:12px;color:var(--color-text-secondary);">
              <span>Div yield: <b style="color:var(--color-text-primary);">{info.get('dividendYield',0)*100:.2f}%</b></span>
              <span>Short float: <b style="color:var(--color-text-primary);">{info.get('shortPercentOfFloat',0)*100:.2f}%</b></span>
              <span>Analyst: <b style="color:#3AC493;">{info.get('recommendationKey','').capitalize()} ({info.get('recommendationMean',0):.2f})</b></span>
            </div>
            <div style="display:flex;gap:6px;">
              <span style="font-size:11px;background:#E6F1FB;color:#0C447C;padding:2px 8px;border-radius:999px;">Target ${target:.0f}</span>
              <span style="font-size:11px;background:{'#EAF3DE' if upside>=0 else '#FCEBEB'};color:{'#27500A' if upside>=0 else '#791F1F'};padding:2px 8px;border-radius:999px;">{upside:+.2f}% upside</span>
            </div>
          </div>
        </div>
        """)

    mo.stop(not confirm_btn.value)

    info1 = yf.Ticker(chosen[0]).info
    info2 = yf.Ticker(chosen[1]).info

    mo.hstack([ticker_card(info1), ticker_card(info2)], gap="1rem")
    return


@app.cell
def _(page_header):
    page_header("Portofolio Process", "02")  
    return


@app.cell
def _(confirm_btn, mo):
    # Cell 2 — run button gate
    mo.stop(not confirm_btn.value)

    run_btn = mo.ui.run_button(label="▶ Run Portfolio Pipeline")
    run_btn
    return (run_btn,)


@app.cell
def _(section_header):
    section_header("Model Parameter Grid Seach Optimization", "02.1")
    return


@app.cell
def _(mo, run_btn):
    # Cell 3 — data loading
    import portfolio
    mo.stop(not run_btn.value, mo.md("Press **Run Portfolio Pipeline** to start."))

    with mo.status.spinner(title="Fetching price data from Alpaca..."):
        X, X_train, X_test = portfolio.data()

    mo.callout(
        mo.md(f"✅ Data loaded — **{X.shape[1]} assets**, **{len(X_train)}** train / **{len(X_test)}** test observations"),
        kind="success"
    )
    return X_test, X_train, portfolio


@app.cell
def _(section_header):
    section_header("Combinatorial Purged Cross-Validation for Backtesting", "02.2")
    return


@app.cell
def _(X_train, mo, portfolio, run_btn):
    # Cell 4 — model fitting
    mo.stop(not run_btn.value)

    with mo.status.spinner(title="Grid-searching HRP & HERC models (this may take a while)..."):
        model_hrp, model_herc, cv = portfolio.portfolio_model(X_train)

    mo.callout(mo.md("✅ Models fitted."), kind="success")
    return cv, model_herc, model_hrp


@app.cell
def _(X_test, cv, mo, model_herc, model_hrp, portfolio, run_btn):
    # Cell 5 — cross-val predictions
    mo.stop(not run_btn.value)

    with mo.status.spinner(title="Running walk-forward cross-validation..."):
        population, pred_hrp, pred_herc = portfolio.cross_predict(model_hrp, model_herc, X_test, cv)

    mo.callout(mo.md("✅ Cross-validation complete."), kind="success")
    return population, pred_herc, pred_hrp


@app.cell
def _(section_header):
    section_header("Portfolio Statistics", "02.3") # subsection
    return


@app.cell
def _(X_test, cv):
    cv.summary(X_test)
    return


@app.cell
def _(RatioMeasure, mo, population, portfolio, pred_herc, pred_hrp, run_btn):
    # Cell 6 — statistics report
    mo.stop(not run_btn.value)

    summary_df = portfolio.statistics_report(population)

    mo.vstack([
        mo.md(f"""
    ### HRP ({len(pred_hrp)} paths)
    - **Average Mean-CVaR ratio:** {pred_hrp.measures_mean(RatioMeasure.CVAR_RATIO):.4f}
    - **Std Mean-CVaR ratio :** {pred_hrp.measures_std(measure=RatioMeasure.CVAR_RATIO):.4f}

    ### HERC ({len(pred_herc)} paths)
    - **Average Mean-CVaR ratio:** {pred_herc.measures_mean(RatioMeasure.CVAR_RATIO):.4f}
    - **Std Mean-CVaR ratio :** {pred_herc.measures_std(measure=RatioMeasure.CVAR_RATIO):.4f}
        """)
    ])
    return


@app.cell
def _(RatioMeasure, mo, np, pd, pred_herc, pred_hrp):
    def pop_metric(pop, fn):
        """Apply fn to each portfolio in the population, return mean across paths."""
        vals = [fn(p) for p in pop]
        return float(np.mean(vals))

    metrics_def = [
        ("Annualized Return",    lambda p: p.annualized_mean,                  False),
        ("Annualized Volatility",lambda p: p.annualized_standard_deviation,    True),
        ("Sharpe Ratio",         lambda p: p.sharpe_ratio,                     False),
        ("CVaR (95%)",           lambda p: p.cvar,                             True),
        ("Mean-CVaR Ratio",      lambda p: p.cvar_ratio,                       False),
        ("Max Drawdown",         lambda p: p.max_drawdown,                     True),
        ("Calmar Ratio",         lambda p: p.calmar_ratio,                     False),
        ("Sortino Ratio",        lambda p: p.sortino_ratio,                    False),
        # CPCV-specific: stability of CVaR ratio across paths
        ("CVaR Ratio Std",       lambda p: p.cvar_ratio,                       True),
    ]

    rows = []
    for name, fn, lower_better in metrics_def:
        if name == "CVaR Ratio Std":
            hrp_val  = float(pred_hrp.measures_std(measure=RatioMeasure.CVAR_RATIO))
            herc_val = float(pred_herc.measures_std(measure=RatioMeasure.CVAR_RATIO))
        else:
            hrp_val  = pop_metric(pred_hrp,  fn)
            herc_val = pop_metric(pred_herc, fn)

        is_pct = name in {"Annualized Return", "Annualized Volatility",
                          "CVaR (95%)", "Max Drawdown"}
        fmt = "{:.2%}" if is_pct else "{:.4f}"

        if lower_better:
            winner = "HRP ✓" if hrp_val < herc_val else "HERC ✓"
        else:
            winner = "HRP ✓" if hrp_val > herc_val else "HERC ✓"

        rows.append({
            "Metric":  name,
            "HRP":     fmt.format(hrp_val),
            "HERC":    fmt.format(herc_val),
            "Winner":  winner,
        })

    comparison_df = pd.DataFrame(rows)
    hrp_wins  = sum(1 for r in rows if r["Winner"].startswith("HRP"))
    herc_wins = sum(1 for r in rows if r["Winner"].startswith("HERC"))
    overall   = "HRP" if hrp_wins >= herc_wins else "HERC"

    # CVaR ratio mean ± std summary (the key CPCV diagnostic)
    hrp_mean  = pred_hrp.measures_mean(measure=RatioMeasure.CVAR_RATIO)
    hrp_std   = pred_hrp.measures_std(measure=RatioMeasure.CVAR_RATIO)
    herc_mean = pred_herc.measures_mean(measure=RatioMeasure.CVAR_RATIO)
    herc_std  = pred_herc.measures_std(measure=RatioMeasure.CVAR_RATIO)

    mo.vstack([
        mo.md("## Head-to-Head: HRP vs HERC (CPCV)"),
        mo.ui.table(comparison_df),
        mo.md(f"""
    ### CVaR Ratio across {len(pred_hrp)} test paths
    | | Mean | Std | Signal |
    |---|---|---|---|
    | HRP  | `{hrp_mean:.4f}` | `{hrp_std:.4f}` | {"✅ more stable" if hrp_std < herc_std else ""} |
    | HERC | `{herc_mean:.4f}` | `{herc_std:.4f}` | {"✅ more stable" if herc_std < hrp_std else ""} |

    *Lower std = more consistent across market regimes*
        """),
        mo.callout(
            mo.md(f"**Overall winner: {overall}** ({hrp_wins} vs {herc_wins} metrics)"),
            kind="success" if overall == "HRP" else "info",
        ),
    ])
    return (overall,)


@app.cell
def _(mo, population):
    # Cell 6c — overlaid returns
    fig_compare = population.plot_cumulative_returns()

    mo.vstack([
        mo.md("## Cumulative Returns: HRP vs HERC"),
        mo.as_html(fig_compare),
    ])
    return


@app.cell
def _(RatioMeasure, mo, pred_herc, pred_hrp):
    # Cell 6d — weight compositions

    best_hrp  = pred_hrp.max_measure(RatioMeasure.CVAR_RATIO)
    best_herc = pred_herc.max_measure(RatioMeasure.CVAR_RATIO)

    fig_hrp_comp  = best_hrp.plot_composition()
    fig_herc_comp = best_herc.plot_composition()

    mo.vstack([
        mo.md("## Portfolio Compositions"),
        mo.hstack([
            mo.vstack([mo.md("### HRP"),  mo.as_html(fig_hrp_comp)]),
            mo.vstack([mo.md("### HERC"), mo.as_html(fig_herc_comp)]),
        ]),
    ])
    return best_herc, best_hrp


@app.cell
def _(RatioMeasure, mo, population):
    fig_cvar = population.plot_distribution(
        measure_list=[RatioMeasure.CVAR_RATIO], tag_list=["HRP", "HERC"], n_bins=50
    )

    mo.as_html(fig_cvar)    
    return


@app.cell
def _(RatioMeasure, mo, population):
    fig_sharpe = population.plot_distribution(
        measure_list=[
            RatioMeasure.ANNUALIZED_SHARPE_RATIO,
            RatioMeasure.ANNUALIZED_SORTINO_RATIO,
        ],
        tag_list=["HRP", "HERC"],
        n_bins=50,
    )

    mo.as_html(fig_sharpe)
    return


@app.cell
def _(best_herc, best_hrp, mo, overall):
    model_choice = mo.ui.radio(
        options={"HRP (best path)": best_hrp, "HERC (best path)": best_herc},
        value="HRP (best path)",
        label="Model to save",
    )
    save_btn = mo.ui.run_button(label="💾 Save selected weights to DuckDB")

    mo.vstack([
        mo.md(f"**Recommended: {overall}** based on head-to-head above"),
        model_choice,
        save_btn,
    ])
    return model_choice, save_btn


@app.cell
def _(mo, model_choice, portfolio, save_btn):
    # Cell 9
    mo.stop(not save_btn.value, mo.md("Select a model and press **Save**."))

    portfolio.portfolio_ratio_duckdb(model=model_choice.value)
    mo.callout(
        mo.md(f"✅ **{model_choice.value if isinstance(model_choice.value, str) else 'Selected'}** weights saved to `quant_trading.db` → `Portfolio` table."),
        kind="success",
    )
    return


@app.cell
def _(duckdb, mo, pd, save_btn):
    mo.stop(not save_btn.value)

    with duckdb.connect('quant_trading.db') as _con:
        portfolio_df = _con.execute('SELECT * FROM Portfolio').df()
        latest_portfolio = portfolio_df.sort_values('id', ascending=False).iloc[0]
        latest_portfolio_df = pd.DataFrame([latest_portfolio])

    mo.ui.dataframe(latest_portfolio_df)
    return (latest_portfolio_df,)


@app.cell
def _(page_header):
    page_header("EDA", "03")  
    return


@app.cell
def _(Figure, adj_close_data, index, mo, pairs, plt, save_btn, split_date):
    mo.stop(not save_btn.value)


    figure = Figure(figsize=(12, 8))
    axis = figure.subplots()
    adj_close_data[pairs[index][1]].plot(ax=axis, color='black')
    adj_close_data[pairs[index][0]].plot(ax=axis, color='royalblue')
    axis.axvline(split_date, color='crimson', linestyle='--', linewidth=1)
    axis.legend([pairs[index][1], pairs[index][0], 'train | test'], prop={'size': 15})
    axis.set_ylabel('Price In U.S. Dollars')

    mo.center(mo.mpl.interactive(figure))
    return


@app.cell
def _(
    Figure,
    OLS,
    add_constant,
    adj_close_data,
    adj_close_train,
    index,
    mo,
    pairs,
    pd,
    plt,
    save_btn,
    split_date,
):
    mo.stop(not save_btn.value)


    adj_close_data_1 = pd.DataFrame(adj_close_data)
    Asset_1 = adj_close_data_1[pairs[index][1]]
    Asset_2 = adj_close_data_1[pairs[index][0]]

    # Hedge ratio is a fitted parameter, so it comes from the train window only.
    # Fitting it on all history and then "backtesting" the resulting spread
    # leaks the test period into the position the strategy would have held.
    results = OLS(
        adj_close_train[pairs[index][0]],
        add_constant(adj_close_train[pairs[index][1]]),
    ).fit()
    hedge_coef = results.params[pairs[index][1]]
    _spread = Asset_2 - hedge_coef * Asset_1

    begin_date = adj_close_data_1.index[0].strftime('%Y-%m-%d')
    end_date = adj_close_data_1.index[-1].strftime('%Y-%m-%d')

    figure1 = Figure(figsize=(15, 10))
    axis1 = figure1.subplots()
    _spread.plot(ax=axis1, color='black')
    axis1.set_xlim(begin_date, end_date)
    # Equilibrium level is a fitted quantity too — take it from train.
    axis1.axhline(_spread.loc[:split_date].mean(), color='red', linestyle=':')
    axis1.axvline(split_date, color='crimson', linestyle='--', linewidth=1)
    axis1.legend(
        [f'Spread Between {pairs[index][1]} and {pairs[index][0]}',
         'Train mean', 'train | test'],
        prop={'size': 12},
    )
    axis1.set_title(
        f'Spread Between {pairs[index][1]} and {pairs[index][0]} '
        f'(hedge ratio {hedge_coef:.4f}, fitted on train)'
    )

    mo.center(mo.mpl.interactive(figure1))
    return Asset_1, Asset_2, adj_close_data_1, begin_date, end_date, hedge_coef


@app.cell
def _(Asset_1, Asset_2, adfuller, hedge_coef, mo, save_btn):
    mo.stop(not save_btn.value)


    price_ratio = Asset_1 / Asset_2
    # Same hedge-ratio spread that the chart above plots. Testing the raw
    # difference Asset_1 - Asset_2 here (as this cell used to) reported the
    # stationarity of a series nobody looks at or trades.
    _spread = Asset_2 - hedge_coef * Asset_1

    def adf_card(label, series):
        result = adfuller(series, autolag='AIC')
        crit_rows = "\n".join(
            f"- **{k}:** {v:.4f}" for k, v in result[4].items()
        )
        return mo.callout(mo.md(f"""
    ### {label}
    - **ADF Statistic:** {result[0]:.4f}
    - **n_lags:** {result[2]}
    - **p-value:** {result[1]:.4f}

    **Critical Values:**
    {crit_rows}
    """), kind="info")

    mo.hstack([
        adf_card("Price Ratio", price_ratio),
        adf_card("Spread", _spread),
    ])
    return (price_ratio,)


@app.cell
def _(Figure, begin_date, end_date, index, mo, np, pairs, plt, price_ratio, save_btn):
    mo.stop(not save_btn.value)


    # pandas .std() (ddof=1) rather than np.std (ddof=0), to match the sample
    # standard deviation used everywhere else in the notebook.
    price_ratio_z_score = (price_ratio - price_ratio.mean()) / price_ratio.std()

    figure2 = Figure(figsize=(15, 10))
    axis2 = figure2.subplots()
    price_ratio_z_score.plot(ax=axis2, color='black')
    axis2.axhline(price_ratio_z_score.mean(), color='darkgrey')
    axis2.axhline(1, color='tomato', linestyle='dashed')
    axis2.axhline(2, color='darkred', alpha=.4)
    axis2.axhline(-1, color='limegreen', linestyle='dashed')
    axis2.axhline(-2, color='darkgreen', alpha=.4)
    axis2.set_xlim(begin_date, end_date)
    axis2.legend([f'Price Ratio Z Score Of {pairs[index][0]} and {pairs[index][1]}', 'Mean', 'Z+=1', 'Z+=2', 'Z-=1', 'Z-=2'])
    axis2.set_title(f"Price Ratio Z Score Between {pairs[index][1]} and {pairs[index][0]}")
    axis2.set_ylabel('Z Score')

    mo.center(mo.mpl.interactive(figure2))
    return


@app.cell
def _(Figure, index, mo, pairs, plt, price_ratio, save_btn):
    mo.stop(not save_btn.value)


    price_ratio_10D_MAVG = price_ratio.rolling(window=10, center=False).mean()
    price_ratio_60D_MAVG = price_ratio.rolling(window=60, center=False).mean()

    figure3 = Figure(figsize=(15, 10))
    axis3 = figure3.subplots()
    price_ratio.plot(ax=axis3, color='black')
    price_ratio_10D_MAVG.plot(ax=axis3, color='magenta', linewidth=2, alpha=.8)
    price_ratio_60D_MAVG.plot(ax=axis3, color='b', linewidth=3)
    axis3.axhline(price_ratio.mean(), color='darkgrey', linestyle='dashed')
    axis3.legend(['Actual Price Ratio', '10d Ratio Moving Avg', '60d Ratio Moving Avg', 'Price Ratio Mean'])
    axis3.set_title(f"Moving Average for prices between {pairs[index][1]} and {pairs[index][0]}")
    axis3.set_ylabel('(Moving) Price Ratio')

    mo.center(mo.mpl.interactive(figure3))
    return price_ratio_10D_MAVG, price_ratio_60D_MAVG


@app.cell
def _(
    Figure,
    begin_date,
    end_date,
    index,
    mo,
    pairs,
    plt,
    price_ratio,
    price_ratio_10D_MAVG,
    price_ratio_60D_MAVG,
    save_btn,
):
    mo.stop(not save_btn.value)


    STD_60 = price_ratio.rolling(window=60, center=False).std()
    Rolling_Z_Score = (price_ratio_10D_MAVG - price_ratio_60D_MAVG) / STD_60

    figure4 = Figure(figsize=(15, 10))
    axis4 = figure4.subplots()
    Rolling_Z_Score.plot(ax=axis4, color='black')
    axis4.set_xlim(begin_date, end_date)
    axis4.axhline(0, color='black')
    axis4.axhline(1, color='tomato', linestyle='dashed')
    axis4.axhline(-1, color='limegreen', linestyle='dashed')
    axis4.legend(['Rolling Z-Score', 'Mean', 'Z=+1', 'Z=-1'])
    axis4.set_title(f"Rolling Z-Score of Price Ratio Between {pairs[index][1]} and {pairs[index][0]}")
    axis4.set_ylabel('Z Score')

    mo.center(mo.mpl.interactive(figure4))
    return (Rolling_Z_Score,)


@app.cell
def _(
    Asset_1,
    Asset_2,
    Figure,
    Rolling_Z_Score,
    index,
    mo,
    pairs,
    plt,
    price_ratio,
    save_btn,
):
    mo.stop(not save_btn.value)


    buy = price_ratio.copy()
    sell = price_ratio.copy()
    buy[Rolling_Z_Score > -1] = 0
    sell[Rolling_Z_Score < 1] = 0

    S1 = Asset_1
    S2 = Asset_2

    buyR = 0 * S1.copy()
    sellR = 0 * S1.copy()

    buyR[buy != 0] = S1[buy != 0]
    sellR[buy != 0] = S2[buy != 0]

    buyR[sell != 0] = S2[sell != 0]
    sellR[sell != 0] = S1[sell != 0]

    figure5 = Figure(figsize=(12, 7))
    axis5 = figure5.subplots()
    S1[60:].plot(ax=axis5, color='b')
    S2[60:].plot(ax=axis5, color='c')
    buyR[60:].plot(ax=axis5, color='g', linestyle='None', marker='^')
    sellR[60:].plot(ax=axis5, color='r', linestyle='None', marker='^')

    axis5.set_ylim(min(S1.min(), S2.min()), max(S1.max(), S2.max()))
    axis5.legend([pairs[index][1], pairs[index][0], 'Buy Signal', 'Sell Signal'])
    axis5.set_title(f"Trading Signals: {pairs[index][1]} vs {pairs[index][0]}")

    mo.center(mo.mpl.interactive(figure5))
    return


@app.cell
def _(page_header):
    page_header("Trading Operation", "04")  
    return


@app.cell
def _(np):
    # Kalman Filter for Pair Trading: avoid Moving Average and have better estimate of z_score
    class MyKalmanFilter:
        def __init__(self, delta=1e-4, R=1e-3, P0=1e4):
            # measurement noise variance
            self.R = R

            # co-variance of process noise(2 dimensions)
            self.Q = delta / (1-delta) * np.eye(2)

            # state (slope, intercept) history as a list of (2 x 1) columns.
            # Appending to a list is O(1); re-concatenating an array every step
            # (the old approach) was O(n^2) over the whole series.
            self._states = [np.zeros((2, 1))]

            # State covariance, initialised diffuse (large) rather than zero.
            # Starting at zero claims total confidence in the [0, 0] seed, so
            # the innovation variance S stays small while the residual is the
            # full price level — the first standardized residuals came out in
            # the tens and tripped the entry threshold on bar 0. A diffuse
            # prior says "the seed is meaningless", which keeps the early
            # z-scores near zero and lets the filter converge in a few steps.
            self.P = P0 * np.eye(2)

        @property
        def x(self):
            # Materialize the full (2 x n) state history on demand for plotting.
            return np.concatenate(self._states, axis=1)

        def step_forward(self, y1, y2):
            # Before entering the equations, let's define H as (1, 2) matrix
            H = np.array([y2, 1])[None]
            # and define z
            z = y1

            ## TIME UPDATE ##
            # first thing is to predict new state as the previous one (2x1)
            x_hat = self._states[-1]

            # then, the uncertainty or covariance prediction
            P_hat = self.P + self.Q

            ## MEASUREMENT UPDATE ##
            # innovation (residual) variance S = H P_hat H^T + R  (1x1 scalar).
            # This is the conditional variance of the spread; sqrt(S) is the
            # natural scale for standardizing the residual into a z-score.
            S = H.dot(P_hat.dot(H.T)) + self.R

            # calc the Kalman gain
            K = P_hat.dot(H.T) / S

            # state update part 1 (measurement estimation)
            z_hat = H.dot(x_hat)
            # state update part 2
            x = x_hat + K.dot(z-z_hat)

            # uncertainty update
            self.P = (np.eye(2)-K.dot(H)).dot(P_hat)

            # append the new state to the history (O(1))
            self._states.append(x)

            return x, P_hat, K, z_hat, S

    return (MyKalmanFilter,)


@app.cell
def _(TradingClient, api_key, api_secret, mo, save_btn):
    mo.stop(not save_btn.value)


    trading_client = TradingClient(api_key, api_secret, paper=True)
    account = trading_client.get_account()

    status = mo.callout(
        mo.md("⚠️ **Account is currently restricted from trading.**"),
        kind="warn"
    ) if account.trading_blocked else mo.callout(
        mo.md("✅ **Account is active and allowed to trade.**"),
        kind="success"
    )

    mo.vstack([
        status,
        mo.md(f"**Buying Power Available:** ${float(account.buying_power):,.2f}")
    ])
    return account, trading_client


@app.cell
def _(
    account,
    hedge_ratio_series,
    index,
    latest_portfolio_df,
    mo,
    np,
    pairs,
    pd,
    series,
):
    CAPITAL_FRACTION = 0.1   # share of the pair's weighted buying power at risk

    def trading_simulation(Asset1, Asset2, window1, window2, stop_loss_pct, Kalman_Filter):
        """Frictionless backtest — no commission, slippage, or borrow cost.

        Matches the paper-trading account this notebook submits to, which is not
        charged fees. The reported P&L is therefore an upper bound on live
        performance, not an estimate of it: it is the result before any
        execution friction. For reference, on the MSEX/CLNE run this measured
        ~106 bps of round-trip friction to break even, against a realistic
        small-cap drag of roughly 24 bps.
        """
        price_ratio = Asset1 / Asset2
        moving_average1 = price_ratio.rolling(window=window1).mean()
        moving_average2 = price_ratio.rolling(window=window2).mean()
        std = price_ratio.rolling(window=window2).std()
        z_score = (moving_average1 - moving_average2) / std

        # Align the signal to the price index by LABEL. `series` is built from
        # its own dropna'd frame, so the old positional reset_index paired
        # signal row N with price row N and would silently misalign the two the
        # moment their lengths diverged.
        if Kalman_Filter:
            signal = series.reindex(Asset1.index)
        else:
            signal = z_score

        profit, profit_high, profit_low = (0, 0, 0)
        ratio_high_sell, ratio_high_buy, ratio_low_buy, ratio_low_sell = (0, 0, 0, 0)
        low_trade_total, high_trade_total = (0, 0)
        high_hit_rate, low_hit_rate = ([], [])
        low_dic, high_dic = ({}, {})
        Asset1_shares, Asset2_shares = (0, 0)
        open_trade = 0
        # Kept (without any cost attached) because the proportional stop-loss
        # is expressed as a percentage of the notional actually put on.
        entry_notional = 0.0
        # Total dollars crossed, entry + exit. No cost is charged against it —
        # it only lets the summary report how much friction the result could
        # absorb before it breaks even, which is the number that tells you
        # whether a frictionless edge would survive real execution.
        _traded_notional = 0.0
        current_trade_type = None
        stopped_out = False
        stop_loss_exits = 0
        # Collect log rows in a list and build the DataFrame once after the loop.
        # Re-concatenating a DataFrame per trade (the old approach) was O(n^2).
        log_rows = []

        # Sizing budget: one capital pool for the pair.
        bp = float(account.buying_power)
        weight_1 = float(latest_portfolio_df[pairs[index][1]].iloc[0])
        weight_2 = float(latest_portfolio_df[pairs[index][0]].iloc[0])
        pair_capital = (weight_1 + weight_2) * bp * CAPITAL_FRACTION
        leg_notional = pair_capital / 2

        # The hedge has to match the spread the SIGNAL trades, otherwise the
        # position is exposed to moves the signal never claimed to predict:
        #   ratio path  : trades A1/A2, which is neutral to equal % moves
        #                 -> equal notional per leg
        #   Kalman path : trades A2 - beta*A1 in levels
        #                 -> beta shares of A1 per share of A2
        # These coincide only when the cointegrating intercept is ~0; with a
        # large intercept the equal-notional hedge drifts materially (a 34%
        # under-hedge on a pair with alpha=100 on a $316 stock).
        betas = (
            hedge_ratio_series.reindex(Asset1.index).to_numpy(dtype=float)
            if Kalman_Filter else None
        )

        # Pull the loop off pandas: .iloc on every bar dominated the runtime.
        dates = Asset1.index
        px1 = Asset1.to_numpy(dtype=float)
        px2 = Asset2.to_numpy(dtype=float)
        sig = signal.to_numpy(dtype=float)

        for _i in range(len(dates)):
            date = dates[_i]
            p1, p2 = px1[_i], px2[_i]
            kalman_val = sig[_i]
            # NaN over the rolling / Kalman warm-up. No signal means no entry,
            # but an already-open position still has to be risk-checked below.
            has_signal = not np.isnan(kalman_val)

            # A stop-out means the spread kept diverging. Re-arming on the very
            # next bar just re-opens the same losing position — the stop has to
            # wait for the spread to actually normalise before trading it again.
            if stopped_out and has_signal and abs(kalman_val) < 0.5:
                stopped_out = False

            # --- Entry Logic ---
            if open_trade == 0 and not stopped_out and has_signal and abs(kalman_val) > 1.25:
                # Whole shares — a fractional quantity is not executable on the
                # GTC orders this log later feeds.
                if betas is not None and np.isfinite(betas[_i]) and betas[_i] > 0:
                    # Hedge-consistent: shares in beta:1, scaled to the budget.
                    _q = pair_capital / (betas[_i] * p1 + p2)
                    Asset2_shares = int(_q)
                    Asset1_shares = int(betas[_i] * _q)
                else:
                    # Equal notional per leg (ratio-spread hedge).
                    Asset1_shares = int(leg_notional // p1)
                    Asset2_shares = int(leg_notional // p2)

                # Too little capital to hold both legs; a one-legged "pair"
                # trade is just an outright directional bet.
                if Asset1_shares > 0 and Asset2_shares > 0:
                    entry_notional = p1 * Asset1_shares + p2 * Asset2_shares
                    _traded_notional += entry_notional
                    open_trade = 1

                    if kalman_val > 1.25:
                        ratio_high_sell = p1 * Asset1_shares
                        ratio_high_buy = p2 * Asset2_shares
                        high_trade_total += 1
                        current_trade_type = 'high'
                        log_rows.extend([
                            {'Date': date, 'Asset': pairs[index][1], 'Action': 'Sell', 'Price': p1, 'Quantity': Asset1_shares, 'Trade_Type': 'High'},
                            {'Date': date, 'Asset': pairs[index][0], 'Action': 'Buy',  'Price': p2, 'Quantity': Asset2_shares, 'Trade_Type': 'High'}
                        ])
                    else:
                        ratio_low_buy  = p1 * Asset1_shares
                        ratio_low_sell = p2 * Asset2_shares
                        low_trade_total += 1
                        current_trade_type = 'low'
                        log_rows.extend([
                            {'Date': date, 'Asset': pairs[index][1], 'Action': 'Buy',  'Price': p1, 'Quantity': Asset1_shares, 'Trade_Type': 'Low'},
                            {'Date': date, 'Asset': pairs[index][0], 'Action': 'Sell', 'Price': p2, 'Quantity': Asset2_shares, 'Trade_Type': 'Low'}
                        ])

            # --- Exit Logic ---
            # Bug 4 Fix: only enter exit block when trade is open
            elif open_trade == 1:
                # Mark-to-market of the *active* side only. The old code scaled
                # every term by count_high / count_low, which — because entries
                # are gated on open_trade == 0 and the counters reset on close —
                # were always exactly 1 while a position was open.
                if current_trade_type == 'high':
                    open_pnl = (ratio_high_sell - p1 * Asset1_shares) + \
                               (p2 * Asset2_shares - ratio_high_buy)
                else:
                    open_pnl = (p1 * Asset1_shares - ratio_low_buy) + \
                               (ratio_low_sell - p2 * Asset2_shares)

                # Stop as a FRACTION of the notional actually put on. An absolute
                # dollar stop is not scale-free: the same $5,000 is a hard stop
                # on a small account and unreachable on a large one, because
                # position size scales with buying power.
                stop_loss_hit = open_pnl < -(stop_loss_pct / 100.0) * entry_notional

                # Close the trade on a stop-loss hit OR a normal mean-reversion
                # exit. Both cases run identical accounting, so they share one block.
                if stop_loss_hit or (has_signal and abs(kalman_val) < 0.5):
                    realised = open_pnl
                    _traded_notional += p1 * Asset1_shares + p2 * Asset2_shares
                    if stop_loss_hit:
                        stopped_out = True
                        stop_loss_exits += 1

                    if current_trade_type == 'high':
                        profit_high += realised
                        high_hit_rate.append(realised)
                        high_dic[date.strftime('%Y-%m-%d')] = realised
                        # Bug 2 Fix: correct closing actions for high trade
                        log_rows.extend([
                            {'Date': date, 'Asset': pairs[index][1], 'Action': 'Buy',  'Price': p1, 'Quantity': Asset1_shares, 'Trade_Type': 'Close'},
                            {'Date': date, 'Asset': pairs[index][0], 'Action': 'Sell', 'Price': p2, 'Quantity': Asset2_shares, 'Trade_Type': 'Close'}
                        ])
                    else:
                        profit_low += realised
                        low_hit_rate.append(realised)
                        low_dic[date.strftime('%Y-%m-%d')] = realised
                        # Bug 2 Fix: correct closing actions for low trade
                        log_rows.extend([
                            {'Date': date, 'Asset': pairs[index][1], 'Action': 'Sell', 'Price': p1, 'Quantity': Asset1_shares, 'Trade_Type': 'Close'},
                            {'Date': date, 'Asset': pairs[index][0], 'Action': 'Buy',  'Price': p2, 'Quantity': Asset2_shares, 'Trade_Type': 'Close'}
                        ])

                    ratio_high_sell, ratio_high_buy, ratio_low_buy, ratio_low_sell = (0, 0, 0, 0)
                    open_trade = 0
                    entry_notional = 0.0
                    current_trade_type = None

        # Build the trade log once from the collected rows (single concat).
        trading_log = pd.DataFrame(
            log_rows,
            columns=['Date', 'Asset', 'Action', 'Price', 'Quantity', 'Trade_Type'],
        )

        # --- Summary Stats ---
        profit = profit_low + profit_high

        # Clamp at 0: with no losing trades, min() is the SMALLEST GAIN, and
        # abs() was reporting that profit as if it were the biggest loss.
        high_biggest_loss = abs(min(min(high_hit_rate), 0.0)) if high_hit_rate else 0
        high_biggest_gain = max(max(high_hit_rate), 0.0) if high_hit_rate else 0
        low_biggest_loss  = abs(min(min(low_hit_rate), 0.0)) if low_hit_rate else 0
        low_biggest_gain  = max(max(low_hit_rate), 0.0) if low_hit_rate else 0

        high_hit_rate_pct = len([x for x in high_hit_rate if x > 0]) / len(high_hit_rate) * 100 if high_hit_rate else 0
        low_hit_rate_pct  = len([x for x in low_hit_rate  if x > 0]) / len(low_hit_rate)  * 100 if low_hit_rate  else 0

        # --- at the end of the function, replace the mo.vstack block ---
        summary_ui = mo.vstack([
            mo.callout(
                mo.md(
                    f"**Gross Profit (out-of-sample)** from "
                    f"`{Asset1.index[0].strftime('%Y-%m-%d')}` to "
                    f"`{Asset1.index[-1].strftime('%Y-%m-%d')}`: **${profit:,.2f}**  \n"
                    f"*Return on ${pair_capital:,.0f} deployed: "
                    f"{profit / pair_capital:+.2%} over "
                    f"{high_trade_total + low_trade_total} trades. "
                    f"Stop-loss exits: {stop_loss_exits}.*  \n"
                    f"*The pair was selected on the train window only — these "
                    f"bars were never used to choose it.*  \n"
                    f"*No commission, slippage, or borrow cost is modelled "
                    f"(paper account). Breakeven friction for this run: "
                    f"{profit / max(_traded_notional, 1e-9) * 1e4:,.0f} bps "
                    f"round-trip across ${_traded_notional:,.0f} traded.*"
                ),
                kind="success" if profit > 0 else "danger"
            ),
            mo.ui.table(pd.DataFrame([
                {"Trade Type": "High Ratio Trades", "Total Trades": high_trade_total, "Hit Rate (%)": f"{high_hit_rate_pct:.2f}", "Biggest Gain ($)": f"{high_biggest_gain:.2f}", "Biggest Loss ($)": f"{high_biggest_loss:.2f}"},
                {"Trade Type": "Low Ratio Trades",  "Total Trades": low_trade_total,  "Hit Rate (%)": f"{low_hit_rate_pct:.2f}",  "Biggest Gain ($)": f"{low_biggest_gain:.2f}",  "Biggest Loss ($)": f"{low_biggest_loss:.2f}"},
            ]))
        ])


        trades_dic = {**high_dic, **low_dic}
        # Base the curve on the capital actually DEPLOYED, not the whole
        # weighted buying power. Only `pair_capital` is ever at risk, so
        # dividing P&L by the full pool understated return on capital 10x.
        # (Realised-at-close only — there is no mark-to-market between trades,
        # so drawdown inside an open position is not visible here.)
        total = pair_capital
        tracker = []
        for key in sorted(trades_dic.keys()):
            total += trades_dic[key]
            tracker.append(total)

        trades = pd.DataFrame({'Date': list(trades_dic.keys()), 'Profit': list(trades_dic.values())})
        growth_tracker = pd.DataFrame({
            'Date': sorted(trades_dic.keys()),
            'Cumulative Value': tracker
        }).set_index('Date')

        return (('High Trades', high_dic), ('Low Trades', low_dic), growth_tracker,
                ('Total Profit:', profit), trades, trading_log, summary_ui)

    return (trading_simulation,)


@app.cell
def _(
    MyKalmanFilter,
    adj_close_data_1,
    index,
    mo,
    np,
    pairs,
    pd,
    save_btn,
    split_date,
):
    mo.stop(not save_btn.value)

    # ── Data Prep ────────────────────────────────────────────────────────────────
    adj_close = (
        adj_close_data_1
        .reset_index()
        .rename(columns={'timestamp': 'ds'})
        .drop_duplicates(subset='ds', keep='last')
    )

    first_stock_data = (
        adj_close[['ds', pairs[index][1]]]
        .rename(columns={pairs[index][1]: 'y'})
        .reset_index(drop=True)
    )
    second_stock_data = (
        adj_close[['ds', pairs[index][0]]]
        .rename(columns={pairs[index][0]: 'y'})
        .reset_index(drop=True)
    )

    # ── Spread / Residual via Kalman Filter ───────────────────────────────────────
    # The backtest and the live signal run on ACTUAL prices only. The Prophet
    # forecasts in section 04.1 are kept purely as projection plots — they are
    # deliberately NOT spliced into the traded series, and this cell no longer
    # waits on them. Computing P&L on forecasted prices is not real performance,
    # and pre-scheduling trades off a deterministic price extrapolation is
    # unsound; the strategy instead reacts to the latest observed spread.
    def actual_series(stock_data):
        """Actual close prices indexed by date."""
        return stock_data.set_index('ds')['y']

    _full_1 = actual_series(first_stock_data)    # pairs[index][1] — INDEPENDENT
    _full_2 = actual_series(second_stock_data)   # pairs[index][0] — DEPENDENT

    _df = pd.DataFrame({
        pairs[index][1]: _full_1,
        pairs[index][0]: _full_2,
    }).dropna()

    # The filter is causal — its estimate at time t uses only data up to t — so
    # it is run over the FULL history and the signal is sliced to the test
    # window afterwards. Running it on test bars alone would instead burn the
    # first stretch of the evaluation period on convergence.
    KALMAN_WARMUP = 30   # bars before the state estimate is worth trading
    Z_WINDOW = 60        # trailing window used to standardise the residual

    mkf = MyKalmanFilter(delta=0.0001, R=0.001)
    _innovations, _betas = [], []
    for _, row in _df.iterrows():
        # DIRECTION: Engle-Granger is asymmetric, and cointegration_checker
        # already picked the orientation whose residual is the more stationary,
        # recording it as (dependent, independent) = (pair[0], pair[1]). The
        # measurement must follow that same orientation. This previously ran the
        # regression the other way round, so the filter tracked a different —
        # and possibly non-stationary — spread from the one that selected the
        # pair and produced the OLS hedge ratio.
        _x, _P, _K, z_hat, _S = mkf.step_forward(
            row[pairs[index][0]],   # y1 = dependent   (measurement)
            row[pairs[index][1]],   # y2 = independent (regressor)
        )
        _innovations.append(row[pairs[index][0]] - z_hat.squeeze())
        _betas.append(float(_x[0, 0]))

    innovations = pd.Series(_innovations, index=_df.index)

    # STANDARDISATION: divide by the REALISED trailing std of the residual, not
    # the filter's claimed sqrt(S). sqrt(S) is a function of the hand-set delta
    # and R rather than of the data — sweeping delta from 1e-4 to 1e-8 moves it
    # ~36x while the realised residual barely changes, so "+/-1.25" silently
    # meant a different thing for every parameter choice. A trailing window is
    # causal, so no bar is standardised using its own future.
    _scale = innovations.rolling(Z_WINDOW).std()

    # SIGN: innovation > 0 means the DEPENDENT leg is rich. trading_simulation
    # reads a positive signal as "Asset1 (= the independent leg) is rich, sell
    # it", so flip to keep that convention intact.
    series = -(innovations / _scale)
    series.iloc[:KALMAN_WARMUP] = np.nan

    # Kalman slope: shares of Asset1 to hold per share of Asset2 to keep the
    # position hedged against the spread it is actually trading.
    hedge_ratio_series = pd.Series(_betas, index=_df.index)

    # Backtest and evaluate on the held-out window only.
    result = _full_1.loc[split_date:]
    result1 = _full_2.loc[split_date:]

    Kalman_Filter = True
    return (
        Kalman_Filter,
        KALMAN_WARMUP,
        Z_WINDOW,
        adj_close,
        first_stock_data,
        hedge_ratio_series,
        innovations,
        mkf,
        result,
        result1,
        second_stock_data,
        series,
    )


@app.cell
def _(mo, save_btn):
    mo.stop(not save_btn.value)

    # Prophet tuning is by far the most expensive thing in this notebook: a
    # 16-point grid crossed with rolling-origin CV, run for both legs. Its only
    # consumer is the pair of projection charts below — no forecast reaches the
    # backtest or the order path — so it is opt-in rather than on by default.
    prophet_toggle = mo.ui.checkbox(
        label="Run Prophet projection (slow — grid search over both legs)"
    )
    prophet_toggle
    return (prophet_toggle,)


@app.cell
def _(
    Figure,
    Prophet,
    adj_close,
    cross_validation,
    first_stock_data,
    index,
    itertools,
    mo,
    np,
    pairs,
    pd,
    performance_metrics,
    plt,
    prophet_toggle,
    second_stock_data,
):
    figs = []

    if prophet_toggle.value:
        # ── Hyperparameter Tuning ─────────────────────────────────────────────
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        }
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

        n_total     = len(adj_close)
        initial_str = f'{int(n_total * 0.70)} days'
        period_str  = f'{int(n_total * 0.70 / 6)} days'
        horizon_str = f'{int(n_total * 0.70 / 3)} days'

        def tune_prophet(data):
            """Return (best_params, tuning_results_df)."""
            rmses = []
            for params in all_params:
                m = Prophet(**params).fit(data)
                df_cv = cross_validation(
                    m,
                    initial=initial_str,
                    period=period_str,
                    horizon=horizon_str,
                    parallel='threads',   # safe on Windows / Marimo
                )
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmses.append(df_p['rmse'].values[0])
            results = pd.DataFrame(all_params)
            results['rmse'] = rmses
            best = all_params[int(np.argmin(rmses))]
            return best, results

        # ── Model Training ────────────────────────────────────────────────────
        def train_prophet_model(data, params):
            model = Prophet(
                interval_width=0.95,
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                holidays_prior_scale=15,
                weekly_seasonality=True,
                yearly_seasonality=True,
                daily_seasonality=False,
            )
            model.add_country_holidays(country_name='US')
            model.fit(data)
            return model

        def generate_forecast(model, periods=365):
            future   = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            return forecast

        # ── Plotting ──────────────────────────────────────────────────────────
        def plot_forecast(model, forecast, title=''):
            fig = Figure(figsize=(10, 4))
            ax = fig.subplots()
            model.plot(forecast, ax=ax, include_legend=True)
            if title:
                ax.set_title(title)
            fig.tight_layout()
            return fig

        with mo.status.spinner(title="Tuning and fitting Prophet models..."):
            for _data, _title in (
                (first_stock_data,  pairs[index][1]),
                (second_stock_data, pairs[index][0]),
            ):
                _best, _ = tune_prophet(_data)
                _model = train_prophet_model(_data, _best)
                figs.append(plot_forecast(_model, generate_forecast(_model), title=_title))
    return (figs,)


@app.cell
def _(section_header):
    section_header("Prediction Graphs", "04.1") # subsection
    return


@app.cell
def _(figs, mo):
    # ── Return to Marimo ──────────────────────────────────────────────────────────
    mo.vstack([mo.mpl.interactive(fig) for fig in figs]) if figs else mo.callout(
        mo.md("Prophet projection is off — tick the box above to generate it."),
        kind="info",
    )
    return


@app.cell
def _(section_header):
    section_header("Kalman Filter for signal processing", "04.2") # subsection
    return


@app.cell
def _(Figure, KALMAN_WARMUP, mkf, mo, plt, save_btn, series):
    mo.stop(not save_btn.value)


    fig1 = Figure(figsize=(20, 10))
    ax1 = fig1.subplots()
    # The warm-up window is already NaN in `series`, so this plots exactly the
    # stretch the backtest is allowed to trade.
    ax1.plot(series)
    ax1.set_title(f"Standardized Kalman Residual (first {KALMAN_WARMUP} bars masked)")

    fig2 = Figure(figsize=(20, 5))
    ax2 = fig2.subplots()
    ax2.plot(mkf.x[0, 1:])
    ax2.set_title("Kalman Filter State")

    mo.vstack([
        mo.center(mo.mpl.interactive(fig1)),
        mo.center(mo.mpl.interactive(fig2))
    ])
    return


@app.cell
def _(section_header):
    section_header("Trading Summary", "04.3") # subsection
    return


@app.cell
def _(mo, save_btn):
    mo.stop(not save_btn.value)

    # One stop-loss for the whole section. The summary and the trade log used to
    # come from two separate runs with different values (5,000 and 50,000), so
    # the P&L shown here described a different strategy from the one whose
    # orders got submitted downstream.
    #
    # Expressed as a PERCENT of the notional actually put on, so the risk limit
    # means the same thing regardless of account size.
    stop_loss_input = mo.ui.number(
        start=0.5, stop=50.0, step=0.5, value=5.0,
        label="Stop loss (% of position notional)",
    )
    stop_loss_input
    return (stop_loss_input,)


@app.cell
def _(
    Kalman_Filter,
    mo,
    result,
    result1,
    save_btn,
    stop_loss_input,
    trading_simulation,
):
    mo.stop(not save_btn.value)

    # Single source of truth — every display and the order path below read from
    # this one run instead of re-simulating with their own parameters.
    simulation = trading_simulation(
        result, result1, 10, 60, stop_loss_input.value, Kalman_Filter
    )
    return (simulation,)


@app.cell
def _(simulation):
    simulation[6]
    return


@app.cell
def _(section_header):
    section_header("Historical Pair Trading Simulation", "04.4") # subsection
    return


@app.cell
def _(simulation):
    trade_log = simulation[5]
    trade_log
    return (trade_log,)


@app.cell
def _(section_header):
    section_header("All Trade Logs", "04.5")
    return


@app.cell
def _(duckdb, index, market_cap_input, mo, pairs, save_btn, sector_input):
    mo.stop(not save_btn.value)

    item = {'sector': sector_input, 'Size': market_cap_input, 'Asset1': pairs[index][0], 'Asset2': pairs[index][1]}
    with duckdb.connect('quant_trading.db') as _con:
        query = '\n        SELECT * FROM Assets\n        WHERE sector = ? AND Size = ? AND Asset1 = ? AND Asset2 = ?\n    '
        result_1 = _con.execute(query, (item['sector'], item['Size'], item['Asset1'], item['Asset2'])).fetchall()
        if result_1:
            print('Entry already exists')
        else:
            last_id = _con.execute('SELECT MAX(id) FROM Assets').fetchone()[0]
            item['id'] = last_id + 1 if last_id is not None else 1
            _con.execute('INSERT INTO Assets (id, sector, Asset1, Asset2, Size) VALUES (?, ?, ?, ?, ?)', (int(item['id']), str(item['sector']), str(item['Asset1']), str(item['Asset2']), str(item['Size'])))
            print('Inserted successfully')
        _items_df = _con.execute('SELECT * FROM Assets').df()

    mo.ui.dataframe(_items_df)
    return


@app.cell
def _(
    MarketOrderRequest,
    OrderSide,
    TimeInForce,
    mo,
    pd,
    result,
    save_btn,
    trade_log,
    trading_client,
):
    mo.stop(not save_btn.value)


    status_items = []
    order_items  = []
    # Outcome per leg, handed to the hub assistant's run record after this cell.
    pair_exec = {"signal_date": None, "hold": False, "rows": []}

    trade_log['Date'] = pd.to_datetime(trade_log['Date'])
    # Reactive execution: act only if the most recent ACTUAL bar fired an
    # entry/exit signal. No forecast prices are involved, so trades follow the
    # spread we actually observed rather than a stale forward schedule.
    signal_date = pd.Timestamp(result.index[-1]).date()
    today_row = trade_log[trade_log['Date'].dt.date == signal_date]
    pair_exec["signal_date"] = str(signal_date)

    if today_row.empty:
        pair_exec["hold"] = True
        status_items.append(mo.callout(mo.md(f"⏸ **Hold** — no signal on the latest bar (`{signal_date}`)."), kind="info"))
    else:
        for _ticker, _action, _qty in zip(today_row['Asset'], today_row['Action'], today_row['Quantity']):
            _leg = {"symbol": _ticker, "action": _action, "qty": _qty, "outcome": "unknown"}
            pair_exec["rows"].append(_leg)
            # Look the symbol up directly. This used to GET /v2/assets and pull
            # down the entire tradable universe to linear-scan it for two names.
            try:
                asset = trading_client.get_asset(_ticker)
            except Exception as e:
                _leg["outcome"] = "skipped: lookup failed"
                order_items.append(mo.callout(mo.md(f"⚠️ Ticker `{_ticker}` lookup failed: {e}"), kind="warn"))
                continue

            asset_info = f"**{asset.symbol}** — {asset.exchange} — Tradable: `{asset.tradable}`"

            if not asset.tradable:
                _leg["outcome"] = "skipped: not tradable"
                order_items.append(mo.callout(mo.md(f"🚫 {asset_info}\nNot tradable, skipping."), kind="warn"))
                continue

            if _action == 'Sell' and not asset.shortable:
                _leg["outcome"] = "skipped: not shortable"
                order_items.append(mo.callout(mo.md(f"🚫 {asset_info}\nCannot be sold short, skipping."), kind="warn"))
                continue

            # Whole shares: GTC market orders reject fractional quantities, and
            # the backtest already sized in whole shares.
            _qty = int(_qty)
            _leg["qty"] = _qty
            if _qty <= 0:
                _leg["outcome"] = "skipped: zero quantity"
                order_items.append(mo.callout(mo.md(f"⚠️ {asset_info}\nQuantity rounds to 0, skipping."), kind="warn"))
                continue

            order_data = MarketOrderRequest(
                symbol=_ticker,
                qty=_qty,
                side=OrderSide.BUY if _action == 'Buy' else OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            try:
                order = trading_client.submit_order(order_data=order_data)
                _leg["outcome"] = "submitted"
                _leg["order_ref"] = str(order.id)
                order_items.append(mo.callout(
                    mo.md(f"✅ {asset_info}\n`{_action}` {_qty} shares — Order ID: `{order.id}`"),
                    kind="success"
                ))
            except Exception as e:
                _leg["outcome"] = "failed"
                order_items.append(mo.callout(
                    mo.md(f"❌ {asset_info}\nOrder failed: {e}"),
                    kind="danger"
                ))

    # Current positions
    portfolio_trading = trading_client.get_all_positions()
    positions_df = pd.DataFrame([
        {"Symbol": p.symbol, "Quantity": p.qty}
        for p in portfolio_trading
    ])

    mo.vstack([
        mo.md("## Trade Execution"),
        *status_items,
        mo.md("### Order Results") if order_items else mo.md(""),
        *order_items,
        mo.md("### Current Positions"),
        mo.ui.table(positions_df) if not positions_df.empty
        else mo.callout(mo.md("No open positions."), kind="info"),
    ])
    return (pair_exec,)


@app.cell
def _(chosen, pair_diagnostics, pair_exec, pairs, save_btn, simulation):
    # Hub assistant run record: screening diagnostics, the backtest, the latest
    # signal, and what happened to each leg. Runs after the orders above and
    # never raises; see agent_diag/record.py.
    import agent_diag.record as _agent_record

    if save_btn.value:
        _agent_record.record_pair_trading(
            chosen, pair_diagnostics, simulation, pair_exec, pairs_available=len(pairs)
        )
    return


@app.cell
def _(mo, pair_flag_cleared, pathlib, save_btn, trade_log):
    mo.stop(not save_btn.value)

    # Depends on `pair_flag_cleared` purely for ordering (see the invalidation
    # cell) and on `trade_log` so the flag only goes up once the simulation has
    # actually produced results for this pair.
    if pair_flag_cleared and trade_log is not None:
        pathlib.Path(".pair_ready").touch()

    mo.vstack([
        mo.md("✅ Pair analysis complete.")
    ])
    return


if __name__ == "__main__":
    app.run()
