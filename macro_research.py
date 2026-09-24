import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", css_file="../../theme.css", html_head_file="../../theme_head.html")


@app.cell
def _():
    import os
    import pathlib
    import sys
    import pandas as pd
    import math
    import requests
    from concurrent.futures import ThreadPoolExecutor
    from alpaca_trade_api.rest import REST, TimeFrame
    import numpy as np
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import FactorAnalysis
    from sklearn.model_selection import KFold, cross_val_score
    import plotly.graph_objects as go_diag
    from plotly.subplots import make_subplots
    import marimo as mo

    # market_cache is shared with the Macro desk, so it lives in <root>/shared.
    _shared = str(pathlib.Path(mo.notebook_dir()).parents[1] / "shared")
    if _shared not in sys.path:
        sys.path.insert(0, _shared)
    import plotly.graph_objects as go
    from datetime import date
    from dotenv import load_dotenv

    # When stdout is a pipe rather than a console, Python encodes it with the
    # locale codec — cp1252 on this machine — and every arrow/sigma this
    # notebook prints raises UnicodeEncodeError. Force UTF-8 on the real
    # streams; marimo's in-notebook capture stream has no reconfigure().
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")

    # The Alpaca/FRED keys live in .env in the Quant root; without this the
    # sector cell only works when they happen to be set in the OS environment.
    # (assign to _ so marimo doesn't render load_dotenv's bool return).
    _ = load_dotenv(mo.notebook_dir().parents[1] / ".env")

    return (
        FactorAnalysis,
        KFold,
        REST,
        StandardScaler,
        ThreadPoolExecutor,
        TimeFrame,
        cross_val_score,
        date,
        go,
        go_diag,
        make_subplots,
        math,
        mo,
        multipletests,
        np,
        os,
        pathlib,
        pd,
        requests,
        smf,
    )


@app.cell
def _(mo):
    mo.Html("""
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

    <div style="padding: 2.5rem 0 2rem; text-align: center;">
      <div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; border-radius: 50%; border: 0.5px solid #1D9E75; margin-bottom: 1rem;">
        <i class="ti ti-chart-candle" style="font-size: 22px; color: #1D9E75;"></i>
      </div>
      <h1 style="font-family: 'DM Serif Display', serif; font-size: 36px; font-weight: 400; font-style: italic; margin: 0 0 6px; letter-spacing: -0.01em; color: var(--color-text-primary);">Macro Research Stress Testing</h1>
      <p style="font-family: 'DM Mono', monospace; font-size: 11px; color: var(--color-text-secondary); letter-spacing: 0.18em; text-transform: uppercase; margin: 0 0 1.25rem;">Validating sector reacts</p>
      <div style="display: inline-flex; align-items: center; gap: 6px; font-family: 'DM Mono', monospace; font-size: 11px; color: #0F6E56; background: #E1F5EE; padding: 4px 14px; border-radius: 999px;">
        <span style="width: 6px; height: 6px; border-radius: 50%; background: #1D9E75; display: inline-block;"></span>
        Live
      </div>
      <div style="margin-top: 1.5rem; width: 40px; height: 0.5px; background: var(--color-border-tertiary); margin-left: auto; margin-right: auto;"></div>
    </div>
    """)
    return


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
def _(page_header):
    page_header("Data Processing", "01")
    return


@app.cell
def _(section_header):
    section_header("Data Collection and Processing", "01.1")
    return


@app.cell
def _(ThreadPoolExecutor, date, mo, os, pathlib, pd, requests):
    URL_BASE = 'https://api.stlouisfed.org/'
    ENDPOINT = 'fred/series/observations'
    URL      = URL_BASE + ENDPOINT
    API_KEY  = os.environ.get('FRED_API_KEY', '')

    START_DATE = '2016-01-01'
    import logging as _logging
    import market_cache as _market_cache
    END_DATE   = _market_cache.market_day()

    MONTHLY = ['INDPRO', 'DGORDER', 'UMCSENT', 'CPIAUCSL', 'CPILFESL', 'PPIACO', 'FEDFUNDS', 'UNRATE', 'CES0500000003']
    WEEKLY  = ['ICSA']
    DAILY   = ['DCOILWTICO', 'GASREGCOVW', 'DGS2', 'DGS10', 'BAA10Y', 'DTWEXBGS']

    # Persistent cache in R2; decoded observations and analytics stay in RAM.
    _cloud_cache = _market_cache.get_cache()
    _fred_session = requests.Session()

    def fetch_fred(series_id, resample=None):
        try:
            raw = _cloud_cache.fred(series_id, START_DATE, END_DATE, API_KEY, _fred_session)
        except _market_cache.CacheError:
            raise
        except Exception as _exc:
            _logging.getLogger("market_cache").warning(
                "FRED unavailable for %s (%s)", series_id, type(_exc).__name__)
            return pd.DataFrame()

        if resample == 'last':
            return raw.resample('ME').last()
        if resample == 'mean':
            return raw.resample('ME').mean()
        return raw

    def fetch_all_fred(series_list, resample=None):
        with ThreadPoolExecutor(max_workers=6) as pool:
            dfs = list(pool.map(lambda s: fetch_fred(s, resample), series_list))
        dfs = [d for d in dfs if not d.empty]
        if not dfs:
            raise RuntimeError(
                "FRED returned nothing for every series in this group. "
                "Is FRED_API_KEY set?"
            )
        return pd.concat(dfs, axis=1)

    df_monthly = fetch_all_fred(MONTHLY, resample='last')
    df_weekly  = fetch_all_fred(WEEKLY,  resample='mean')
    df_daily   = fetch_all_fred(DAILY,   resample='last')

    df_levels_all = pd.concat([df_monthly, df_weekly, df_daily], axis=1)

    # Publication lags are staggered (DGORDER/UMCSENT trail the daily series by
    # ~2 months). A blanket .dropna() therefore silently discards the most recent
    # — and most decision-relevant — months. Carry the slow series forward over a
    # bounded window instead, and surface the true vintage of each input.
    FFILL_LIMIT = 3
    _vintage = df_levels_all.apply(lambda c: c.last_valid_index())
    df = (df_levels_all
            .dropna(how='all')
            .ffill(limit=FFILL_LIMIT)
            .dropna())

    _stalest    = _vintage.min()
    _months_ff  = (df.index[-1].to_period('M') - _stalest.to_period('M')).n
    _vintage_md = "\n".join(
        f"- `{s}` → {d.date()}" + ("  ⟵ stalest" if d == _stalest else "")
        for s, d in _vintage.sort_values().items()
    )

    mo.callout(mo.md(
        f"""✅ **FRED macro variables collected** — {df.shape[0]} months × {df.shape[1]} series,
through **{df.index[-1].date()}**.

Slowest-publishing series is {_months_ff} month(s) behind and is forward-filled
(limit {FFILL_LIMIT}). Per-series vintage:

{_vintage_md}"""
    ), kind="success")
    return (df,)


@app.cell
def _(StandardScaler, df, mo, np, pd):
    # ── Stationarity ─────────────────────────────────────────────────────────
    # ADF on the raw levels rejects a unit root for only 4 of the 16 series
    # (CPIAUCSL p=0.99, CES0500000003 p=0.999, DGS10 p=0.83, ...). Standardising
    # levels rescales but does NOT detrend, so a factor model fitted on them is
    # dominated by a shared time trend — on this sample PC1 takes 63.2% of the
    # variance in levels vs 30.6% in changes. Regressing stationary sector
    # returns on I(1) factor scores is an unbalanced regression whose R² and
    # t-stats are not interpretable, and HC3/HAC does not repair that.
    #
    # So everything downstream runs on monthly CHANGES:
    #   rate-like series (already %)     -> simple difference (percentage points)
    #   indices / prices / counts (> 0)  -> log difference (% change)
    # A level series that goes non-positive anywhere (e.g. DCOILWTICO printed a
    # negative spot in Apr-2020) falls back to a simple difference.
    RATE_LIKE = {'FEDFUNDS', 'UNRATE', 'DGS2', 'DGS10', 'BAA10Y'}

    # The COVID collapse is a handful of months that dominate this sample:
    # 5 of 124 months carry 94% of UNRATE's squared variation and 92% of ICSA's.
    # Defined here so the regression and the scenario scaling agree on the window.
    COVID_START, COVID_END = '2020-02-01', '2020-06-30'

    def to_stationary(s, name):
        """Return (changes, human label, display unit, unit scale factor)."""
        if name in RATE_LIKE:
            return s.diff(), 'Δ (pp)', 'pp', 1.0
        if (s <= 0).any():
            return s.diff(), 'Δ (level, non-positive present)', 'units', 1.0
        return np.log(s).diff(), 'Δlog (%)', '%', 100.0

    _cols, _how, chg_units = {}, {}, {}
    for _c in df.columns:
        _cols[_c], _how[_c], _u, _sc = to_stationary(df[_c], _c)
        chg_units[_c] = (_u, _sc)

    df_chg = pd.DataFrame(_cols, index=df.index).dropna()

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_chg),
        index=df_chg.index,
        columns=df_chg.columns
    )

    _transform_md = "\n".join(f"- `{k}` → {v}" for k, v in _how.items())
    mo.callout(mo.md(
        f"""✅ **Transformed to stationary changes, then standardised** —
{df_chg.shape[0]} months × {df_chg.shape[1]} series.

{_transform_md}"""
    ), kind="success")
    return COVID_END, COVID_START, chg_units, df_chg, df_scaled


@app.cell
def _(FactorAnalysis, KFold, cross_val_score, df_scaled, mo, pd):
    # ── Step 1: choose n_components by cross-validated log-likelihood ─────────
    # Previously this borrowed a 90%-cumulative-variance rule from a PCA fit,
    # which is a PCA diagnostic answering a different question (total variance
    # reproduced) than the one FA poses (common vs unique variance). FA does
    # expose a proper likelihood via .score(), so select on held-out fit — this
    # penalises the extra factors that a variance threshold happily accepts and
    # keeps the sector regressions from being overfitted on ~120 monthly rows.
    MAX_FACTORS = min(10, df_scaled.shape[1] - 1)
    _cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_loglik = pd.Series(
        {
            k: cross_val_score(
                FactorAnalysis(n_components=k, random_state=42),   # rotation is
                df_scaled, cv=_cv                                  # likelihood-
            ).mean()                                               # invariant
            for k in range(1, MAX_FACTORS + 1)
        },
        name='cv_loglik',
    )
    cv_loglik.index.name = 'n_factors'

    n_factors = int(cv_loglik.idxmax())
    print(f"Chosen n_factors (max CV log-likelihood): {n_factors}")
    print(cv_loglik.round(3).to_string())

    # ── Step 2: fit Factor Analysis with Varimax rotation ─────────────────────
    fa = FactorAnalysis(n_components=n_factors, rotation='varimax', random_state=42)
    fa_scores = fa.fit_transform(df_scaled)          # shape: T × n_factors

    # Loadings matrix: shape n_vars × n_factors
    loadings_fa = pd.DataFrame(
        fa.components_.T,                            # components_ is n_factors × n_vars
        index=df_scaled.columns,
        columns=[f'F{i+1}' for i in range(n_factors)]
    )

    print("\nVarimax-rotated loadings (top driver per factor clearly dominant):")
    print(loadings_fa.abs().round(3))

    _best = cv_loglik.max()
    mo.callout(mo.md(
        f"""✅ **Factor model selected by cross-validated log-likelihood** —
{n_factors} factor(s), held-out score {_best:.3f}, chosen from 1–{MAX_FACTORS}."""
    ), kind="success")
    return cv_loglik, fa, fa_scores, loadings_fa, n_factors


@app.cell
def _(df_scaled, fa_scores, loadings_fa, n_factors, pd):
    def auto_label(loadings, factor_col, n_top=2):
            """Return 'SERIES1_SERIES2' from the top-n absolute loadings."""
            top = loadings[factor_col].abs().nlargest(n_top).index.tolist()
            return '_'.join(top)

    # Two factors can share their top-2 drivers (they differ in sign or in the
    # 3rd+ loading). Left unchecked that yields duplicate column names in
    # df_factors / loadings_named, and every later `.loc[sector, beta_col]`
    # silently returns a frame instead of a scalar. Keep labels unique.
    factor_labels, _seen_labels = {}, set()
    for i in range(n_factors):
        _fkey  = f'F{i+1}'
        _label = auto_label(loadings_fa, _fkey)
        if _label in _seen_labels:
            _label = f'{_label}_{_fkey}'
        _seen_labels.add(_label)
        factor_labels[_fkey] = _label
    print("Auto-generated factor labels:")
    for k, v in factor_labels.items():
        print(f"  {k} → {v}")

    # Build factor score DataFrame with auto-generated column names
    df_factors = pd.DataFrame(
        fa_scores,
        index=df_scaled.index[:len(fa_scores)],   # align to df_scaled index
        columns=list(factor_labels.values())
    )

    # Rename loadings columns to match
    loadings_named = loadings_fa.rename(columns=factor_labels)
    return df_factors, loadings_named


@app.cell
def _(section_header):
    section_header("Factor Analysis", "01.2")
    return


@app.cell
def _(go_diag, loadings_named, n_factors):
    _z    = loadings_named.values.tolist()
    _x    = loadings_named.columns.tolist()
    _y    = loadings_named.index.tolist()

    _fig = go_diag.Figure(go_diag.Heatmap(
        z=_z, x=_x, y=_y,
        colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title='Loading'),
        text=[[f'{v:.2f}' for v in row] for row in _z],
        texttemplate='%{text}',
    ))
    _fig.update_layout(
        title='FA Varimax Loadings Heatmap',
        height=max(350, len(_y) * 28),
        width=max(500, n_factors * 120),
        yaxis=dict(autorange='reversed'),
        font=dict(family='DM Mono', size=11),
        paper_bgcolor='#0A0E17',
        plot_bgcolor='#0A0E17',
        font_color='#E2E8F0',
    )
    _fig
    return


@app.cell
def _(loadings_named, mo):
    # print("\n--- Factor Drivers (Varimax) ---")
    # for _fc in loadings_named.columns:
    #     _top3 = loadings_named[_fc].abs().nlargest(3)
    #     print(f"\n{_fc} — top drivers:")
    #     print(_top3.round(3))

    _VOID, _BORDER = '#0A0E17', '#1E293B'
    _TEXT_HI, _TEXT_MID, _TEXT_DIM = '#E2E8F0', '#94A3B8', '#64748B'
    _AQUA, _BEAR_RED, _BULL_AMB = '#22D3EE', '#F87171', '#FBBF24'

    _cards = []
    for _fc in loadings_named.columns:
        _top3 = loadings_named[_fc].abs().nlargest(3)
        _max = _top3.max()

        _rows = ""
        for _name, _val in _top3.items():
            _pct = (_val / _max) * 100
            _rows += f"""
            <div style='margin-bottom:8px;'>
                <div style='display:flex;justify-content:space-between;font-family:DM Mono;font-size:11px;color:{_TEXT_MID};margin-bottom:3px;'>
                    <span>{_name}</span>
                    <span style='color:{_TEXT_HI};'>{_val:.3f}</span>
                </div>
                <div style='height:4px;background:{_BORDER};border-radius:2px;overflow:hidden;'>
                    <div style='width:{_pct:.0f}%;height:100%;background:{_AQUA};'></div>
                </div>
            </div>"""

        _cards.append(f"""
        <div style='background:{_VOID};border:1px solid {_BORDER};border-radius:8px;
                    padding:16px 18px;min-width:220px;flex:1;'>
            <div style='font-family:DM Serif Display;color:{_TEXT_HI};font-size:16px;
                        margin-bottom:12px;letter-spacing:0.2px;'>{_fc}</div>
            {_rows}
        </div>""")

    mo.Html(f"""
    <div style='font-family:DM Mono;'>
        <div style='color:{_TEXT_DIM};font-size:11px;text-transform:uppercase;
                    letter-spacing:1.5px;margin-bottom:12px;'>
            Factor Drivers — Varimax
        </div>
        <div style='display:flex;gap:14px;flex-wrap:wrap;'>
            {''.join(_cards)}
        </div>
    </div>
    """)
    return


@app.cell
def _(REST, TimeFrame, os, pd):
    api_key    = os.environ.get('ALPACA_API_KEY', '')
    api_secret = os.environ.get('ALPACA_API_SECRET', '')
    base_url   = 'https://paper-api.alpaca.markets/v2'
    api = REST(api_key, api_secret, base_url)
    tickers = ["XLRE", "XLF", "XLE", "XLV", "XLI", "IYZ", "XLY", "XLU", "XLP", "XLB", "XLK"]
    start_date = '2016-01-01'
    data_dict = {}
    for _ticker in tickers:
        try:
            # adjustment='all' -> split AND dividend adjusted. The Alpaca default
            # is 'raw', which drops distributions entirely; since the master table
            # is ranked by Alpha that systematically penalised the high-yield
            # sectors (XLU/XLRE/XLP yield 3-4% vs XLK ~0.7%, i.e. ~25bp/month).
            bars = api.get_bars(
                _ticker, TimeFrame.Month, start=start_date, adjustment='all'
            ).df
            if not bars.empty:
                bars = bars.tz_localize(None)
                bars = bars[['close']]
                bars = bars.rename(columns={'close': _ticker})
                data_dict[_ticker] = bars
        except Exception as e:
            print(f'Failed to fetch data for {_ticker}: {e}')
    close_prices   = pd.concat(data_dict.values(), axis=1)
    adj_close_data = close_prices.dropna(axis=1)

    _dropped = sorted(set(close_prices.columns) - set(adj_close_data.columns))
    if _dropped:
        print(f'Dropped (incomplete history): {_dropped}')
    _never = sorted(set(tickers) - set(close_prices.columns))
    if _never:
        print(f'Never fetched: {_never}')

    adj_close_data.index = pd.to_datetime(adj_close_data.index).to_period('M').to_timestamp('M')
    return adj_close_data, tickers


@app.cell
def _(adj_close_data, df_factors, mo, np):
    returns = np.log(adj_close_data / adj_close_data.shift(1)).dropna()

    # `result` is CONTEMPORANEOUS: month-t returns on month-t factor scores.
    # That is the right estimand for a what-if stress test ("if macro realises X,
    # what does the sector do?"), but it is NOT a forecast — CPI/INDPRO for month
    # t only publish mid-month t+1, so these betas are not tradable as-is.
    #
    # `result_lagged` is the predictive variant (returns on t-1 factors, which
    # are knowable at t). Both are fitted and both R² are reported, so the split
    # between coincident and genuinely forecastable variation is explicit rather
    # than assumed.
    result        = df_factors.join(returns, how='inner')
    result_lagged = df_factors.shift(1).join(returns, how='inner').dropna()

    mo.callout(mo.md(
        f"""✅ **Sector ETF returns collected** (dividend-adjusted) —
{returns.shape[1]} sectors, {result.shape[0]} aligned months.

Betas below are **contemporaneous** (a conditional-response model, not a
forecast). The lagged R² column shows how much survives a 1-month publication
lag."""
    ), kind="success")
    return result, result_lagged


@app.cell
def _(section_header):
    section_header("Regression Statistics and Analysis", "01.4")
    return


@app.cell
def _(COVID_END, COVID_START, df_factors, mo, result, result_lagged, smf, tickers):
    _VOID, _BORDER = '#0A0E17', '#1E293B'
    _TEXT_HI, _TEXT_MID, _TEXT_DIM = '#E2E8F0', '#94A3B8', '#64748B'
    _AQUA, _BEAR_RED, _BULL_AMB = '#22D3EE', '#F87171', '#FBBF24'

    factor_cols = df_factors.columns.tolist()

    def sanitise(name):
        return ''.join(c if (c.isalnum() or c == '_') else '_' for c in name)

    safe_map      = {col: sanitise(col) for col in factor_cols}
    result_safe   = result.rename(columns=safe_map)
    result_lag_sf = result_lagged.rename(columns=safe_map)
    safe_factors  = list(safe_map.values())

    # Newey-West lag truncation, standard 4*(T/100)^(2/9) rule.
    HAC_MAXLAGS = max(1, int(4 * (len(result_safe) / 100) ** (2 / 9)))

    # ── COVID leverage ───────────────────────────────────────────────────────
    # One indicator per crisis month, so those months are fitted exactly by their
    # own dummy and contribute nothing to the factor slopes. Betas therefore
    # describe the normal-times transmission channel rather than a relationship
    # that ran through lockdown policy — without deleting the observations, which
    # are still needed to calibrate how large a severe shock is (see the scenario
    # scale selector).
    #
    # Motivation, measured: rolling-36m beta SD is ~3x the HAC SE at the median
    # and 20-50x for the labour factor; 44 of 55 betas flip sign across windows;
    # 23 of 55 shift >2 SE when these 5 months are removed.
    def add_covid_dummies(frame):
        out, cols = frame.copy(), []
        mask = (out.index >= COVID_START) & (out.index <= COVID_END)
        for _d in out.index[mask]:
            _c = f'covid_{_d:%Y_%m}'
            out[_c] = (out.index == _d).astype(float)
            cols.append(_c)
        return out, cols

    result_safe,   covid_cols = add_covid_dummies(result_safe)
    result_lag_sf, _          = add_covid_dummies(result_lag_sf)
    print(f"COVID month dummies ({len(covid_cols)}): {covid_cols}")

    reg_results   = {}
    _accordion_items = {}

    for ticker in tickers:
        if ticker not in result_safe.columns:
            continue

        # Always quote the LHS — the old conditional only quoted tickers
        # containing '-' or ' ', so it never fired and would break on the first
        # symbol that needed it.
        formula = f'Q("{ticker}") ~ ' + ' + '.join(safe_factors + covid_cols)

        # HC3 corrects heteroskedasticity only. Persistent monthly macro factors
        # leave serially correlated residuals, which makes HC3 SEs too small and
        # over-rejects. HAC (Newey-West) handles both.
        model_ols = smf.ols(formula, data=result_safe).fit(
            cov_type='HAC', cov_kwds={'maxlags': HAC_MAXLAGS, 'use_correction': True}
        )
        model_lag = smf.ols(formula, data=result_lag_sf).fit(
            cov_type='HAC', cov_kwds={'maxlags': HAC_MAXLAGS, 'use_correction': True}
        )

        reg_results[ticker] = {
            'model':        model_ols,
            'safe_factors': safe_factors,
            'orig_factors': factor_cols,
            # Retained for the Monte Carlo: the two uncertainty sources the fan
            # chart previously ignored entirely.
            'resid_var':    model_ols.mse_resid,
            'beta_cov':     model_ols.cov_params().loc[safe_factors, safe_factors].values,
            'r2_lagged':    model_lag.rsquared,
        }

        # --- styled coefficient rows (skip intercept) ---
        _coef_rows = ""
        for _safe, _orig in zip(safe_factors, factor_cols):
            _b     = model_ols.params.get(_safe, float('nan'))
            _p     = model_ols.pvalues.get(_safe, float('nan'))
            _color = _BULL_AMB if _b >= 0 else _BEAR_RED
            _sig   = "\u25cf\u25cf\u25cf" if _p < 0.01 else "\u25cf\u25cf" if _p < 0.05 else "\u25cf" if _p < 0.1 else "\u00b7"
            _coef_rows += f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:5px 0;border-bottom:1px solid {_BORDER};font-family:DM Mono;font-size:11px;'>
                <span style='color:{_TEXT_MID};'>{_orig}</span>
                <span style='display:flex;gap:10px;align-items:center;'>
                    <span style='color:{_TEXT_DIM};font-size:10px;'>{_sig}</span>
                    <span style='color:{_color};width:64px;text-align:right;'>{_b:+.4f}</span>
                </span>
            </div>"""

        _r2     = model_ols.rsquared
        _adj_r2 = model_ols.rsquared_adj
        _fstat  = model_ols.fvalue
        _fp     = model_ols.f_pvalue
        _nobs   = int(model_ols.nobs)

        _card_html = f"""
        <div style='background:{_VOID};border:1px solid {_BORDER};border-radius:8px;
                    padding:16px 18px;'>
            <div style='display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px;'>
                <span style='font-family:DM Serif Display;color:{_TEXT_HI};font-size:16px;'>{ticker}</span>
                <span style='font-family:DM Mono;color:{_AQUA};font-size:11px;'>R\u00b2 {_r2:.3f}</span>
            </div>
            {_coef_rows}
            <div style='display:flex;justify-content:space-between;margin-top:10px;
                        font-family:DM Mono;font-size:10px;color:{_TEXT_DIM};'>
                <span>Adj R\u00b2 {_adj_r2:.3f}</span>
                <span title='R\u00b2 using t-1 factors (publication-lag safe)'>Lagged R\u00b2 {reg_results[ticker]['r2_lagged']:.3f}</span>
                <span>F {_fstat:.1f} (p={_fp:.3f})</span>
                <span>n={_nobs}</span>
            </div>
        </div>"""

        # --- raw statsmodels summary, styled to fit dark terminal palette ---
        _raw_summary_html = f"""
        <div style='background:{_VOID};border:1px solid {_BORDER};border-top:none;
                    border-radius:0 0 8px 8px;padding:14px 18px;
                    font-family:DM Mono;font-size:10.5px;color:{_TEXT_MID};
                    overflow-x:auto;white-space:pre;'>
    {model_ols.summary().as_text()}
        </div>"""

        _accordion_items[ticker] = mo.vstack([
            mo.Html(_card_html),
            mo.Html(_raw_summary_html),
        ], gap=0)

    mo.vstack([
        mo.Html(f"""
            <div style='color:{_TEXT_DIM};font-size:11px;text-transform:uppercase;
                        letter-spacing:1.5px;font-family:DM Mono;'>
                Sector Factor Regressions — HAC (Newey-West, {HAC_MAXLAGS} lags) ·
                contemporaneous · dots = raw p, significance below is FDR-adjusted
            </div>
        """),
        mo.accordion(_accordion_items),
    ])
    return HAC_MAXLAGS, factor_cols, reg_results


@app.cell
def _(factor_cols, multipletests, pd, reg_results):
    ALPHA_THRESHOLD = 0.1   # relaxed from 0.05 — standard for macro attribution with monthly data

    master_data = []
    for ticker_var, info in reg_results.items():
        model_var  = info['model']
        _orig      = info['orig_factors']
        _safe      = info['safe_factors']

        row = {
            'Sector':        ticker_var,
            'R_squared':     model_var.rsquared,
            'Adj_R_squared': model_var.rsquared_adj,
            'R2_lagged':     info['r2_lagged'],
            'Alpha':         model_var.params.get('Intercept', 0),
            'Alpha_pvalue':  model_var.pvalues.get('Intercept', 1),
        }

        for orig, safe in zip(_orig, _safe):
            row[f'{orig}_Beta'] = model_var.params.get(safe, 0)
            row[f'{orig}_pval'] = model_var.pvalues.get(safe, 1)

        master_data.append(row)

    df_master = (pd.DataFrame(master_data)
                    .set_index('Sector')
                    .sort_values('Alpha', ascending=False))

    beta_cols = [f'{f}_Beta' for f in factor_cols]
    pval_cols = [f'{f}_pval' for f in factor_cols]
    sig_cols  = [f'{f}_sig'  for f in factor_cols]

    # ── Multiple testing ─────────────────────────────────────────────────────
    # This is n_sectors × n_factors simultaneous t-tests (e.g. 11 × 4 = 44). At a
    # raw 0.1 cutoff roughly 4 of those fire by chance alone, and every surviving
    # beta feeds the impact scores and the MC. Control the false-discovery rate
    # across the whole family rather than testing each coefficient in isolation.
    _pvals_flat = df_master[pval_cols].values.ravel()
    _reject, _padj, _, _ = multipletests(
        _pvals_flat, alpha=ALPHA_THRESHOLD, method='fdr_bh'
    )
    _n_factors_cols = len(pval_cols)
    for _j, _f in enumerate(factor_cols):
        df_master[f'{_f}_padj'] = _padj.reshape(-1, _n_factors_cols)[:, _j]
        df_master[f'{_f}_sig']  = _reject.reshape(-1, _n_factors_cols)[:, _j]

    padj_cols = [f'{f}_padj' for f in factor_cols]

    _n_raw = int((df_master[pval_cols].values < ALPHA_THRESHOLD).sum())
    _n_fdr = int(df_master[sig_cols].values.sum())
    print(f"Significant betas: {_n_raw} raw (p<{ALPHA_THRESHOLD}) "
          f"-> {_n_fdr} after Benjamini-Hochberg FDR "
          f"({len(_pvals_flat)} tests)")

    print("\n=== MASTER SECTOR ATTRIBUTION TABLE ===")
    print(df_master[['Alpha'] + beta_cols + ['Adj_R_squared', 'R2_lagged']].round(4))
    return ALPHA_THRESHOLD, beta_cols, df_master, padj_cols, sig_cols


@app.cell
def _(beta_cols, df_factors, df_master, factor_cols, pd, sig_cols):
    # Latest available factor scores (most recent row of the factor DataFrame)
    latest_factors = df_factors.iloc[-1]
    latest_date    = df_factors.index[-1]
    print(f"Factor scores as of: {latest_date.date()} (structural regime — lagged ~4-8 weeks)")
    print(latest_factors.round(4))

    impact_rows = []
    for sector in df_master.index:
        contrib = {}
        gross   = 0.0
        net     = 0.0
        for factor, sig_col, beta_col in zip(factor_cols, sig_cols, beta_cols):
            beta_comp = df_master.loc[sector, beta_col]
            is_sig    = df_master.loc[sector, sig_col]
            fscore    = latest_factors[factor]
            raw       = beta_comp * fscore
            effective = raw if is_sig else 0.0   # zero-out insignificant betas
            contrib[factor] = effective
            gross += abs(effective)
            net   += effective

        # Net and Gross answer different questions and were previously collapsed
        # into one column: the old `Total_Impact` summed ABSOLUTE contributions,
        # so a sector at (+0.5, -0.5) — net-flat — outranked one at (+0.9).
        # Net = directional tailwind/headwind; Gross = macro exposure magnitude.
        impact_rows.append({
            'Sector':       sector,
            'Net_Impact':   net,
            'Gross_Impact': gross,
            **contrib,
        })

    df_impact = (pd.DataFrame(impact_rows)
                    .set_index('Sector')
                    .sort_values('Net_Impact', ascending=False))

    print("\n=== SECTOR MACRO IMPACT SCORES (FDR-significant betas only) ===")
    print("Net = directional; Gross = exposure magnitude. Ranked by Net.")
    print(df_impact.round(4))

    return df_impact, latest_date, latest_factors


@app.cell
def _(ALPHA_THRESHOLD, beta_cols, df_master, factor_cols, go, sig_cols):
    # Mask insignificant betas to zero for display
    _betas = df_master[beta_cols].copy()
    _sigs  = df_master[sig_cols].values
    _betas_masked = _betas.where(_sigs, other=0.0)

    _sectors  = df_master.index.tolist()
    _factors  = factor_cols
    _z        = _betas_masked.values.tolist()
    _text     = [[f'{v:.3f}{"*" if _sigs[r][c] else ""}' 
                    for c, v in enumerate(row)]
                    for r, row in enumerate(_betas_masked.values.tolist())]

    _fig_heat = go.Figure(go.Heatmap(
        z=_z, x=_factors, y=_sectors,
        colorscale='RdBu', zmid=0,
        text=_text, texttemplate='%{text}',
        colorbar=dict(title='Beta (sig.)', tickfont=dict(family='DM Mono', size=10)),
        hovertemplate='<b>%{y}</b> × <b>%{x}</b><br>β = %{text}<extra></extra>',
    ))
    _fig_heat.update_layout(
        title=f'Sector Factor Betas (starred = BH-FDR q < {ALPHA_THRESHOLD}, others zeroed)',
        height=max(350, len(_sectors) * 40),
        yaxis=dict(autorange='reversed'),
        paper_bgcolor='#0A0E17', plot_bgcolor='#0A0E17',
        font=dict(family='DM Mono', size=11, color='#E2E8F0'),
        margin=dict(l=80, r=20, t=50, b=120),
        xaxis=dict(tickangle=-35),
    )
    _fig_heat
    return


@app.cell
def _(df_impact, factor_cols, go, latest_date):
    # Colour tokens
    VOID     = '#0A0E17'
    SURFACE  = '#111827'
    CARD     = '#1A2035'
    BORDER   = '#2D3A58'
    AQUA     = '#0FF4C6'
    BULL_AMB = '#F7B731'
    BEAR_RED = '#FF4D6D'
    TEXT_DIM = '#4A5568'
    TEXT_MID = '#8896B3'
    TEXT_HI  = '#E2E8F0'

    # Colour palette for factors
    _palette = ['#0FF4C6','#F7B731','#FF4D6D','#7B61FF','#4ECDC4',
                '#FF6B6B','#A8E6CF','#FFD93D','#6BCB77','#4D96FF']
    _factor_colors = {f: _palette[i % len(_palette)] for i, f in enumerate(factor_cols)}

    _sectors_ranked = df_impact.index.tolist()

    # Stacked bar: each factor's signed contribution
    _fig_impact = go.Figure()

    for _f in factor_cols:
        _vals = df_impact[_f].tolist()
        _fig_impact.add_trace(go.Bar(
            name=_f,
            y=_sectors_ranked,
            x=_vals,
            orientation='h',
            marker=dict(color=_factor_colors[_f], opacity=0.85,
                        line=dict(color='rgba(0,0,0,0)', width=0)),
            hovertemplate=f'<b>%{{y}}</b><br>{_f}: %{{x:.4f}}<extra></extra>',
        ))

    _fig_impact.update_layout(
        barmode='relative',          # stacked signed bars
        title=f'Current Macro Impact by Sector  //  regime as of {latest_date.date()}  //  ranked by net',
        paper_bgcolor=VOID, plot_bgcolor=VOID,
        font=dict(family='DM Mono', size=11, color=TEXT_HI),
        legend=dict(bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
                    font=dict(size=9), orientation='h', y=-0.18),
        yaxis=dict(autorange='reversed', gridcolor='rgba(0,0,0,0)',
                    tickfont=dict(family='DM Mono', size=11, color=TEXT_MID)),
        xaxis=dict(gridcolor=BORDER, gridwidth=0.5, zeroline=True,
                    zerolinecolor=TEXT_DIM, zerolinewidth=1,
                    tickfont=dict(family='DM Mono', size=10, color=TEXT_DIM),
                    title='Signed Impact (β × factor score)'),
        height=max(380, len(_sectors_ranked) * 38),
        margin=dict(l=10, r=20, t=50, b=80),
    )
    _fig_impact.show()
    return AQUA, BEAR_RED, BORDER, BULL_AMB, TEXT_DIM, TEXT_HI, TEXT_MID, VOID


@app.cell
def _(page_header):
    page_header("Conditional Monte Carlo Stress Testing", "02")
    return


@app.cell
def _(mo):
    # How big is "1.5σ"? The answer differs by ~4x depending on whether the COVID
    # months are in the σ estimate, and previously that choice was made silently
    # by the data rather than deliberately by the user.
    #
    #   Normal times  — σ from the 119 non-crisis months. A severe-but-ordinary
    #                   move. Use this for the regime call the sector gating acts on.
    #   Crisis replay — σ from the full sample, i.e. anchored to Feb-Jun 2020.
    #                   Use this for tail sizing, not for a directional view.
    #
    # Betas are identical either way (COVID months are dummied out of the slopes),
    # so switching mode changes only the shock magnitude, never the sensitivity.
    scale_mode = mo.ui.radio(
        options={
            'Normal times  (σ excluding Feb–Jun 2020)': 'normal',
            'Crisis replay (σ including Feb–Jun 2020)': 'crisis',
        },
        value='Normal times  (σ excluding Feb–Jun 2020)',
        label='**Scenario scale** — how large is a 1.5σ shock?',
    )
    scale_mode
    return (scale_mode,)


@app.cell
def _(
    AQUA,
    BEAR_RED,
    BORDER,
    BULL_AMB,
    TEXT_DIM,
    TEXT_HI,
    COVID_END,
    COVID_START,
    TEXT_MID,
    VOID,
    chg_units,
    df_chg,
    df_factors,
    df_impact,
    df_master,
    df_scaled,
    fa,
    go,
    latest_factors,
    loadings_named,
    make_subplots,
    math,
    mo,
    np,
    pd,
    reg_results,
    scale_mode,
):
    """
    Scenario MC — stress-test top-2 raw macro DRIVERS per factor
    =============================================================

      1. For each active factor, identify its top-2 loading series
      2. Shock those standardised series by ±SHOCK_SIGMA
      3. Propagate through the macro correlation structure and project onto the
         factors → a shocked score for ALL factors simultaneously
      4. Simulate sector impact with a full uncertainty decomposition
      5. Fan chart + delta table, adaptive to however many scenarios are generated

    Why the MC is a real MC now
    ---------------------------
    The previous version drew factor_samples ~ N(shocked, 0.10 · Σ_hist) and then
    applied the linear map `@ betas.T`. A linear map of a Gaussian is Gaussian, so
    those 10,000 draws reproduced a closed form — no information gained — and the
    0.10 was a magic number. Worse, the band omitted the two dominant sources of
    uncertainty. All three are now modelled:

      (a) factor    — the shock pins one driver; the rest of the macro vector is
                      only conditionally determined. Covariance comes from the
                      Schur complement in raw z-space, projected onto the factors.
                      No tuning constant.
      (b) parameter — betas are estimates. Drawn from N(β̂, Var(β̂)) using the HAC
                      covariance from the regression cell.
      (c) residual  — the 1−R² of each sector regression, previously invisible.
                      This is typically the LARGEST term for monthly equity
                      returns, so the old bands were far too narrow to read as
                      return forecasts.

    Because (b) multiplies (a), the product is genuinely non-Gaussian and
    simulation is now the right tool rather than an expensive detour.

    Upstream variables:
      df_scaled      : pd.DataFrame   — standardised macro CHANGES (FA input)
      fa             : FactorAnalysis — fitted model, for the exact score map
      df_factors     : pd.DataFrame   — rotated factor scores (Varimax)
      df_master      : pd.DataFrame   — sector betas + FDR sig flags
      reg_results    : dict           — per-sector resid_var / beta_cov
      loadings_named : pd.DataFrame   — loadings, rows=macro series, cols=factors
      latest_factors : pd.Series      — most recent factor score vector
    """

    # ── Colour palette for driver scenarios (cycles if > 5 drivers) ──────────
    DRIVER_COLOR_PAIRS = [
        (AQUA,      BEAR_RED),
        (BULL_AMB,  '#A78BFA'),
        ('#F97316', '#EC4899'),
        ('#34D399', '#F43F5E'),
        ('#60A5FA', '#FBBF24'),
        ('#E879F9', '#38BDF8'),
    ]

    SHOCK_SIGMA  = 1.5
    N_SIM        = 10_000
    TOP_N        = 2        # how many drivers per factor to stress-test
    MC_SEED      = 42       # results were previously irreproducible across reloads

    _rng = np.random.default_rng(MC_SEED)

    # ── Exact FA score map ───────────────────────────────────────────────────
    # sklearn computes scores as  F = (Z - mean_) @ M  with
    #   M = (Λᵀ/ψ)ᵀ · (I + Λᵀψ⁻¹Λ)⁻¹
    # Building M explicitly lets us (i) verify the shock projection below and
    # (ii) push a raw-space covariance onto the factors.
    _Wpsi      = fa.components_ / fa.noise_variance_
    SCORE_MAP  = _Wpsi.T @ np.linalg.inv(
        np.eye(fa.components_.shape[0]) + _Wpsi @ fa.components_.T
    )                                                    # (n_series, n_factors)

    # Correlation of the standardised macro changes == their covariance.
    MACRO_CORR = pd.DataFrame(
        np.corrcoef(df_scaled.values, rowvar=False),
        index=df_scaled.columns, columns=df_scaled.columns,
    )

    # ── Scenario scale ───────────────────────────────────────────────────────
    # df_scaled is z-scored on the FULL sample, so 1 z-unit == 1 full-sample σ.
    # Under 'normal', shrink the shock to the ex-COVID σ of each driver:
    #     1.5 normal-σ  ==  1.5 · (σ_ex / σ_full) full-sample z-units.
    # Betas are unaffected (COVID months are dummied out of the slopes), so this
    # changes only the magnitude being asked about.
    SCALE_MODE   = scale_mode.value
    _covid_mask  = (df_chg.index >= COVID_START) & (df_chg.index <= COVID_END)
    _sd_full     = df_chg.std()
    _sd_ex       = df_chg[~_covid_mask].std()

    SHOCK_MULT = {
        c: (1.0 if SCALE_MODE == 'crisis' else float(_sd_ex[c] / _sd_full[c]))
        for c in df_chg.columns
    }

    def native_shock(driver_col, direction, shock_sigma=SHOCK_SIGMA):
        """Express a shock in the driver's own units, e.g. '+0.36pp' or '+10.3%'."""
        unit, unit_scale = chg_units[driver_col]
        z    = direction * shock_sigma * SHOCK_MULT[driver_col]
        move = z * _sd_full[driver_col] * unit_scale
        return f"{move:+.2f}{unit}", z

    print(f"[scale]    mode = {SCALE_MODE!r}  "
          f"(shock multipliers {min(SHOCK_MULT.values()):.2f}–{max(SHOCK_MULT.values()):.2f})")
    for _d in ['UNRATE', 'ICSA', 'DGS10']:
        if _d in SHOCK_MULT:
            print(f"[scale]      {_d:12s} ±{SHOCK_SIGMA}σ = {native_shock(_d, +1)[0]}")


    # =============================================================================
    # STEP 1 — Derive top-N drivers for each active factor
    # =============================================================================

    def get_top_drivers(
        loadings_named: pd.DataFrame,
        factor_cols:    list[str],
        top_n:          int = TOP_N,
    ) -> dict[str, list[str]]:
        """
        Returns {factor_col: [driver1, driver2, ...]} ordered by |loading| descending.
        Only includes drivers whose series are present in the FA input.
        """
        driver_map = {}
        for fc in factor_cols:
            if fc not in loadings_named.columns:
                print(f"[drivers] WARNING — {fc} not in loadings_named, skipping")
                continue
            top = (
                loadings_named[fc]
                .abs()
                .nlargest(top_n + 2)          # fetch extra in case some are absent
                .index.tolist()
            )
            # Filter to series actually present in the standardised macro panel
            available = [s for s in top if s in df_scaled.columns][:top_n]
            driver_map[fc] = available
            print(f"[drivers]  {fc:35s} → top-{top_n}: {available}")
        return driver_map


    # =============================================================================
    # STEP 2 — Build scenario defs from driver map
    # =============================================================================

    def build_driver_scenarios(
        driver_map:   dict[str, list[str]],
        color_pairs:  list = DRIVER_COLOR_PAIRS,
        shock_sigma:  float = SHOCK_SIGMA,
    ) -> list[dict]:
        """
        One +shock and one -shock scenario per (factor, driver) pair.

        If the same raw series is the top driver of two factors it deliberately
        appears once per factor, so the spillover is visible under each label;
        `short` is disambiguated in that case.

        Each scenario dict carries:
          short        : unique key
          label        : display label for subplot title
          color        : hex colour
          driver_col   : raw macro series name (in df)
          factor_label : which factor this scenario is anchored to
          direction    : +1 or -1
        """
        scenarios = []
        color_idx = 0

        for fc, drivers in driver_map.items():
            for driver in drivers:
                pos_color, neg_color = color_pairs[color_idx % len(color_pairs)]
                color_idx += 1

                for direction, color, suffix, arrow in [
                    (+1, pos_color, 'up', '↑'),
                    (-1, neg_color, 'dn', '↓'),
                ]:
                    short = f"{driver}_{suffix}"
                    # Make short unique if same driver appears in multiple factors
                    base_shorts = [s['short'] for s in scenarios]
                    if short in base_shorts:
                        short = f"{driver}_{fc[:6]}_{suffix}"

                    # Label in the driver's own units as well as sigmas — "1.5σ"
                    # alone hid the fact that, on the full sample, it meant a
                    # 1.50pp one-month jump in unemployment.
                    native, _ = native_shock(driver, direction, shock_sigma)
                    scenarios.append({
                        'short':        short,
                        'label':        f"{driver} {arrow}{shock_sigma}σ = {native}\n[{fc}]",
                        'color':        color,
                        'driver_col':   driver,
                        'factor_label': fc,
                        'direction':    direction,
                        'native':       native,
                    })

        print(f"[scenarios]  {len(scenarios)} driver scenarios: {[s['short'] for s in scenarios]}")
        return scenarios


    # =============================================================================
    # STEP 3 — Project raw shock through loading matrix → shocked factor scores
    # =============================================================================

    def project_driver_shock(
        driver_col:     str,
        direction:      float,
        loadings_named: pd.DataFrame,
        factor_cols:    list[str],
        shock_sigma:    float = SHOCK_SIGMA,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Shock one standardised macro driver by ±shock_sigma, propagate it through
        the macro correlation structure, and project onto every factor.

        Mean
        ----
        Scores are  F = Z @ M  (M = SCORE_MAP), and a shock to driver k does not
        move k alone — the other series co-move. Conditioning on Z_k = c gives
            E[Z | Z_k = c] = Σ[:, k] / Σ[k, k] · c
        so the factor delta is  Δf = (Σ[:, k] · c) @ M.

        This simplifies exactly: for the FA-implied covariance Σ = ΛΛᵀ + Ψ,
            Σ @ M = (ΛΛᵀ + Ψ) ψ⁻¹Λ(I + Λᵀψ⁻¹Λ)⁻¹ = Λ(Λᵀψ⁻¹Λ + I)(I + Λᵀψ⁻¹Λ)⁻¹ = Λ
        so  Δf = c · Λ[k, :], i.e. the loading row. Verified numerically against
        Σ @ M below (max abs deviation printed once at startup).

        Note this is a CORRELATED shock, not a ceteris-paribus one: it answers
        "oil moves 1.5σ and everything co-moves as it historically does", which
        is the realistic scenario for stress testing.

        Covariance
        ----------
        Pinning Z_k leaves the rest of the macro vector uncertain. The Schur
        complement gives that residual covariance in raw space,
            Σ_cond = Σ - Σ[:, k] Σ[k, :] / Σ[k, k]
        which projects onto the factors as  Mᵀ Σ_cond M. This replaces the old
        hand-tuned `RESIDUAL_SCALE = 0.10 · Σ_hist`.

        Returns
        -------
        (shocked_scores, cond_cov) : (n_factors,), (n_factors, n_factors)
        """
        mu_factors = df_factors[factor_cols].mean().values.copy()
        n_f        = len(factor_cols)

        if driver_col not in loadings_named.index:
            print(f"[project] WARNING — {driver_col} not in loadings, returning mean")
            return mu_factors, np.eye(n_f) * 1e-8

        # Standardised shock in full-sample z-units, rescaled for the selected
        # scenario mode (the old code computed a raw-unit `raw_shock` and then
        # never used it).
        _, z_shock = native_shock(driver_col, direction, shock_sigma)

        # SCORE_MAP rows follow df_scaled.columns (== MACRO_CORR.columns), so only
        # the factor axis needs subsetting to the active factors.
        _keep = [loadings_named.columns.get_loc(c) for c in factor_cols]
        M_act = SCORE_MAP[:, _keep]                         # (n_series, n_active)

        k       = MACRO_CORR.columns.get_loc(driver_col)
        corr    = MACRO_CORR.values
        delta_z = corr[:, k] / corr[k, k] * z_shock         # correlated shock

        delta_factors = delta_z @ M_act

        # Schur complement -> residual macro covariance with driver k pinned
        cov_cond_raw = corr - np.outer(corr[:, k], corr[k, :]) / corr[k, k]
        cond_cov     = M_act.T @ cov_cond_raw @ M_act
        cond_cov     = (cond_cov + cond_cov.T) / 2
        cond_cov    += np.eye(len(factor_cols)) * 1e-10

        shocked_scores = mu_factors + delta_factors
        print(f"[project]  {driver_col:20s} {'+' if direction>0 else ''}{direction*shock_sigma:.1f}σ  "
              f"→ Δf = {np.round(delta_factors, 4)}")
        return shocked_scores, cond_cov


    # =============================================================================
    # STEP 4 — Monte Carlo with a full uncertainty decomposition
    # =============================================================================

    def run_conditional_mc(
        df_master:      pd.DataFrame,
        factor_cols:    list[str],
        beta_cols:      list[str],
        sig_cols:       list[str],
        shocked_scores: np.ndarray,   # (n_factors,)  mean factor vector under shock
        cond_cov:       np.ndarray,   # (n_factors, n_factors)  Schur residual cov
        n_sim:          int = N_SIM,
    ) -> tuple[np.ndarray, list[str], pd.DataFrame]:
        """
        Simulate the sector impact distribution under a shocked factor vector,
        propagating factor, parameter AND residual uncertainty.

        For sector s:
            impact = β_s' · f  +  ε_s
            f  ~ N(shocked_scores, cond_cov)          (a) factor
            β_s ~ N(β̂_s, Var(β̂_s))  masked by FDR    (b) parameter
            ε_s ~ N(0, σ²_resid,s)                    (c) residual

        β and f are drawn independently and multiplied, so the product is not
        Gaussian — which is precisely why this needs simulating rather than a
        closed form.

        Returns
        -------
        impact_samples : (n_sim, n_sectors)
        sectors        : list[str]
        variance_share : DataFrame — contribution of (a)/(b)/(c) per sector
        """
        sectors   = df_master.index.tolist()
        n_factors = len(factor_cols)

        # The regressions are fitted on ALL factors, but only the factors with
        # non-zero current impact are simulated. Map the active subset back onto
        # the stored (full-width) HAC covariance.
        _all_factors = reg_results[sectors[0]]['orig_factors']
        _act_idx     = [_all_factors.index(f) for f in factor_cols]

        # (a) factor draws, shared across sectors (one macro world per sim path)
        factor_samples = _rng.multivariate_normal(
            shocked_scores, cond_cov, size=n_sim
        )                                                   # (n_sim, n_factors)

        sig_mask  = df_master[sig_cols].values.astype(float)   # (n_sectors, n_factors)
        betas_hat = df_master[beta_cols].values * sig_mask

        impact_samples = np.empty((n_sim, len(sectors)))
        var_rows       = []

        for i, sector in enumerate(sectors):
            info      = reg_results[sector]
            mask_i    = sig_mask[i]
            beta_mu   = betas_hat[i]

            # (b) parameter draws — HAC covariance, zeroed on FDR-insignificant
            # factors so a masked beta stays exactly 0 rather than wobbling.
            beta_cov_i = (info['beta_cov'][np.ix_(_act_idx, _act_idx)]
                          * np.outer(mask_i, mask_i))
            beta_cov_i = (beta_cov_i + beta_cov_i.T) / 2
            beta_draws = _rng.multivariate_normal(
                beta_mu, beta_cov_i + np.eye(n_factors) * 1e-12, size=n_sim
            )                                               # (n_sim, n_factors)

            # (c) residual draws — the 1-R² the old fan chart omitted entirely
            resid_sd    = float(np.sqrt(max(info['resid_var'], 0.0)))
            resid_draws = _rng.normal(0.0, resid_sd, size=n_sim)

            impact_samples[:, i] = (
                np.einsum('ij,ij->i', beta_draws, factor_samples) + resid_draws
            )

            # Variance attribution, holding the other two sources at their mean
            v_factor = float(np.var(factor_samples @ beta_mu))
            v_param  = float(np.var(np.einsum('ij,j->i', beta_draws, shocked_scores)))
            v_resid  = resid_sd ** 2
            v_tot    = v_factor + v_param + v_resid
            var_rows.append({
                'Sector':   sector,
                'Factor_%': 100 * v_factor / v_tot if v_tot else 0.0,
                'Param_%':  100 * v_param  / v_tot if v_tot else 0.0,
                'Resid_%':  100 * v_resid  / v_tot if v_tot else 0.0,
            })

        variance_share = pd.DataFrame(var_rows).set_index('Sector')
        return impact_samples, sectors, variance_share


    # =============================================================================
    # STEP 5 — Run all driver scenarios
    # =============================================================================

    def run_all_driver_scenarios(
        scenarios_def:  list[dict],
        df_master:      pd.DataFrame,
        loadings_named: pd.DataFrame,
        factor_cols:    list[str],
        beta_cols:      list[str],
        sig_cols:       list[str],
        n_sim:          int = N_SIM,
    ) -> dict:
        scenario_outputs = {}

        for sc in scenarios_def:
            # Correlated driver shock → shocked factor vector + residual covariance
            shocked_scores, cond_cov = project_driver_shock(
                driver_col     = sc['driver_col'],
                direction      = sc['direction'],
                loadings_named = loadings_named,
                factor_cols    = factor_cols,
                shock_sigma    = SHOCK_SIGMA,
            )

            samples, sectors, var_share = run_conditional_mc(
                df_master      = df_master,
                factor_cols    = factor_cols,
                beta_cols      = beta_cols,
                sig_cols       = sig_cols,
                shocked_scores = shocked_scores,
                cond_cov       = cond_cov,
                n_sim          = n_sim,
            )

            scenario_outputs[sc['short']] = {
                'samples':        samples,
                'sectors':        sectors,
                'color':          sc['color'],
                'label':          sc['label'],
                'shocked_scores': shocked_scores,
                'driver_col':     sc['driver_col'],
                'factor_label':   sc['factor_label'],
                'direction':      sc['direction'],
                'native':         sc['native'],
                'variance_share': var_share,
            }

        return scenario_outputs


    # =============================================================================
    # STEP 6 — Fan chart (adaptive grid, same visual language as Code 2)
    # =============================================================================

    def build_driver_fan_chart(scenario_outputs: dict) -> go.Figure:
        sc_keys = list(scenario_outputs.keys())
        n_sc    = len(sc_keys)
        n_cols  = 2
        n_rows  = math.ceil(n_sc / n_cols)

        positions = [
            (r + 1, c + 1)
            for r in range(n_rows)
            for c in range(n_cols)
        ][:n_sc]

        subplot_titles = []
        for k in sc_keys:
            lbl = scenario_outputs[k]['label'].replace('\n', '  //  ')
            subplot_titles.append(lbl)

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.08,
            vertical_spacing=max(0.08, 0.28 / n_rows),
        )

        for (row_pos, col), key in zip(positions, sc_keys):
            out     = scenario_outputs[key]
            samples = out['samples']
            color   = out['color']
            secs    = out['sectors']

            p05 = np.percentile(samples,  5, axis=0)
            p25 = np.percentile(samples, 25, axis=0)
            p50 = np.percentile(samples, 50, axis=0)
            p75 = np.percentile(samples, 75, axis=0)
            p95 = np.percentile(samples, 95, axis=0)

            order       = np.argsort(p50)[::-1]
            secs_sorted = [secs[i] for i in order]
            show_legend = (row_pos == 1 and col == 1)

            # 90% CI band
            fig.add_trace(go.Bar(
                y=secs_sorted, x=(p95 - p05)[order], base=p05[order],
                orientation='h',
                marker=dict(color=color, opacity=0.15, line=dict(width=0)),
                name='90% CI', legendgroup='ci90', showlegend=show_legend,
                hovertemplate='p5: %{base:.4f}  p95: %{x:.4f}<extra>90% CI</extra>',
            ), row=row_pos, col=col)

            # IQR band
            fig.add_trace(go.Bar(
                y=secs_sorted, x=(p75 - p25)[order], base=p25[order],
                orientation='h',
                marker=dict(color=color, opacity=0.45, line=dict(width=0)),
                name='IQR', legendgroup='iqr', showlegend=show_legend,
                hovertemplate='p25: %{base:.4f}  p75: %{x:.4f}<extra>IQR</extra>',
            ), row=row_pos, col=col)

            # Median marker
            fig.add_trace(go.Scatter(
                y=secs_sorted, x=p50[order], mode='markers',
                marker=dict(color=color, size=7, symbol='diamond',
                            line=dict(color=VOID, width=1)),
                name='Median', legendgroup='median', showlegend=show_legend,
                hovertemplate='<b>%{y}</b><br>Median: %{x:.4f}<extra></extra>',
            ), row=row_pos, col=col)

            fig.add_vline(
                x=0, line_color='rgba(255,255,255,0.2)',
                line_width=1, row=row_pos, col=col,
            )

        fig.update_xaxes(
            gridcolor=BORDER, gridwidth=0.5,
            zeroline=True, zerolinecolor=TEXT_DIM, zerolinewidth=1,
            tickfont=dict(family='DM Mono', size=9, color=TEXT_DIM),
            # Now a simulated monthly log-return: β·f + ε, not just the β·f
            # point estimate the old bands showed.
            title_text='Simulated monthly log-return (β·f + ε)',
            title_font=dict(size=9, color=TEXT_DIM),
        )
        fig.update_yaxes(
            autorange='reversed',
            gridcolor='rgba(0,0,0,0)',
            tickfont=dict(family='DM Mono', size=10, color=TEXT_MID),
        )
        fig.update_layout(
            paper_bgcolor=VOID,
            plot_bgcolor=VOID,
            font=dict(family='DM Mono', color=TEXT_MID),
            legend=dict(
                bgcolor=VOID, bordercolor=BORDER, borderwidth=1,
                font=dict(size=10, color=TEXT_MID),
            ),
            height=400 * n_rows,
            margin=dict(t=60, b=40, l=20, r=20),
        )
        return fig


    # =============================================================================
    # STEP 7 — Delta table HTML (driver-aware column headers)
    # =============================================================================

    def build_driver_delta_html(
        scenario_outputs: dict,
        df_master:        pd.DataFrame,
        beta_cols:        list[str],
        sig_cols:         list[str],
        latest_factors:   np.ndarray,
        factor_cols:      list[str],
    ) -> str:
        sc_keys = list(scenario_outputs.keys())

        # Current point estimate
        # Slice latest_factors to only the active factors (by position in factor_cols)
        all_factor_cols   = list(df_factors.columns)
        active_idx        = [all_factor_cols.index(f) for f in factor_cols if f in all_factor_cols]
        latest_factors_active = np.array(latest_factors)[active_idx]   # (n_active_factors,)

        betas_eff_curr     = df_master[beta_cols].values * df_master[sig_cols].values.astype(float)
        current_impact_arr = betas_eff_curr @ latest_factors_active
        current_impact     = dict(zip(df_master.index.tolist(), current_impact_arr))

        # Build delta rows
        delta_rows = []
        for key in sc_keys:
            out = scenario_outputs[key]
            p50 = np.percentile(out['samples'], 50, axis=0)
            for s, med in zip(out['sectors'], p50):
                delta_rows.append({
                    'Scenario': key,
                    'Sector':   s,
                    'Delta':    round(med - current_impact.get(s, 0), 4),
                })

        df_delta = (
            pd.DataFrame(delta_rows)
            .pivot_table(index='Sector', columns='Scenario', values='Delta')
            .reindex(columns=sc_keys)
            .round(4)
        )

        def _cell_style(v: float) -> str:
            if v > 0.002:  return f'background:{AQUA}22;color:{AQUA}'
            if v < -0.002: return f'background:{BEAR_RED}22;color:{BEAR_RED}'
            return f'color:{TEXT_MID}'

        # Annotate each column header with driver + factor label + direction.
        # Read the direction off the scenario rather than string-matching '_up'
        # in the key, which would misfire on any driver whose name contains it.
        def _col_header(key: str) -> str:
            out = scenario_outputs[key]
            arrow = '↑' if out['direction'] > 0 else '↓'
            return (f"{out['driver_col']}<br>{arrow} {out['native']}"
                    f"<br>[{out['factor_label'][:12]}]")

        header_cells = ''.join(
            f'<th style="font-family:DM Mono;font-size:8px;letter-spacing:.10em;'
            f'text-transform:uppercase;color:{TEXT_DIM};padding:6px 8px;'
            f'text-align:right;border-bottom:2px solid {BORDER};white-space:nowrap">'
            f'{_col_header(k)}</th>'
            for k in sc_keys
        )

        rows_html = ''
        for sector in df_delta.index:
            rows_html += (
                f'<tr><td style="font-family:DM Mono;font-size:11px;color:{TEXT_HI};'
                f'padding:5px 12px;border-bottom:1px solid {BORDER};white-space:nowrap">'
                f'{sector}</td>'
            )
            for key in sc_keys:
                raw = df_delta.loc[sector, key] if key in df_delta.columns else 0
                v   = float(raw) if (raw is not None and raw == raw) else 0.0
                sign = '+' if v > 0 else ''
                rows_html += (
                    f'<td style="font-family:DM Mono;font-size:11px;{_cell_style(v)};'
                    f'padding:5px 8px;text-align:right;'
                    f'border-bottom:1px solid {BORDER}">{sign}{v:.4f}</td>'
                )
            rows_html += '</tr>'

        # Shocked factor score summary per scenario
        shocked_note_rows = ''
        for key in sc_keys:
            out = scenario_outputs[key]
            sc_vals = '  '.join(
                f"{fc[:12]}={out['shocked_scores'][i]:+.3f}"
                for i, fc in enumerate(factor_cols)
            )
            shocked_note_rows += (
                f'<div style="font-family:DM Mono;font-size:8px;color:{TEXT_DIM};'
                f'padding:2px 0">'
                f'<span style="color:{out["color"]}">{key}</span>  →  {sc_vals}</div>'
            )

        html = f"""
    <div style="background:{VOID};border:1px solid {BORDER};border-radius:3px;
                margin-top:20px;overflow:hidden;">
      <div style="padding:14px 20px 10px;border-bottom:1px solid {BORDER}">
        <span style="font-family:'DM Serif Display',serif;font-size:15px;color:{TEXT_HI}">
          Driver Shock Delta vs Current
        </span>
        <span style="font-family:DM Mono;font-size:9px;color:{TEXT_DIM};
                     margin-left:12px;letter-spacing:.15em">
          // median scenario impact minus current point estimate  ·  top-{TOP_N} drivers per factor  ·  ±{SHOCK_SIGMA}σ correlated shock ({SCALE_MODE} scale, shown in native units) projected via loadings  ·  {N_SIM:,} sims, seed {MC_SEED}
        </span>
      </div>
      <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;background:{VOID}">
          <thead>
            <tr>
              <th style="font-family:DM Mono;font-size:9px;letter-spacing:.12em;
                         text-transform:uppercase;color:{TEXT_DIM};padding:6px 12px;
                         text-align:left;border-bottom:2px solid {BORDER}">Sector</th>
              {header_cells}
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
      <div style="padding:10px 20px;border-top:1px solid {BORDER}">
        <div style="font-family:DM Mono;font-size:8px;color:{TEXT_DIM};margin-bottom:6px;
                    letter-spacing:.12em;text-transform:uppercase">
          Shocked factor scores per scenario (all factors, full spillover)
        </div>
        {shocked_note_rows}
      </div>
      <div style="padding:8px 20px;font-family:DM Mono;font-size:9px;
                  color:{TEXT_DIM};border-top:1px solid {BORDER}">
        ↑ teal = macro tailwind strengthens  ·  ↓ red = headwind deepens
        ·  MC uncertainty = factor (Schur conditional) + parameter (HAC β cov)
        + residual (1−R²); no tuning constant
        ·  β estimated with Feb–Jun 2020 dummied out, so scale mode changes shock
        size only, never sensitivity
      </div>
    </div>
    """
        return html


    # =============================================================================
    # MAIN — wire everything together
    # =============================================================================

    # ── 1. Detect active factors from df_impact ──────────────────────────────
    EXCLUDE_COLS  = {'Net_Impact', 'Gross_Impact'}
    factor_cols_sce = [
        c for c in df_impact.columns
        if c not in EXCLUDE_COLS and df_impact[c].abs().sum() > 1e-12
    ]
    print(f"[factors]  Active: {factor_cols_sce}")

    # ── 2. Match beta/sig cols from df_master ────────────────────────────────
    cols_lower    = {c.lower(): c for c in df_master.columns}
    beta_cols_sce = [cols_lower[f"{f}_beta".lower()] for f in factor_cols_sce
                     if f"{f}_beta".lower() in cols_lower]
    sig_cols_sce  = [cols_lower[f"{f}_sig".lower()]  for f in factor_cols_sce
                     if f"{f}_sig".lower()  in cols_lower]
    print(f"[betas]    {beta_cols_sce}")
    print(f"[sigs]     {sig_cols_sce}")

    # ── 2b. Verify the shock projection identity  Σ @ M == Λ ─────────────────
    # The correlated-shock delta reduces to the loading row only because the FA
    # model implies Σ = ΛΛᵀ + Ψ. Assert it numerically rather than trusting the
    # algebra, so a future change to the FA fit cannot silently break the
    # scenarios.
    # The identity is exact only for the model-implied Σ = ΛΛᵀ + Ψ; with k < p the
    # fitted model does not reproduce the sample Σ exactly, so compare the
    # deviation RELATIVE to typical loading magnitude rather than to zero.
    _Lam_ref   = loadings_named.loc[MACRO_CORR.columns].values
    _ident_dev = float(np.abs(MACRO_CORR.values @ SCORE_MAP - _Lam_ref).max())
    _ident_rel = _ident_dev / float(np.abs(_Lam_ref).mean())
    print(f"[check]    max |Σ@M - Λ| = {_ident_dev:.2e}  "
          f"({_ident_rel:.2%} of mean |Λ|) — "
          f"{'OK' if _ident_rel < 1e-2 else 'WARNING: projection identity degraded'}")

    # ── 3. Top-2 drivers per factor ──────────────────────────────────────────
    driver_map = get_top_drivers(loadings_named, factor_cols_sce, top_n=TOP_N)

    # ── 4. Build driver scenario defs ────────────────────────────────────────
    scenarios_def = build_driver_scenarios(driver_map)

    # ── 5. Run MC for all driver scenarios ──────────────────────────────────
    scenario_outputs = run_all_driver_scenarios(
        scenarios_def  = scenarios_def,
        df_master      = df_master,
        loadings_named = loadings_named,
        factor_cols    = factor_cols_sce,
        beta_cols      = beta_cols_sce,
        sig_cols       = sig_cols_sce,
        n_sim          = N_SIM,
    )

    # ── 6. Fan chart ─────────────────────────────────────────────────────────
    fig_driver_cmc = build_driver_fan_chart(scenario_outputs)

    # ── 7. Delta table ───────────────────────────────────────────────────────
    driver_delta_html = build_driver_delta_html(
        scenario_outputs = scenario_outputs,
        df_master        = df_master,
        beta_cols        = beta_cols_sce,
        sig_cols         = sig_cols_sce,
        latest_factors   = latest_factors,
        factor_cols      = factor_cols_sce,   # ← active factors only, used for slicing
    )

    # ── 8. Variance attribution ──────────────────────────────────────────────
    # Makes explicit how much of each fan-chart band is macro signal versus
    # estimation error versus plain idiosyncratic noise. For monthly equity
    # returns Resid_% is usually dominant — that is the honest answer, and it is
    # what the previous 10%-of-covariance band concealed.
    _var_share = (
        scenario_outputs[next(iter(scenario_outputs))]['variance_share']
        .round(1)
        .sort_values('Resid_%', ascending=False)
    )
    print("\n=== IMPACT VARIANCE ATTRIBUTION (first scenario, % of total) ===")
    print(_var_share.to_string())

    _var_rows = ''.join(
        f'<tr>'
        f'<td style="font-family:DM Mono;font-size:11px;color:{TEXT_HI};padding:4px 12px;'
        f'border-bottom:1px solid {BORDER}">{_s}</td>'
        + ''.join(
            f'<td style="font-family:DM Mono;font-size:11px;color:{TEXT_MID};'
            f'padding:4px 12px;text-align:right;border-bottom:1px solid {BORDER}">'
            f'{_r[_c]:.1f}%</td>'
            for _c in ['Factor_%', 'Param_%', 'Resid_%']
        )
        + '</tr>'
        for _s, _r in _var_share.iterrows()
    )

    _var_html = f"""
    <div style="background:{VOID};border:1px solid {BORDER};border-radius:3px;
                margin-top:20px;overflow:hidden;">
      <div style="padding:14px 20px 10px;border-bottom:1px solid {BORDER}">
        <span style="font-family:'DM Serif Display',serif;font-size:15px;color:{TEXT_HI}">
          Where the Uncertainty Comes From
        </span>
        <span style="font-family:DM Mono;font-size:9px;color:{TEXT_DIM};
                     margin-left:12px;letter-spacing:.15em">
          // share of simulated impact variance  ·  factor · parameter · residual
        </span>
      </div>
      <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;background:{VOID}">
          <thead><tr>
            <th style="font-family:DM Mono;font-size:9px;letter-spacing:.12em;
                       text-transform:uppercase;color:{TEXT_DIM};padding:6px 12px;
                       text-align:left;border-bottom:2px solid {BORDER}">Sector</th>
            {''.join(
                f'<th style="font-family:DM Mono;font-size:9px;letter-spacing:.12em;'
                f'text-transform:uppercase;color:{TEXT_DIM};padding:6px 12px;'
                f'text-align:right;border-bottom:2px solid {BORDER}">{_h}</th>'
                for _h in ['Factor', 'Parameter', 'Residual']
            )}
          </tr></thead>
          <tbody>{_var_rows}</tbody>
        </table>
      </div>
    </div>
    """

    # ── 9. Render (Marimo) ───────────────────────────────────────────────────
    mo.vstack([
        mo.ui.plotly(fig_driver_cmc),
        mo.Html(driver_delta_html),
        mo.Html(_var_html),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
