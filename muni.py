import marimo

__generated_with = "0.23.15"
app = marimo.App(
    width="medium",
    css_file="../../theme.css",
    html_head_file="../../theme_head.html",
)


@app.cell
def _():
    import pathlib

    import numpy as np
    import rateslib as rl
    import requests
    from requests.adapters import HTTPAdapter, Retry

    import marimo as mo
    import sys as _sys

    # market_cache is shared with the Macro desk, so it lives in <root>/shared.
    _shared = str(pathlib.Path(mo.notebook_dir()).parents[1] / "shared")
    if _shared not in _sys.path:
        _sys.path.insert(0, _shared)

    import bondfile

    # rateslib prices the assumed fixed-rate schedules and calibrates the
    # descriptive discount curve. Holdings alone do not supply call or tax terms.
    return HTTPAdapter, Retry, bondfile, mo, np, pathlib, requests, rl


@app.cell
def _(HTTPAdapter, Retry, bondfile, pathlib, requests):
    """
    SPDR publishes each fund's complete holdings daily, and every row carries
    par value and market value — so price falls out as market/par*100. That is
    the free per-CUSIP municipal bond price source this desk runs on.

    The fund is never analysed here and never bought. It is the delivery
    mechanism, exactly as FedInvest is for Treasuries.

    Three files, not because they slice by maturity alone: TFI is the broad
    national muni market, HYMB is the high-yield muni slice (below-investment-
    grade credits the broad fund is too clean to hold), and SHM is short
    maturities. Together they widen the credit and maturity range a single
    fund would miss.
    """
    _BASE = ("https://www.ssga.com/us/en/intermediary/library-content/products"
             "/fund-data/etfs/us/holdings-daily-us-en")

    TICKERS = ("tfi", "hymb", "shm")
    import logging as _logging
    import market_cache as _market_cache
    import bondfile as _bondfile

    session = requests.Session()
    session.mount("https://", HTTPAdapter(
        max_retries=Retry(total=3, backoff_factor=1,
                          status_forcelist=[429, 500, 502, 503, 504],
                          allowed_methods=["GET"]),
    ))
    session.headers.update({
        # A default python-requests UA is served an HTML shell on several
        # issuer sites. Send something browser-shaped.
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    })

    def fetch(tickers):
        """Validated holdings in private R2, with daily in-memory reuse."""
        _cache = _market_cache.get_cache()
        files, failed = {}, []
        for t in tickers:
            try:
                files[t] = _cache.holdings(t, session, _bondfile.read_holdings)
            except Exception as _exc:
                _logging.getLogger("market_cache").warning(
                    "SSGA holdings unavailable for %s (%s)", t, type(_exc).__name__)
                failed.append(t)
        return files, failed

    return TICKERS, fetch


@app.cell
def _(TICKERS, bondfile, fetch):
    # One rateslib spec for the sector, named once and shared. bondfile.add_yield
    # prices every bond's OBSERVED yield under it, and the analytics cell prices
    # every bond's benchmark yield under it. Two literals could drift, and a residual
    # computed across two different day-count/frequency conventions would be a
    # convention gap wearing richness as a costume.
    SPEC = "us_muni"
    _files, failed = fetch(TICKERS)
    if not _files:
        raise RuntimeError(
            "no SPDR holdings files could be fetched — check the URL pattern "
            "in the fetch cell, or whether the issuer now blocks direct download "
            "(the spec records EDGAR N-PORT as the fallback)")
    as_of, universe, dropped = bondfile.build_sector(_files, SPEC)
    return SPEC, as_of, dropped, failed, universe


@app.cell
def _(SPEC, as_of, bondfile, np, rl, universe):
    """Descriptive cross-credit YTM benchmark, not an OAS or fair-value model.

    Sparse synthetic instruments summarize median coupon/yield by tenor.
    Their prices are schedule-aware, but median yields across different
    issuers, coupons and options are only an approximation to a market curve.
    No ratings, call schedules or issuer controls are available in this feed.
    """
    MIN_FIT_YEARS, MAX_FIT_YEARS = bondfile.MIN_YEARS_TO_MATURITY, 31.0
    MIN_SANE_YIELD, MAX_SANE_YIELD = bondfile.YTM_FLOOR, bondfile.DISTRESSED_YTM
    _settle = as_of.to_pydatetime()
    # Match the observed-yield schedule anchor exactly. No business-day roll
    # is appropriate for this artificial historical schedule start.
    from pandas import DateOffset as _DateOffset
    _effective = (as_of - _DateOffset(years=5)).to_pydatetime()
    CURVE_CONVENTION = "act365f"

    def _years(maturities):
        return [rl.dcf(_settle, m.to_pydatetime(), CURVE_CONVENTION)
                for m in maturities]

    def _tenor_date(years):
        from datetime import timedelta
        return _settle + timedelta(days=round(float(years) * 365))

    _fit = universe[universe["in_fit"] & np.isfinite(universe["ytm"])].copy()
    _fit["years"] = _years(_fit["maturity"])
    _fit = _fit[_fit["years"].between(MIN_FIT_YEARS, MAX_FIT_YEARS, inclusive="neither")]
    _fit = _fit.sort_values("years")
    if len(_fit) < 30:
        raise ValueError(f"only {len(_fit)} eligible bonds; need at least 30")

    _TENORS = (0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30)
    _buckets, dropped_buckets = {}, []
    for _t in _TENORS:
        _b = _fit[_fit["years"].between(_t * 0.75, _t * 1.25)]
        _target = float(_b["ytm"].median()) if len(_b) else float("nan")
        _reason = ("too few bonds" if len(_b) < 5 else
                   "median yield outside the sanity band" if not
                   MIN_SANE_YIELD < _target < MAX_SANE_YIELD else "")
        if _reason:
            dropped_buckets.append(dict(tenor=_t, reason=_reason, n=len(_b), target=_target))
            continue
        _rep = _b.iloc[int((_b["years"] - _b["years"].median()).abs().argmin())]
        _date = _rep["maturity"].to_pydatetime()
        if _date in _buckets:
            dropped_buckets.append(dict(tenor=_t, reason="duplicate anchor maturity", n=len(_b), target=_target))
            continue
        _buckets[_date] = (float(_b["Coupon"].median()), _target, _b.index)
    if len(_buckets) < 3:
        raise ValueError(f"only {len(_buckets)} surviving anchors; need at least 3: {dropped_buckets}")
    _buckets = dict(sorted(_buckets.items()))
    anchor_tenors = tuple(rl.dcf(_settle, d, CURVE_CONVENTION) for d in _buckets)
    MIN_FIT_YEARS, MAX_FIT_YEARS = min(anchor_tenors), max(anchor_tenors)
    # Node dates must coincide with representative maturities. A realistic
    # initial DF avoids a zero-yield starting point in the nested YTM solver.
    _cv = rl.Curve(nodes={_settle: 1.0, **{
        d: (1 + target / 200) ** (-2 * rl.dcf(_settle, d, CURVE_CONVENTION))
        for d, (_, target, _) in _buckets.items()}},
        interpolation="log_linear", convention=CURVE_CONVENTION,
        calendar="nyc", id="muni")
    _pricing = dict(curves=_cv, metric="ytm", settlement=_settle)
    _insts = [rl.FixedRateBond(effective=_effective, termination=d,
                              fixed_rate=coupon, spec=SPEC, curves=_cv.id)
              for d, (coupon, _, _) in _buckets.items()]
    _targets = [target for _, target, _ in _buckets.values()]
    anchor_yields = tuple(_targets)
    n_anchors = len(_insts)
    _solver = rl.Solver(curves=[_cv], instruments=[(i, _pricing) for i in _insts],
                        s=_targets, id="munifit")
    _errors = np.array([float(i.rate(**_pricing)) for i in _insts]) - _targets
    if not np.isfinite(_errors).all() or np.max(np.abs(_errors)) > 1e-5:
        raise ValueError("calibration failed to reprice its targets within 0.001bp")
    _dfs = np.array([float(_cv[d]) for d in _buckets])
    if not (np.isfinite(_dfs) & (_dfs > 0)).all():
        raise ValueError("calibration produced invalid discount factors")

    reference_coupon = float(_fit["Coupon"].median())

    def curve(years, coupon=reference_coupon):
        """Reference-coupon YTM; unsupported tenors are explicitly missing."""
        return np.array([
            float(rl.FixedRateBond(effective=_effective, termination=_tenor_date(y),
                                  fixed_rate=coupon, spec=SPEC).rate(**_pricing))
            if MIN_FIT_YEARS <= y <= MAX_FIT_YEARS else float("nan")
            for y in np.atleast_1d(np.asarray(years, dtype=float))])

    # Yield curves may be flat, inverted or negative. Validate numerical
    # health rather than imposing a market view on the slope.
    _grid = curve(np.linspace(MIN_FIT_YEARS, MAX_FIT_YEARS, 50))
    if not np.isfinite(_grid).all():
        raise ValueError("calibrated curve has non-finite yields")
    fitted = universe.copy()
    fitted["years"] = _years(fitted["maturity"])
    _source_indices = set().union(*(set(idx) for _, _, idx in _buckets.values()))
    fitted["is_fit_source"] = fitted.index.isin(_source_indices)

    def _fair_ytm(row):
        if (not row["in_fit"] or not
                MIN_FIT_YEARS <= row["years"] <= MAX_FIT_YEARS):
            return float("nan")
        try:
            return float(rl.FixedRateBond(effective=_effective,
                termination=row["maturity"].to_pydatetime(),
                fixed_rate=float(row["Coupon"]), spec=SPEC).rate(**_pricing))
        except Exception:
            return float("nan")

    fitted["curve_ytm"] = [_fair_ytm(r) for _, r in fitted.iterrows()]
    _valid = np.isfinite(fitted["ytm"]) & np.isfinite(fitted["curve_ytm"])
    _inside = fitted["years"].between(MIN_FIT_YEARS, MAX_FIT_YEARS)
    fitted["is_ranked"] = fitted["in_fit"] & _inside & _valid
    fitted["resid_bp"] = ((fitted["ytm"] - fitted["curve_ytm"]) * 100).where(fitted["is_ranked"])
    fitted["bucket"] = fitted["excluded_reason"].where(
        ~fitted["in_fit"], np.where(~_inside, "outside_window",
                                  np.where(_valid, "ranked", "no_yield")))
    _resid = fitted.loc[fitted["is_ranked"], "resid_bp"]
    if len(_resid) < 2:
        raise ValueError("fewer than two finite residuals inside the supported curve range")
    resid_sd_bp = float(_resid.std())
    return (MAX_FIT_YEARS, MIN_FIT_YEARS, anchor_tenors, anchor_yields,
            curve, dropped_buckets, fitted, n_anchors, reference_coupon, resid_sd_bp)


@app.cell
def _(mo):
    """The reader's federal marginal rate. A municipal desk that assumes one
    bracket answers the wrong question for everyone in a different one."""
    bracket = mo.ui.dropdown(
        options={"0%": 0.0, "10%": 0.10, "12%": 0.12, "22%": 0.22, "24%": 0.24,
                 "32%": 0.32, "35%": 0.35, "37%": 0.37},
        value="24%", label="Federal marginal rate")
    assume_exempt = mo.ui.checkbox(
        value=False, label="Show illustrative TEY assuming the entire YTM is federally exempt")
    mo.vstack([
        bracket, assume_exempt,
        mo.md(
            "*TEY is an optional **federal-only scenario**, not a bond-specific "
            "after-tax return. Tax status, AMT, original-issue discount, taxable "
            "market discount, premium amortization and state/local taxes are "
            "not determined by this holdings feed. No TEY is shown by default.*"),
    ])
    return assume_exempt, bracket


@app.cell
def _(assume_exempt, bondfile, bracket, fitted):
    """Optional gross-up scenario; the feed cannot establish actual tax treatment."""
    taxed = fitted.copy()
    taxed["tey"] = (bondfile.tax_equivalent_yield(taxed["ytm"], bracket.value)
                    .where(taxed["in_fit"]) if assume_exempt.value else float("nan"))
    return (taxed,)


@app.cell
def _(
    MAX_FIT_YEARS,
    MIN_FIT_YEARS,
    anchor_tenors,
    anchor_yields,
    as_of,
    assume_exempt,
    bondfile,
    bracket,
    curve,
    dropped,
    dropped_buckets,
    failed,
    fitted,
    mo,
    n_anchors,
    np,
    reference_coupon,
    resid_sd_bp,
    taxed,
):
    """The desk, rendered as a section of the same publication as treasury.py.

    Same devices — live dot, rule to the margin, serif italic clamp, double
    green rule, gold datum, colour only past one standard deviation — and the
    same stylesheet, copied verbatim rather than reinterpreted. The corporate
    desk is its twin: everything here is the same code but for the tax layer,
    which is the one thing a municipal desk cannot do without.

    Two signature drawings, both direct translations of the Treasury desk's:

      the spread fan     treasury.py hangs every bill below par by its
                         discount. Par is a real gold line there because a
                         bill genuinely is a promise of 100. Here the datum
                         is benchmark off the fitted municipal curve, and
                         each bond hangs off it by its own residual — cheap
                         above, rich below. Same device, different datum.
      the credit surface the working instrument, exactly as the bill curve is
                         there: every ranked bond as a dot at its tenor and
                         observed yield, with the calibrated curve drawn
                         through the cloud and its anchor tenors marked.

    Nothing unranked is drawn on either. resid_bp is NaN outside the ranked
    set by construction, and a bond the curve was never allowed to see has no
    place on a chart about distance from that curve.

    This cell reads `taxed` rather than `fitted` so the bracket dropdown
    genuinely drives it: change the bracket and the fourth stat tile, the
    verdict line and the TEY column all move together, because all three come
    off the same `bracket.value`."""
    # Every count on this page comes off `bucket`, which is one label per bond.
    # It used to come off a mix of independent booleans and reason strings:
    # `is_variable`, `is_distressed`, and `excluded_reason == "no_yield"`
    # overlapped, so this desk rendered "0 variable-rate, 25 distressed,
    # 26 no-yield" — 51 exclusions against 33 bonds actually excluded, because
    # 18 distressed bonds were also counted as no-yield.
    _n = fitted["bucket"].value_counts()
    _n_ranked = int(_n.get("ranked", 0))
    _n_window = int(_n.get("outside_window", 0))
    _n_var = int(_n.get("variable_rate", 0))
    _n_dist = int(_n.get("distressed", 0))
    # A yield rateslib could not solve, or one outside the sanity band and so
    # corrupted rather than distressed. Never surfacing this would mean a
    # future data/library break could empty the curve and the page would say
    # nothing about why.
    _n_noyield = int(_n.get("no_yield", 0))
    if _n.sum() != len(fitted) or set(_n.index) - {
            "ranked", "outside_window", *bondfile.EXCLUDED_REASONS}:
        raise ValueError(
            f"exclusion tally covers {int(_n.sum())} of {len(fitted):,} bonds "
            f"({_n.to_dict()}) — the buckets are no longer a partition, so "
            "the numbers on this page would not reconcile")

    # ── The design system ───────────────────────────────────────────────
    # treasury.py's FI_CSS, copied verbatim — same tokens, same class names,
    # one palette. Inventing a second palette here is precisely how a
    # publication stops reading as one publication, and the token block is the
    # hub's own (main.py :root) rather than anything chosen on this desk.
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

    # The only extension: the two SVG figures this desk has and the Treasury
    # desk does not, plus the page's one orchestrated moment.
    #
    # The stems fade and rise as the fan loads, staggered by tenor so the fan
    # draws front-end first. The stagger runs on a dozen tenor BANDS rather
    # than on each bond: the left-to-right draw reads identically and the
    # browser animates twelve elements instead of four thousand. Nothing else
    # on the page moves — a desk that animates its tables is a desk asking to
    # be distrusted — and the whole animation lives inside
    # `prefers-reduced-motion: no-preference`, with a `reduce` rule that
    # switches it off outright. Declaring it under `no-preference` rather than
    # merely overriding it under `reduce` also means a browser that cannot
    # answer the question gets the still page.
    FI_EXTRA = """<style>
    .fi-fan text, .fi-surface text { font-family: var(--mono); }
    /* Hover belongs to the individual dot, never to the tenor band it happens
       to be animated in — the bands are a batching device, not a target. */
    .fi-fan circle { transition: r 0.12s ease; }
    .fi-fan circle:hover { r: 3.4; }
    @media (prefers-reduced-motion: no-preference) {
      .fi-stem { animation: fi-hang 0.5s cubic-bezier(0.22,0.7,0.3,1) backwards; }
      @keyframes fi-hang {
        from { opacity: 0; transform: translateY(-6px); }
        to   { opacity: 1; transform: none; }
      }
    }
    @media (prefers-reduced-motion: reduce) {
      .fi-stem { animation: none; opacity: 1; }
    }
    </style>"""

    # Green reads cheap/bid, sienna rich/offer, gold is benchmark. Slate is
    # "inside the noise", and it is the default.
    _GREEN, _SIENNA, _GOLD = "#1D9E75", "#D8683C", "#C9A961"
    _SLATE, _RULE = "#93A49B", "#26332C"

    # Only dislocations beyond one standard deviation earn a colour. Painting
    # every ±1bp residual green or sienna dresses quote noise up as conviction,
    # and on a blended IG/high-yield municipal fit — where a standard deviation
    # runs several times the corporate desk's — it would paint the entire page.
    _sig = resid_sd_bp

    def _tone(_v, gate=True):
        """Cheap (yields more than fair) reads green; rich reads sienna."""
        if _v is None or np.isnan(_v):
            return _SLATE
        if gate and abs(_v) < _sig:
            return _SLATE
        return _GREEN if _v > 0 else _SIENNA

    def _handle(_y, dp=3):
        """Split a number the way a rates screen reads it: handle, then tail."""
        _s = f"{_y:.{dp}f}"
        return _s[:-1], _s[-1]

    def _esc(_s):
        """Issuer names come out of a vendor file and are not trusted markup."""
        return (str(_s).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))

    # THE set both drawings are built from. `is_ranked` is the curve's own
    # domain; the extra notna is for the handful of bonds rateslib could rank
    # but not solve a fair yield for — they have no residual, so they have no
    # position on either axis and must not be invented one.
    _drawn = (taxed[taxed["is_ranked"] & taxed["resid_bp"].notna()]
              .sort_values("years"))
    _n_drawn = len(_drawn)
    _unplaced = _n_ranked - _n_drawn

    # Square-root tenor axis, exactly as treasury.py's rail and curve use.
    # Linear-in-years buries the two- to five-year part of a municipal curve —
    # where most of the universe actually sits — in the first tenth of the
    # width. Both figures share it, so a bond is at the same x in each.
    _W, _PADL, _PADR = 1000, 64, 26
    _maxyr = float(_drawn["years"].max())
    _xs = lambda _y: (_PADL + (np.sqrt(max(float(_y), 0.0)) / np.sqrt(_maxyr))
                      * (_W - _PADL - _PADR))

    # ── Signature: the spread fan ───────────────────────────────────────
    # The fitted curve, flattened to a gold datum line, with every ranked bond
    # hanging off it by its own residual — up for cheap, down for rich.
    _FH, _DAT, _HALF = 152, 74, 54
    # Scale the stems off the 98th percentile of |residual|, not the maximum.
    # A handful of thousand-basis-point outliers would otherwise squash the
    # whole universe into a two-pixel band around the datum and turn the
    # signature into a hairline — and on this desk, where a deep-discount
    # long-dated credit can sit an order of magnitude off the fit, that is not
    # hypothetical. Stems past the scale are drawn full length, and the caption
    # says where the scale stops so a clipped stem is not read as an exact
    # length.
    _cap = max(float(np.nanpercentile(np.abs(_drawn["resid_bp"].to_numpy()), 98)),
               1.0)
    # Twelve tenor bands, and the load animation runs on the BANDS. Staggering
    # each bond individually animates four thousand elements and stutters the
    # whole page; a dozen groups draw left-to-right identically because the
    # bands are cut on the same x the stagger was.
    _NBANDS = 12
    _bands = [[] for _ in range(_NBANDS)]
    for _, _r in _drawn.iterrows():
        _x = _xs(_r["years"])
        _v = float(_r["resid_bp"])
        _y2 = _DAT - max(-1.0, min(1.0, _v / _cap)) * _HALF
        _mark = f'<line x1="{_x:.1f}" y1="{_DAT}" x2="{_x:.1f}" y2="{_y2:.1f}"/>'
        # The terminal dot is reserved for bonds that carry a verdict. A slate
        # dot on a slate stem is a node that says nothing the stem has not
        # already said — and four thousand of them turn the signature into
        # uniform speckle, which is the opposite of what it is for. Every bond
        # still draws its stem, so nothing is hidden; only the redundant
        # marker and its equally redundant tooltip go.
        #
        # On this desk that leaves a mostly bare fan, and it should. A 1sd of
        # ~110bp on a fit blending investment-grade and high-yield municipals
        # is the truth about this signal's quality, and the flat grey fan is
        # the strongest evidence for the dispersion caveat above it. Dressing
        # it up would be the desk lying about how much it knows.
        if abs(_v) >= _sig:
            _mark += (f'<circle cx="{_x:.1f}" cy="{_y2:.1f}" r="2.2" '
                      f'fill="{_tone(_v)}"><title>{_esc(str(_r["Name"])[:46])} · '
                      f'{_r["years"]:.1f}y · {_v:+.0f}bp</title></circle>')
        _bands[min(int((_x - _PADL) / (_W - _PADL - _PADR) * _NBANDS),
                   _NBANDS - 1)].append(_mark)
    _stems = [f'<g class="fi-stem" style="animation-delay:'
              f'{_i / (_NBANDS - 1) * 0.55:.2f}s">' + "".join(_g) + '</g>'
              for _i, _g in enumerate(_bands) if _g]
    _fan = (
        f'<svg class="fi-fan" viewBox="0 0 {_W} {_FH}" width="100%" role="img" '
        f'aria-label="Every ranked municipal bond hanging off the fitted curve '
        f'by its residual — positive gaps above the line, negative below">'
        f'<line x1="{_PADL}" y1="{_DAT}" x2="{_W - _PADR}" y2="{_DAT}" '
        f'stroke="{_GOLD}" stroke-width="1"/>'
        f'<text x="0" y="{_DAT - 5}" fill="{_GOLD}" font-size="10">benchmark</text>'
        f'<text x="0" y="{_FH - 6}" fill="{_SLATE}" font-size="10">'
        f'±{_cap:.0f}bp</text>'
        # Stem colour and weight are set once on the parent and inherited.
        f'<g stroke="{_GREEN}" stroke-opacity="0.22" stroke-width="1">'
        + "".join(_stems) + '</g></svg>')

    # ── The curve, as the working instrument ────────────────────────────
    # Every ranked bond at its tenor and its OBSERVED tax-exempt yield, with
    # the calibrated curve through the cloud and its anchor tenors marked, so
    # the residual the fan draws can be read back off the thing it came from.
    _CH, _CT, _CB = 340, 26, 44
    _ts = np.linspace(max(MIN_FIT_YEARS, float(_drawn["years"].min())),
                      min(MAX_FIT_YEARS, _maxyr), 32)
    _cvals = curve(_ts)
    # Each calibration instrument has its own bucket-median coupon. Its
    # calibrated YTM need not equal the reference-coupon curve at that tenor.
    _anchor_ys = np.asarray(anchor_yields, dtype=float)
    # The y-axis is set off the 0.5/99.5 percentiles of observed yield, not
    # the extremes: the ranked set legitimately reaches toward the distressed
    # threshold, and one 20% mark would flatten the whole cloud. The few dots
    # beyond the frame are clipped rather than moved or dropped.
    _p_lo, _p_hi = (float(_v) for _v
                    in np.nanpercentile(_drawn["ytm"].to_numpy(dtype=float),
                                        [0.5, 99.5]))
    _lo = min(_p_lo, float(_cvals.min()), float(_anchor_ys.min())) - 0.15
    _hi = max(_p_hi, float(_cvals.max()), float(_anchor_ys.max())) + 0.15
    _ys = lambda _v: _CT + (_hi - _v) / (_hi - _lo) * (_CH - _CT - _CB)

    _rng = _hi - _lo
    _tickstep = 0.25 if _rng <= 2.5 else 0.5 if _rng <= 5 else 1.0
    _gridlines, _tick = [], float(np.ceil(_lo / _tickstep) * _tickstep)
    while _tick <= _hi:
        _gy = _ys(_tick)
        _gridlines.append(
            f'<line x1="{_PADL}" y1="{_gy:.1f}" x2="{_W - _PADR}" y2="{_gy:.1f}" '
            f'stroke="{_RULE}" stroke-width="1"/>'
            f'<text x="{_PADL - 10}" y="{_gy + 3:.1f}" fill="{_SLATE}" '
            f'text-anchor="end" font-size="10">{_tick:.2f}</text>')
        _tick += _tickstep

    # Every ranked bond gets a dot — the density of this cloud is the whole
    # argument for drawing it — but only the ones beyond a standard deviation
    # get a <title>. A tooltip on a bond sitting inside the noise band tells
    # the reader nothing it has not already been told by the dot being slate.
    def _dot(_r):
        _v = float(_r["resid_bp"])
        _c = (f'<circle cx="{_xs(_r["years"]):.1f}" cy="{_ys(_r["ytm"]):.1f}" '
              f'r="1.6" fill="{_tone(_v)}"')
        if abs(_v) < _sig:
            return _c + '/>'
        return (_c + f'><title>{_esc(str(_r["Name"])[:46])} · '
                f'{_r["years"]:.1f}y · {_r["ytm"]:.3f}%</title></circle>')

    _cloud = "".join(_dot(_r) for _, _r in _drawn.iterrows())
    _fitline = " ".join(f"{_xs(_t):.1f},{_ys(_v):.1f}"
                        for _t, _v in zip(_ts, _cvals))
    _anchors = "".join(
        f'<rect x="{_xs(_t) - 2.6:.1f}" y="{_ys(_v) - 2.6:.1f}" width="5.2" '
        f'height="5.2" fill="{_GOLD}"><title>{_t:g}y anchor · '
        f'{_v:.3f}%</title></rect>'
        for _t, _v in zip(anchor_tenors, _anchor_ys))
    # A term structure is read in tenors, not calendar dates.
    _tenor_labels = "".join(
        f'<text x="{_xs(_t):.1f}" y="{_CH - 14}" fill="{_SLATE}" '
        f'text-anchor="middle" font-size="10">{_lab}</text>'
        for _t, _lab in ((1, "1y"), (2, "2y"), (5, "5y"), (10, "10y"),
                         (20, "20y"), (30, "30y")) if _t <= _maxyr)
    _surface = (
        f'<svg class="fi-surface" viewBox="0 0 {_W} {_CH}" width="100%" '
        f'role="img" aria-label="Observed yield against tenor for every ranked '
        f'municipal bond, with the calibrated curve through it">'
        f'<clipPath id="fi-surface-clip"><rect x="{_PADL - 6}" y="{_CT - 8}" '
        f'width="{_W - _PADR - _PADL + 12}" height="{_CH - _CB - _CT + 10}"/>'
        f'</clipPath>'
        + "".join(_gridlines)
        # fill-opacity on the parent, not opacity: it inherits per dot, so
        # overlapping bonds still darken and the cloud keeps its density.
        + f'<g clip-path="url(#fi-surface-clip)" fill-opacity="0.5">'
          f'{_cloud}</g>'
        + f'<polyline points="{_fitline}" fill="none" stroke="{_GOLD}" '
          f'stroke-width="1.8"/>' + _anchors + _tenor_labels + '</svg>')

    # ── The day's read ──────────────────────────────────────────────────
    # The fourth tile is the whole reason this desk exists separately from the
    # corporate one: a municipal yield is not comparable to a taxable yield
    # until it has been grossed up at the reader's own bracket, and the
    # bracket is the reader's to choose. bondfile.tax_equivalent_yield is the
    # same function the per-bond TEY column uses — one definition, so the
    # headline and the table can never disagree.
    _c2, _c10, _c30 = (float(_v) for _v in curve([2, 10, 30]))
    _tey10 = (bondfile.tax_equivalent_yield(_c10, bracket.value)
              if assume_exempt.value else float("nan"))
    _brk = f"{bracket.value:.0%}"

    def _tile(_lbl, _val, _sub, unit="%", dp=3):
        _h, _t = _handle(_val, dp) if np.isfinite(_val) else ("N/A", "")
        return (f'<div class="fi-stat"><div class="fi-lbl">{_lbl}</div>'
                f'<div class="fi-num">{_h}<span>{_t}</span><em>{unit}</em></div>'
                f'<div class="fi-sub">{_sub}</div></div>')

    _stat_html = (_tile("2y", _c2, "benchmark; N/A outside anchors")
                  + _tile("10y", _c10, "benchmark; N/A outside anchors")
                  + _tile("30y", _c30, "benchmark; N/A outside anchors")
                  + _tile("10y TEY scenario", _tey10, f"full exemption assumed at {_brk}"
                          if assume_exempt.value else "enable exemption assumption above"))
    _slope = (_c30 - _c2) * 100
    _slope_text = f"{_slope:+.0f} bp" if np.isfinite(_slope) else "N/A (outside anchors)"
    _tey_text = (f"Illustrative 10y TEY: {_tey10:.3f}% at {_brk}, assuming full federal exemption."
                 if np.isfinite(_tey10) else "10y TEY scenario unavailable or disabled.")
    _n_signal = int((_drawn["resid_bp"].abs() >= _sig).sum())

    def _rows(_df):
        _out = []
        for _, _r in _df.iterrows():
            _v = float(_r["resid_bp"])
            _out.append(
                f'<tr><td class="fi-cusip">{_esc(str(_r["Name"])[:24])}</td>'
                f'<td class="fi-r fi-dim">{_r["years"]:.1f}y</td>'
                f'<td class="fi-r">{_r["ytm"]:.2f}</td>'
                f'<td class="fi-r" style="color:{_GOLD}">{format(_r["tey"], ".2f") if np.isfinite(_r["tey"]) else "N/A"}</td>'
                f'<td class="fi-r" style="color:{_tone(_v)}">{_v:+.0f}</td></tr>')
        return ('<div class="fi-scroll"><table class="fi-tbl"><thead><tr>'
                '<th>Bond</th><th class="fi-r">Term</th>'
                '<th class="fi-r">YTM est.</th><th class="fi-r">TEY scenario</th>'
                '<th class="fi-r">YTM gap bp</th>'
                '</tr></thead><tbody>' + "".join(_out) + '</tbody></table></div>')

    _cheap = _drawn[_drawn["resid_bp"] > 0].nlargest(12, "resid_bp")
    _rich = _drawn[_drawn["resid_bp"] < 0].nsmallest(12, "resid_bp")

    # The masthead, and then the caveats immediately beneath it — before any
    # chart, so a reader meets the limits of the data before the drawing that
    # makes it look authoritative.
    _head = mo.Html(FI_CSS + FI_EXTRA + f"""
    <div class="fi">
      <div class="fi-eyebrow">US municipal bonds · holdings-derived marks ·
        valuation date {as_of:%d %b %Y}</div>
      <h2 class="fi-title">The municipal desk</h2>
      <div class="fi-deck">{len(fitted):,} bonds, with {_n_ranked:,} finite
        YTM comparisons inside the supported anchor range. The curve summarizes
        mixed credits and states. Colour marks gaps exceeding the cross-sectional
        dispersion of {_sig:.0f}bp; it does not measure statistical significance
        or establish a trading opportunity.</div>
    </div>""")

    _caveats = mo.md(
        "**Descriptive research benchmark, not fair value or an executable quote.** "
        "Market value / par is treated as a clean price; the feed does not "
        "confirm that basis. Accrued interest included in the mark would bias YTM. "
        "Settlement is the holdings valuation date, not a current trade settlement. "
        "YTM and modified duration assume regular semiannual fixed coupons, bullet "
        "redemption at par and payment of every contractual cash flow. Original "
        "schedules, calls, puts, sinking funds and default status are unavailable: "
        "**YTM is not yield-to-worst and modified duration is not effective duration.** "
        "Ratings, issuer state, tax status and liquidity are not controlled. A "
        "positive YTM gap can compensate for credit or call risk, not mispricing. "
        "The sample contains holdings of the selected funds, not the whole market. "
        "**TEY is disabled by default.** Enabling it assumes the entire YTM is "
        "federally exempt. Market discount may be taxable; AMT, OID, premium "
        "amortization and state/local taxes require bond-specific information. "
        "The scenario does not establish an after-tax return or a buy threshold.")

    # Everything the caveats have earned the right to show: the fan, the four
    # numbers, the verdict, the surface, and the two ends of the ranking.
    _body = mo.Html(f"""
    <div class="fi">
      <div class="fi-rail">{_fan}
        <div class="fi-cap">{_n_drawn:,} bonds hung off benchmark ·
          1sd = {_sig:.0f}bp · stems clipped at ±{_cap:.0f}bp"""
        + (f" · {_unplaced:,} ranked but unpriced by the curve, so not drawn"
           if _unplaced else "")
        + f"""</div></div>
      <div class="fi-stats">{_stat_html}</div>
      <div class="fi-slope">{_tey_text}
        Two-to-thirty-year reference-coupon YTM difference: {_slope_text}.</div>
      <div class="fi-panel">{_surface}
        <div class="fi-cap">
          <span class="fi-k fi-k-line" style="border-color:{_GOLD}"></span>
            Reference curve ({reference_coupon:.2f}% coupon)
          <span class="fi-k fi-k-dot" style="background:{_GOLD};
            border-radius:0"></span>Calibration target (bucket-specific coupon)
          <span class="fi-k fi-k-dot"></span>Above benchmark
          <span class="fi-k fi-k-dot" style="background:{_SIENNA}"></span>Below benchmark
          <span class="fi-k fi-k-dot" style="background:{_SLATE}"></span>
            Inside 1 sd</div>
        <div class="fi-note">Each bond's YTM gap uses its own coupon on the
          fitted discount curve. Its vertical distance from the reference-coupon
          line can differ from the gap shown in the table.</div></div>
      <div class="fi-grid">
        <div><div class="fi-h">Largest positive YTM gaps</div>
          <div class="fi-note">Yield above the blended benchmark.
            Credit quality, calls, taxation and liquidity can explain the gap.
            TEY, if enabled, assumes full federal exemption at {_brk}.</div>{_rows(_cheap)}</div>
        <div><div class="fi-h">Largest negative YTM gaps</div>
          <div class="fi-note">Yield below the blended benchmark.
            {_n_signal:,} of {_n_drawn:,} compared bonds have absolute gaps
            greater than or equal to the {_sig:.0f}bp dispersion.</div>
          {_rows(_rich)}</div>
      </div>
    </div>""")

    _notes = [_head, _caveats, _body]
    if failed:
        _notes.append(mo.md(
            f"⚠️ {len(failed)} of 3 holdings files failed to load "
            f"({', '.join(failed)}) — the universe below is incomplete"))
    if not dropped.empty:
        # Counts per reason only. The rows were once named individually here,
        # but the muni files drop ~270 rows — mostly SSGA's own disclaimer and
        # footer lines, which carry no identifier — and naming them buried the
        # page in text that told the reader nothing. The count still holds the
        # line against a silent drop: a number that moves is visible.
        _counts = dropped["reason"].value_counts()
        _notes.append(mo.md(
            f"**Dropped {len(dropped):,} rows before pricing** — "
            + ", ".join(f"{v:,} {k}" for k, v in _counts.items()) + "."))
    # A dropped calibration bucket changes the SHAPE of the curve every
    # residual on this page is measured against, so it is said out loud in the
    # same sentence style as every other exclusion. A curve that quietly lost
    # its front anchor and interpolated across the gap is exactly the kind of
    # thing this desk must never leave for the reader to guess at.
    _bucket_note = ""
    if dropped_buckets:
        _which = "; ".join(
            f"{_d['tenor']:g}y ({_d['reason']}"
            + (f", {_d['n']:,} bonds medianing {_d['target']:.2f}%"
               if _d["n"] else ", no bonds")
            + ")"
            for _d in dropped_buckets)
        _bucket_note = (
            f" **The curve is anchored at {n_anchors} of "
            f"{n_anchors + len(dropped_buckets)} tenors**, having dropped "
            f"{_which} — a bucket too thin, or medianing outside the sanity "
            "band, is not a point on a credit curve and is refused rather "
            "than calibrated to. Benchmark at and around those tenors is "
            "interpolated only within the remaining anchor range. No benchmark "
            "yield or residual is shown outside that range.")
    _notes.append(mo.md(
        f"**Held out of the curve fit:** {_n_var:,} variable-rate, "
        f"{_n_dist:,} distressed, {_n_noyield:,} unpriced. One reason each — "
        f"these are cut from a single mutually-exclusive label — so with the "
        f"{_n_ranked + _n_window:,} fit-eligible bonds they account for all "
        f"{len(fitted):,} rows above, and every one of them is named in a tab "
        f"below. Of the fit-eligible bonds, **{_n_window:,} sit outside the "
        f"[{MIN_FIT_YEARS:.2f}, {MAX_FIT_YEARS:.2f}]-year supported range "
        f"calibrated on**, so they carry no residual and are not ranked; "
        f"**{_n_ranked:,}** have finite comparisons; "
        f"**{int(fitted['is_fit_source'].sum()):,}** supplied surviving bucket "
        f"statistics for **{n_anchors}** synthetic calibration instruments.{_bucket_note}"))
    mo.vstack(_notes)
    return FI_CSS, FI_EXTRA


@app.cell
def _(MAX_FIT_YEARS, MIN_FIT_YEARS, mo, taxed):
    """Seven tabs, each carrying only the columns that mean something for it.

    Five of them — On the curve, Outside the window, Variable rate, Distressed,
    Unpriced — partition the universe exactly: every bond appears in exactly
    one, and the check below refuses to render if that stops being true. It is
    a partition by construction, not by luck: all five are cut from `bucket`,
    one label per bond. The old four-tab version claimed the same thing while
    slicing on independent booleans, so a distressed FRN would have appeared in
    two tabs, and eight bonds inside the headline count — CAS 3.875 08/01/26,
    NHPFAC 11/15/61, MDS 3.000 08/01/26 among them — appeared in none and were
    named nowhere on this page.

    Cheapest and Richest are the extremes of On the curve and repeat its rows
    deliberately.

    Unpriced shows its ytm and tey precisely because they are the evidence: a
    yield of -521% is how you can see the row is a bad mark rather than a bond,
    and hiding it would just re-create the invisible bucket."""
    _COLS = ["Identifier", "Name", "Coupon", "maturity", "years",
             "price", "ytm", "tey", "mod_duration", "resid_bp", "source"]

    def _tab(df, sort, ascending=True):
        return mo.ui.table(
            df[[c for c in _COLS if c in df.columns]]
               .sort_values(sort, ascending=ascending, ignore_index=True)
              .rename(columns={"tey": "TEY scenario", "ytm": "YTM estimate",
                               "mod_duration": "Modified duration (no options)",
                               "resid_bp": "YTM gap bp"}),
            page_size=15)

    # label, sort column, ascending
    _PARTITION = {
        "ranked": ("Compared with benchmark", "years", True),
        "outside_window": (f"Outside the [{MIN_FIT_YEARS:.2f}, {MAX_FIT_YEARS:.2f}]y "
                           "fit window", "years", True),
        "variable_rate": ("Variable rate", "years", True),
        "distressed": ("Distressed", "ytm", False),
        "no_yield": ("Unpriced", "ytm", True),
    }
    _by = {_k: taxed[taxed["bucket"] == _k] for _k in _PARTITION}
    _covered = sum(len(_v) for _v in _by.values())
    if _covered != len(taxed):
        raise ValueError(
            f"the tabs name {_covered:,} of {len(taxed):,} bonds — "
            f"buckets present: {sorted(set(taxed['bucket']))}. A bond inside "
            "the headline count that appears in no tab is named nowhere on "
            "this page, which is exactly what this desk must never do")

    _ranked = _by["ranked"]
    _cheap = _ranked[_ranked["resid_bp"] > 0].nlargest(25, "resid_bp")
    _rich = _ranked[_ranked["resid_bp"] < 0].nsmallest(25, "resid_bp")

    mo.ui.tabs({
        f"Positive YTM gaps ({len(_cheap)})": _tab(_cheap, "resid_bp", False),
        f"Negative YTM gaps ({len(_rich)})": _tab(_rich, "resid_bp", True),
        **{f"{_lbl} ({len(_by[_k]):,})": _tab(_by[_k], _sort, _asc)
           for _k, (_lbl, _sort, _asc) in _PARTITION.items()},
    })
    return


if __name__ == "__main__":
    app.run()
