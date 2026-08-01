import marimo

__generated_with = "0.23.14"
app = marimo.App(
    width="medium",
    css_file="theme.css",
    html_head_file="theme_head.html",
)


@app.cell
def _():
    import pathlib

    import numpy as np
    import rateslib as rl
    import requests
    from requests.adapters import HTTPAdapter, Retry

    import marimo as mo

    import bondfile

    # rateslib is back, and earning it this time: the analytics cell builds an
    # rl.Curve, calibrates it with rl.Solver, and reads every fair yield out of
    # FixedRateBond.rate. It was dropped once because that cell fitted its
    # curve with np.polyfit instead, so rateslib imported — and printed its
    # multi-line licence notice on every desk open — for nothing. numpy stays,
    # but only for array plumbing and NaN handling; no financial calculation on
    # this desk is hand-rolled any more. Still no pandas: every frame this
    # notebook touches comes out of bondfile already built.
    return HTTPAdapter, Retry, bondfile, mo, np, pathlib, requests, rl


@app.cell
def _(HTTPAdapter, Retry, pathlib, requests):
    """
    SPDR publishes each fund's complete holdings daily, and every row carries
    par value and market value — so price falls out as market/par*100. That is
    the free per-CUSIP corporate bond price source this desk runs on.

    The fund is never analysed here and never bought. It is the delivery
    mechanism, exactly as FedInvest is for Treasuries.

    Three files, because they are short, intermediate and broad slices of the
    same market: together they cover ~6,900 unique investment grade bonds.
    """
    _BASE = ("https://www.ssga.com/us/en/intermediary/library-content/products"
             "/fund-data/etfs/us/holdings-daily-us-en")

    TICKERS = ("spbo", "spib", "spsb")
    _CACHE = pathlib.Path(".ssga-cache")

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
        """Raw bytes per ticker, cached to disk by day.

        Returns (files, failed). A file that fails leaves its ticker in
        `failed` rather than raising: two of three files still make a usable
        desk, and the render cell says which one is missing.
        """
        _CACHE.mkdir(exist_ok=True)
        files, failed = {}, []
        import datetime as _dt
        stamp = _dt.date.today().isoformat()
        for t in tickers:
            cached = _CACHE / f"{t}-{stamp}.xlsx"
            if cached.exists():
                files[t] = cached.read_bytes()
                continue
            try:
                r = session.get(f"{_BASE}-{t}.xlsx", timeout=(10, 60))
                r.raise_for_status()
                if not r.content[:2] == b"PK":
                    # An OOXML file starts with PK. Anything else is the HTML
                    # shell an issuer serves when it decides you are a bot.
                    raise ValueError("not an OOXML file — served HTML?")
                tmp = cached.with_suffix(".tmp")
                tmp.write_bytes(r.content)
                tmp.replace(cached)
                files[t] = r.content
            except Exception:
                failed.append(t)
        return files, failed

    return TICKERS, fetch, session


@app.cell
def _(TICKERS, bondfile, fetch):
    # One rateslib spec for the sector, named once and shared. bondfile.add_yield
    # prices every bond's OBSERVED yield under it, and the analytics cell prices
    # every bond's FAIR yield under it. Two literals could drift, and a residual
    # computed across two different day-count/frequency conventions would be a
    # convention gap wearing richness as a costume.
    SPEC = "us_corp"
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
    """
    One credit curve across the fit-eligible universe, then every bond's
    richness measured against it.

    The curve is rateslib's own: a discount `rl.Curve` calibrated by
    `rl.Solver`, with every fair yield read back out of it through
    `FixedRateBond.rate(metric="ytm")`. No financial calculation in this cell
    is hand-rolled. It used to be: a cubic through log-tenor via `np.polyfit`,
    read back with `np.polyval`, and a tenor axis measured in `days / 365.25`
    — a divisor that is not a market convention and matched nothing else on
    the desk.

    The replacement is not merely tidier, it is more correct. A fair yield off
    a discount curve depends on the bond's own COUPON schedule, which a
    tenor-only polynomial cannot express at all: two bonds maturing the same
    day with 3% and 7% coupons have genuinely different yields, and the cubic
    handed them the same fair value and called the difference richness.

    Nodes are far sparser than the bond count, deliberately: a curve threaded
    through every quote fits each bond exactly and reports zero richness. This
    smooths, so a bond sitting off the fitted curve is saying something.

    Only `in_fit` bonds calibrate it. Variable-rate notes are excluded because
    their coupon resets and a fixed-coupon yield misprices them; distressed
    bonds because they trade on recovery, not on spread.

    MIN_FIT_YEARS/MAX_FIT_YEARS are defined once and used for BOTH the
    calibration filter and the ranking gate below, so the two can never drift
    apart. curve_ytm is still computed for every bond outside this band, now
    by genuine extrapolation off the calibrated curve rather than by clipping
    the tenor — a fair-yield estimate is still useful context to show next to
    a 40-year bond. But resid_bp, the ranking signal,
    is NaN'd for any bond outside the band, exactly as it already is for
    in_fit == False: the curve was not calibrated on that tenor, so a bond
    sitting there is not "off the curve", it is off the curve's domain, and
    must not be ranked as if it were a genuine dislocation.

    MIN_FIT_YEARS is bondfile.MIN_YEARS_TO_MATURITY, and that is the whole
    point of it being one constant. This cell used to claim it "serves as the
    near-maturity guard treasury.py's bill cell applies explicitly". It did
    not: it gated calibration and ranking only, never add_yield, so a bond
    inside 91 days still got a yield and still rendered it. MONHGR 5.000
    10/01/26, 63 days out and marked 101.70, solved to -4.85% and showed that
    number in the On-the-curve tab beside genuine yields. The guard now runs
    where it has to, in bondfile.clean_rows, before any yield is computed —
    and because the two use the same constant, the claim is now true rather
    than merely written down. The window filter below is consequently
    belt-and-braces at the near end and load-bearing only at the far end.

    MIN_SANE_YIELD/MAX_SANE_YIELD (0.5%-15%) are a data-corruption trip wire,
    not a market-regime assertion. They exist to catch a units error or a
    parse bug (e.g. a yield reported as 470% or 0.047%) before it silently
    feeds the rich/cheap table, not to assert what "normal" rates look like.
    A genuine cutting cycle putting front-end IG under 3%, or munis (whose
    coupon is tax-exempt and so structurally price through corporates)
    sitting well below that, are legitimate market states and must render,
    not raise. Keep this band wide for that reason, and do not tighten it
    later on the assumption it encodes a rate view - it doesn't. The slope
    check is the real fit-quality signal; this is only a last-ditch sanity
    floor/ceiling against corrupted data.
    """
    MIN_FIT_YEARS, MAX_FIT_YEARS = bondfile.MIN_YEARS_TO_MATURITY, 31
    MIN_SANE_YIELD, MAX_SANE_YIELD = 0.5, 15.0

    # The tenor axis and the discount curve MUST share a day count, and this
    # is the one constant that makes them. `years` decides only WHERE a bond
    # sits on the curve; the Curve itself discounts under its own
    # `convention`. If the two disagree, a bond is placed at one point on the
    # tenor axis and priced at another, and the residual quietly picks up a
    # pure convention error that reads as richness. So CURVE_CONVENTION feeds
    # rl.dcf here and rl.Curve(convention=...) below, and nothing else sets
    # either. The old `(maturity - as_of).days / 365.25` agreed with nothing:
    # 365.25 is a calendar average, not a market convention.
    CURVE_CONVENTION = "act365f"

    _settle = as_of.to_pydatetime()
    # The same anchor bondfile.add_yield uses, for the reason set out in its
    # docstring: `effective` is arbitrary for a held-to-maturity yield, but the
    # settlement-to-maturity leg has to actually EXIST in rateslib's generated
    # coupon schedule. Anchoring it near maturity truncates the schedule to the
    # final coupon periods and solves a materially wrong yield (6.02% against
    # an expected 5.15%, on this project's own fixtures). Five years back is
    # safely before every kept bond's maturity, and rl.add_tenor rather than a
    # raw timedelta because this is calendar arithmetic.
    _effective = rl.add_tenor(_settle, "-5y", "F", "nyc")

    def _years(maturities):
        """Tenor in years, on rateslib's day count rather than a hand divisor."""
        return [rl.dcf(_settle, _m.to_pydatetime(), CURVE_CONVENTION)
                for _m in maturities]

    def _tenor_date(years):
        """The date a tenor lands on. 365 is act365f's own denominator, so this
        inverts rl.dcf under CURVE_CONVENTION rather than introducing a second,
        subtly different day count on the way back."""
        return rl.add_tenor(_settle, f"{int(round(float(years) * 365))}d",
                            "F", "nyc")

    _fit = universe[universe["in_fit"] & universe["ytm"].notna()].copy()
    _fit["years"] = _years(_fit["maturity"])
    _fit = _fit[(_fit["years"] > MIN_FIT_YEARS) & (_fit["years"] < MAX_FIT_YEARS)]
    _fit = _fit.sort_values("years")

    # A degenerate fit — too few bonds to say anything about a curve, or a fit
    # that doesn't look like a credit curve — must fail loudly rather than
    # render.
    _MIN_FIT_BONDS = 30
    if len(_fit) < _MIN_FIT_BONDS:
        raise ValueError(
            f"only {len(_fit)} bonds in [{MIN_FIT_YEARS}, {MAX_FIT_YEARS}]y "
            f"window — need at least {_MIN_FIT_BONDS} to calibrate a stable "
            "credit curve; check upstream universe/in_fit filtering")

    # Calibrate to a handful of BUCKETED instruments, not to every bond.
    #
    # This is a capacity limit, not a preference. Measured on this universe:
    # a Solver given 25 bonds converges in 0.5s, 100 in 1.6s, and 400 does not
    # converge at all — `max_iter: 50 exceeded, f0: nan`. There are thousands
    # of fit-eligible bonds here, so bond-per-instrument is not merely slow,
    # it is unreachable. Ten bucket medians converge in ~0.1s.
    #
    # Bucketing is also what keeps the curve SMOOTH, which is the whole point:
    # a node per bond would fit every quote exactly and report zero richness.
    _TENORS = (0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30)
    # ±25% of the tenor, and a bucket has to clear TWO gates to become a node.
    #
    # Size, because a node calibrated to two bonds is a node calibrated to two
    # bonds' idiosyncrasies. And the target LEVEL, because size alone does not
    # catch the failure this desk actually hit: the muni 0.5y bucket held ten
    # bonds — comfortably over the size floor — and medianed -0.625%, and the
    # Solver faithfully calibrated the front of the curve to it. Every short
    # bond's resid_bp was then measured against a negative fair value, and
    # nothing said so, because the runtime band assertion below only sampled
    # 2y/5y/10y/30y and never looked at the front.
    #
    # A bucket median outside [MIN_SANE_YIELD, MAX_SANE_YIELD] is corrupt input
    # for calibration purposes, on exactly the reading of that band used
    # everywhere else here: a trip wire for bad data, not a view on rates. Drop
    # the bucket and carry on with the rest, as a too-thin bucket already is —
    # one bad tenor must not cost the desk its whole curve. Both kinds of drop
    # are counted and named on the page; see `dropped_buckets`.
    _MIN_BUCKET_BONDS = 5
    _buckets, dropped_buckets = {}, []
    for _t in _TENORS:
        _b = _fit[(_fit["years"] >= _t * 0.75) & (_fit["years"] <= _t * 1.25)]
        if len(_b) < _MIN_BUCKET_BONDS:
            dropped_buckets.append(
                {"tenor": _t, "reason": "too few bonds", "n": len(_b),
                 "target": float(_b["ytm"].median()) if len(_b) else float("nan")})
            continue
        _target = float(_b["ytm"].median())
        if not MIN_SANE_YIELD <= _target <= MAX_SANE_YIELD:
            dropped_buckets.append(
                {"tenor": _t, "reason": "median yield outside the sane band",
                 "n": len(_b), "target": _target})
            continue
        _buckets[_t] = (_b, _target)

    # Losing the odd tenor is survivable; losing most of them is not a curve.
    _MIN_ANCHORS = 3
    if len(_buckets) < _MIN_ANCHORS:
        raise ValueError(
            f"only {len(_buckets)} of {len(_TENORS)} tenor buckets survived "
            f"({dropped_buckets}) — need at least {_MIN_ANCHORS} anchors to "
            "calibrate a curve worth reading")

    # A discount curve, log-linear in DF, exactly as the Treasury desk's bill
    # and coupon cells build theirs. LineCurve is not an option here: pricing a
    # FixedRateBond off one raises `disc_curve cannot be inferred from a
    # non-DF based curve`.
    _cv = rl.Curve(
        nodes={_settle: 1.0, **{_tenor_date(_t): 1.0 for _t in _buckets}},
        interpolation="log_linear", convention=CURVE_CONVENTION,
        calendar="nyc", id="credit")
    _pricing = {"curves": _cv, "metric": "ytm", "settlement": _settle}

    _insts, _targets, _labels = [], [], []
    for _t, (_b, _target) in _buckets.items():
        # One representative bond per bucket: the bond nearest the bucket's
        # median tenor — its real maturity date, so no date is reconstructed
        # from a year count — carrying the bucket's median coupon, targeted at
        # the bucket's median yield (already validated above). Medians, not
        # means: a bucket of thousands of credits has a long tail and a mean
        # chases it.
        _rep = _b.iloc[int((_b["years"] - _b["years"].median()).abs().argmin())]
        _insts.append(rl.FixedRateBond(
            effective=_effective,
            termination=_rep["maturity"].to_pydatetime(),
            fixed_rate=float(_b["Coupon"].median()), spec=SPEC,
            curves=_cv.id))
        _targets.append(_target)
        _labels.append(f"{_t:g}y")
    n_anchors = len(_insts)
    # The tenors the curve is actually pinned at, exported because the credit
    # surface marks them on the fitted line: a reader looking at a residual
    # ought to be able to see whether it is measured against an anchor or
    # against interpolation between two of them.
    anchor_tenors = tuple(_buckets)

    # Each instrument entry is a 2-tuple (Instrument, pricing kwargs). A
    # 3-tuple raises.
    _solver = rl.Solver(
        curves=[_cv], instruments=[(_i, _pricing) for _i in _insts],
        s=_targets, instrument_labels=_labels, id="creditfit")

    # Reading the curve at a bare tenor needs a stated reference coupon,
    # because a fair yield off a discount curve genuinely depends on the
    # coupon — that dependence is the reason for using a discount curve at
    # all, and the thing the old cubic could not see. The fit's median coupon
    # is the honest choice, and it is used only for the sanity checkpoints
    # below; per-bond fair value uses each bond's own coupon.
    _REF_COUPON = float(_fit["Coupon"].median())

    def curve(years, coupon=_REF_COUPON):
        """Fair yield at a tenor, in percent, off the calibrated curve.

        Clipped to the calibration window so a caller asking for 40y gets the
        curve's own far end rather than an unbounded extrapolation."""
        return np.array([
            float(rl.FixedRateBond(
                effective=_effective, termination=_tenor_date(_y),
                fixed_rate=coupon, spec=SPEC, curves=_cv.id).rate(**_pricing))
            for _y in np.atleast_1d(np.clip(np.asarray(years, dtype=float),
                                            MIN_FIT_YEARS, MAX_FIT_YEARS))])

    # Sanity-check the fit itself before anything downstream trusts it. A
    # corrupt curve rendering silently is worse than the notebook failing.
    #
    # Two different checks, deliberately of different strictness AND of
    # deliberately different reach:
    #   - slope: an order invariant (2y < 10y < 30y). A real fitting bug
    #     violates this; a legitimate market move essentially never does.
    #     This is the strong signal and stays exactly as tight as it sounds,
    #     sampled at three points because a SHAPE invariant is what three
    #     well-separated points are for.
    #   - level: MIN_SANE_YIELD/MAX_SANE_YIELD is only a data-corruption trip
    #     wire (catches a units error, e.g. 470% or 0.047%, or a garbage
    #     fetch) — not a market-regime assertion. It must survive any
    #     realistic curve, corporate or municipal, across rate regimes, so it
    #     is kept deliberately wide rather than tuned to "normal" levels.
    #
    # The level check runs across the WHOLE tenor grid, not the four sampled
    # checkpoints it used to. Sampling 2y/5y/10y/30y left the front end
    # unguarded, which is precisely where this desk's real failure happened: a
    # -0.625% front bucket calibrated a negative front end, every short bond's
    # resid_bp was measured against it, and the page rendered without a word.
    # A band check that cannot see the region most likely to break is not a
    # band check. Every tenor in the grid is evaluated — including the ones
    # whose bucket was dropped, because the curve still interpolates or
    # extrapolates a fair value there and bonds are still ranked against it.
    _checkpoints = {t: float(curve(t)[0]) for t in (2, 5, 10, 30)}
    if not (_checkpoints[2] < _checkpoints[10] < _checkpoints[30]):
        raise ValueError(
            f"fitted curve is not upward-sloping: {_checkpoints} — "
            "something upstream (bad yields, wrong window) is corrupting the fit")
    _grid_ytm = {_t: float(curve(_t)[0]) for _t in _TENORS}
    _insane = {_t: round(_v, 4) for _t, _v in _grid_ytm.items()
               if not MIN_SANE_YIELD <= _v <= MAX_SANE_YIELD}
    if _insane:
        raise ValueError(
            f"calibrated curve outside the sane [{MIN_SANE_YIELD}, "
            f"{MAX_SANE_YIELD}]% band at {_insane} (whole grid: "
            f"{ {_t: round(_v, 4) for _t, _v in _grid_ytm.items()} }) — this is "
            "a corruption check, not a rate-regime assertion, so check upstream "
            "yield data/units before trusting this curve")

    fitted = universe.copy()
    fitted["years"] = _years(fitted["maturity"])

    def _fair_ytm(row):
        """One bond's fair yield off the calibrated curve, priced by rateslib
        on that bond's OWN coupon schedule.

        ~3ms a bond, so about 20s across this universe, and deliberately not
        interpolated from a handful of tenor points: interpolating would put
        the hand-rolled numpy straight back into the central number, and would
        throw away exactly the coupon dependence that makes a discount curve
        worth calibrating.

        A bond rateslib cannot solve returns NaN rather than raising — one bad
        row must never take down a 6,000-row desk, the same contract
        bondfile.add_yield keeps."""
        try:
            return float(rl.FixedRateBond(
                effective=_effective,
                termination=row["maturity"].to_pydatetime(),
                fixed_rate=float(row["Coupon"]), spec=SPEC,
                curves=_cv.id).rate(**_pricing))
        except Exception:
            return float("nan")

    fitted["curve_ytm"] = [_fair_ytm(_r) for _, _r in fitted.iterrows()]
    # Richness in YIELD bp. A fixed price threshold would mean wildly different
    # dislocations at 2 years and at 30.
    fitted["resid_bp"] = (fitted["ytm"] - fitted["curve_ytm"]) * 100
    # Never rank a bond the curve was not allowed to see: neither a bond
    # excluded from calibration (in_fit == False) nor one outside the tenor
    # window the curve was actually fit on, even if it is in_fit.
    fitted["is_ranked"] = (fitted["in_fit"]
                           & (fitted["years"] > MIN_FIT_YEARS)
                           & (fitted["years"] < MAX_FIT_YEARS))
    fitted.loc[~fitted["is_ranked"], "resid_bp"] = np.nan

    # `is_ranked` is exactly the set the curve was calibrated from — assert
    # it, because the masthead and the tab labels are about to count off it
    # and a silent divergence would put a number on the page that no bond
    # backs.
    if int(fitted["is_ranked"].sum()) != len(_fit):
        raise ValueError(
            f"{int(fitted['is_ranked'].sum())} bonds marked ranked but the "
            f"curve was calibrated from {len(_fit)} — the ranking gate and "
            "the calibration filter have drifted apart")

    # One label per bond, and the single source everything downstream counts
    # and tabs off. bondfile.excluded_reason is a true partition of the
    # universe; this splits its in-fit half by whether the bond is inside the
    # tenor window the curve was actually fitted on. Counting some buckets off
    # booleans and others off strings is what produced a rendered tally that
    # did not reconcile with its own masthead.
    fitted["bucket"] = fitted["excluded_reason"].where(
        ~fitted["in_fit"],
        np.where(fitted["is_ranked"], "ranked", "outside_window"))

    # The dispersion of the ranking signal. It is the gate on colour — nothing
    # inside one standard deviation of fair earns green or sienna anywhere on
    # this page — so it is computed once here rather than re-derived by each
    # thing that needs it.
    resid_sd_bp = float(fitted.loc[fitted["is_ranked"], "resid_bp"].std())
    return (MAX_FIT_YEARS, MIN_FIT_YEARS, anchor_tenors, curve, dropped_buckets,
            fitted, n_anchors, resid_sd_bp)


@app.cell
def _(MAX_FIT_YEARS, MIN_FIT_YEARS, anchor_tenors, as_of, curve, dropped,
      dropped_buckets, failed, fitted, mo, n_anchors, np, resid_sd_bp):
    """The desk, rendered as a section of the same publication as treasury.py.

    Same devices — live dot, rule to the margin, serif italic clamp, double
    green rule, gold datum, colour only past one standard deviation — and the
    same stylesheet, copied verbatim rather than reinterpreted.

    Two signature drawings, both direct translations of the Treasury desk's:

      the spread fan     treasury.py hangs every bill below par by its
                         discount. Par is a real gold line there because a
                         bill genuinely is a promise of 100. Here the datum
                         is fair value off the fitted credit curve, and each
                         bond hangs off it by its own residual — cheap above,
                         rich below. Same device, different datum.
      the credit surface the working instrument, exactly as the bill curve is
                         there: every ranked bond as a dot at its tenor and
                         observed yield, with the calibrated curve drawn
                         through the cloud and its anchor tenors marked.

    Nothing unranked is drawn on either. resid_bp is NaN outside the ranked
    set by construction, and a bond the curve was never allowed to see has no
    place on a chart about distance from that curve."""
    # Every count on this page comes off `bucket`, which is one label per bond.
    # It used to come off a mix of independent booleans and reason strings:
    # `is_variable`, `is_distressed`, and `excluded_reason == "no_yield"`
    # overlapped, so the muni desk rendered "0 variable-rate, 25 distressed,
    # 26 no-yield" — 51 exclusions against 33 bonds actually excluded.
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
            "ranked", "outside_window", *fitted["excluded_reason"].unique()}:
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
    # browser animates twelve elements instead of six thousand. Nothing else
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

    # Green reads cheap/bid, sienna rich/offer, gold is fair value. Slate is
    # "inside the noise", and it is the default.
    _GREEN, _SIENNA, _GOLD = "#1D9E75", "#D8683C", "#C9A961"
    _SLATE, _RULE = "#93A49B", "#26332C"

    # Only dislocations beyond one standard deviation earn a colour. Painting
    # every ±1bp residual green or sienna dresses quote noise up as conviction,
    # and on a 6,000-bond universe it would paint the entire page.
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
    _drawn = (fitted[fitted["is_ranked"] & fitted["resid_bp"].notna()]
              .sort_values("years"))
    _n_drawn = len(_drawn)
    _unplaced = _n_ranked - _n_drawn

    # Square-root tenor axis, exactly as treasury.py's rail and curve use.
    # Linear-in-years buries the two- to five-year part of a credit curve —
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
    # A handful of 2,000bp outliers would otherwise squash 5,900 bonds into a
    # two-pixel band around the datum and turn the signature into a hairline.
    # Stems past the scale are drawn full length, and the caption says where
    # the scale stops so a clipped stem is not read as an exact length.
    _cap = max(float(np.nanpercentile(np.abs(_drawn["resid_bp"].to_numpy()), 98)),
               1.0)
    # Twelve tenor bands, and the load animation runs on the BANDS. Staggering
    # each bond individually animates six thousand elements and stutters the
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
        # already said — and six thousand of them turn the signature into
        # uniform speckle, which is the opposite of what it is for. Every bond
        # still draws its stem, so nothing is hidden; only the redundant
        # marker and its equally redundant tooltip go.
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
        f'aria-label="Every ranked corporate bond hanging off the fitted curve '
        f'by its residual — cheap above the line, rich below">'
        f'<line x1="{_PADL}" y1="{_DAT}" x2="{_W - _PADR}" y2="{_DAT}" '
        f'stroke="{_GOLD}" stroke-width="1"/>'
        f'<text x="0" y="{_DAT - 5}" fill="{_GOLD}" font-size="10">fair value</text>'
        f'<text x="0" y="{_FH - 6}" fill="{_SLATE}" font-size="10">'
        f'±{_cap:.0f}bp</text>'
        # Stem colour and weight are set once on the parent and inherited.
        f'<g stroke="{_GREEN}" stroke-opacity="0.22" stroke-width="1">'
        + "".join(_stems) + '</g></svg>')

    # ── The curve, as the working instrument ────────────────────────────
    # Every ranked bond at its tenor and its OBSERVED yield, with the
    # calibrated curve through the cloud and its anchor tenors marked, so the
    # residual the fan draws can be read back off the thing it came from.
    _CH, _CT, _CB = 340, 26, 44
    _ts = np.linspace(max(MIN_FIT_YEARS, float(_drawn["years"].min())),
                      min(MAX_FIT_YEARS, _maxyr), 32)
    _cvals = curve(_ts)
    _anchor_ys = curve(list(anchor_tenors))
    # The y-axis is set off the 0.5/99.5 percentiles of observed yield, not
    # the extremes: the ranked set legitimately reaches toward the distressed
    # threshold, and one 20% mark would flatten the whole cloud. The few dots
    # beyond the frame are clipped rather than moved or dropped.
    _p_lo, _p_hi = (float(_v) for _v
                    in np.nanpercentile(_drawn["ytm"].to_numpy(dtype=float),
                                        [0.5, 99.5]))
    _lo = min(_p_lo, float(_cvals.min())) - 0.15
    _hi = max(_p_hi, float(_cvals.max())) + 0.15
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
        f'corporate bond, with the calibrated credit curve through it">'
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
    _c2, _c10, _c30 = (float(_v) for _v in curve([2, 10, 30]))

    def _tile(_lbl, _val, _sub, unit="%", dp=3):
        _h, _t = _handle(_val, dp)
        return (f'<div class="fi-stat"><div class="fi-lbl">{_lbl}</div>'
                f'<div class="fi-num">{_h}<span>{_t}</span><em>{unit}</em></div>'
                f'<div class="fi-sub">{_sub}</div></div>')

    _stat_html = (_tile("2y", _c2, "fitted curve")
                  + _tile("10y", _c10, "fitted curve")
                  + _tile("30y", _c30, "fitted curve")
                  + _tile("Dispersion", _sig, "1 sd", unit="bp", dp=1))
    _slope = (_c30 - _c2) * 100
    _n_signal = int((_drawn["resid_bp"].abs() >= _sig).sum())

    def _rows(_df):
        _out = []
        for _, _r in _df.iterrows():
            _v = float(_r["resid_bp"])
            _out.append(
                f'<tr><td class="fi-cusip">{_esc(str(_r["Name"])[:26])}</td>'
                f'<td class="fi-r fi-dim">{_r["years"]:.1f}y</td>'
                f'<td class="fi-r">{_r["ytm"]:.2f}</td>'
                f'<td class="fi-r" style="color:{_tone(_v)}">{_v:+.0f}</td></tr>')
        return ('<div class="fi-scroll"><table class="fi-tbl"><thead><tr>'
                '<th>Bond</th><th class="fi-r">Term</th>'
                '<th class="fi-r">Yield</th><th class="fi-r">Edge bp</th>'
                '</tr></thead><tbody>' + "".join(_out) + '</tbody></table></div>')

    _cheap = _drawn.nlargest(12, "resid_bp")
    _rich = _drawn.nsmallest(12, "resid_bp")

    # The masthead, and then the caveats immediately beneath it — before any
    # chart, so a reader meets the limits of the data before the drawing that
    # makes it look authoritative.
    _head = mo.Html(FI_CSS + FI_EXTRA + f"""
    <div class="fi">
      <div class="fi-eyebrow">US corporate bonds · evaluated prices ·
        marks of {as_of:%d %b %Y}</div>
      <h2 class="fi-title">The corporate desk</h2>
      <div class="fi-deck">{len(fitted):,} bonds marked per 100 face by one
        pricing vendor and fitted to a single credit curve; {_n_ranked:,} of
        them sit inside the tenor window the curve was calibrated on and are
        ranked against it, {_n_var:,} are variable-rate and are not. Colour is
        reserved for the bonds more than {_sig:.0f}bp from fair — green cheap,
        sienna rich. Everything inside that is noise at this spread.</div>
    </div>""")

    # Every limitation the reader needs in order not to be misled. These are
    # rendered, not merely documented, because a desk that hides them lies.
    _caveats = mo.md(
        "**These are evaluated prices, not executed trades.** They are the "
        "fund's pricing-vendor marks — good marks, but not a last-trade tape. "
        "**A retail fill will be worse:** odd-lot spreads on corporate bonds "
        "are real and wide. Rich/cheap identifies candidates, it does not "
        "promise a price. **The universe is benchmark-eligible bonds**, so "
        "small and illiquid issues are absent. **All marks come from one "
        "vendor** and are not cross-checked. **The curve is fit across "
        "every credit quality at once** — the holdings file carries no "
        "rating field, so \"cheapest\" often means lower-rated rather "
        "than mispriced, and \"richest\" often means higher-rated. Read "
        "the ranking as a starting point for research, not as a verdict "
        "on value.")

    # Everything the caveats have earned the right to show: the fan, the four
    # numbers, the verdict, the surface, and the two ends of the ranking.
    _body = mo.Html(f"""
    <div class="fi">
      <div class="fi-rail">{_fan}
        <div class="fi-cap">{_n_drawn:,} bonds hung off fair value ·
          1sd = {_sig:.0f}bp · stems clipped at ±{_cap:.0f}bp"""
        + (f" · {_unplaced:,} ranked but unpriced by the curve, so not drawn"
           if _unplaced else "")
        + f"""</div></div>
      <div class="fi-stats">{_stat_html}</div>
      <div class="fi-slope">Two years to thirty
        <b style="color:{_tone(_slope)}">{_slope:+.0f} bp</b> ·
        the credit curve pays that much for twenty-eight more years of
        duration, and <b>{_n_signal:,}</b> of {_n_drawn:,} ranked bonds sit
        further than {_sig:.0f}bp from it — the rest is inside the noise.</div>
      <div class="fi-panel">{_surface}
        <div class="fi-cap">
          <span class="fi-k fi-k-line" style="border-color:{_GOLD}"></span>
            Calibrated curve
          <span class="fi-k fi-k-dot" style="background:{_GOLD};
            border-radius:0"></span>Anchor tenor
          <span class="fi-k fi-k-dot"></span>Cheap
          <span class="fi-k fi-k-dot" style="background:{_SIENNA}"></span>Rich
          <span class="fi-k fi-k-dot" style="background:{_SLATE}"></span>
            Inside 1 sd</div></div>
      <div class="fi-grid">
        <div><div class="fi-h">Cheapest to the curve</div>
          <div class="fi-note">Yield above fair. The holdings file carries no
            rating, so read the top of this list as <b>lower-rated</b> until
            research says otherwise.</div>{_rows(_cheap)}</div>
        <div><div class="fi-h">Richest to the curve</div>
          <div class="fi-note">Yield below fair — and for the same reason,
            usually <b>higher-rated</b> rather than expensive.</div>
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
        # but most of the ~280 dropped are SSGA's own disclaimer and footer
        # lines, which carry no identifier, and naming them buried the page in
        # text that told the reader nothing. The count still holds the line
        # against a silent drop: a number that moves is visible. The cost is
        # that a malformed-identifier row which is genuinely a bond — SPSB's
        # EQUINIX rows are the known case — is no longer visible here.
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
            "than calibrated to. Fair value at and around those tenors is "
            "interpolated or extrapolated from the anchors that remain, so "
            "read residuals there with correspondingly less weight.")
    _notes.append(mo.md(
        f"**Held out of the curve fit:** {_n_var:,} variable-rate, "
        f"{_n_dist:,} distressed, {_n_noyield:,} unpriced. One reason each — "
        f"these are cut from a single mutually-exclusive label — so with the "
        f"{_n_ranked + _n_window:,} fit-eligible bonds they account for all "
        f"{len(fitted):,} rows above, and every one of them is named in a tab "
        f"below. Of the fit-eligible bonds, **{_n_window:,} sit outside the "
        f"[{MIN_FIT_YEARS}, {MAX_FIT_YEARS}]-year window the curve was "
        f"calibrated on**, so they carry no residual and are not ranked; "
        f"**{_n_ranked:,}** actually calibrated the curve.{_bucket_note}"))
    mo.vstack(_notes)
    return FI_CSS, FI_EXTRA


@app.cell
def _(MAX_FIT_YEARS, MIN_FIT_YEARS, fitted, mo):
    """Seven tabs, each carrying only the columns that mean something for it.

    Five of them — On the curve, Outside the window, Variable rate, Distressed,
    Unpriced — partition the universe exactly: every bond appears in exactly
    one, and the check below refuses to render if that stops being true. It is
    a partition by construction, not by luck: all five are cut from `bucket`,
    one label per bond. The old four-tab version claimed the same thing while
    slicing on independent booleans, so a distressed FRN would have appeared in
    two tabs, and eight muni bonds inside the headline count appeared in none.

    Cheapest and Richest are the extremes of On the curve and repeat its rows
    deliberately."""
    _COLS = ["Identifier", "Name", "Coupon", "maturity", "years",
             "price", "ytm", "mod_duration", "resid_bp", "source"]

    def _tab(df, sort, ascending=True):
        return mo.ui.table(
            df[[c for c in _COLS if c in df.columns]]
              .sort_values(sort, ascending=ascending, ignore_index=True),
            page_size=15)

    # label, sort column, ascending
    _PARTITION = {
        "ranked": ("On the curve", "years", True),
        "outside_window": (f"Outside the [{MIN_FIT_YEARS}, {MAX_FIT_YEARS}]y "
                           "fit window", "years", True),
        "variable_rate": ("Variable rate", "years", True),
        "distressed": ("Distressed", "ytm", False),
        "no_yield": ("Unpriced", "ytm", True),
    }
    _by = {_k: fitted[fitted["bucket"] == _k] for _k in _PARTITION}
    _covered = sum(len(_v) for _v in _by.values())
    if _covered != len(fitted):
        raise ValueError(
            f"the tabs name {_covered:,} of {len(fitted):,} bonds — "
            f"buckets present: {sorted(set(fitted['bucket']))}. A bond inside "
            "the headline count that appears in no tab is named nowhere on "
            "this page, which is exactly what this desk must never do")

    _ranked = _by["ranked"]
    _cheap = _ranked.nlargest(25, "resid_bp")
    _rich = _ranked.nsmallest(25, "resid_bp")

    mo.ui.tabs({
        f"Cheapest ({len(_cheap)})": _tab(_cheap, "resid_bp", False),
        f"Richest ({len(_rich)})": _tab(_rich, "resid_bp", True),
        **{f"{_lbl} ({len(_by[_k]):,})": _tab(_by[_k], _sort, _asc)
           for _k, (_lbl, _sort, _asc) in _PARTITION.items()},
    })
    return


if __name__ == "__main__":
    app.run()
