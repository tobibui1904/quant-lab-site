"""Pure parsing and pricing for SPDR bond-fund holdings files.

These files are the free per-CUSIP price source for corporate and municipal
bonds: each row carries par value and market value, so price falls out as
market_value / par_value * 100. The fund itself is never analysed — it is the
delivery mechanism, the way FedInvest is for Treasuries.

Everything here is pure. No HTTP, no caching, no rendering: those live in the
desk notebooks. This module exists separately because marimo cells are
anonymous `def _()` and cannot be imported, and these functions must be tested.
"""
import io
import re

import pandas as pd
import rateslib as rl

# Row 4 holds the header; rows 0-3 are a preamble carrying the as-of date.
HEADER_ROW = 4

REQUIRED = ("Name", "Identifier", "Coupon", "Par Value",
            "Market Value", "Maturity")

# ISIN: two country letters, nine alphanumerics, one check digit. Cash and money
# market rows fail this — they carry "-" or a 9-character pseudo-CUSIP such as
# "999USDZ92" — which is exactly why the identifier filter must run first.
ISIN = re.compile(r"[A-Z]{2}[0-9A-Z]{9}[0-9]")


def _assert_schema(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"holdings layout changed; missing {missing}, saw {list(df.columns)}")
    return df


def read_holdings(source):
    """Return (as_of, raw rows). Performs no filtering.

    `source` is a path or raw bytes. The as-of date comes from the preamble,
    never from today's date: the file publishes with a lag, and dating a curve
    wrongly shifts every yield on it.
    """
    buf = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    preamble = pd.read_excel(buf, header=None, nrows=HEADER_ROW)
    if isinstance(buf, io.BytesIO):
        buf.seek(0)
    body = pd.read_excel(buf, skiprows=HEADER_ROW)
    _assert_schema(body)

    as_of = None
    for cell in preamble.astype(str).to_numpy().ravel():
        m = re.search(r"As of (\d{1,2}-[A-Za-z]{3}-\d{4})", cell)
        if m:
            as_of = pd.to_datetime(m.group(1), format="%d-%b-%Y")
            break
    if as_of is None:
        raise ValueError("no 'As of' date found in the holdings preamble")
    return as_of, body


read_holdings.assert_schema = _assert_schema


# A bond maturing inside this window is a money-market instrument, not a point
# on a credit curve: a hairline gap between the vendor's mark and redemption
# annualizes into an absurd yield. `treasury.py` applies exactly this guard to
# bills, dropping anything inside seven days of settlement. 0.25y (~91 days) is
# the corporate/muni analogue, and is deliberately the same number as the desks'
# MIN_FIT_YEARS: the desks' docstrings claim the near-edge of the fit window
# serves as the near-maturity guard, and that claim is only true if the guard
# actually runs here, before a yield is ever computed or displayed. Live muni
# data makes the point: MONHGR 5.000 10/01/26, 63 days out and marked 101.70,
# solved to ytm -4.85% — inside YTM_FLOOR, so it rendered in the On-the-curve
# tab in the same column as every genuine yield.
MIN_YEARS_TO_MATURITY = 0.25


def clean_rows(df, as_of):
    """Split rows into (kept, dropped). Never drops anything silently.

    Order is load-bearing:
      1. identifier   — removes cash and money-market rows, which price at
                        exactly 100.00 and would otherwise sit undetected in
                        the middle of the price distribution.
      2. amounts      — par and market value must both be positive. This is
                        also what makes `add_price`'s division by par safe.
      3. maturity     — must parse, must be in the future (already-matured
                        bonds appear in the file with ordinary prices), and
                        must be at least MIN_YEARS_TO_MATURITY away.

    No price band. Zero-coupon municipals legitimately price in single digits
    at 30-40 years, so a band deletes real bonds. Yield sanity comes later,
    once a yield can actually be computed.
    """
    work = df.copy()
    work["reason"] = pd.NA

    ident = work["Identifier"].astype(str).str.strip()
    work.loc[~ident.str.fullmatch(ISIN) & work["reason"].isna(),
             "reason"] = "identifier"

    par = pd.to_numeric(work["Par Value"], errors="coerce")
    mv = pd.to_numeric(work["Market Value"], errors="coerce")
    work.loc[~((par > 0) & (mv > 0)) & work["reason"].isna(),
             "reason"] = "amounts"

    # Explicit format: an ambiguous date must fail loudly, not be reinterpreted.
    mat = pd.to_datetime(work["Maturity"], format="%m/%d/%Y", errors="coerce")
    work.loc[mat.isna() & work["reason"].isna(), "reason"] = "no_maturity"
    work.loc[(mat <= as_of) & work["reason"].isna(), "reason"] = "matured"
    near = as_of + pd.Timedelta(days=MIN_YEARS_TO_MATURITY * 365.25)
    work.loc[(mat <= near) & work["reason"].isna(), "reason"] = "near_maturity"

    work["maturity"] = mat
    work["par"] = par
    work["market_value"] = mv
    kept = work[work["reason"].isna()].drop(columns=["reason"])
    dropped = work[work["reason"].notna()]
    return kept.reset_index(drop=True), dropped.reset_index(drop=True)


def add_price(df):
    """Price per 100 face. This is the whole trick: the holdings file reports
    par and market value, and their ratio is the bond's price."""
    out = df.copy()
    out["price"] = out["market_value"] / out["par"] * 100.0
    return out


# rateslib ships the conventions; do not hand-roll day counts.
#   us_corp — 30/360, semiannual
#   us_muni — the municipal convention
SPECS = ("us_corp", "us_muni")


def add_yield(df, as_of, spec):
    """Yield to maturity and modified duration per bond, via rateslib.

    `effective` is deliberately arbitrary: for a bond held to maturity only
    the settlement-to-maturity leg enters the yield, and the holdings file
    carries no issue date. But that leg has to actually exist in rateslib's
    generated coupon schedule for the yield/duration solve to be correct, so
    `effective` is anchored to settlement (five years before `as_of`) rather
    than to maturity. Anchoring it to maturity instead — e.g. one year before
    maturity — silently truncates the schedule to whatever few coupon periods
    fall in that final year, dropping every coupon between settlement and
    then; rateslib then solves a materially wrong yield to hit the given
    price (verified against this project's own test fixtures: the
    maturity-anchored version returned ytm=6.02% against an expected 5.15%,
    and 3.37% against an expected 3.07%). Five years comfortably clears the
    ~90-day-before-settlement threshold at which the solve stabilizes, and is
    always safely before `termination` because `clean_rows` only keeps bonds
    maturing after `as_of`. A bond whose yield cannot be solved returns NaN
    rather than raising, so one bad row cannot take down a 5,000-row desk.
    """
    if spec not in SPECS:
        raise ValueError(f"spec must be one of {SPECS}, got {spec!r}")
    settle = as_of.to_pydatetime()
    effective = (as_of - pd.DateOffset(years=5)).to_pydatetime()
    ytms, durs = [], []
    for _, row in df.iterrows():
        try:
            price = float(row["price"])
            # Zero (or negative) price has no finite yield: YTM only approaches
            # infinity as price approaches zero, it is never reached. Guarding
            # explicitly here matters because rateslib's solver does not
            # reliably raise on this input — with a long enough coupon
            # schedule it can converge on a nonsense four-digit "yield"
            # instead of failing, which the broad except below would not catch.
            if price <= 0:
                raise ValueError("price must be positive; yield is undefined at zero")
            bond = rl.FixedRateBond(
                effective=effective,
                termination=row["maturity"].to_pydatetime(),
                fixed_rate=float(row["Coupon"]), spec=spec)
            y = float(bond.ytm(price=price, settlement=settle))
            d = float(bond.duration(ytm=y, settlement=settle, metric="modified"))
        except Exception:
            y, d = float("nan"), float("nan")
        ytms.append(y)
        durs.append(d)
    out = df.copy()
    out["ytm"] = ytms
    out["mod_duration"] = durs
    return out


# A yield this high is a recovery play, not a point on a credit curve. Set well
# above the 11.6% that legitimate 30-year zero-coupon municipals reach, so the
# threshold cannot quietly delete them.
DISTRESSED_YTM = 25.0

# The yield-sanity band. Below the floor or above the ceiling the row is not a
# bond at all — it is corrupted input, typically a unit mismatch.
#
# The floor is the spec's -5%: a mark that implies paying materially more than
# every remaining cashflow is a bad mark, not a bond.
#
# The ceiling is NOT the spec's original 50%, and this file and the spec now
# agree on why. That 50% was written when the yield band was the only high-side
# gate. It is not: DISTRESSED_YTM at 25% now owns the whole ordinary high side,
# and distressed bonds are kept — in the universe, out of the fit — rather than
# discarded. So anything a 50% (or the code's earlier 200%) ceiling would catch
# is a distressed bond, and tagging it "unit mismatch" is simply wrong: measured
# on live HYMB data, 18 of the 25 genuinely distressed municipals solve ABOVE
# 200%, KANDEV 5.000 11/15/46 at price 0.000611 among them at 1,019%, and the
# worst real mark (DIRGEN 5.750 02/15/38 at 0.000998) reaches 1,801%.
#
# The ceiling's remaining job is therefore only to catch a number no positive
# price in a holdings file can produce. 10,000% sits ~5.5x above the worst real
# distressed mark and far below anything a genuine bond can reach, so it fires
# on corrupted input and on nothing else.
YTM_FLOOR, YTM_CEILING = -5.0, 10_000.0

# Non-capturing: a capturing group makes str.contains warn on every call.
_VARIABLE = re.compile(r"\b(?:VAR|FRN)\b")

# Every value `excluded_reason` can take. "" means the bond is in the fit.
EXCLUDED_REASONS = ("variable_rate", "distressed", "no_yield")


def classify(df):
    """Mark which bonds may calibrate the credit curve, and why the rest may not.

    Three exclusions, all of which stay visible in the universe:

    variable_rate  A fixed-to-float note's coupon is fixed only to its reset
                   date. A yield assuming today's coupon runs to maturity is
                   wrong for 14% of the corporate universe. Pricing them needs
                   a forward curve and each note's reset schedule, neither of
                   which the holdings file carries.
    distressed     Marked near zero and yielding on recovery, not on spread.
    no_yield       Yield unsolvable, or outside [YTM_FLOOR, YTM_CEILING] and so
                   corrupted input rather than a bond.

    `excluded_reason` is a TRUE PARTITION of the universe: every row carries
    exactly one of those three strings or "" for in-fit, and `in_fit`,
    `is_variable` and `is_distressed` are all derived from it, so a caller can
    count buckets off `excluded_reason` alone and have them sum to len(df).
    They did not always: `is_variable` and `is_distressed` were independent
    booleans, so a distressed FRN counted twice, and an unconditional
    `reason[unusable] = "no_yield"` overwrote "distressed" on every recovery
    mark above the ceiling.

    Precedence, most fundamental reason first:

      variable_rate > distressed > no_yield

    variable_rate wins because a fixed-coupon yield on a note whose coupon
    resets is not a yield at all — whatever it solved to, distressed or absurd,
    it is the wrong number for the wrong instrument. distressed then wins over
    no_yield because a near-zero recovery mark solving to 1,019% is a correct
    yield on a real holding, not the "unit mismatch" no_yield is defined as;
    calling it no_yield hid KANDEV — the spec's own canonical distressed
    example — inside a bucket meaning "we could not price this".
    """
    out = df.copy()
    name = out["Name"].astype(str).str.upper()
    ytm = pd.to_numeric(out["ytm"], errors="coerce")

    variable = name.str.contains(_VARIABLE)
    # Sane means "the solver returned a number that could describe a bond".
    # distressed is bounded above by the ceiling so that the ceiling stays a
    # live gate rather than an unreachable branch under the precedence above.
    sane = ytm.notna() & (ytm > YTM_FLOOR) & (ytm < YTM_CEILING)

    reason = pd.Series("", index=out.index, dtype="object")
    reason[~sane] = "no_yield"
    reason[sane & (ytm >= DISTRESSED_YTM)] = "distressed"
    reason[variable] = "variable_rate"

    out["excluded_reason"] = reason
    out["in_fit"] = reason == ""
    out["is_variable"] = reason == "variable_rate"
    out["is_distressed"] = reason == "distressed"
    return out


def build_sector(files, spec):
    """Assemble one sector's bond universe from several holdings files.

    Deduplicates on identifier, keeping the largest position, on the reasoning
    that the fund holding more of a bond is the better mark. The three files of
    a sector overlap heavily by design — they are short, intermediate and broad
    slices of the same market.

    If the files disagree on as-of date the OLDEST wins, because pricing part of
    a curve a day forward of the rest is worse than pricing all of it a day late.

    `files` maps ticker -> a path or raw bytes (production hands over bytes
    fetched via HTTP; read_holdings accepts either).
    """
    if not files:
        raise ValueError("build_sector requires at least one holdings file")

    as_ofs, frames, drops = [], [], []
    for ticker, source in files.items():
        as_of, raw = read_holdings(source)
        as_ofs.append(as_of)
        kept, dropped = clean_rows(raw, as_of)
        kept = kept.assign(source=ticker)
        frames.append(kept)
        drops.append(dropped.assign(source=ticker))

    as_of = min(as_ofs)
    universe = pd.concat(frames, ignore_index=True)
    # Deduplicate BEFORE pricing, not after. The three files of a sector overlap
    # heavily, so pricing first meant solving a yield for every duplicate and
    # throwing 37% of the solves away: corporate ran 10,914 rateslib solves to
    # keep 6,870. The result is identical either way — dedup keys on Identifier
    # and market_value, and add_price/add_yield/classify touch neither, and are
    # per-row pure — so this is purely the same work in a cheaper order.
    universe = (universe.sort_values("market_value", ascending=False)
                        .drop_duplicates(subset="Identifier", keep="first")
                        .reset_index(drop=True))
    universe = add_yield(add_price(universe), as_of, spec)
    universe = classify(universe)
    # `files` is non-empty (guarded above), so `drops` always has at least one
    # frame here — no empty-input fallback needed.
    dropped = pd.concat(drops, ignore_index=True)
    return as_of, universe, dropped


def tax_equivalent_yield(ytm, marginal_rate):
    """What a taxable bond would have to yield to match this tax-exempt one.

    This is the number that actually decides a retail municipal trade: a 3%
    muni beats a 4.5% corporate for a top-bracket buyer and loses for a
    low-bracket one. Federal only — state and local exemptions vary by issuer
    and by the holder's residence, and the holdings file identifies neither.
    """
    if not 0.0 <= marginal_rate < 1.0:
        raise ValueError(f"marginal_rate must be in [0, 1), got {marginal_rate}")
    return ytm / (1.0 - marginal_rate)
