<div align="center">

# 🏛️ Treasury Desk

**Bills, notes, TIPS, FRNs and STRIPS — curves fitted with rateslib.**

![rateslib](https://img.shields.io/badge/rateslib-1D9E75?style=flat-square&labelColor=0C110F) ![TreasuryDirect](https://img.shields.io/badge/TreasuryDirect-1D9E75?style=flat-square&labelColor=0C110F) ![FedInvest](https://img.shields.io/badge/FedInvest-1D9E75?style=flat-square&labelColor=0C110F)

<sub>A desk of <a href="https://tobibui1904.github.io/quant-lab-site/">Quant Lab</a> · branch <code>fixed-income</code></sub>

</div>

---

> Every marketable Treasury security, priced end-of-day, with yields computed under the right convention for each instrument type.

Pulls the FedInvest end-of-day file, cleans it, and fits yield curves. The conventions matter: bills quote bond-equivalent yield, notes use street convention, FRNs have no fixed YTM at all.

### Why the conventions matter

A naive `yield()` across the whole file is wrong for most of it:

| Instrument | Treatment |
|---|---|
| Bills | discount quote converted to **bond-equivalent yield** |
| Notes / Bonds | **street convention**, next NYC business day settlement |
| FRNs | **no YTM** — the coupon floats, so a fixed yield is meaningless |
| TIPS | real yield; principal indexes to CPI |
| STRIPS | zero-coupon, priced off the discount factor directly |

Rows that are matured or unpriced resolve to `NaN` rather than a fabricated
number, and a zero end-of-day price is treated as missing rather than as a real
price of zero.

### Data

`fedinvest_out/` holds the fetched end-of-day files; `.tdterms.parquet` caches
security terms so a re-run does not re-download everything. Curve fitting is
`rateslib`; the fetch is parallelised with `concurrent.futures` because the
source is slow per-request.

### Files on this branch

| |
|---|
| `.tdterms.parquet` |
| `fedinvest_out/` |
| `fixed_income.py` |

<sub>This branch also carries the published site from `main`.</sub>

---

<details>
<summary><b>Running this desk</b></summary>

<br>

These are [marimo](https://marimo.io) notebooks, not scripts. Each one is a
reactive dashboard: cells re-execute when their inputs change, so there is no
hidden run order to remember.

```bash
pip install marimo
marimo run fixed_income.py          # read-only dashboard
marimo edit fixed_income.py         # editable notebook
```

Served together, all desks live behind one FastAPI hub with a password gate
(branch [`hub`](../../tree/hub)). Credentials come from a local `.env` that is
never committed -- see `.env.example` on the `hub` branch for the variable
names.

</details>

<details>
<summary><b>Honest limitations</b></summary>

<br>

End-of-day pricing only — this is not a live curve. FedInvest publishes with a lag, and the file occasionally carries stale or zero prices that are dropped rather than interpolated. Fitted curves are only as good as the day's liquidity in the underlying issues.

</details>

---

<div align="center">
<sub>Part of <b>Quant Lab</b> by <a href="https://github.com/tobibui1904">Tobi Bui</a> ·
<a href="https://tobibui1904.github.io/quant-lab-site/">Live site</a></sub>
</div>
