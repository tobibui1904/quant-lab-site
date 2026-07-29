<div align="center">

# 🕶️ OTC Tape Desk

**Off-exchange volume, dark pools and short interest from FINRA.**

![FINRA](https://img.shields.io/badge/FINRA-1D9E75?style=flat-square&labelColor=0C110F) ![pandas](https://img.shields.io/badge/pandas-1D9E75?style=flat-square&labelColor=0C110F)

<sub>A desk of <a href="https://tobibui1904.github.io/quant-lab-site/">Quant Lab</a> · branch <code>otc-tape</code></sub>

</div>

---

> Roughly half of US equity volume prints off-exchange. This desk reads that half.

Queries FINRA for OTC and ATS volume, daily short volume, and bi-monthly short interest, then normalises the reporting lags so the series can actually be compared.

### What FINRA actually gives you

| Dataset | Granularity | Lag |
|---|---|---|
| OTC / ATS volume | weekly, per venue | ~2-4 weeks |
| Daily short volume | daily | ~1 day |
| Short interest | bi-monthly | ~1 week after settlement |

The lags differ by dataset, which is the main trap: charting them on one axis
without aligning publication dates produces a chart that looks meaningful and
is not.

### A note on the API

FINRA's Query API is the useful surface here — `otcMarket` carries the
off-exchange data. There is no per-CUSIP fixed income endpoint on the standard
key, so Treasury work lives on the [`fixed-income`](../../tree/fixed-income)
branch against FedInvest instead.

### Files on this branch

| |
|---|
| `OTC_Track.py` |

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
marimo run OTC_Track.py          # read-only dashboard
marimo edit OTC_Track.py         # editable notebook
```

Served together, all desks live behind one FastAPI hub with a password gate
(branch [`hub`](../../tree/hub)). Credentials come from a local `.env` that is
never committed -- see `.env.example` on the `hub` branch for the variable
names.

</details>

<details>
<summary><b>Honest limitations</b></summary>

<br>

Short volume is not short interest — it counts shares sold short intraday, most of which are covered by the close, and it is routinely misread as a bearish gauge. Dark pool volume indicates where trades printed, not direction or intent.

</details>

---

<div align="center">
<sub>Part of <b>Quant Lab</b> by <a href="https://github.com/tobibui1904">Tobi Bui</a> ·
<a href="https://tobibui1904.github.io/quant-lab-site/">Live site</a></sub>
</div>
