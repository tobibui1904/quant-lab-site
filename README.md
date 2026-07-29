<div align="center">

# ⚖️ Pair Trading Desk

**Cointegration screens, forecast-assisted spreads, live execution.**

![statsmodels](https://img.shields.io/badge/statsmodels-1D9E75?style=flat-square&labelColor=0C110F) ![Prophet](https://img.shields.io/badge/Prophet-1D9E75?style=flat-square&labelColor=0C110F) ![skfolio](https://img.shields.io/badge/skfolio-1D9E75?style=flat-square&labelColor=0C110F) ![Alpaca](https://img.shields.io/badge/Alpaca-1D9E75?style=flat-square&labelColor=0C110F)

<sub>A desk of <a href="https://tobibui1904.github.io/quant-lab-site/">Quant Lab</a> · branch <code>pairs-trading</code></sub>

</div>

---

> Screens the universe for genuinely cointegrated pairs — with a multiple-testing correction, because scanning thousands of pairs manufactures false positives by the hundred.

Screens candidates with `tradingview_screener`, tests cointegration, corrects for multiple comparisons, forecasts the spread, and executes the two legs through Alpaca.

### Method

1. **Screen** — `tradingview_screener` narrows the universe to liquid candidates.
2. **Test** — Engle-Granger via `statsmodels.tsa.stattools.coint`, with the
   hedge ratio from an OLS fit.
3. **Correct** — `statsmodels.stats.multitest` applies a false-discovery-rate
   correction. Scanning *n* pairs at p<0.05 yields roughly 0.05·*n* spurious
   "cointegrated" pairs; without this step the screen is noise.
4. **Forecast** — `prophet` models the spread, with `prophet.diagnostics`
   cross-validation rather than a single in-sample fit.
5. **Size** — `skfolio` for allocation across accepted pairs.
6. **Execute** — both legs through Alpaca.

Results persist to `quant_trading.db` (branch [`portfolio`](../../tree/portfolio)),
and `joblib` caches the expensive screen so a re-run is not a re-scan.

### Files on this branch

| |
|---|
| `known_pair_trading.py` |

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
marimo run known_pair_trading.py          # read-only dashboard
marimo edit known_pair_trading.py         # editable notebook
```

Served together, all desks live behind one FastAPI hub with a password gate
(branch [`hub`](../../tree/hub)). Credentials come from a local `.env` that is
never committed -- see `.env.example` on the `hub` branch for the variable
names.

</details>

<details>
<summary><b>Honest limitations</b></summary>

<br>

Cointegration is a statement about the sample you tested, not a promise about the future — pairs decouple, often exactly when the spread looks most attractive. The FDR correction reduces false discoveries but cannot eliminate them. Prophet is a trend/seasonality model and has no notion of a structural break.

</details>

---

<div align="center">
<sub>Part of <b>Quant Lab</b> by <a href="https://github.com/tobibui1904">Tobi Bui</a> ·
<a href="https://tobibui1904.github.io/quant-lab-site/">Live site</a></sub>
</div>
