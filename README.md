<div align="center">

# 📉 LETF Desk

**Leveraged-ETF hedge overlays, backtested against the decay.**

![yfinance](https://img.shields.io/badge/yfinance-1D9E75?style=flat-square&labelColor=0C110F) ![Alpaca](https://img.shields.io/badge/Alpaca-1D9E75?style=flat-square&labelColor=0C110F) ![Plotly](https://img.shields.io/badge/Plotly-1D9E75?style=flat-square&labelColor=0C110F)

<sub>A desk of <a href="https://tobibui1904.github.io/quant-lab-site/">Quant Lab</a> · branch <code>letf</code></sub>

</div>

---

> A 3x inverse ETF is not a 3x short. Daily rebalancing means the tracking error compounds against you, and this desk measures that rather than ignoring it.

Backtests SQQQ-style overlays as hedges: what the protection actually cost over the holding period, versus what a naive multiple would suggest.

### The decay problem

Leveraged ETFs reset exposure **daily**. Over any period longer than a day, the
return is path-dependent: in a choppy market a 3x fund can lose money while the
underlying index is flat. That is a structural feature of daily rebalancing, not
a fee.

This desk therefore evaluates overlays on realised paths, not on a multiplier
assumption:

- entry and exit timing across the holding period
- realised drag versus the naive `3 × index` expectation
- hedge effectiveness during drawdowns specifically, where it is supposed to earn its cost

`ETF_triple.txt` lists the leveraged tickers under consideration.

### Files on this branch

| |
|---|
| `ETF_triple.txt` |
| `letf.py` |

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
marimo run letf.py          # read-only dashboard
marimo edit letf.py         # editable notebook
```

Served together, all desks live behind one FastAPI hub with a password gate
(branch [`hub`](../../tree/hub)). Credentials come from a local `.env` that is
never committed -- see `.env.example` on the `hub` branch for the variable
names.

</details>

<details>
<summary><b>Honest limitations</b></summary>

<br>

Backtests inherit survivorship and look-ahead risk, and this one assumes fills at close without slippage or borrow cost. Hedge effectiveness measured on past drawdowns says little about the shape of the next one. Leveraged products are unsuitable as long-horizon holdings by design.

</details>

---

<div align="center">
<sub>Part of <b>Quant Lab</b> by <a href="https://github.com/tobibui1904">Tobi Bui</a> ·
<a href="https://tobibui1904.github.io/quant-lab-site/">Live site</a></sub>
</div>
