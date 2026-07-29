<div align="center">

# 🎯 Options Desk

**Defined-risk vertical debit spreads — bull call and bear put.**

![Alpaca Options](https://img.shields.io/badge/Alpaca%20Options-1D9E75?style=flat-square&labelColor=0C110F) ![SciPy](https://img.shields.io/badge/SciPy-1D9E75?style=flat-square&labelColor=0C110F) ![Black--Scholes](https://img.shields.io/badge/Black----Scholes-1D9E75?style=flat-square&labelColor=0C110F)

<sub>A desk of <a href="https://tobibui1904.github.io/quant-lab-site/">Quant Lab</a> · branch <code>options-spreads</code></sub>

</div>

---

> Two mirrored desks. Both are debit spreads, so maximum loss is the premium paid and is known before entry.

`bull_call_spread.py` expresses the bullish leg, `bear_put_spread.py` the bearish one. Each prices the chain, picks strikes, plots the payoff, and can place the two-leg order through Alpaca.

### The two structures

| | Bull Call | Bear Put |
|---|---|---|
| View | up | down |
| Buy | lower-strike call | higher-strike put |
| Sell | higher-strike call | lower-strike put |
| Max loss | net debit | net debit |
| Max gain | strike width − debit | strike width − debit |

Selling the far leg caps the upside but pays for much of the near leg — the
trade is a bounded bet, which is the point.

### Method

Greeks and implied vol come from `scipy.stats` and `scipy.optimize` over the
Alpaca option chain. Payoff diagrams are plotted at expiry and at present
value, so the difference between "right eventually" and "right now" is visible.

These desks are **gated in the hub** — they unlock only when the fundamental
desk's verdict names the matching structure.

### Files on this branch

| |
|---|
| `bear_put_spread.py` |
| `bull_call_spread.py` |

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
marimo run bull_call_spread.py          # read-only dashboard
marimo edit bull_call_spread.py         # editable notebook
```

Served together, all desks live behind one FastAPI hub with a password gate
(branch [`hub`](../../tree/hub)). Credentials come from a local `.env` that is
never committed -- see `.env.example` on the `hub` branch for the variable
names.

</details>

<details>
<summary><b>Honest limitations</b></summary>

<br>

Defined risk is not small risk: the entire debit can be lost, and usually is when the view is wrong. Chain liquidity varies enormously by underlying and expiry, and mid-price fills assumed in the payoff plots are optimistic for wide spreads. Early assignment on the short leg is not modelled.

</details>

---

<div align="center">
<sub>Part of <b>Quant Lab</b> by <a href="https://github.com/tobibui1904">Tobi Bui</a> ·
<a href="https://tobibui1904.github.io/quant-lab-site/">Live site</a></sub>
</div>
