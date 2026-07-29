<div align="center">

# 📊 Portfolio & Allocation

**Cross-desk allocation with hierarchical risk parity.**

![skfolio](https://img.shields.io/badge/skfolio-1D9E75?style=flat-square&labelColor=0C110F) ![DuckDB](https://img.shields.io/badge/DuckDB-1D9E75?style=flat-square&labelColor=0C110F) ![scikit--learn](https://img.shields.io/badge/scikit----learn-1D9E75?style=flat-square&labelColor=0C110F)

<sub>A desk of <a href="https://tobibui1904.github.io/quant-lab-site/">Quant Lab</a> · branch <code>portfolio</code></sub>

</div>

---

> Individual desks produce signals. This is what decides how much of the book each one gets.

Reads positions and returns from DuckDB, then allocates with `skfolio` — clustering correlated assets before optimising, rather than inverting a covariance matrix that is barely invertible.

### Why not mean-variance

Classical Markowitz optimisation on real return data is famously unstable: the
covariance matrix is near-singular, and tiny changes in estimated returns swing
the weights wildly. `skfolio`'s clustering approach (`skfolio.cluster`,
`skfolio.distance`) groups correlated assets first and allocates across the
hierarchy, which is far less sensitive to estimation error.

Selection uses `skfolio.model_selection` with walk-forward splits, so the
allocation is scored on data it was not fitted on.

### Storage

`quant_trading.db` is a DuckDB file with `Assets`, `Portfolio` and
`Trading_Log` tables, shared with the
[`pairs-trading`](../../tree/pairs-trading) desk.

### Files on this branch

| |
|---|
| `portfolio.py` |
| `quant_trading.db` |
| `trading.py` |

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
marimo run portfolio.py          # read-only dashboard
marimo edit portfolio.py         # editable notebook
```

Served together, all desks live behind one FastAPI hub with a password gate
(branch [`hub`](../../tree/hub)). Credentials come from a local `.env` that is
never committed -- see `.env.example` on the `hub` branch for the variable
names.

</details>

<details>
<summary><b>Honest limitations</b></summary>

<br>

Allocation is only as good as the return series it is fitted on, and walk-forward validation still reuses one historical path. Hierarchical methods reduce estimation sensitivity but do not remove it. The committed database is a point-in-time snapshot, not a live book.

</details>

---

<div align="center">
<sub>Part of <b>Quant Lab</b> by <a href="https://github.com/tobibui1904">Tobi Bui</a> ·
<a href="https://tobibui1904.github.io/quant-lab-site/">Live site</a></sub>
</div>
