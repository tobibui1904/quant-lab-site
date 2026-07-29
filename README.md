<div align="center">

# 🌡️ Macro Desk

**FRED factor models — sector betas under macro stress.**

![FRED](https://img.shields.io/badge/FRED-1D9E75?style=flat-square&labelColor=0C110F) ![scikit--learn](https://img.shields.io/badge/scikit----learn-1D9E75?style=flat-square&labelColor=0C110F) ![statsmodels](https://img.shields.io/badge/statsmodels-1D9E75?style=flat-square&labelColor=0C110F)

<sub>A desk of <a href="https://tobibui1904.github.io/quant-lab-site/">Quant Lab</a> · branch <code>macro</code></sub>

</div>

---

> Which sectors actually move when rates, inflation or unemployment surprise — estimated, not assumed.

Pulls FRED series, reduces them to orthogonal factors, and regresses sector returns on those factors to get betas that can be stressed.

### Method

1. **Fetch** FRED series (rates, CPI, unemployment, spreads) in parallel.
2. **Orthogonalise** with `sklearn.decomposition` — raw macro series are heavily
   collinear, and regressing on them directly produces unstable, uninterpretable
   coefficients.
3. **Regress** sector returns on the factors via `statsmodels.formula.api`.
4. **Correct** with `statsmodels.stats.multitest` — testing many sectors against
   many factors is a multiple-comparisons problem, and uncorrected p-values here
   would be actively misleading.
5. **Stress** — shock a factor, read the implied sector response off the betas.

`SUUR0000SA0.txt` is a BLS CPI series carried alongside the FRED pulls.

### Caching

FRED responses cache to `.fredcache/` locally (not committed — it regenerates
on demand and is pure duplication of a public API).

### Files on this branch

| |
|---|
| `SUUR0000SA0.txt` |
| `macro_research.py` |

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
marimo run macro_research.py          # read-only dashboard
marimo edit macro_research.py         # editable notebook
```

Served together, all desks live behind one FastAPI hub with a password gate
(branch [`hub`](../../tree/hub)). Credentials come from a local `.env` that is
never committed -- see `.env.example` on the `hub` branch for the variable
names.

</details>

<details>
<summary><b>Honest limitations</b></summary>

<br>

Betas are estimated over a historical window and are not stable across regimes — a rate beta fitted through a hiking cycle does not describe a cutting one. Factor models describe co-movement, not causation, and a stress scenario built from them assumes the historical relationship survives the stress.

</details>

---

<div align="center">
<sub>Part of <b>Quant Lab</b> by <a href="https://github.com/tobibui1904">Tobi Bui</a> ·
<a href="https://tobibui1904.github.io/quant-lab-site/">Live site</a></sub>
</div>
