<div align="center">

# 🔮 Prediction Markets Desk

**A Polymarket desk driven by an LLM broker over MCP.**

![Polymarket](https://img.shields.io/badge/Polymarket-1D9E75?style=flat-square&labelColor=0C110F) ![MCP](https://img.shields.io/badge/MCP-1D9E75?style=flat-square&labelColor=0C110F) ![Ollama](https://img.shields.io/badge/Ollama-1D9E75?style=flat-square&labelColor=0C110F)

<sub>A desk of <a href="https://tobibui1904.github.io/quant-lab-site/">Quant Lab</a> · branch <code>prediction-markets</code></sub>

</div>

---

> An LLM reads the market, forms a view, and places it — with the tool surface constrained by a schema so it cannot invent an order.

Connects to Polymarket through an MCP server. A local model reasons about resolution criteria and pricing; `jsonschema` validates every tool call before it executes.

### Architecture

```
ollama (local LLM)
   ↕  reasons about markets, proposes actions
MCP client  ──stdio──▶  MCP server
   ↕  every call validated against a JSON schema first
Polymarket
```

The schema layer is the load-bearing part. A language model asked to trade will
otherwise hallucinate market ids, sizes, or entire endpoints; validating each
proposed call against a declared schema turns a malformed action into a rejected
one rather than a submitted one.

### Why prediction markets

Prices are explicit probabilities, which makes calibration measurable in a way
equity prices never are — a market at 0.30 should resolve YES about 30% of the
time, and you can check.

### Files on this branch

| |
|---|
| `pred_market.py` |

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
marimo run pred_market.py          # read-only dashboard
marimo edit pred_market.py         # editable notebook
```

Served together, all desks live behind one FastAPI hub with a password gate
(branch [`hub`](../../tree/hub)). Credentials come from a local `.env` that is
never committed -- see `.env.example` on the `hub` branch for the variable
names.

</details>

<details>
<summary><b>Honest limitations</b></summary>

<br>

Thin books and wide spreads dominate outside headline markets, so quoted probability is often stale. Resolution criteria carry real ambiguity, and an LLM reading them will not reliably notice the edge cases that decide payout. Access is jurisdiction-dependent.

</details>

---

<div align="center">
<sub>Part of <b>Quant Lab</b> by <a href="https://github.com/tobibui1904">Tobi Bui</a> ·
<a href="https://tobibui1904.github.io/quant-lab-site/">Live site</a></sub>
</div>
