<div align="center">

# 💱 Forex Desk

**EUR/USD ensemble combining price models with news sentiment, executed through OANDA.**

![OANDA](https://img.shields.io/badge/OANDA-1D9E75?style=flat-square&labelColor=0C110F) ![Transformers](https://img.shields.io/badge/Transformers-1D9E75?style=flat-square&labelColor=0C110F) ![Ollama](https://img.shields.io/badge/Ollama-1D9E75?style=flat-square&labelColor=0C110F) ![PyTorch](https://img.shields.io/badge/PyTorch-1D9E75?style=flat-square&labelColor=0C110F)

<sub>A desk of <a href="https://tobibui1904.github.io/quant-lab-site/">Quant Lab</a> · branch <code>forex</code></sub>

</div>

---

> Price history alone is a thin signal on a major pair. This desk adds a language model reading the news feed and blends the two.

An ML ensemble scores EUR/USD direction, a local LLM summarises incoming headlines into a sentiment term, and the combined view is executed against OANDA's practice endpoint.

### Method

Three signals are combined:

1. **Price models** — `meridianalgo.forex_ml` plus a small PyTorch model over recent bars.
2. **News sentiment** — `feedparser` pulls the wire; a HuggingFace `transformers`
   model and a local `ollama` model turn headlines into a directional score.
3. **Execution** — OANDA v3 REST, practice account by default.

Requests are issued through `aiohttp` with an explicit resolver, because the
news sources are slow and blocking the notebook on them makes the dashboard
unusable.

### Live marking

Positions and P&L are marked at **top-of-book** (`bids[0]` / `asks[0]`), not at
OANDA's closeout prices. Closeout spreads widen sharply outside session hours
and make an otherwise flat position look like a loss.

### Files on this branch

| |
|---|
| `forex.py` |

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
marimo run forex.py          # read-only dashboard
marimo edit forex.py         # editable notebook
```

Served together, all desks live behind one FastAPI hub with a password gate
(branch [`hub`](../../tree/hub)). Credentials come from a local `.env` that is
never committed -- see `.env.example` on the `hub` branch for the variable
names.

</details>

<details>
<summary><b>Honest limitations</b></summary>

<br>

Sentiment from headlines is a weak, laggy signal and the LLM scoring it is not calibrated — it produces a number, not a probability. The pair is also the most efficient in FX, so any edge here is thin by construction. Runs against OANDA's practice environment.

</details>

---

<div align="center">
<sub>Part of <b>Quant Lab</b> by <a href="https://github.com/tobibui1904">Tobi Bui</a> ·
<a href="https://tobibui1904.github.io/quant-lab-site/">Live site</a></sub>
</div>
