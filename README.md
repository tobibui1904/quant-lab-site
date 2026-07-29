<div align="center">

# 🪙 Crypto Desk

**Multi-agent reinforcement learning trading BTC and ETH on Alpaca.**

![PyTorch](https://img.shields.io/badge/PyTorch-1D9E75?style=flat-square&labelColor=0C110F) ![MAPPO](https://img.shields.io/badge/MAPPO-1D9E75?style=flat-square&labelColor=0C110F) ![LSTM](https://img.shields.io/badge/LSTM-1D9E75?style=flat-square&labelColor=0C110F) ![Alpaca](https://img.shields.io/badge/Alpaca-1D9E75?style=flat-square&labelColor=0C110F)

<sub>A desk of <a href="https://tobibui1904.github.io/quant-lab-site/">Quant Lab</a> · branch <code>crypto</code></sub>

</div>

---

> Two cooperating policy networks share one reward signal and learn to trade BTC and ETH together, rather than as two unrelated bets.

`crypto_MAPPO_LSTM.py` is the training loop. `crypto_inference.py` loads a trained checkpoint and runs it live against Alpaca's crypto venue. Ten checkpoints from cross-validated folds ship with the branch so results can be reproduced without retraining.

### Method

**MAPPO** — Multi-Agent Proximal Policy Optimisation. Each asset gets its own
actor, but a single centralised critic scores the joint state, so the agents
learn a portfolio policy instead of two independent ones. The actors are LSTMs,
so position sizing depends on the recent path of prices rather than a single
snapshot.

Training uses PPO's clipped surrogate objective with a shared advantage
estimate, `torch.distributions` for the stochastic policy, and Adam.

### Checkpoints

| File | What it is |
|---|---|
| `mappo_lstm.pt` | first full run |
| `mappo_lstm_best.pt` | best training reward |
| `mappo_lstm_valbest.pt` / `mappo_lstm_valsel.pt` | selected on validation, not training |
| `mappo_lstm_fold{1,2,3}.pt` | per-fold final weights |
| `mappo_lstm_best_fold{1,2,3}.pt` | per-fold best weights |

Three folds exist because a single train/test split on one crypto regime tells
you almost nothing — the folds are there to show how much the result moves.

### Data

Alpaca's historical crypto API supplies the bars; the same client places live
orders at inference time.

### Files on this branch

| |
|---|
| `crypto_MAPPO_LSTM.py` |
| `crypto_inference.py` |
| `mappo_lstm.pt` |
| `mappo_lstm_best.pt` |
| `mappo_lstm_best_fold1.pt` |
| `mappo_lstm_best_fold2.pt` |
| `mappo_lstm_best_fold3.pt` |
| `mappo_lstm_fold1.pt` |
| `mappo_lstm_fold2.pt` |
| `mappo_lstm_fold3.pt` |
| `mappo_lstm_valbest.pt` |
| `mappo_lstm_valsel.pt` |

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
marimo run crypto_inference.py          # read-only dashboard
marimo edit crypto_inference.py         # editable notebook
```

Served together, all desks live behind one FastAPI hub with a password gate
(branch [`hub`](../../tree/hub)). Credentials come from a local `.env` that is
never committed -- see `.env.example` on the `hub` branch for the variable
names.

</details>

<details>
<summary><b>Honest limitations</b></summary>

<br>

Crypto regimes shift faster than the training window. A policy fitted through one volatility regime can be confidently wrong in the next — which is exactly why the validation-selected and per-fold checkpoints are published alongside the headline one. Nothing here models exchange outages, funding costs, or slippage at size.

</details>

---

<div align="center">
<sub>Part of <b>Quant Lab</b> by <a href="https://github.com/tobibui1904">Tobi Bui</a> ·
<a href="https://tobibui1904.github.io/quant-lab-site/">Live site</a></sub>
</div>
