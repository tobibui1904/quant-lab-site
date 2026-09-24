import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", css_file="theme.css", html_head_file="theme_head.html")


@app.cell
def _():
    # Import standard library modules
    from datetime import datetime, timedelta
    from datetime import date
    from zoneinfo import ZoneInfo
    import json
    import os
    import sys
    import marimo as mo

    # When stdout is a pipe rather than a console, Python encodes it with the
    # locale codec — cp1252 on this machine — and every arrow/sigma this
    # notebook prints raises UnicodeEncodeError. Force UTF-8 on the real
    # streams; marimo's in-notebook capture stream has no reconfigure().
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")

    # Import third-party modules
    from dotenv import load_dotenv
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Import Alpaca modules
    import alpaca
    from alpaca.data.historical.crypto import CryptoHistoricalDataClient
    from alpaca.data.requests import (
        CryptoBarsRequest,
        CryptoLatestQuoteRequest,
        CryptoQuoteRequest,
        CryptoTradesRequest,
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import (
        AssetClass,
        AssetStatus,
        OrderSide,
        OrderType,
        QueryOrderStatus,
        TimeInForce,
        AssetClass
    )

    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, GetOrdersRequest

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    from collections import deque
    from dateutil.relativedelta import relativedelta

    # Load Alpaca credentials from .env in the notebook's own directory
    # (assign to _ so marimo doesn't render load_dotenv's bool return).
    _ = load_dotenv(mo.notebook_dir() / ".env")

    return (
        AssetClass,
        Categorical,
        CryptoBarsRequest,
        CryptoHistoricalDataClient,
        GetOrdersRequest,
        MarketOrderRequest,
        OrderSide,
        OrderType,
        QueryOrderStatus,
        TimeFrame,
        TimeInForce,
        TradingClient,
        ZoneInfo,
        date,
        datetime,
        go,
        make_subplots,
        mo,
        nn,
        np,
        optim,
        os,
        pd,
        timedelta,
        torch,
    )


@app.cell
def _(np, pd):
    short_window = 12
    long_window  = 26
    sma_window   = 50

    def add_indicators(df, prefix):
        h = df[f'{prefix}_high']
        l = df[f'{prefix}_low']
        c = df[f'{prefix}_close']

        # EMAs / SMA
        df[f'{prefix}_ema_short'] = c.ewm(span=short_window, adjust=False).mean()
        df[f'{prefix}_ema_long']  = c.ewm(span=long_window,  adjust=False).mean()
        df[f'{prefix}_sma']       = c.rolling(window=sma_window).mean()

        # ATR
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        df[f'{prefix}_ATR_14'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

        # ADX
        up     = h - h.shift(1)
        down   = l.shift(1) - l
        dm_pos = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
        dm_neg = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
        atr_s  = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        di_pos = 100 * dm_pos.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr_s
        di_neg = 100 * dm_neg.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr_s
        dx     = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg)
        df[f'{prefix}_ADX_14'] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        df[f'{prefix}_DMP_14'] = di_pos
        df[f'{prefix}_DMN_14'] = di_neg

        return df

    return (add_indicators,)


@app.cell
def _(mo):
    mo.Html("""
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

    <div style="padding: 2.5rem 0 2rem; text-align: center;">
      <div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; border-radius: 50%; border: 0.5px solid #1D9E75; margin-bottom: 1rem;">
        <i class="ti ti-chart-candle" style="font-size: 22px; color: #1D9E75;"></i>
      </div>
      <h1 style="font-family: 'DM Serif Display', serif; font-size: 36px; font-weight: 400; font-style: italic; margin: 0 0 6px; letter-spacing: -0.01em; color: var(--color-text-primary);">Crypto MAPPO LSTM Trading Strategy</h1>
      <p style="font-family: 'DM Mono', monospace; font-size: 11px; color: var(--color-text-secondary); letter-spacing: 0.18em; text-transform: uppercase; margin: 0 0 1.25rem;">Made with Customized LSTM model</p>
      <div style="display: inline-flex; align-items: center; gap: 6px; font-family: 'DM Mono', monospace; font-size: 11px; color: #0F6E56; background: #E1F5EE; padding: 4px 14px; border-radius: 999px;">
        <span style="width: 6px; height: 6px; border-radius: 50%; background: #1D9E75; display: inline-block;"></span>
        Live
      </div>
      <div style="margin-top: 1.5rem; width: 40px; height: 0.5px; background: var(--color-border-tertiary); margin-left: auto; margin-right: auto;"></div>
    </div>
    """)
    return


@app.cell
def _(Categorical, nn, np, optim, pd, torch):
    """
    MAPPO + LSTM Pipeline for Multi-Asset Crypto Trading
    Pairs  : BTC/USD, ETH/USD, ETH/BTC
    Agent 0 → trades BTC/USD
    Agent 1 → trades ETH/USD
    Agent 2 → trades ETH/BTC
    Central critic sees all three markets simultaneously.

    Expected DataFrame columns (prefixed by pair):
      btc_open, btc_high, btc_low, btc_close, btc_volume, btc_trade_count,
      btc_vwap, btc_ema_short, btc_ema_long, btc_sma,
      btc_ADX_14, btc_DMP_14, btc_DMN_14, btc_ATR_14,
      eth_open, eth_high, ...  (same 14 cols)
      etb_open, etb_high, ...  (ETH/BTC, same 14 cols)
    """

    # ─────────────────────────────────────────────
    # 1. CONFIG
    # ─────────────────────────────────────────────

    class Config:
        # Pairs and per-pair feature columns (unprefixed)
        PAIRS = ['btc', 'eth', 'etb']           # btc=BTC/USD, eth=ETH/USD, etb=ETH/BTC
        BASE_FEATURES = [
            'open', 'high', 'low', 'close', 'volume',
            'trade_count', 'vwap', 'ema_short', 'ema_long',
            'sma', 'ADX_14', 'DMP_14', 'DMN_14', 'ATR_14'
        ]
        N_FEATURES  = len(BASE_FEATURES)        # features per agent (14)
        N_AGENTS    = len(PAIRS)                # one agent per pair (3)

        # Environment
        WINDOW_SIZE  = 20
        N_ACTIONS    = 3                        # 0=hold, 1=buy, 2=sell
        INITIAL_CASH = 100_000.0                 # per agent, in USD
        TRADE_SIZE   = 0.1
        TX_COST      = 0.001

        # LSTM
        LSTM_HIDDEN  = 64
        LSTM_LAYERS  = 1
        FC_HIDDEN    = 64

        # PPO
        CLIP_EPS     = 0.2
        GAMMA        = 0.99
        GAE_LAMBDA   = 0.95
        ENTROPY_COEF = 0.01
        VALUE_COEF   = 0.5
        LR           = 3e-4
        EPOCHS       = 4
        BATCH_SIZE   = 64
        ROLLOUT_LEN  = 128

        DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'


    cfg = Config()


    # ─────────────────────────────────────────────
    # 2. PREPROCESSING
    # ─────────────────────────────────────────────

    def preprocess(df: pd.DataFrame, scaler: dict = None):
        """
        Expects columns prefixed by pair: btc_close, eth_ADX_14, etc.

        If `scaler` is provided (loaded from the training checkpoint) it is used
        to TRANSFORM inputs with the exact stats seen during training. If None,
        min/max are fit on this window (legacy behavior) — only correct for
        checkpoints that predate scaler persistence.

        Returns: norm_data, raw_closes, T
        All pairs are inner-joined on index so T is consistent.
        """
        # Build per-pair DataFrames
        pair_dfs = {}
        for pair in cfg.PAIRS:
            cols = {f'{pair}_{f}': f for f in cfg.BASE_FEATURES}
            pair_df = df[[c for c in cols]].rename(columns=cols).dropna()
            pair_dfs[pair] = pair_df

        # Align all pairs to common timestamps
        common_idx = pair_dfs[cfg.PAIRS[0]].index
        for pair in cfg.PAIRS[1:]:
            common_idx = common_idx.intersection(pair_dfs[pair].index)

        fit = scaler is None
        if fit:
            scaler = {}

        norm_data  = {}
        raw_closes = {}
        for pair in cfg.PAIRS:
            aligned = pair_dfs[pair].loc[common_idx]
            raw_closes[pair] = aligned['close'].values.astype(np.float32)
            if fit:
                scaler[pair] = {'min': aligned.min().to_dict(),
                                'max': aligned.max().to_dict()}
            mn = pd.Series(scaler[pair]['min'])
            mx = pd.Series(scaler[pair]['max'])
            norm = (aligned - mn) / (mx - mn + 1e-8)
            norm_data[pair] = norm[cfg.BASE_FEATURES].values.astype(np.float32)

        return norm_data, raw_closes, len(common_idx)


    # ─────────────────────────────────────────────
    # 3. ENVIRONMENT
    # ─────────────────────────────────────────────

    class MultiAssetEnv:
        """
        Multi-asset environment: 3 agents, each trades its own pair.
          Agent 0 → BTC/USD   Agent 1 → ETH/USD   Agent 2 → ETH/BTC

        Local obs  (actor input) : own pair window  (W * N_FEATURES,)
        Global obs (critic input): all pairs concat (W * N_FEATURES * N_AGENTS,)
        Reward per agent         : PnL change on own position
        """

        def __init__(self, norm_data: dict, raw_closes: dict, T: int):
            self.norm_data  = norm_data    # {pair: (T, N_FEATURES)}
            self.raw_closes = raw_closes   # {pair: (T,)}
            self.T          = T
            self.pairs      = cfg.PAIRS
            self.obs_dim    = cfg.WINDOW_SIZE * cfg.N_FEATURES

        def reset(self):
            self.t              = cfg.WINDOW_SIZE
            self.cash           = {p: cfg.INITIAL_CASH for p in self.pairs}
            self.holdings       = {p: 0.0              for p in self.pairs}
            self.prev_portfolio = {p: cfg.INITIAL_CASH for p in self.pairs}
            return self._get_obs()

        def _get_obs(self):
            local_obs = []
            for pair in self.pairs:
                window = self.norm_data[pair][self.t - cfg.WINDOW_SIZE : self.t]
                local_obs.append(window.flatten())
            global_obs = np.concatenate(local_obs)
            return local_obs, global_obs

        def _price(self, pair):
            return max(float(self.raw_closes[pair][self.t]), 1e-8)

        def step(self, actions):
            """
            actions : list of int, one per agent/pair  (0=hold, 1=buy, 2=sell)
            Returns : (local_obs, global_obs), rewards, done
            """
            rewards = []
            for i, pair in enumerate(self.pairs):
                price  = self._price(pair)
                action = actions[i]

                if action == 1:
                    amount = self.cash[pair] * cfg.TRADE_SIZE
                    cost   = amount * (1 + cfg.TX_COST)
                    if self.cash[pair] >= cost:
                        self.holdings[pair] += amount / price
                        self.cash[pair]     -= cost

                elif action == 2:
                    if self.holdings[pair] > 0:
                        qty      = self.holdings[pair] * cfg.TRADE_SIZE
                        proceeds = qty * price * (1 - cfg.TX_COST)
                        self.holdings[pair] -= qty
                        self.cash[pair]     += proceeds

                portfolio = self.cash[pair] + self.holdings[pair] * price
                rewards.append(portfolio - self.prev_portfolio[pair])
                self.prev_portfolio[pair] = portfolio

            self.t += 1
            done = self.t >= self.T - 1

            if done:
                zero = np.zeros(self.obs_dim)
                obs  = ([zero] * cfg.N_AGENTS, np.zeros(self.obs_dim * cfg.N_AGENTS))
            else:
                obs = self._get_obs()

            return obs, rewards, done


    # ─────────────────────────────────────────────
    # 4a. LSTM ACTOR  (one per agent, decentralized)
    # ─────────────────────────────────────────────

    class LSTMActor(nn.Module):
        """
        Decentralized actor: sees only its own observation.
        Input: (batch, seq_len, n_features)
        Output: action logits (batch, n_actions)
        """

        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size  = cfg.N_FEATURES,
                hidden_size = cfg.LSTM_HIDDEN,
                num_layers  = cfg.LSTM_LAYERS,
                batch_first = True,
                dropout     = 0.1 if cfg.LSTM_LAYERS > 1 else 0.0
            )
            self.actor = nn.Sequential(
                nn.Linear(cfg.LSTM_HIDDEN, cfg.FC_HIDDEN),
                nn.Tanh(),
                nn.Linear(cfg.FC_HIDDEN, cfg.N_ACTIONS),
            )
            self._init_weights()

        def _init_weights(self):
            for name, p in self.lstm.named_parameters():
                if 'weight' in name: nn.init.orthogonal_(p)
                elif 'bias' in name: nn.init.zeros_(p)
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
            nn.init.zeros_(self.actor[-1].bias)

        def forward(self, x, hidden=None):
            out, hidden = self.lstm(x, hidden)
            logits = self.actor(out[:, -1, :])
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
            return logits, hidden

        def get_action(self, x, hidden=None):
            with torch.no_grad():
                logits, hidden = self.forward(x, hidden)
            dist   = Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), hidden


    # ─────────────────────────────────────────────
    # 4b. LSTM CENTRALIZED CRITIC  (shared, sees ALL agents)
    # ─────────────────────────────────────────────

    class LSTMCentralCritic(nn.Module):
        """
        Centralized critic (CTDE): during training sees the concatenation of
        ALL agents' observations → single value estimate per agent.

        Input:  global_obs (batch, seq_len, n_features * n_agents)
        Output: values     (batch, n_agents)
        """

        def __init__(self):
            super().__init__()
            global_input = cfg.N_FEATURES * cfg.N_AGENTS
            self.lstm = nn.LSTM(
                input_size  = global_input,
                hidden_size = cfg.LSTM_HIDDEN * 2,      # wider — sees more info
                num_layers  = cfg.LSTM_LAYERS,
                batch_first = True,
                dropout     = 0.1 if cfg.LSTM_LAYERS > 1 else 0.0
            )
            self.critic = nn.Sequential(
                nn.Linear(cfg.LSTM_HIDDEN * 2, cfg.FC_HIDDEN),
                nn.Tanh(),
                nn.Linear(cfg.FC_HIDDEN, cfg.N_AGENTS),  # one value per agent
            )
            self._init_weights()

        def _init_weights(self):
            for name, p in self.lstm.named_parameters():
                if 'weight' in name: nn.init.orthogonal_(p)
                elif 'bias' in name: nn.init.zeros_(p)
            nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
            nn.init.zeros_(self.critic[-1].bias)

        def forward(self, global_obs, hidden=None):
            """
            global_obs: (batch, seq_len, n_features * n_agents)
            returns:    (batch, n_agents)
            """
            out, hidden = self.lstm(global_obs, hidden)
            values = self.critic(out[:, -1, :])
            return values, hidden

        def get_values(self, global_obs, hidden=None):
            """Single-step inference. Returns (n_agents,) numpy array."""
            with torch.no_grad():
                values, hidden = self.forward(global_obs, hidden)
            return values.squeeze(0).cpu().numpy(), hidden



    # ─────────────────────────────────────────────
    # 5. ROLLOUT BUFFER
    # ─────────────────────────────────────────────

    class RolloutBuffer:
        """Stores transitions for one rollout, per agent."""

        def __init__(self):
            self.reset()

        def reset(self):
            self.obs        = []   # local obs (this agent only)
            self.global_obs = []   # all agents' obs concatenated (for central critic)
            self.actions    = []
            self.log_probs  = []
            self.rewards    = []
            self.values     = []   # from central critic
            self.dones      = []

        def add(self, obs, global_obs, action, log_prob, reward, value, done):
            self.obs.append(obs)
            self.global_obs.append(global_obs)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            self.values.append(value)
            self.dones.append(done)

        def compute_returns(self, last_value):
            """GAE advantage + discounted returns."""
            T          = len(self.rewards)
            advantages = np.zeros(T, dtype=np.float32)
            returns    = np.zeros(T, dtype=np.float32)
            gae        = 0.0

            for t in reversed(range(T)):
                next_val  = last_value if t == T - 1 else self.values[t + 1]
                next_done = self.dones[t]
                delta     = self.rewards[t] + cfg.GAMMA * next_val * (1 - next_done) - self.values[t]
                gae       = delta + cfg.GAMMA * cfg.GAE_LAMBDA * (1 - next_done) * gae
                advantages[t] = gae
                returns[t]    = gae + self.values[t]

            return advantages, returns


    # ─────────────────────────────────────────────
    # 6. MAPPO TRAINER
    # ─────────────────────────────────────────────

    class MAPPOTrainer:
        """
        True MAPPO — Centralized Training, Decentralized Execution (CTDE):
          - N decentralized actors  : each sees only its own obs
          - 1 centralized critic    : sees ALL agents' obs, outputs value per agent
        """

        def __init__(self):
            self.actors  = [LSTMActor().to(cfg.DEVICE) for _ in range(cfg.N_AGENTS)]
            self.critic  = LSTMCentralCritic().to(cfg.DEVICE)

            self.actor_optims = [optim.Adam(a.parameters(), lr=cfg.LR) for a in self.actors]
            self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.LR)

        def update(self, buffers: list, last_values: list):
            """
            buffers     : list of RolloutBuffer, one per agent
            last_values : list of float, bootstrap value per agent
            """
            # ── Compute advantages per agent ──────────────────────
            all_adv, all_ret = [], []
            for i, buf in enumerate(buffers):
                adv, ret = buf.compute_returns(last_values[i])
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ret = (ret - ret.mean()) / (ret.std() + 1e-8)
                all_adv.append(adv)
                all_ret.append(ret)

            # ── Build tensors ─────────────────────────────────────
            # Local obs per agent: (T, W, F)
            local_obs = [
                torch.tensor(np.array(buf.obs), dtype=torch.float32)
                  .to(cfg.DEVICE)
                  .view(-1, cfg.WINDOW_SIZE, cfg.N_FEATURES)
                for buf in buffers
            ]
            # Global obs for critic: (T, W, F*N)
            global_obs = torch.tensor(
                np.array(buffers[0].global_obs), dtype=torch.float32
            ).to(cfg.DEVICE).view(-1, cfg.WINDOW_SIZE, cfg.N_FEATURES * cfg.N_AGENTS)

            actions  = [torch.tensor(buf.actions,    dtype=torch.long).to(cfg.DEVICE)  for buf in buffers]
            old_lps  = [torch.clamp(
                            torch.tensor(buf.log_probs, dtype=torch.float32).to(cfg.DEVICE),
                            -10.0, 0.0)
                        for buf in buffers]
            adv_ts   = [torch.tensor(a, dtype=torch.float32).to(cfg.DEVICE) for a in all_adv]
            ret_ts   = [torch.tensor(r, dtype=torch.float32).to(cfg.DEVICE) for r in all_ret]

            T = global_obs.shape[0]

            for _ in range(cfg.EPOCHS):
                idx = torch.randperm(T)
                for start in range(0, T, cfg.BATCH_SIZE):
                    b = idx[start : start + cfg.BATCH_SIZE]

                    # ── Centralized critic update ──────────────────
                    values_all, _ = self.critic(global_obs[b])   # (B, N_AGENTS)
                    critic_loss   = sum(
                        nn.functional.mse_loss(values_all[:, i], ret_ts[i][b])
                        for i in range(cfg.N_AGENTS)
                    )
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.critic_optim.step()

                    # ── Decentralized actor updates ────────────────
                    for i in range(cfg.N_AGENTS):
                        logits, _ = self.actors[i](local_obs[i][b])
                        dist      = Categorical(logits=logits)
                        lp        = dist.log_prob(actions[i][b])
                        entropy   = dist.entropy().mean()

                        ratio     = torch.clamp((lp - old_lps[i][b]).exp(), 0.0, 10.0)
                        clip      = torch.clamp(ratio, 1 - cfg.CLIP_EPS, 1 + cfg.CLIP_EPS)
                        pol_loss  = -torch.min(ratio * adv_ts[i][b],
                                               clip  * adv_ts[i][b]).mean()

                        loss = pol_loss - cfg.ENTROPY_COEF * entropy
                        if torch.isnan(loss):
                            continue

                        self.actor_optims[i].zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
                        self.actor_optims[i].step()

        def save(self, path='mappo_lstm.pt'):
            torch.save({
                **{f'actor_{i}': self.actors[i].state_dict() for i in range(cfg.N_AGENTS)},
                'critic': self.critic.state_dict()
            }, path)
            print(f'Saved → {path}')

        def load(self, path='mappo_lstm.pt'):
            ckpt = torch.load(path, map_location=cfg.DEVICE)
            for i in range(cfg.N_AGENTS):
                self.actors[i].load_state_dict(ckpt[f'actor_{i}'])
            self.critic.load_state_dict(ckpt['critic'])
            self.scaler = ckpt.get('scaler', None)   # None for legacy checkpoints
            print(f'Loaded ← {path}')
            return self.scaler

    return MAPPOTrainer, MultiAssetEnv, cfg, preprocess


@app.cell
def _(
    CryptoBarsRequest,
    CryptoHistoricalDataClient,
    MAPPOTrainer,
    MultiAssetEnv,
    TimeFrame,
    ZoneInfo,
    add_indicators,
    cfg,
    datetime,
    pd,
    preprocess,
    timedelta,
    torch,
):
    # ─────────────────────────────────────────────────────────────────
    # inference_live.py
    # Load mappo_lstm_best.pt, fetch current Alpaca data,
    # run decentralized inference, save results + HTML dashboard
    # ─────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────
    # 1. FETCH + ENGINEER FEATURES
    # ────────────────────────────────────────────

    client_latest = CryptoHistoricalDataClient()

    # Set the timezone
    timezone = ZoneInfo('America/New_York')

    # Get current date in US/Eastern timezone
    today = datetime.now(timezone).date()
    start_date = today - timedelta(days=100)

    request_params_latest = CryptoBarsRequest(
        symbol_or_symbols=["BTC/USD", "ETH/USD", "ETH/BTC"],
        timeframe=TimeFrame.Day,
        start=start_date,
        end=today
    )

    bars_latest = client_latest.get_crypto_bars(request_params_latest)

    df_latest = bars_latest.df

    btc_latest = df_latest.xs('BTC/USD', level='symbol').add_prefix('btc_')
    eth_latest = df_latest.xs('ETH/USD', level='symbol').add_prefix('eth_')
    etb_latest = df_latest.xs('ETH/BTC', level='symbol').add_prefix('etb_')

    df_multi_latest = (
        btc_latest.join(eth_latest, how='inner')
           .join(etb_latest, how='inner')
    )

    for prefix_latest in ['btc', 'eth', 'etb']:
        df_multi_tec_latest = add_indicators(df_multi_latest, prefix_latest)

    df_multi_tec_latest = df_multi_tec_latest.dropna()

    # ─────────────────────────────────────────────
    # 2. INFERENCE
    # ─────────────────────────────────────────────

    ACTION_MAP   = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    ACTION_COLOR = {'HOLD': '#888787', 'BUY': '#1D9E75', 'SELL': '#D85A30'}


    def run_inference(df: pd.DataFrame,
                      checkpoint: str = 'mappo_lstm_best.pt') -> pd.DataFrame:
        """
        Load checkpoint, run decentralized inference over df.
        Returns a DataFrame with date, per-pair price, action, and a running
        portfolio value per agent.

        One decision per bar, each from its OWN window [t-W, t) evaluated
        statelessly (matches how the model is trained) and executed at close[t].
        Actions are chosen with argmax — deterministic, so the same data always
        yields the same live signal. Every bar in [W, T) is scored, including the
        final bar (today), which is the one live execution acts on.
        """
        trainer = MAPPOTrainer()
        trainer.load(checkpoint)
        for actor in trainer.actors:
            actor.eval()

        # Transform with the training scaler persisted in the checkpoint.
        norm_data, raw_closes, T = preprocess(df, getattr(trainer, 'scaler', None))
        W = cfg.WINDOW_SIZE

        # The aligned index equals df.index here (all pairs are inner-joined and
        # jointly dropna'd upstream), so df.index[t] lines up with raw_closes[t].
        dates = df.index

        records  = []
        cash     = {p: cfg.INITIAL_CASH for p in cfg.PAIRS}
        holdings = {p: 0.0              for p in cfg.PAIRS}

        for t in range(W, T):
            row = {'date': dates[t]}
            for i, pair in enumerate(cfg.PAIRS):
                window = norm_data[pair][t - W:t]
                obs_t  = torch.tensor(
                    window, dtype=torch.float32
                ).to(cfg.DEVICE).view(1, W, cfg.N_FEATURES)
                with torch.no_grad():
                    logits, _ = trainer.actors[i](obs_t)        # stateless
                action_idx = int(logits.argmax(dim=-1).item())  # deterministic

                price = float(raw_closes[pair][t])
                qty   = 0.0
                if action_idx == 1:        # BUY
                    amount = cash[pair] * cfg.TRADE_SIZE
                    cost   = amount * (1 + cfg.TX_COST)
                    if cash[pair] >= cost:
                        qty = amount / price
                        holdings[pair] += qty
                        cash[pair]     -= cost
                elif action_idx == 2:      # SELL
                    if holdings[pair] > 0:
                        qty      = holdings[pair] * cfg.TRADE_SIZE   # positive
                        proceeds = qty * price * (1 - cfg.TX_COST)
                        holdings[pair] -= qty
                        cash[pair]     += proceeds

                row[f'{pair}_price']     = price
                row[f'{pair}_action']    = ACTION_MAP[action_idx]
                row[f'{pair}_quantity']  = qty
                row[f'{pair}_portfolio'] = cash[pair] + holdings[pair] * price
            records.append(row)

        results = pd.DataFrame(records)
        results['date'] = pd.to_datetime(results['date'])
        return results

    # ─────────────────────────────────────────────
    # 3. ENTRY POINT
    # ─────────────────────────────────────────────

    if __name__ == '__main__':
        # Run inference with best checkpoint on latest data
        results = run_inference(df_multi_tec_latest, checkpoint='mappo_lstm_best.pt')

        cols = ['date'] + [f'{p}_{s}' for p in cfg.PAIRS
                       for s in ['price', 'action', 'quantity', 'portfolio']]
        print(results[cols].to_string(index=False))
    return (results,)


@app.cell
def _(go, make_subplots, mo, pd, results):
    def build_dashboard(results: pd.DataFrame) -> mo.Html:
        results = results.copy()
        results['date'] = pd.to_datetime(results['date'])

        PAIRS     = [col[:-6] for col in results.columns if col.endswith('_price')]
        PAIR_DISPLAY = {'btc': 'BTC/USD', 'eth': 'ETH/USD', 'etb': 'ETH/BTC'}
        PAIR_COLORS  = {'btc': '#4f8ef7', 'eth': '#a78bfa', 'etb': '#f7c948'}
        ACTION_COLOR = {'HOLD': '#888787', 'BUY': '#1D9E75', 'SELL': '#D85A30'}

        DARK = dict(
            paper_bgcolor='#0a0b0e',
            plot_bgcolor='#111318',
            font=dict(family='DM Mono, monospace', color='#e8eaf0', size=11),
            xaxis=dict(gridcolor='#1e2130', zerolinecolor='#1e2130'),
            yaxis=dict(gridcolor='#1e2130', zerolinecolor='#1e2130'),
            margin=dict(l=50, r=20, t=40, b=40),
        )

        tabs = {}

        for pair in PAIRS:
            label  = PAIR_DISPLAY.get(pair, pair.upper())
            color  = PAIR_COLORS.get(pair, '#ffffff')
            prices = results[f'{pair}_price']
            pf     = results[f'{pair}_portfolio']
            acts   = results[f'{pair}_action']
            dates  = results['date']

            # ── split signals ──────────────────────────────────────
            buy_dates  = dates[acts == 'BUY']
            buy_prices = prices[acts == 'BUY']
            sell_dates  = dates[acts == 'SELL']
            sell_prices = prices[acts == 'SELL']

            # ── fig 1 : price + signals ────────────────────────────
            fig1 = make_subplots(rows=1, cols=1)

            fig1.add_trace(go.Scatter(
                x=dates, y=prices,
                mode='lines',
                name='Price',
                line=dict(color=color, width=1.5),
                hovertemplate='%{x|%b %d}<br>$%{y:,.4f}<extra></extra>',
            ))
            fig1.add_trace(go.Scatter(
                x=buy_dates, y=buy_prices,
                mode='markers',
                name='BUY',
                marker=dict(
                    symbol='triangle-up', size=10,
                    color='#1D9E75', line=dict(width=0)
                ),
                hovertemplate='BUY<br>%{x|%b %d}<br>$%{y:,.4f}<extra></extra>',
            ))
            fig1.add_trace(go.Scatter(
                x=sell_dates, y=sell_prices,
                mode='markers',
                name='SELL',
                marker=dict(
                    symbol='triangle-down', size=10,
                    color='#D85A30', line=dict(width=0)
                ),
                hovertemplate='SELL<br>%{x|%b %d}<br>$%{y:,.4f}<extra></extra>',
            ))

            fig1.update_layout(
                **DARK,
                title=dict(text=f'{label} — Price & Signals', font=dict(size=13), x=0),
                legend=dict(
                    orientation='h', y=1.08, x=0,
                    font=dict(size=10), bgcolor='rgba(0,0,0,0)',
                ),
                height=320,
            )

            # ── fig 2 : portfolio value ────────────────────────────
            pct_change = (pf.iloc[-1] - pf.iloc[0]) / pf.iloc[0] * 100
            pf_color   = '#1D9E75' if pct_change >= 0 else '#D85A30'

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=dates, y=pf,
                mode='lines',
                name='Portfolio',
                line=dict(color=pf_color, width=1.5),
                fill='tozeroy',
                fillcolor=f"rgba({int(pf_color[1:3],16)},{int(pf_color[3:5],16)},{int(pf_color[5:7],16)},0.094)",
                hovertemplate='%{x|%b %d}<br>$%{y:,.2f}<extra></extra>',
            ))
            fig2.update_layout(
                **DARK,
                title=dict(text=f'{label} — Portfolio Value', font=dict(size=13), x=0),
                height=260,
                showlegend=False,
            )

            # ── stat cards ────────────────────────────────────────
            last_price = prices.iloc[-1]
            last_pf    = pf.iloc[-1]
            last_act   = acts.iloc[-1]
            sign       = '+' if pct_change >= 0 else ''
            act_col    = ACTION_COLOR[last_act]

            stats_html = f"""
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px;">
              <div style="background:#111318;border:1px solid #1e2130;border-radius:6px;padding:14px 16px;">
                <div style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#5a6080;margin-bottom:6px;">Last Price</div>
                <div style="font-size:22px;font-family:'DM Serif Display',serif;color:#e8eaf0;">${last_price:,.4f}</div>
              </div>
              <div style="background:#111318;border:1px solid #1e2130;border-radius:6px;padding:14px 16px;">
                <div style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#5a6080;margin-bottom:6px;">Portfolio</div>
                <div style="font-size:22px;font-family:'DM Serif Display',serif;color:#e8eaf0;">${last_pf:,.2f}</div>
                <div style="font-size:10px;color:{pf_color};margin-top:2px;">{sign}{pct_change:.2f}% all-time</div>
              </div>
              <div style="background:#111318;border:1px solid #1e2130;border-radius:6px;padding:14px 16px;">
                <div style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#5a6080;margin-bottom:6px;">Last Signal</div>
                <div style="font-size:22px;font-family:'DM Serif Display',serif;color:{act_col};">{last_act}</div>
              </div>
            </div>
            """

            # ── signal table (last 15 rows) ───────────────────────
            recent = results[['date', f'{pair}_price', f'{pair}_action', f'{pair}_portfolio']].tail(15)
            rows = ""
            for _, r in recent.iterrows():
                a    = r[f'{pair}_action']
                acol = ACTION_COLOR[a]
                rows += f"""
                <tr style="border-bottom:1px solid #15171f;">
                  <td style="padding:7px 10px;">{r['date'].strftime('%b %d')}</td>
                  <td style="padding:7px 10px;">${r[f'{pair}_price']:,.4f}</td>
                  <td style="padding:7px 10px;">
                    <span style="display:inline-block;padding:1px 8px;border-radius:3px;
                      font-size:10px;font-weight:500;letter-spacing:.08em;
                      background:{'#0f3327' if a=='BUY' else '#3a1a0d' if a=='SELL' else '#1a1c24'};
                      color:{acol};">{a}</span>
                  </td>
                  <td style="padding:7px 10px;">${r[f'{pair}_portfolio']:,.2f}</td>
                </tr>"""

            table_html = f"""
            <div style="background:#111318;border:1px solid #1e2130;border-radius:6px;padding:20px;margin-top:16px;">
              <div style="font-size:10px;font-weight:500;letter-spacing:.12em;text-transform:uppercase;
                   color:#5a6080;margin-bottom:14px;">Recent Signals</div>
              <div style="overflow-x:auto;">
                <table style="width:100%;border-collapse:collapse;font-size:12px;font-family:'DM Mono',monospace;">
                  <thead>
                    <tr>
                      <th style="text-align:left;font-size:10px;letter-spacing:.1em;text-transform:uppercase;
                          color:#5a6080;padding:6px 10px;border-bottom:1px solid #1e2130;">Date</th>
                      <th style="text-align:left;font-size:10px;letter-spacing:.1em;text-transform:uppercase;
                          color:#5a6080;padding:6px 10px;border-bottom:1px solid #1e2130;">Price</th>
                      <th style="text-align:left;font-size:10px;letter-spacing:.1em;text-transform:uppercase;
                          color:#5a6080;padding:6px 10px;border-bottom:1px solid #1e2130;">Action</th>
                      <th style="text-align:left;font-size:10px;letter-spacing:.1em;text-transform:uppercase;
                          color:#5a6080;padding:6px 10px;border-bottom:1px solid #1e2130;">Portfolio</th>
                    </tr>
                  </thead>
                  <tbody style="color:#e8eaf0;">{rows}</tbody>
                </table>
              </div>
            </div>
            """

            tabs[label] = mo.vstack([
                mo.Html(stats_html),
                mo.ui.plotly(fig1),
                mo.ui.plotly(fig2),
                mo.Html(table_html),
            ], gap='0')

        # ── header ────────────────────────────────────────────────
        last_date = results['date'].iloc[-1].strftime('%b %d, %Y')
        header = mo.Html(f"""
        <div style="display:flex;align-items:baseline;gap:16px;margin-bottom:24px;
             border-bottom:1px solid #1e2130;padding-bottom:16px;">
          <h1 style="font-family:'DM Serif Display',serif;font-size:26px;
               letter-spacing:.02em;color:#e8eaf0;">MAPPO · LSTM</h1>
          <span style="font-family:'DM Mono',monospace;font-size:10px;font-weight:500;
                letter-spacing:.12em;text-transform:uppercase;background:#4f8ef7;
                color:#0a0b0e;padding:2px 8px;border-radius:3px;">Live Inference</span>
          <span style="margin-left:auto;color:#5a6080;font-size:11px;
                font-family:'DM Mono',monospace;">as of {last_date}</span>
        </div>
        """)

        return mo.vstack([
            header,
            mo.ui.tabs(tabs),
        ], gap='0')


    # ── Marimo cell ───────────────────────────────────────────────────
    dashboard = build_dashboard(results)
    dashboard
    return


@app.cell
def _(
    AssetClass,
    GetOrdersRequest,
    MarketOrderRequest,
    OrderSide,
    OrderType,
    QueryOrderStatus,
    TimeInForce,
    TradingClient,
    date,
    os,
    pd,
    results,
):
    ##############################################################################
    # CONFIG
    ##############################################################################
    # Credentials from .env (loaded in the import cell). No hardcoded secrets.
    _api_key    = os.getenv('ALPACA_API_KEY')
    _api_secret = os.getenv('ALPACA_API_SECRET')
    if not _api_key or not _api_secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Copy .env.example to .env and set "
            "ALPACA_API_KEY and ALPACA_API_SECRET."
        )
    trade_client = TradingClient(
        api_key=_api_key,
        secret_key=_api_secret,
        paper=True
    )


    ##############################################################################
    # POSITION HELPERS
    ##############################################################################

    def get_position_qty(symbol: str) -> float:
        """
        Returns current quantity owned.
        Returns 0 if no position exists.
        """

        try:
            position = trade_client.get_open_position(symbol)
            return float(position.qty)

        except Exception:
            return 0.0


    def print_positions():

        print("\nCURRENT POSITIONS")

        try:
            positions = trade_client.get_all_positions()

            if len(positions) == 0:
                print("No open positions")

            for p in positions:
                print(
                    f"{p.symbol:<10}"
                    f" Qty={p.qty:<15}"
                    f" MV={p.market_value}"
                )

        except Exception as e:
            print("Position Error:", e)


    ##############################################################################
    # ORDER HELPERS
    ##############################################################################

    def submit_buy(symbol: str, qty: float):

        if qty <= 0:
            return

        try:

            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
            )

            order = trade_client.submit_order(req)

        except Exception as e:
            print(f"BUY ERROR [{symbol}] -> {e}")


    def submit_sell(symbol: str, qty: float):

        owned_qty = get_position_qty(symbol)

        if owned_qty <= 0:
            print(f"No position in {symbol}. Sell skipped.")
            return

        qty_to_sell = min(qty, owned_qty)

        try:

            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty_to_sell,
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
            )

            order = trade_client.submit_order(req)

            print(
                f"SELL submitted:"
                f" {symbol}"
                f" qty={qty_to_sell}"
                f" order_id={order.id}"
            )

        except Exception as e:
            print(f"SELL ERROR [{symbol}] -> {e}")


    ##############################################################################
    # SIGNAL EXECUTION
    ##############################################################################

    def execute_signal(
        symbol: str,
        action: str,
        quantity: float
    ):

        action = str(action).upper()

        if action == "BUY":

            submit_buy(
                symbol=symbol,
                qty=quantity
            )

        elif action == "SELL":

            submit_sell(
                symbol=symbol,
                qty=quantity
            )

        else:

            print(
                f"{symbol}: HOLD "
                f"(signal={action})"
            )


    ##############################################################################
    # MAIN STRATEGY EXECUTION
    ##############################################################################

    def execute_today_signals(results: pd.DataFrame):

        today = date.today()

        results["date"] = pd.to_datetime(
            results["date"]
        ).dt.date

        today_row = results[
            results["date"] == today
        ]

        if today_row.empty:
            raise ValueError(
                f"No signals found for {today}"
            )

        row = today_row.iloc[0]

        execute_signal(
            symbol="BTC/USD",
            action=row["btc_action"],
            quantity=float(row["btc_quantity"])
        )

        execute_signal(
            symbol="ETH/USD",
            action=row["eth_action"],
            quantity=float(row["eth_quantity"])
        )

        execute_signal(
            symbol="ETH/BTC",
            action=row["etb_action"],
            quantity=float(row["etb_quantity"])
        )


    ##############################################################################
    # ORDER HISTORY
    ##############################################################################

    def print_crypto_order_history():

        req = GetOrdersRequest(
            status=QueryOrderStatus.ALL
        )

        orders = trade_client.get_orders(req)

        print("\nCRYPTO ORDER HISTORY")

        for o in orders:

            if o.asset_class != AssetClass.CRYPTO:
                continue

            print(
                f"ID={o.id} | "
                f"Symbol={o.symbol} | "
                f"Side={o.side} | "
                f"Qty={o.qty} | "
                f"Status={o.status}"
            )


    ##############################################################################
    # NOTE: This cell only DEFINES helpers + trade_client. Order submission and
    # display happen exactly once, in the dashboard cell below (render_execution_log
    # / render_positions / render_order_history). Executing here too would submit
    # every signal twice.
    ##############################################################################
    return execute_signal, trade_client


@app.cell
def _(mo):
    # Explicit guard: today's orders are submitted only when this button is
    # clicked, never on an incidental reactive re-run of the notebook.
    exec_button = mo.ui.run_button(label="⚡ Execute Today's Signals")
    exec_button
    return (exec_button,)


@app.cell
def _(
    AssetClass,
    GetOrdersRequest,
    QueryOrderStatus,
    date,
    exec_button,
    execute_signal,
    mo,
    pd,
    results,
    trade_client,
):
    import agent_diag.record as _agent_record  # hub assistant run record

    ##############################################################################
    # DISPLAY HELPERS
    ##############################################################################

    def _th(label, width=""):
        w = f"width:{width};" if width else ""
        return f'<th style="text-align:left;{w}font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#5a6080;padding:6px 10px;border-bottom:1px solid #1e2130;">{label}</th>'

    def _td(content, color="#e8eaf0", align="left"):
        return f'<td style="padding:7px 10px;text-align:{align};color:{color};">{content}</td>'


    def render_positions() -> mo.Html:
        try:
            positions = trade_client.get_all_positions()
            if not positions:
                rows = '<tr><td colspan="3" style="padding:12px 10px;color:#5a6080;">No open positions</td></tr>'
            else:
                rows = ""
                for p in positions:
                    mv = float(p.market_value)
                    mv_color = "#1D9E75" if mv >= 0 else "#D85A30"
                    rows += f"<tr style='border-bottom:1px solid #15171f;'>{_td(p.symbol)}{_td(p.qty)}{_td(f'${mv:,.2f}', color=mv_color, align='right')}</tr>"
        except Exception as e:
            rows = f'<tr><td colspan="3" style="padding:12px 10px;color:#D85A30;">Error: {e}</td></tr>'

        headers = f"<tr>{_th('Symbol','20%')}{_th('Qty','40%')}{_th('Market Value','40%')}</tr>"

        return mo.Html(f"""
        <div style="background:#111318;border:1px solid #1e2130;border-radius:6px;padding:20px;margin-bottom:16px;">
          <div style="font-size:10px;font-weight:500;letter-spacing:.12em;text-transform:uppercase;color:#5a6080;margin-bottom:14px;">Current Positions</div>
          <table style="width:100%;border-collapse:collapse;font-size:12px;font-family:'DM Mono',monospace;table-layout:fixed;">
            <thead>{headers}</thead>
            <tbody style="color:#e8eaf0;">{rows}</tbody>
          </table>
        </div>
        """)


    def render_order_history() -> mo.Html:
        ACTION_COLOR = {"buy": "#1D9E75", "sell": "#D85A30"}
        STATUS_BG    = {"filled": "#0f3327", "canceled": "#3a1a0d", "new": "#1a1c24", "partially_filled": "#1a1c24"}

        try:
            orders = trade_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL))
            crypto_orders = [o for o in orders if o.asset_class == AssetClass.CRYPTO]
        except Exception as e:
            return mo.Html(f'<div style="color:#D85A30;font-family:DM Mono,monospace;font-size:12px;">Order fetch error: {e}</div>')

        if not crypto_orders:
            rows = '<tr><td colspan="5" style="padding:12px 10px;color:#5a6080;">No crypto orders found</td></tr>'
        else:
            rows = ""
            for o in crypto_orders:
                side      = str(o.side.value).lower()
                status    = str(o.status.value).lower()
                side_col  = ACTION_COLOR.get(side, "#e8eaf0")
                status_bg = STATUS_BG.get(status, "#1a1c24")
                side_badge   = f'<span style="display:inline-block;padding:1px 8px;border-radius:3px;font-size:10px;font-weight:500;letter-spacing:.08em;background:{"#0f3327" if side=="buy" else "#3a1a0d"};color:{side_col};">{side.upper()}</span>'
                status_badge = f'<span style="display:inline-block;padding:1px 8px;border-radius:3px;font-size:10px;background:{status_bg};color:#e8eaf0;">{status}</span>'
                rows += f"""<tr style="border-bottom:1px solid #15171f;">
                  {_td(str(o.id)[:8]+'…', color='#5a6080')}
                  {_td(o.symbol)}
                  {_td(side_badge)}
                  {_td(o.qty, align='right')}
                  {_td(status_badge, align='right')}
                </tr>"""

        headers = f"<tr>{_th('Order ID','15%')}{_th('Symbol','20%')}{_th('Side','15%')}{_th('Qty','25%')}{_th('Status','25%')}</tr>"

        return mo.Html(f"""
        <div style="background:#111318;border:1px solid #1e2130;border-radius:6px;padding:20px;">
          <div style="font-size:10px;font-weight:500;letter-spacing:.12em;text-transform:uppercase;color:#5a6080;margin-bottom:14px;">Crypto Order History</div>
          <table style="width:100%;border-collapse:collapse;font-size:12px;font-family:'DM Mono',monospace;table-layout:fixed;">
            <thead>{headers}</thead>
            <tbody style="color:#e8eaf0;">{rows}</tbody>
          </table>
        </div>
        """)


    def render_execution_log(results: pd.DataFrame) -> mo.Html:
        """Runs execute_today_signals and captures per-signal status as HTML."""
        today = date.today()
        results["date"] = pd.to_datetime(results["date"]).dt.date
        today_row = results[results["date"] == today]

        if today_row.empty:
            return mo.Html(f'<div style="color:#D85A30;font-family:DM Mono,monospace;font-size:12px;">No signals for {today}</div>')

        row       = today_row.iloc[0]
        ACTION_COLOR = {"BUY": "#1D9E75", "SELL": "#D85A30", "HOLD": "#888787"}
        PAIR_DISPLAY = {"btc": "BTC/USD", "eth": "ETH/USD", "etb": "ETH/BTC"}

        # Outcome per pair, handed to the hub assistant's run record below.
        exec_rows = []

        cards = ""
        for pair, symbol in [("btc","BTC/USD"), ("eth","ETH/USD"), ("etb","ETH/BTC")]:
            action = str(row[f"{pair}_action"]).upper()
            qty    = float(row[f"{pair}_quantity"])
            color  = ACTION_COLOR.get(action, "#e8eaf0")
            label  = PAIR_DISPLAY[pair]

            leg = {"symbol": symbol, "action": action, "quantity": qty, "outcome": "unknown"}
            exec_rows.append(leg)
            try:
                execute_signal(symbol=symbol, action=action, quantity=qty)
                leg["outcome"] = "submitted" if action in ("BUY", "SELL") else "skipped: hold"
                status_html = '<span style="color:#1D9E75;">✓ submitted</span>'
            except Exception as e:
                leg["outcome"] = "failed"
                status_html = f'<span style="color:#D85A30;">✗ {e}</span>'

            cards += f"""
            <div style="background:#111318;border:1px solid #1e2130;border-radius:6px;padding:14px 16px;">
              <div style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#5a6080;margin-bottom:6px;">{label}</div>
              <div style="display:flex;align-items:baseline;gap:10px;">
                <span style="font-size:20px;font-family:'DM Serif Display',serif;color:{color};">{action}</span>
                <span style="font-size:12px;font-family:'DM Mono',monospace;color:#e8eaf0;">qty {qty}</span>
              </div>
              <div style="font-size:11px;font-family:'DM Mono',monospace;margin-top:6px;">{status_html}</div>
            </div>"""

        # Hub assistant run record: this bar's signals and what each submission
        # did. Runs after every order above and never raises.
        _agent_record.record_crypto(results, exec_rows, executed=True)

        return mo.Html(f"""
        <div style="background:#0a0b0e;border:1px solid #1e2130;border-radius:6px;padding:20px;margin-bottom:16px;">
          <div style="font-size:10px;font-weight:500;letter-spacing:.12em;text-transform:uppercase;
               color:#5a6080;margin-bottom:14px;">Signal Execution · {today}</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">{cards}</div>
        </div>
        """)


    ##############################################################################
    # MARIMO CELL
    ##############################################################################

    # Positions + order history are read-only and always shown. Order submission
    # (render_execution_log) runs ONLY when the button has been clicked.
    if exec_button.value:
        exec_section = render_execution_log(results)
    else:
        # Signals only: recorded before anything is submitted, and updated by
        # render_execution_log once the button is pressed.
        _agent_record.record_crypto(results, None, executed=False)
        exec_section = mo.callout(
            mo.md("Press **⚡ Execute Today's Signals** above to submit today's orders."),
            kind="info",
        )

    execution_view = mo.vstack([
        exec_section,
        render_positions(),
        render_order_history(),
    ], gap='0')

    execution_view
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
