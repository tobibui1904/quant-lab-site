# Import third-party modules
# Import standard library modules
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Import third-party modules
import numpy as np
import pandas as pd

# Import Alpaca modules
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import (
    CryptoBarsRequest
)
from alpaca.data.timeframe import TimeFrame

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from dateutil.relativedelta import relativedelta

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
    INITIAL_CASH = 10_000.0                 # per agent, in USD
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


def set_seed(seed: int = 42):
    """Seed all RNGs so training and action sampling are reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame, scaler: dict = None):
    """
    Expects columns prefixed by pair: btc_close, eth_ADX_14, etc.

    Normalization: if `scaler` is None the min/max are FIT on this df (training);
    otherwise the provided scaler is used to TRANSFORM (inference/validation) so
    the model always sees the same scaling it was trained under. Recomputing
    min/max per window would shift the input distribution between train and
    inference and leak look-ahead into historical bars.

    Returns:
      norm_data  : dict {pair: np.ndarray (T, N_FEATURES)}  normalized obs per agent
      raw_closes : dict {pair: np.ndarray (T,)}             original prices for PnL
      T          : int                                      number of aligned bars
      scaler     : dict {pair: {'min': {feat: val}, 'max': {feat: val}}}
    All pairs are inner-joined on index so T is consistent.
    """
    # Build per-pair DataFrames
    pair_dfs = {}
    for pair in cfg.PAIRS:
        cols = {f'{pair}_{f}': f for f in cfg.BASE_FEATURES}
        pair_dfs[pair] = df[[c for c in cols]].rename(columns=cols).dropna()

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
            # Store as plain float dicts so the scaler is JSON/checkpoint safe.
            scaler[pair] = {'min': aligned.min().to_dict(),
                            'max': aligned.max().to_dict()}
        mn = pd.Series(scaler[pair]['min'])
        mx = pd.Series(scaler[pair]['max'])
        norm = (aligned - mn) / (mx - mn + 1e-8)
        norm_data[pair] = norm[cfg.BASE_FEATURES].values.astype(np.float32)

    return norm_data, raw_closes, len(common_idx), scaler

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
            # Reward as a FRACTION of starting capital so rewards, values, and
            # returns share one ~unit scale. With raw-dollar rewards the critic
            # (trained on ~unit-scale returns) is dwarfed in the GAE delta and
            # provides no baseline.
            rewards.append((portfolio - self.prev_portfolio[pair]) / cfg.INITIAL_CASH)
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
            # Normalize advantages only (standard PPO). The critic target `ret`
            # is left on its true (fractional-return) scale so the learned value
            # matches the rewards used in GAE and actually reduces variance.
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
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

    def save(self, path='mappo_lstm.pt', scaler=None):
        # Persist the normalization scaler alongside the weights so inference
        # transforms inputs with the exact stats seen during training.
        torch.save({
            **{f'actor_{i}': self.actors[i].state_dict() for i in range(cfg.N_AGENTS)},
            'critic': self.critic.state_dict(),
            'scaler': scaler if scaler is not None else getattr(self, 'scaler', None),
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


# ─────────────────────────────────────────────
# 7. TRAINING LOOP
# ─────────────────────────────────────────────

def evaluate(df: pd.DataFrame, trainer, scaler) -> float:
    """
    Greedy (argmax) rollout on `df` using the fitted `scaler`. Returns the mean
    per-agent portfolio return (fraction of starting capital). Used for honest,
    validation-based checkpoint selection — never trains on this data.
    """
    norm_data, raw_closes, T, _ = preprocess(df, scaler)
    env = MultiAssetEnv(norm_data, raw_closes, T)
    (local_obs, _) = env.reset()

    for actor in trainer.actors:
        actor.eval()

    while True:
        actions = []
        for i in range(cfg.N_AGENTS):
            obs_t = torch.tensor(local_obs[i], dtype=torch.float32).to(cfg.DEVICE)
            obs_t = obs_t.view(1, cfg.WINDOW_SIZE, cfg.N_FEATURES)
            with torch.no_grad():
                logits, _ = trainer.actors[i](obs_t)
            actions.append(int(logits.argmax(dim=-1).item()))
        (local_obs, _), _, done = env.step(actions)
        if done:
            break

    rets = [(env.cash[p] + env.holdings[p] * env._price(p)) / cfg.INITIAL_CASH - 1.0
            for p in cfg.PAIRS]

    for actor in trainer.actors:
        actor.train()   # restore training mode for the next episode

    return float(np.mean(rets))


def train(df: pd.DataFrame, n_episodes: int = 50, val_df: pd.DataFrame = None,
          save_path: str = 'mappo_lstm.pt', best_path: str = 'mappo_lstm_best.pt'):
    set_seed()
    norm_data, raw_closes, T, scaler = preprocess(df)   # fit scaler on training data
    env     = MultiAssetEnv(norm_data, raw_closes, T)
    trainer = MAPPOTrainer()
    trainer.scaler = scaler
    buffers = [RolloutBuffer() for _ in range(cfg.N_AGENTS)]

    history = {'ep': [], 'avg_reward': [], 'rolling_avg': []}
    for i, pair in enumerate(cfg.PAIRS):
        history[f'{pair}_reward'] = []
        history[f'{pair}_actions'] = []

    best_avg      = -np.inf
    best_episode  = 0
    reward_window = deque(maxlen=10)

    print(f"Training on {cfg.DEVICE} | {n_episodes} episodes | {cfg.N_AGENTS} agents")
    print(f"Pairs: {' | '.join(p.upper() for p in cfg.PAIRS)}\n")
    header = f"{'EP':>5} {'AVG':>10} {'ROLL10':>10} {'BEST':>10}  " +              "  ".join(f"{p.upper():>12}" for p in cfg.PAIRS)
    print(header)
    print("─" * len(header))

    for ep in range(1, n_episodes + 1):
        (local_obs, global_obs) = env.reset()
        ep_rewards    = [0.0] * cfg.N_AGENTS
        action_counts = [[0, 0, 0] for _ in range(cfg.N_AGENTS)]
        [b.reset() for b in buffers]
        step = 0

        while True:
            actions, log_probs = [], []

            # Decentralized actors — each sees only its own pair.
            # Stateless: every 20-step window is evaluated from a fresh hidden
            # state, exactly as the PPO update re-evaluates it (hidden=None on
            # shuffled minibatches). Carrying hidden here would make the stored
            # log-probs/values inconsistent with the update and corrupt the ratio.
            for i in range(cfg.N_AGENTS):
                obs_t = torch.tensor(local_obs[i], dtype=torch.float32).to(cfg.DEVICE)
                obs_t = obs_t.view(1, cfg.WINDOW_SIZE, cfg.N_FEATURES)
                a, lp, _ = trainer.actors[i].get_action(obs_t, None)
                actions.append(a)
                log_probs.append(lp)
                action_counts[i][a] += 1

            # Centralized critic — sees all pairs (also stateless)
            global_t = torch.tensor(global_obs, dtype=torch.float32).to(cfg.DEVICE)
            global_t = global_t.view(1, cfg.WINDOW_SIZE, cfg.N_FEATURES * cfg.N_AGENTS)
            values, _ = trainer.critic.get_values(global_t, None)

            (next_local, next_global), rewards, done = env.step(actions)

            for i in range(cfg.N_AGENTS):
                r = np.clip(rewards[i], -10.0, 10.0)   # fractional reward: guard only
                buffers[i].add(local_obs[i], global_obs,
                               actions[i], log_probs[i],
                               r, float(values[i]), float(done))
                ep_rewards[i] += rewards[i]

            local_obs, global_obs = next_local, next_global
            step += 1

            if step % cfg.ROLLOUT_LEN == 0 or done:
                if done:
                    last_vals = [0.0] * cfg.N_AGENTS
                else:
                    g = torch.tensor(global_obs, dtype=torch.float32).to(cfg.DEVICE)
                    g = g.view(1, cfg.WINDOW_SIZE, cfg.N_FEATURES * cfg.N_AGENTS)
                    last_vals, _ = trainer.critic.get_values(g, None)
                    last_vals = last_vals.tolist()

                trainer.update(buffers, last_vals)
                [b.reset() for b in buffers]

            if done:
                break

        avg_r = float(np.mean(ep_rewards))
        reward_window.append(avg_r)
        rolling = float(np.mean(reward_window))

        history['ep'].append(ep)
        history['avg_reward'].append(avg_r)
        history['rolling_avg'].append(rolling)
        for i, pair in enumerate(cfg.PAIRS):
            history[f'{pair}_reward'].append(ep_rewards[i])
            total = max(sum(action_counts[i]), 1)
            history[f'{pair}_actions'].append(
                [round(100 * c / total, 1) for c in action_counts[i]]
            )

        # Checkpoint selection: on held-out validation return when a val set is
        # given (honest), otherwise fall back to the training rolling reward.
        sel_metric = evaluate(val_df, trainer, scaler) if val_df is not None else rolling
        improved = ''
        if sel_metric > best_avg:
            best_avg = sel_metric
            best_episode = ep
            trainer.save(best_path, scaler=scaler)
            improved = ' ✓'

        pair_str = '  '.join(f'{ep_rewards[i]:>12.4f}' for i in range(cfg.N_AGENTS))
        print(f"{ep:>5} {avg_r:>10.4f} {rolling:>10.4f} {best_avg:>10.4f}  {pair_str}{improved}")

        if ep % 50 == 0:
            print(f"\n  Action dist (hold% / buy% / sell%):")
            for i, pair in enumerate(cfg.PAIRS):
                ac = history[f'{pair}_actions'][-1]
                print(f"    {pair.upper():>3}: hold={ac[0]}%  buy={ac[1]}%  sell={ac[2]}%")
            print()

    trainer.save(save_path, scaler=scaler)
    return trainer, pd.DataFrame({
        k: v for k, v in history.items()
        if k in ('ep', 'avg_reward', 'rolling_avg',
                 *[f'{p}_reward' for p in cfg.PAIRS])
    }), best_episode


# ─────────────────────────────────────────────
# 8. INFERENCE
# ─────────────────────────────────────────────

def predict(df: pd.DataFrame, trainer: MAPPOTrainer):
    """
    Decentralized execution: each actor uses only its own pair obs.
    Returns DataFrame with step, per-pair price, and per-agent action.
    """
    # Transform with the training scaler (falls back to fitting if a legacy
    # checkpoint without a saved scaler was loaded).
    norm_data, raw_closes, T, _ = preprocess(df, getattr(trainer, 'scaler', None))
    env      = MultiAssetEnv(norm_data, raw_closes, T)
    (local_obs, _) = env.reset()
    records    = []
    action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}

    for actor in trainer.actors:
        actor.eval()

    while True:
        actions = []
        for i in range(cfg.N_AGENTS):
            obs_t = torch.tensor(local_obs[i], dtype=torch.float32).to(cfg.DEVICE)
            obs_t = obs_t.view(1, cfg.WINDOW_SIZE, cfg.N_FEATURES)
            # Stateless window eval — must match how the actor was trained.
            a, _, _ = trainer.actors[i].get_action(obs_t, None)
            actions.append(a)

        row = {'step': env.t}
        for i, pair in enumerate(cfg.PAIRS):
            row[f'{pair}_price']  = float(raw_closes[pair][env.t])
            row[f'{pair}_action'] = action_map[actions[i]]

        records.append(row)
        (local_obs, _), _, done = env.step(actions)
        if done:
            break

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 9. ENTRY POINT
# ─────────────────────────────────────────────
def walk_forward_splits(df: pd.DataFrame,
                        min_train_years: int = 2,
                        test_months: int = 6):
    """
    Yields (train_df, test_df, fold_label) for each walk-forward fold.
    Requires a DatetimeIndex on df.
    """
    start = df.index.min()
    end   = df.index.max()

    train_end = start + relativedelta(years=min_train_years)

    fold = 1
    while True:
        test_end = train_end + relativedelta(months=test_months)
        if test_end > end:
            break

        train_df = df.loc[start : train_end - pd.Timedelta(days=1)]
        test_df  = df.loc[train_end : test_end - pd.Timedelta(days=1)]

        label = (f"Fold {fold} | "
                 f"Train: {start.date()} → {train_df.index.max().date()} "
                 f"({len(train_df)} bars) | "
                 f"Test: {test_df.index.min().date()} → {test_df.index.max().date()} "
                 f"({len(test_df)} bars)")

        yield train_df, test_df, label

        train_end = test_end   # expand window by one test period
        fold += 1


def run_walk_forward(df_multi_tec: pd.DataFrame, n_episodes: int = 100):
    """
    Full walk-forward loop: retrain from scratch on each fold,
    collect test predictions and metrics.
    """
    all_results = []
    all_metrics = []

    for fold_i, (train_df, test_df, label) in enumerate(walk_forward_splits(df_multi_tec), 1):
        print(f"\n{'─'*60}")
        print(label)
        print(f"{'─'*60}")

        # Hold out the tail of this fold's train window as validation for
        # checkpoint selection (never touch the test window for selection).
        cut     = int(len(train_df) * 0.85)
        tr_df   = train_df.iloc[:cut]
        val_df  = train_df.iloc[cut:]

        # Fold-specific paths so folds don't overwrite each other's checkpoints.
        best_path = f'mappo_lstm_best_fold{fold_i}.pt'
        save_path = f'mappo_lstm_fold{fold_i}.pt'

        # Retrain from scratch each fold, selecting the best on validation.
        trainer, log_df, _ = train(tr_df, n_episodes=n_episodes, val_df=val_df,
                                   save_path=save_path, best_path=best_path)

        # Load the validation-selected best checkpoint before scoring the test set.
        trainer.load(best_path)

        # Evaluate on unseen test window
        results = predict(test_df, trainer)
        results['fold'] = label

        # Simple PnL metric: count buy/sell signal balance per pair
        metrics = {'fold': label}
        for pair in ['btc', 'eth', 'etb']:
            actions = results[f'{pair}_action']
            metrics[f'{pair}_buy_pct']  = (actions == 'BUY').mean()  * 100
            metrics[f'{pair}_sell_pct'] = (actions == 'SELL').mean() * 100
            metrics[f'{pair}_hold_pct'] = (actions == 'HOLD').mean() * 100

        all_results.append(results)
        all_metrics.append(metrics)

        print(f"Test signals sampled:")
        print(results[['step','btc_action','eth_action','etb_action']].head(10))

    results_df = pd.concat(all_results, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)

    print("\n\nWalk-forward summary:")
    print(metrics_df.to_string(index=False))

    return results_df, metrics_df


# ── Entry point ───────────────────────────────────────────────────
if __name__ == '__main__':
    client_latest = CryptoHistoricalDataClient()

    # Set the timezone
    timezone = ZoneInfo('America/New_York')

    # Get current date in US/Eastern timezone
    today = datetime.now(timezone).date()
    start_date = today - timedelta(days=1500)

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
    results_df, metrics_df = run_walk_forward(df_multi_tec_latest, n_episodes=100)

    # ── Final deployment model (best result for live) ─────────────────────
    # Walk-forward above validates the approach. For the live model we want it
    # trained on ALL data INCLUDING the most recent bars — those matter most for
    # live trading. Standard recipe: (1) hold out the tail as validation to pick
    # the episode count E* that generalizes best, then (2) refit on the FULL
    # dataset for E* episodes and deploy that final checkpoint. This wastes no
    # recent data yet avoids overfit episode-count selection.
    cut = int(len(df_multi_tec_latest) * 0.85)

    # Phase 1 — find E* on held-out validation.
    _, _, best_ep = train(
        df_multi_tec_latest.iloc[:cut],
        n_episodes=100,
        val_df=df_multi_tec_latest.iloc[cut:],
        save_path='mappo_lstm_valsel.pt',
        best_path='mappo_lstm_valbest.pt',
    )
    best_ep = max(best_ep, 1)
    print(f"\nBest validation episode: E* = {best_ep} — refitting on all data.")

    # Phase 2 — refit on ALL data for E* episodes; the FINAL checkpoint (saved to
    # save_path) is the deployed model, so point save_path at mappo_lstm_best.pt.
    final_trainer, _, _ = train(
        df_multi_tec_latest,
        n_episodes=best_ep,
        val_df=None,
        save_path='mappo_lstm_best.pt',
        best_path='mappo_lstm_valbest.pt',
    )