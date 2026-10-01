"""Causal MAPPO-LSTM shared by training, evaluation and the crypto notebook.
Observe completed close t, transact at next open, mark/reward at next close.
"""
from __future__ import annotations
from dataclasses import dataclass
import io
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical

VERSION = 3
PAIRS = ('btc', 'eth', 'etb')
SYMBOLS = ('BTC/USD', 'ETH/USD', 'ETH/BTC')
WINDOW = 20
N_FEATURES = 14
INITIAL_CASH = 10000.
TRADE_SIZE = .1
FEE = 0.0  # Requested zero-fee simulation; no fixed slippage surcharge.
DATA_SOURCE = 'alpaca-us-direct-v1'
ARCHITECTURE = {'version': VERSION, 'window': WINDOW, 'features': N_FEATURES,
    'hidden': 32, 'portfolio_features': 2, 'fee': FEE, 'trade_size': TRADE_SIZE,
    'features_kind': 'causal-relative-v4-contiguous', 'data_source': DATA_SOURCE}


class IncompatibleCheckpoint(ValueError):
    """Saved training assumptions differ from the current model configuration."""


@dataclass
class MarketData:
    index: pd.DatetimeIndex
    features: dict
    closes: np.ndarray
    opens: np.ndarray
    btc: np.ndarray
    btc_open: np.ndarray

    def slice(self, start, end):
        return MarketData(self.index[start:end], {p:a[start:end] for p,a in self.features.items()},
            self.closes[:,start:end], self.opens[:,start:end], self.btc[start:end], self.btc_open[start:end])


def prepare(frame):
    """Scale-free causal features; no fitted statistics from future observations."""
    frame = frame.sort_index().copy()
    if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_unique:
        raise ValueError('Expected unique chronological bar timestamps')
    # Older provider gaps must not become single-day returns or rolling windows.
    # Keep all supplied history in the cache; learn from its continuous recent tail.
    gaps=np.flatnonzero(frame.index[1:]-frame.index[:-1] != pd.Timedelta(days=1))
    if len(gaps): frame=frame.iloc[gaps[-1]+1:]
    parts = {}
    for p in PAIRS:
        raw = frame[[f'{p}_{c}' for c in ('open','high','low','close','volume','trade_count','vwap')]]
        if not np.isfinite(raw.to_numpy()).all() or (raw.iloc[:,:4] <= 0).any().any():
            raise ValueError('Bars must contain finite positive prices')
        c,h,l = frame[f'{p}_close'],frame[f'{p}_high'],frame[f'{p}_low']
        prev = c.shift(1)
        tr = pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
        up,down=h.diff(),-l.diff()
        dmp=pd.Series(np.where((up>down)&(up>0),up,0.),index=frame.index)
        dmn=pd.Series(np.where((down>up)&(down>0),down,0.),index=frame.index)
        dip=dmp.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/atr.clip(lower=1e-12)
        din=dmn.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/atr.clip(lower=1e-12)
        adx=((dip-din).abs()/(dip+din).clip(lower=1e-12)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
        volume=frame[f'{p}_volume']; count=frame[f'{p}_trade_count']
        parts[p]=pd.concat([
            frame[f'{p}_open']/prev-1, h/c-1, l/c-1, np.log(c/prev),
            np.log1p(volume)-np.log1p(volume.rolling(20).mean()),
            np.log1p(count)-np.log1p(count.rolling(20).mean()),
            frame[f'{p}_vwap']/c-1, c/c.ewm(span=12,adjust=False).mean()-1,
            c/c.ewm(span=26,adjust=False).mean()-1, c/c.rolling(50).mean()-1,
            adx,dip,din,atr/c],axis=1)
    valid = np.logical_and.reduce([np.isfinite(x.to_numpy()).all(axis=1) for x in parts.values()])
    idx = frame.index[valid]
    if len(idx) < WINDOW+2:
        raise ValueError('Insufficient completed aligned bars (need at least 71 raw daily bars)')
    if len(idx)>1 and not (idx[1:]-idx[:-1] == pd.Timedelta(days=1)).all():
        raise ValueError('Missing aligned daily bars; repair history before training')
    f={p:parts[p].loc[idx].to_numpy(dtype=np.float32) for p in PAIRS}
    closes=np.stack([frame.loc[idx,f'{p}_close'].to_numpy(float) for p in PAIRS])
    opens=np.stack([frame.loc[idx,f'{p}_open'].to_numpy(float) for p in PAIRS])
    return MarketData(idx,f,closes,opens,closes[0],opens[0])


class MultiAssetEnv:
    def __init__(self, data, start=WINDOW-1, end=None):
        self.data=data; self.start=start
        self.end=len(data.index)-1 if end is None else end
        if start<WINDOW-1 or self.end<=start or self.end>=len(data.index):
            raise ValueError('Invalid episode bounds')
        self.reset()

    def reset(self):
        self.t=self.start
        self.cash=np.array([INITIAL_CASH,INITIAL_CASH,INITIAL_CASH/self.data.btc[self.t]])
        self.holdings=np.zeros(3); self.done=False; self.trades=0
        self.trade_counts=np.zeros(3,dtype=int)
        return self.observation()

    def conversion(self, t):
        return np.array([1.,1.,self.data.btc[t]])

    def nav(self):
        return (self.cash+self.holdings*self.data.closes[:,self.t])*self.conversion(self.t)

    def observation(self):
        local=[self.data.features[p][self.t-WINDOW+1:self.t+1] for p in PAIRS]
        equity=np.maximum(self.cash+self.holdings*self.data.closes[:,self.t],1e-12)
        state=np.stack([self.cash/equity, self.holdings*self.data.closes[:,self.t]/equity],axis=1).astype(np.float32)
        return local,np.concatenate(local,axis=1),state

    def step(self, actions):
        if self.done: raise ValueError('Episode already ended')
        before=self.nav(); self.t+=1
        for i,action in enumerate(actions):
            price=self.data.opens[i,self.t]
            if action==1 and self.cash[i]>1e-12:
                spend=self.cash[i]*TRADE_SIZE
                self.holdings[i]+=spend/(price*(1+FEE)); self.cash[i]-=spend; self.trades+=1
                self.trade_counts[i]+=1
            elif action==2 and self.holdings[i]>1e-12:
                qty=self.holdings[i]*TRADE_SIZE
                self.cash[i]+=qty*price*(1-FEE); self.holdings[i]-=qty; self.trades+=1
                self.trade_counts[i]+=1
        reward=(self.nav()-before)/INITIAL_CASH; self.done=self.t>=self.end
        return self.observation(),reward,self.done


class LSTMActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(N_FEATURES,32,batch_first=True)
        self.head=nn.Sequential(nn.Linear(34,32),nn.Tanh(),nn.Linear(32,3))
        nn.init.orthogonal_(self.head[-1].weight,gain=.01); nn.init.zeros_(self.head[-1].bias)

    def forward(self, window, state):
        seq,_=self.lstm(window)
        logits=self.head(torch.cat([seq[:,-1],state],dim=1))
        mask=torch.stack([torch.ones_like(state[:,0],dtype=torch.bool),state[:,0]>1e-8,state[:,1]>1e-8],dim=1)
        return logits.masked_fill(~mask,-1e9)


class CentralCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(N_FEATURES*3,64,batch_first=True)
        self.head=nn.Sequential(nn.Linear(70,64),nn.Tanh(),nn.Linear(64,3))

    def forward(self, window, state):
        seq,_=self.lstm(window)
        return self.head(torch.cat([seq[:,-1],state.flatten(1)],dim=1))


class MAPPOTrainer:
    def __init__(self, seed=42):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.actors=[LSTMActor() for _ in PAIRS]; self.critic=CentralCritic()
        self.actor_optims=[torch.optim.Adam(a.parameters(),lr=3e-4) for a in self.actors]
        self.critic_optim=torch.optim.Adam(self.critic.parameters(),lr=3e-4)
        self.metadata={}; self.generator=torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        local,glob,state=obs; st=torch.as_tensor(state).unsqueeze(0)
        actions=[]; lps=[]
        for i,actor in enumerate(self.actors):
            logits=actor(torch.as_tensor(local[i]).unsqueeze(0),st[:,i]); dist=Categorical(logits=logits)
            action=logits.argmax(1) if deterministic else torch.multinomial(dist.probs,1,generator=self.generator).flatten()
            actions.append(int(action.item())); lps.append(float(dist.log_prob(action).item()))
        values=self.critic(torch.as_tensor(glob).unsqueeze(0),st)[0].numpy()
        return actions,np.array(lps),values

    def update(self, transitions, bootstrap):
        rewards=np.array([x[3] for x in transitions]); values=np.array([x[4] for x in transitions])
        adv=np.zeros_like(rewards); gae=np.zeros(3)
        for t in reversed(range(len(transitions))):
            nxt=bootstrap if t==len(transitions)-1 else values[t+1]; mask=1-float(transitions[t][5])
            gae=rewards[t]+.99*nxt*mask-values[t]+.99*.95*mask*gae; adv[t]=gae
        returns=torch.tensor(adv+values,dtype=torch.float32)
        advantages=torch.tensor((adv-adv.mean(0))/(adv.std(0)+1e-8),dtype=torch.float32)
        local=[torch.tensor(np.stack([x[0][0][i] for x in transitions])) for i in range(3)]
        glob=torch.tensor(np.stack([x[0][1] for x in transitions])); state=torch.tensor(np.stack([x[0][2] for x in transitions]))
        actions=torch.tensor(np.array([x[1] for x in transitions])); old=torch.tensor(np.array([x[2] for x in transitions]),dtype=torch.float32)
        for _ in range(3):
            indices=torch.randperm(len(transitions),generator=self.generator)
            for start in range(0,len(indices),64):
                b=indices[start:start+64]
                loss=nn.functional.mse_loss(self.critic(glob[b],state[b]),returns[b])
                self.critic_optim.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(),.5); self.critic_optim.step()
                for i,actor in enumerate(self.actors):
                    dist=Categorical(logits=actor(local[i][b],state[b,i]))
                    ratio=(dist.log_prob(actions[b,i])-old[b,i]).exp()
                    objective=torch.minimum(ratio*advantages[b,i],ratio.clamp(.8,1.2)*advantages[b,i])
                    loss=-objective.mean()-.01*dist.entropy().mean()
                    if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                    self.actor_optims[i].zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(),.5); self.actor_optims[i].step()

    def dumps(self, metadata=None):
        stream=io.BytesIO()
        torch.save({'architecture':ARCHITECTURE,'actors':[a.state_dict() for a in self.actors],
            'critic':self.critic.state_dict(),'actor_optims':[o.state_dict() for o in self.actor_optims],
            'critic_optim':self.critic_optim.state_dict(),'rng':self.generator.get_state(),
            'metadata':metadata if metadata is not None else self.metadata},stream)
        return stream.getvalue()

    @classmethod
    def loads(cls,data):
        ckpt=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
        if ckpt.get('architecture')!=ARCHITECTURE: raise IncompatibleCheckpoint('Incompatible checkpoint version or training assumptions; fresh training required')
        result=cls()
        for i in range(3):
            result.actors[i].load_state_dict(ckpt['actors'][i]); result.actor_optims[i].load_state_dict(ckpt['actor_optims'][i])
        result.critic.load_state_dict(ckpt['critic']); result.critic_optim.load_state_dict(ckpt['critic_optim'])
        result.generator.set_state(ckpt['rng']); result.metadata=ckpt['metadata']
        return result


def fit(data, trainer=None, episodes=8, start=WINDOW-1, end=None, on_episode=None):
    trainer=trainer or MAPPOTrainer()
    for ep in range(episodes):
        env=MultiAssetEnv(data,start,end); obs=env.observation(); buffer=[]
        while not env.done:
            actions,lps,values=trainer.act(obs); nxt,reward,done=env.step(actions)
            buffer.append((obs,actions,lps,reward,values,done)); obs=nxt
            if len(buffer)>=128 or done:
                bootstrap=np.zeros(3) if done else trainer.act(obs,deterministic=True)[2]
                trainer.update(buffer,bootstrap); buffer=[]
        if on_episode: on_episode(ep+1,trainer)
    return trainer


def evaluate(data,trainer,start=WINDOW-1,end=None):
    env=MultiAssetEnv(data,start,end); history=[env.nav()]; correct=[]
    while not env.done:
        actions,_,_=trainer.act(env.observation(),deterministic=True); old=env.t
        env.step(actions); history.append(env.nav()); changes=data.closes[:,env.t]/data.closes[:,old]-1
        correct.extend(bool(changes[i]>0) if a==1 else bool(changes[i]<0) for i,a in enumerate(actions) if a in (1,2))
    nav=np.array(history); returns=nav[-1]/INITIAL_CASH-1; dd=1-nav/np.maximum.accumulate(nav,axis=0)
    conversion=np.array([1.,1.,data.btc[env.end]/data.btc[start]])
    cash=conversion-1
    quote_nav=nav.copy()
    quote_nav[:,2]/=data.btc[start:env.end+1]/data.btc[start]
    quote_returns=quote_nav[-1]/INITIAL_CASH-1
    quote_dd=1-quote_nav/np.maximum.accumulate(quote_nav,axis=0)
    portfolio=nav.mean(axis=1)
    usd_portfolio=nav[:,:2].mean(axis=1)
    return {'return':float(returns.mean()),'max_drawdown':float((1-portfolio/np.maximum.accumulate(portfolio)).max()),
        'cash':float(cash.mean()),'trades':env.trades,
        'direction_accuracy':float(np.mean(correct)) if correct else None,
        'bars':len(history)-1,'start':data.index[start+1].isoformat(),'end':data.index[env.end].isoformat(),
        'tradable':{'return':float(returns[:2].mean()),'cash':0.,
                    'max_drawdown':float((1-usd_portfolio/np.maximum.accumulate(usd_portfolio)).max()),
                    'trades':int(env.trade_counts[:2].sum())},
        'pairs':{p:{'return':float(returns[i]),'max_drawdown':float(dd[:,i].max()),'cash':float(cash[i]),
                    'quote_return':float(quote_returns[i]),'quote_max_drawdown':float(quote_dd[:,i].max()),
                    'trades':int(env.trade_counts[i])} for i,p in enumerate(PAIRS)}}


def signals(data,trainer,portfolio=None):
    env=MultiAssetEnv(data); env.t=len(data.index)-1
    if portfolio is not None:
        env.cash=np.array(portfolio['cash'],dtype=float); env.holdings=np.array(portfolio['holdings'],dtype=float)
    else: env.cash=np.array([INITIAL_CASH,INITIAL_CASH,INITIAL_CASH/data.btc[-1]])
    actions,_,_=trainer.act(env.observation(),deterministic=True)
    return [{'pair':p,'symbol':SYMBOLS[i],'action':('HOLD','BUY','SELL')[a],
             'bar_time':data.index[-1].isoformat(),'reference_price':float(data.closes[i,-1])}
            for i,(p,a) in enumerate(zip(PAIRS,actions))]


if __name__=='__main__':
    from crypto_pipeline import refresh_once
    print(refresh_once())
