"""
PortfolioEnv — a gymnasium.Env wrapping the portfolio backtest engine.

Design decisions
----------------
* Inherits from both gymnasium.Env and BasePortfolioStrategy so it reuses
  rebalance_to_weights() and the buy/sell helpers without duplication.
  on_portfolio_bar() is implemented as a no-op because the RL agent drives
  actions through step(), not through the strategy hook.

* Observation: Box(n_symbols, lookback, n_features=5).
  Each feature (OHLCV) is min-max normalised within the lookback window so
  values always lie in [0, 1] regardless of price level, making it possible
  to train across different symbols and time periods without rescaling.
  Early steps where fewer than `lookback` bars are available are zero-padded.

* Action: Box(low=0, high=1, shape=(n_symbols,)).
  Raw values are softmaxed inside step() to produce target portfolio weights.
  Using bounded [0,1] keeps the action space compatible with most DRL samplers
  out of the box; softmax inside step() ensures weights always sum to 1.

* Reward: log(equity_t / equity_{t-1}) — log portfolio return per step.
  This is additive over time (cumulative log return = sum of step rewards),
  numerically stable, and has a natural zero point (no change).

* reward_shaping: a callable (raw_reward: float, info: dict) -> float injected
  at construction time. Defaults to identity. Use it to add Sharpe penalties,
  drawdown penalties, or any shaped reward without subclassing.

* Data alignment mirrors PortfolioBacktestEngine: only trading days present
  in ALL symbols are used (inner join). This is done once in __init__ so
  reset() is cheap.
"""

from __future__ import annotations

from typing import Callable, Optional, Union
import numpy as np
import pandas as pd
import gymnasium
from gymnasium import spaces

from gym.broker.simulated import SimulatedBroker
from strategies.portfolio_base import BasePortfolioStrategy

_FEATURES = ["open", "high", "low", "close", "volume"]
N_FEATURES = len(_FEATURES)


class PortfolioEnv(gymnasium.Env, BasePortfolioStrategy):
    """
    Single-episode gymnasium environment for multi-asset portfolio management.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        OHLCV DataFrames keyed by symbol. Columns must include open/high/low/close/volume.
        DataFrames are aligned to their common trading days on construction.
    initial_cash : float
        Starting cash for each episode.
    lookback : int
        Number of past bars in each observation. Default 20.
    min_trade_value : float
        Minimum dollar trade size passed to rebalance_to_weights(). Default $10.
    commission_per_share : float
        Per-share commission forwarded to SimulatedBroker. Default 0.
    reward_shaping : callable, optional
        (raw_log_return: float, info: dict) -> float. Defaults to identity.
        Called after each step; return value is used as the env reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        initial_cash: float = 100_000.0,
        lookback: int = 20,
        min_trade_value: float = 10.0,
        commission_per_share: float = 0.0,
        reward_shaping: Optional[Callable[[float, dict], float]] = None,
    ):
        gymnasium.Env.__init__(self)
        BasePortfolioStrategy.__init__(self)

        # Align to common trading days (same logic as PortfolioBacktestEngine)
        common_index = None
        for df in data.values():
            common_index = df.index if common_index is None else common_index.intersection(df.index)

        self._aligned: dict[str, pd.DataFrame] = {
            sym: df.loc[common_index] for sym, df in data.items()
        }
        self._common_index: pd.DatetimeIndex = common_index

        self.symbols: list[str] = list(self._aligned.keys())
        self.n_symbols: int = len(self.symbols)
        self.lookback: int = lookback
        self.initial_cash: float = initial_cash
        self.min_trade_value: float = min_trade_value
        self.commission_per_share: float = commission_per_share
        self._reward_shaping: Callable = reward_shaping or (lambda r, info: r)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_symbols, lookback, N_FEATURES),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_symbols,),
            dtype=np.float32,
        )

        # Initialised properly in reset()
        self._step: int = 0
        self._prev_equity: float = initial_cash
        self.broker: Optional[SimulatedBroker] = None

    # ── gymnasium.Env interface ───────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        # gymnasium.Env.reset handles seeding (sets self.np_random).
        # Called explicitly to avoid MRO ambiguity from multiple inheritance.
        gymnasium.Env.reset(self, seed=seed)

        self.broker = SimulatedBroker(self.initial_cash, self.commission_per_share)
        self._step = 0
        self._prev_equity = self.initial_cash

        obs = self._build_obs()
        info = self._make_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one bar:
        1. Softmax the action to get portfolio weights.
        2. Call rebalance_to_weights() — submits orders to _pending.
        3. Fill orders via process_bar() for each symbol (at bar's open).
        4. Compute log-return reward; call reward_shaping hook.
        5. Advance step counter; build next observation.

        Note: fills happen at the *current* bar's open price (SimulatedBroker
        behaviour). The agent sees the same bar's OHLCV in its observation, so
        there is a slight look-ahead inherent to the existing engine design.
        This is consistent with how PortfolioBacktestEngine works.
        """
        bars = {sym: self._aligned[sym].iloc[self._step] for sym in self.symbols}
        timestamp = self._common_index[self._step]

        # Translate raw action → weights → orders
        weights = self._softmax(action)
        self.rebalance_to_weights(weights, bars, self.min_trade_value)

        # Fill orders (market orders fill at bar's open)
        for sym in self.symbols:
            self.broker.process_bar(sym, bars[sym], timestamp=timestamp)

        # Portfolio value after fills
        prices = {sym: float(bars[sym]["close"]) for sym in self.symbols}
        equity = self.broker.portfolio.equity(prices)

        # Log-return reward
        raw_reward = float(np.log(equity / self._prev_equity)) if self._prev_equity > 0 else 0.0
        self._prev_equity = equity

        self._step += 1
        terminated = self._step >= len(self._common_index)

        info = self._make_info(prices=prices, equity=equity, timestamp=timestamp)
        reward = self._reward_shaping(raw_reward, info)

        obs = self._build_obs()
        return obs, reward, terminated, False, info

    def render(self):
        pass  # visual rendering not implemented; use app.py for visualisation

    # ── BasePortfolioStrategy abstract method ─────────────────────────────────

    def on_portfolio_bar(self, bars: dict, history: dict) -> None:
        """No-op: the RL agent drives actions via step(), not this hook."""

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def _build_obs(self) -> np.ndarray:
        """
        Build (n_symbols, lookback, n_features) observation.
        Each feature column is min-max normalised within the window.
        Steps with fewer than `lookback` prior bars are zero-padded at the front.
        """
        obs = np.zeros((self.n_symbols, self.lookback, N_FEATURES), dtype=np.float32)
        start = max(0, self._step - self.lookback)
        end = self._step  # exclusive; 0 at reset gives all-zero obs

        if end == 0:
            return obs

        for i, sym in enumerate(self.symbols):
            window = self._aligned[sym].iloc[start:end][_FEATURES].values.astype(np.float32)
            mn = window.min(axis=0, keepdims=True)
            mx = window.max(axis=0, keepdims=True)
            scale = np.where(mx - mn > 1e-8, mx - mn, 1.0)
            normed = (window - mn) / scale
            obs[i, -len(normed):, :] = normed  # right-align; leading rows stay 0

        return obs

    def _make_info(self, prices=None, equity=None, timestamp=None) -> dict:
        if equity is None:
            equity = self.initial_cash
        if prices is None:
            prices = {}
        if timestamp is None and self._step < len(self._common_index):
            timestamp = self._common_index[self._step]

        pos_weights = {}
        if self.broker is not None and equity > 0:
            for sym in self.symbols:
                pos = self.broker.portfolio.positions.get(sym)
                pos_weights[sym] = (pos.qty * prices.get(sym, 0.0) / equity) if pos else 0.0

        return {
            "equity": equity,
            "weights": pos_weights,
            "date": timestamp,
        }
