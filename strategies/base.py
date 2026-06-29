from abc import ABC, abstractmethod
import pandas as pd
from gym.broker.simulated import SimulatedBroker


class BaseStrategy(ABC):
    """
    All strategies inherit from this class.
    Implement `on_bar` to define trading logic.
    """

    def __init__(self):
        self.broker: SimulatedBroker = None  # injected by the backtest engine
        self.symbol: str = None
        self._bar_index: int = 0

    def on_start(self, data: pd.DataFrame):
        """Called once before the first bar. Override for setup."""

    @abstractmethod
    def on_bar(self, bar: pd.Series, history: pd.DataFrame):
        """
        Called once per bar during backtesting or live simulation.
        bar: current OHLCV bar
        history: all bars up to and including the current one
        """

    def on_end(self, data: pd.DataFrame):
        """Called once after the last bar. Override for teardown."""

    def buy(self, qty: float = 1, limit_price: float = None):
        if limit_price:
            self.broker.submit_order(self.symbol, qty, "limit", limit_price)
        else:
            self.broker.submit_order(self.symbol, qty, "market")

    def sell(self, qty: float = 1, limit_price: float = None):
        if limit_price:
            self.broker.submit_order(self.symbol, -qty, "limit", limit_price)
        else:
            self.broker.submit_order(self.symbol, -qty, "market")

    @property
    def position(self):
        return self.broker.portfolio.positions.get(self.symbol)

    @property
    def cash(self) -> float:
        return self.broker.portfolio.cash
