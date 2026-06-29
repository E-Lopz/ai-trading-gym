from abc import ABC, abstractmethod
import pandas as pd


class BasePortfolioStrategy(ABC):
    """
    Base class for multi-symbol strategies.
    Receives all symbol bars simultaneously so capital can be allocated across them.
    """

    def __init__(self):
        self.broker = None
        self.symbols: list[str] = []
        self._bar_index: int = 0

    def on_start(self, data: dict[str, pd.DataFrame]):
        """Called once before the first bar with the full aligned dataset."""

    @abstractmethod
    def on_portfolio_bar(self, bars: dict[str, pd.Series], history: dict[str, pd.DataFrame]):
        """
        Called once per timestep with the current bar for every symbol.
        bars    — {symbol: current OHLCV Series}
        history — {symbol: DataFrame of all bars up to and including now}
        """

    def on_end(self, data: dict[str, pd.DataFrame]):
        """Called once after the last bar."""

    def buy(self, symbol: str, qty: float, limit_price: float = None):
        if limit_price:
            self.broker.submit_order(symbol, qty, "limit", limit_price)
        else:
            self.broker.submit_order(symbol, qty, "market")

    def sell(self, symbol: str, qty: float, limit_price: float = None):
        if limit_price:
            self.broker.submit_order(symbol, -qty, "limit", limit_price)
        else:
            self.broker.submit_order(symbol, -qty, "market")

    def position(self, symbol: str):
        return self.broker.portfolio.positions.get(symbol)

    @property
    def cash(self) -> float:
        return self.broker.portfolio.cash

    def portfolio_value(self, prices: dict[str, float]) -> float:
        return self.broker.portfolio.equity(prices)
