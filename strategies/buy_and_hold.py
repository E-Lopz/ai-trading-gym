import pandas as pd
from strategies.base import BaseStrategy


class BuyAndHold(BaseStrategy):
    """Buys on the first bar and holds. Baseline benchmark."""

    def __init__(self, equity_fraction: float = 0.95):
        super().__init__()
        self.equity_fraction = equity_fraction

    def on_bar(self, bar: pd.Series, history: pd.DataFrame):
        if len(history) == 1 and self.position is None:
            shares = int(self.cash * self.equity_fraction / bar["close"])
            if shares > 0:
                self.buy(shares)
