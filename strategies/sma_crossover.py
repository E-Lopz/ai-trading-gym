import pandas as pd
from strategies.base import BaseStrategy


class SMACrossover(BaseStrategy):
    """
    Simple moving average crossover: buy when fast SMA crosses above slow SMA, sell otherwise.
    Exists purely as a working skeleton to validate the backtest pipeline.
    """

    def __init__(self, fast: int = 10, slow: int = 30):
        super().__init__()
        self.fast = fast
        self.slow = slow

    def on_bar(self, bar: pd.Series, history: pd.DataFrame):
        if len(history) < self.slow:
            return

        fast_sma = history["close"].iloc[-self.fast :].mean()
        slow_sma = history["close"].iloc[-self.slow :].mean()

        in_position = self.position is not None and self.position.qty > 0

        if fast_sma > slow_sma and not in_position:
            shares = int(self.cash * 0.95 / bar["close"])
            if shares > 0:
                self.buy(shares)

        elif fast_sma < slow_sma and in_position:
            self.sell(self.position.qty)
