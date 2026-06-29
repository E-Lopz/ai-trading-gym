import pandas as pd
from strategies.base import BaseStrategy


def _rsi(series: pd.Series, period: int) -> float:
    delta = series.diff().dropna()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, float("inf"))
    return 100 - (100 / (1 + rs.iloc[-1]))


class RSIMeanReversion(BaseStrategy):
    """
    Buy when RSI drops below oversold threshold, sell when it rises above overbought.
    """

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__()
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def on_bar(self, bar: pd.Series, history: pd.DataFrame):
        if len(history) < self.period + 1:
            return

        rsi = _rsi(history["close"], self.period)
        in_position = self.position is not None and self.position.qty > 0

        if rsi < self.oversold and not in_position:
            shares = int(self.cash * 0.95 / bar["close"])
            if shares > 0:
                self.buy(shares)

        elif rsi > self.overbought and in_position:
            self.sell(self.position.qty)
