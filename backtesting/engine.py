import pandas as pd
from strategies.base import BaseStrategy
from gym.broker.simulated import SimulatedBroker
from backtesting.results import BacktestResult


class BacktestEngine:
    """
    Feeds historical bars to a strategy one at a time and tracks equity.
    """

    def __init__(self, initial_cash: float = 100_000.0, commission_per_share: float = 0.0):
        self.initial_cash = initial_cash
        self.commission_per_share = commission_per_share

    def run(self, strategy: BaseStrategy, data: pd.DataFrame, symbol: str) -> BacktestResult:
        broker = SimulatedBroker(self.initial_cash, self.commission_per_share)
        strategy.broker = broker
        strategy.symbol = symbol
        strategy._bar_index = 0

        strategy.on_start(data)

        equity_series: dict = {}

        for i, (timestamp, bar) in enumerate(data.iterrows()):
            strategy._bar_index = i
            history = data.iloc[: i + 1]

            strategy.on_bar(bar, history)
            broker.process_bar(symbol, bar, timestamp=timestamp)

            prices = {symbol: bar["close"]}
            equity_series[timestamp] = broker.portfolio.equity(prices)

        strategy.on_end(data)

        equity_curve = pd.Series(equity_series, name="equity")

        return BacktestResult(
            strategy_name=type(strategy).__name__,
            symbol=symbol,
            equity_curve=equity_curve,
            fills=broker.fills,
        )
