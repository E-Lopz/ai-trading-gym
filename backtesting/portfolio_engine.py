from dataclasses import dataclass, field
import pandas as pd
from gym.broker.simulated import SimulatedBroker
from strategies.portfolio_base import BasePortfolioStrategy


@dataclass
class PortfolioBacktestResult:
    strategy_name: str
    symbols: list[str]
    equity_curve: pd.Series
    fills: list
    allocation_history: pd.DataFrame  # columns = symbols + "cash", indexed by date

    @property
    def metrics(self) -> dict:
        from evaluation.metrics import compute_metrics
        return compute_metrics(self.equity_curve)

    def print_summary(self):
        from tabulate import tabulate
        m = self.metrics
        print(f"\n=== {self.strategy_name} | {', '.join(self.symbols)} ===")
        print(tabulate([[k, v] for k, v in m.items()], headers=["Metric", "Value"], tablefmt="simple"))


class PortfolioBacktestEngine:
    """
    Feeds aligned multi-symbol bars to a BasePortfolioStrategy and tracks equity.
    All symbols share a single cash pool.
    """

    def __init__(self, initial_cash: float = 100_000.0, commission_per_share: float = 0.0):
        self.initial_cash = initial_cash
        self.commission_per_share = commission_per_share

    def run(self, strategy: BasePortfolioStrategy, data: dict[str, pd.DataFrame]) -> PortfolioBacktestResult:
        # Align all symbols to their common trading days
        common_index = None
        for df in data.values():
            common_index = df.index if common_index is None else common_index.intersection(df.index)

        aligned = {sym: df.loc[common_index] for sym, df in data.items()}
        symbols = list(aligned.keys())

        broker = SimulatedBroker(self.initial_cash, self.commission_per_share)
        strategy.broker = broker
        strategy.symbols = symbols
        strategy._bar_index = 0

        strategy.on_start(aligned)

        equity_series: dict = {}
        allocation_rows: dict = {}

        for i, timestamp in enumerate(common_index):
            strategy._bar_index = i
            bars = {sym: aligned[sym].loc[timestamp] for sym in symbols}
            history = {sym: aligned[sym].iloc[: i + 1] for sym in symbols}

            strategy.on_portfolio_bar(bars, history)

            for sym, bar in bars.items():
                broker.process_bar(sym, bar, timestamp=timestamp)

            prices = {sym: bars[sym]["close"] for sym in symbols}
            equity = broker.portfolio.equity(prices)
            equity_series[timestamp] = equity

            # Record dollar allocation per symbol + cash
            row = {sym: broker.portfolio.positions[sym].qty * prices[sym]
                   if sym in broker.portfolio.positions else 0.0
                   for sym in symbols}
            row["cash"] = broker.portfolio.cash
            allocation_rows[timestamp] = row

        strategy.on_end(aligned)

        return PortfolioBacktestResult(
            strategy_name=type(strategy).__name__,
            symbols=symbols,
            equity_curve=pd.Series(equity_series, name="equity"),
            fills=broker.fills,
            allocation_history=pd.DataFrame.from_dict(allocation_rows, orient="index"),
        )
