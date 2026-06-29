from dataclasses import dataclass
import pandas as pd
from tabulate import tabulate
from evaluation.metrics import compute_metrics


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    equity_curve: pd.Series
    fills: list

    @property
    def metrics(self) -> dict:
        return compute_metrics(self.equity_curve)

    def summary(self) -> str:
        m = self.metrics
        rows = [[k, v] for k, v in m.items()]
        header = f"\n=== {self.strategy_name} on {self.symbol} ===\n"
        return header + tabulate(rows, headers=["Metric", "Value"], tablefmt="simple")

    def print_summary(self):
        print(self.summary())
