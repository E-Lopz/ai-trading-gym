import pandas as pd
from strategies.portfolio_base import BasePortfolioStrategy


class MultiSMAPortfolio(BasePortfolioStrategy):
    """
    Runs SMA crossover independently on each symbol.
    Capital is split equally at start; each symbol only uses its own slice.
    """

    def __init__(self, fast: int = 10, slow: int = 30, equity_fraction: float = 0.95):
        super().__init__()
        self.fast = fast
        self.slow = slow
        self.equity_fraction = equity_fraction
        self._initial_allocation: dict[str, float] = {}

    def on_start(self, data: dict[str, pd.DataFrame]):
        # We don't know initial cash here, so defer to first bar
        pass

    def on_portfolio_bar(self, bars: dict[str, pd.Series], history: dict[str, pd.DataFrame]):
        # Set per-symbol capital budget on the first bar
        if not self._initial_allocation:
            prices = {sym: bar["close"] for sym, bar in bars.items()}
            total = self.broker.portfolio.equity(prices)
            per_sym = (total * self.equity_fraction) / len(self.symbols)
            self._initial_allocation = {sym: per_sym for sym in self.symbols}

        for sym, bar in bars.items():
            hist = history[sym]
            if len(hist) < self.slow:
                continue

            fast_sma = hist["close"].iloc[-self.fast:].mean()
            slow_sma = hist["close"].iloc[-self.slow:].mean()
            pos = self.position(sym)
            in_position = pos is not None and pos.qty > 0

            if fast_sma > slow_sma and not in_position:
                budget = self._initial_allocation[sym]
                shares = int(min(budget, self.cash * 0.98) / bar["close"])
                if shares > 0:
                    self.buy(sym, shares)

            elif fast_sma < slow_sma and in_position:
                self.sell(sym, pos.qty)
