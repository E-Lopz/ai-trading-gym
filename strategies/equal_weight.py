import pandas as pd
from strategies.portfolio_base import BasePortfolioStrategy


class EqualWeightPortfolio(BasePortfolioStrategy):
    """
    Buys an equal dollar allocation of every symbol on the first bar,
    then rebalances back to equal weight every `rebalance_days` trading days.
    """

    def __init__(self, rebalance_days: int = 21, equity_fraction: float = 0.95):
        super().__init__()
        self.rebalance_days = rebalance_days
        self.equity_fraction = equity_fraction
        self._last_rebalance = -rebalance_days  # trigger on bar 0

    def on_portfolio_bar(self, bars: dict[str, pd.Series], history: dict[str, pd.DataFrame]):
        if self._bar_index - self._last_rebalance < self.rebalance_days:
            return

        self._last_rebalance = self._bar_index
        prices = {sym: bar["close"] for sym, bar in bars.items()}
        total_equity = self.broker.portfolio.equity(prices)
        target_per_symbol = (total_equity * self.equity_fraction) / len(self.symbols)

        # Sell excess first to free cash, then buy shortfalls
        orders = []
        for sym, bar in bars.items():
            pos = self.position(sym)
            current_value = (pos.qty * bar["close"]) if pos else 0.0
            diff = target_per_symbol - current_value
            qty = int(diff / bar["close"])
            if qty != 0:
                orders.append((sym, qty, bar["close"]))

        for sym, qty, price in sorted(orders, key=lambda x: x[1]):  # sells first
            if qty < 0:
                self.sell(sym, abs(qty))
        for sym, qty, price in orders:
            if qty > 0:
                self.buy(sym, qty)
