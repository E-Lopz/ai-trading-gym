from abc import ABC, abstractmethod
from typing import Union
import numpy as np
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

    def rebalance_to_weights(
        self,
        weights: Union[dict[str, float], np.ndarray],
        bars: dict[str, pd.Series],
        min_trade_value: float = 10.0,
    ) -> None:
        """
        Translate a target weight vector into buy/sell orders against the current bars.

        weights          — target allocation per symbol, either a dict keyed by symbol
                           or a 1-D ndarray parallel to self.symbols. Negative weights
                           are clipped to 0 (long-only). Weights are renormalised to
                           sum to 1 before use. All-zero input is a no-op.
        bars             — current OHLCV bars (same dict passed to on_portfolio_bar).
                           Close prices are used to value existing positions and size
                           new orders; fills happen at each bar's open (engine behaviour).
        min_trade_value  — orders whose absolute dollar value is below this threshold
                           are skipped to avoid transaction cost noise (default $10).

        Sell orders are submitted before buy orders so that, for symbols the engine
        processes earlier in its iteration, freed cash is available for subsequent buys.
        Cross-symbol cash availability within the same bar is not guaranteed because
        the engine calls process_bar per symbol in its own iteration order.
        """
        # -- normalise weights to dict ------------------------------------------
        if isinstance(weights, np.ndarray):
            weights = dict(zip(self.symbols, weights.tolist()))

        # clip negatives, guard against all-zero
        weights = {sym: max(0.0, w) for sym, w in weights.items()}
        total = sum(weights.values())
        if total < 1e-9:
            return
        weights = {sym: w / total for sym, w in weights.items()}

        # -- compute equity and current position values --------------------------
        prices = {sym: float(bar["close"]) for sym, bar in bars.items()}
        equity = self.broker.portfolio.equity(prices)

        # -- build sell / buy lists (sells first to free cash) ------------------
        sells: list[tuple[str, int]] = []
        buys: list[tuple[str, int]] = []

        for sym in self.symbols:
            price = prices[sym]
            target_value = weights.get(sym, 0.0) * equity
            pos = self.broker.portfolio.positions.get(sym)
            current_value = pos.qty * price if pos else 0.0
            diff_value = target_value - current_value

            if abs(diff_value) < min_trade_value:
                continue

            # int() truncates toward zero: -3.8 → -3, 3.8 → 3
            diff_qty = int(diff_value / price)
            if diff_qty == 0:
                continue

            if diff_qty < 0:
                # cap sell at current holding to stay long-only
                max_sell = int(pos.qty) if pos else 0
                sell_qty = min(abs(diff_qty), max_sell)
                if sell_qty > 0:
                    sells.append((sym, sell_qty))
            else:
                buys.append((sym, diff_qty))

        for sym, qty in sells:
            self.sell(sym, qty)
        for sym, qty in buys:
            self.buy(sym, qty)
