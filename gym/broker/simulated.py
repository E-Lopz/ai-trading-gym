from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from gym.portfolio import Portfolio, Position


@dataclass
class Order:
    symbol: str
    qty: float          # positive = buy, negative = sell
    order_type: str     # "market" | "limit"
    limit_price: Optional[float] = None
    order_id: int = 0


@dataclass
class Fill:
    order: Order
    fill_price: float
    commission: float
    timestamp: object = None  # bar timestamp, set by process_bar


class SimulatedBroker:
    """
    Simulates order execution against a price bar.
    Market orders fill at the open of the next bar.
    Limit orders fill if the bar's low ≤ limit (buy) or high ≥ limit (sell).
    """

    def __init__(self, initial_cash: float, commission_per_share: float = 0.0):
        self.portfolio = Portfolio(cash=initial_cash)
        self.commission_per_share = commission_per_share
        self._pending: list[Order] = []
        self._order_counter = 0
        self.fills: list[Fill] = []

    def submit_order(self, symbol: str, qty: float, order_type: str = "market", limit_price: Optional[float] = None) -> Order:
        self._order_counter += 1
        order = Order(symbol, qty, order_type, limit_price, self._order_counter)
        self._pending.append(order)
        return order

    def process_bar(self, symbol: str, bar: pd.Series, timestamp=None) -> list[Fill]:
        """Call once per bar to execute pending orders against OHLCV data."""
        executed: list[Fill] = []
        still_pending: list[Order] = []

        for order in self._pending:
            if order.symbol != symbol:
                still_pending.append(order)
                continue

            fill_price: Optional[float] = None

            if order.order_type == "market":
                fill_price = bar["open"]
            elif order.order_type == "limit" and order.limit_price is not None:
                if order.qty > 0 and bar["low"] <= order.limit_price:
                    fill_price = min(order.limit_price, bar["open"])
                elif order.qty < 0 and bar["high"] >= order.limit_price:
                    fill_price = max(order.limit_price, bar["open"])

            if fill_price is not None:
                commission = abs(order.qty) * self.commission_per_share
                try:
                    self.portfolio.apply_fill(order.symbol, order.qty, fill_price, commission)
                    fill = Fill(order, fill_price, commission, timestamp)
                    self.fills.append(fill)
                    executed.append(fill)
                except ValueError:
                    still_pending.append(order)  # keep if insufficient cash
            else:
                still_pending.append(order)

        self._pending = still_pending
        return executed

    def cancel_all(self):
        self._pending.clear()
