from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float

    @property
    def market_value(self) -> float:
        return self.qty * self.avg_entry_price


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    def equity(self, prices: dict[str, float]) -> float:
        position_value = sum(
            pos.qty * prices.get(pos.symbol, pos.avg_entry_price)
            for pos in self.positions.values()
        )
        return self.cash + position_value

    def apply_fill(self, symbol: str, qty: float, price: float, commission: float = 0.0):
        """Update portfolio after a fill. Positive qty = buy, negative = sell."""
        cost = qty * price + commission
        if self.cash < cost:
            raise ValueError(f"Insufficient cash: need {cost:.2f}, have {self.cash:.2f}")

        self.cash -= cost

        if symbol in self.positions:
            pos = self.positions[symbol]
            new_qty = pos.qty + qty
            if abs(new_qty) < 1e-9:
                del self.positions[symbol]
            else:
                total_cost = pos.qty * pos.avg_entry_price + qty * price
                pos.qty = new_qty
                pos.avg_entry_price = total_cost / new_qty
        elif qty > 0:
            self.positions[symbol] = Position(symbol, qty, price)
