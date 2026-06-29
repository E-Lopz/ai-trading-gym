"""
LiveBroker — same interface as SimulatedBroker but routes to Alpaca.
Strategies call self.buy()/self.sell() identically in backtest and live.
"""
import logging
from gym.portfolio import Position
from gym.broker.alpaca_broker import AlpacaBroker

logger = logging.getLogger(__name__)


class LivePortfolio:
    """Reads cash and positions live from Alpaca on each access."""

    def __init__(self, alpaca: AlpacaBroker):
        self._alpaca = alpaca

    @property
    def cash(self) -> float:
        return float(self._alpaca.account().cash)

    @property
    def positions(self) -> dict[str, Position]:
        result = {}
        for p in self._alpaca.get_positions():
            result[p.symbol] = Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
            )
        return result


class LiveBroker:
    """
    Drop-in replacement for SimulatedBroker during live/paper trading.
    Injected into BaseStrategy.broker so strategies need zero changes.
    """

    def __init__(self):
        self._alpaca = AlpacaBroker()
        self.portfolio = LivePortfolio(self._alpaca)

    def submit_order(self, symbol: str, qty: float, order_type: str = "market", limit_price: float = None):
        side = "buy" if qty > 0 else "sell"
        abs_qty = abs(qty)

        if order_type == "market":
            order = self._alpaca.submit_market_order(symbol, abs_qty, side)
        else:
            if limit_price is None:
                raise ValueError("limit_price required for limit orders")
            order = self._alpaca.submit_limit_order(symbol, abs_qty, limit_price, side)

        logger.info(f"Order submitted: {side.upper()} {abs_qty} {symbol} @ {order_type} → id={order.id}")
        return order

    def process_bar(self, symbol: str, bar) -> list:
        # Alpaca handles fills — nothing to simulate
        return []
