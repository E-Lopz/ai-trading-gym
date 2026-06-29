from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER


class AlpacaBroker:
    """
    Thin wrapper around Alpaca's TradingClient.
    Works for both paper (ALPACA_PAPER=true) and live accounts.
    """

    def __init__(self):
        if not ALPACA_API_KEY:
            raise EnvironmentError(
                "ALPACA_API_KEY not set. Copy .env.example → .env and fill in credentials."
            )
        self._client = TradingClient(
            ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER
        )

    def account(self):
        return self._client.get_account()

    def submit_market_order(self, symbol: str, qty: float, side: str = "buy"):
        request = MarketOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        return self._client.submit_order(request)

    def submit_limit_order(self, symbol: str, qty: float, limit_price: float, side: str = "buy"):
        request = LimitOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
        )
        return self._client.submit_order(request)

    def get_positions(self):
        return self._client.get_all_positions()

    def close_all_positions(self):
        return self._client.close_all_positions(cancel_orders=True)
