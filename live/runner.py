"""
Live trading runner — connects a strategy to Alpaca (paper or live).
Not wired to strategies.BaseStrategy yet; placeholder for Phase 2.
"""
import time
import logging
from datetime import datetime
from gym.broker.alpaca_broker import AlpacaBroker

logger = logging.getLogger(__name__)


class LiveRunner:
    """
    Polls Alpaca for the latest bar and calls strategy.on_bar.
    Intended for daily bar strategies. Sub-minute strategies should use WebSocket streaming instead.
    """

    def __init__(self, strategy, symbol: str, poll_interval_seconds: int = 60):
        self.strategy = strategy
        self.symbol = symbol
        self.poll_interval = poll_interval_seconds
        self.broker = AlpacaBroker()

    def run(self):
        logger.info(f"Starting live runner for {self.symbol} (paper={self.broker._client._use_raw_data})")
        account = self.broker.account()
        logger.info(f"Account equity: {account.equity}")

        # TODO: wire strategy.broker → AlpacaBroker adapter and call on_bar each tick
        raise NotImplementedError(
            "LiveRunner execution loop not yet implemented. "
            "Implement the strategy → AlpacaBroker adapter and call on_bar each poll cycle."
        )
