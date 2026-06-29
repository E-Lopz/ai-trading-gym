"""
Live trading runner — polls Alpaca for new bars and drives any BaseStrategy.

Usage (paper trading):
    python -m live.runner --symbol AAPL --strategy sma
    python -m live.runner --symbol TSLA --strategy rsi --interval 1d --poll 60

Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in .env (ALPACA_PAPER=true for paper).
"""
import argparse
import logging
import time
from datetime import datetime, timezone

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestBarRequest

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
from data.fetchers import YFinanceFetcher
from gym.broker.live_broker import LiveBroker
from strategies.sma_crossover import SMACrossover
from strategies.rsi import RSIMeanReversion
from strategies.buy_and_hold import BuyAndHold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

STRATEGY_MAP = {
    "sma":   SMACrossover(fast=10, slow=30),
    "sma50": SMACrossover(fast=20, slow=50),
    "rsi":   RSIMeanReversion(),
    "hold":  BuyAndHold(),
}

WARMUP_BARS = 100  # bars of history fed to on_start so indicators have context


def _fetch_warmup(symbol: str, warmup_bars: int) -> pd.DataFrame:
    """Pull recent daily history via yfinance for indicator warmup."""
    # Request extra days to account for weekends/holidays
    import pandas as pd
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    # ~1.5× trading days to calendar days
    start = (pd.Timestamp.now() - pd.Timedelta(days=int(warmup_bars * 1.5))).strftime("%Y-%m-%d")
    fetcher = YFinanceFetcher()
    df = fetcher.fetch(symbol, start=start, end=end, interval="1d", cache=False)
    return df.tail(warmup_bars)


def _latest_bar(symbol: str, client: StockHistoricalDataClient) -> pd.Series:
    """Fetch the most recent completed bar from Alpaca."""
    req = StockLatestBarRequest(symbol_or_symbols=symbol)
    bar = client.get_stock_latest_bar(req)[symbol]
    return pd.Series({
        "open":   bar.open,
        "high":   bar.high,
        "low":    bar.low,
        "close":  bar.close,
        "volume": bar.volume,
    }, name=bar.timestamp.replace(tzinfo=None))


def run(symbol: str, strategy_key: str, poll_interval: int):
    if not ALPACA_API_KEY:
        raise EnvironmentError("ALPACA_API_KEY not set. Copy .env.example → .env")

    strategy = STRATEGY_MAP[strategy_key]
    broker = LiveBroker()
    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    strategy.broker = broker
    strategy.symbol = symbol

    mode = "PAPER" if ALPACA_PAPER else "LIVE"
    account = broker._alpaca.account()
    logger.info(f"[{mode}] Account equity=${float(account.equity):,.2f}  cash=${float(account.cash):,.2f}")

    logger.info(f"Fetching {WARMUP_BARS} warmup bars for {symbol}...")
    history = _fetch_warmup(symbol, WARMUP_BARS)
    logger.info(f"Warmup: {len(history)} bars  ({history.index[0].date()} → {history.index[-1].date()})")

    strategy.on_start(history)

    last_bar_time: datetime = None
    logger.info(f"Polling every {poll_interval}s for new {symbol} bars  (Ctrl+C to stop)")

    try:
        while True:
            try:
                bar = _latest_bar(symbol, data_client)
                bar_time = bar.name  # timestamp is the Series name

                if last_bar_time is None or bar_time > last_bar_time:
                    last_bar_time = bar_time
                    history = pd.concat([history, bar.to_frame().T])
                    history = history[~history.index.duplicated(keep="last")]

                    pos = broker.portfolio.positions.get(symbol)
                    cash = broker.portfolio.cash
                    pos_str = f"pos={pos.qty:.0f}@{pos.avg_entry_price:.2f}" if pos else "flat"
                    logger.info(
                        f"New bar {bar_time.date()}  "
                        f"O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f}  "
                        f"cash=${cash:,.2f}  {pos_str}"
                    )

                    strategy._bar_index += 1
                    strategy.on_bar(bar, history)
                else:
                    logger.debug(f"No new bar (latest={bar_time.date()}), sleeping...")

            except Exception as exc:
                logger.error(f"Error during poll cycle: {exc}", exc_info=True)

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
        strategy.on_end(history)


def main():
    parser = argparse.ArgumentParser(description="Live/paper trading runner")
    parser.add_argument("--symbol",   default="AAPL")
    parser.add_argument("--strategy", default="sma", choices=list(STRATEGY_MAP))
    parser.add_argument("--interval", default="1d", help="Bar interval (informational — poll detects new bars)")
    parser.add_argument("--poll",     default=60, type=int, help="Poll interval in seconds")
    args = parser.parse_args()

    run(args.symbol, args.strategy, args.poll)


if __name__ == "__main__":
    main()
