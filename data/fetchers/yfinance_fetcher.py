import pandas as pd
import yfinance as yf
from pathlib import Path
from config import DATA_RAW


class YFinanceFetcher:
    """Fetches bulk historical OHLCV data from Yahoo Finance."""

    VALID_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"}

    def fetch(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
        cache: bool = True,
    ) -> pd.DataFrame:
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"interval must be one of {self.VALID_INTERVALS}")

        cache_path = DATA_RAW / f"{symbol}_{interval}_{start}_{end}.parquet"

        if cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {symbol} ({start} → {end})")

        df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
        df = df[["Open", "High", "Low", "Close", "Volume"]].rename(
            columns=str.lower
        )

        if cache:
            DATA_RAW.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)

        return df
