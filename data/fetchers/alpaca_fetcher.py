from datetime import datetime
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, DATA_RAW


_TIMEFRAME_MAP = {
    "1m": TimeFrame(1, TimeFrameUnit.Minute),
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "1d": TimeFrame(1, TimeFrameUnit.Day),
}


class AlpacaFetcher:
    """Fetches recent OHLCV bars from Alpaca (requires API credentials)."""

    def __init__(self):
        if not ALPACA_API_KEY:
            raise EnvironmentError(
                "ALPACA_API_KEY not set. Copy .env.example → .env and fill in credentials."
            )
        self._client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    def fetch(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
        cache: bool = True,
    ) -> pd.DataFrame:
        if interval not in _TIMEFRAME_MAP:
            raise ValueError(f"interval must be one of {list(_TIMEFRAME_MAP)}")

        cache_path = DATA_RAW / f"alpaca_{symbol}_{interval}_{start}_{end}.parquet"

        if cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_TIMEFRAME_MAP[interval],
            start=datetime.fromisoformat(start),
            end=datetime.fromisoformat(end),
        )
        bars = self._client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            raise ValueError(f"No data returned for {symbol} ({start} → {end})")

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)

        df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
        df = df[["open", "high", "low", "close", "volume"]]

        if cache:
            DATA_RAW.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)

        return df
