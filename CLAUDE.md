# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A framework for developing, backtesting, and live-trading algorithmic strategies on stocks.
Goal: test many methods (rule-based and ML) on real data in simulation, rank them, train models,
and deploy the best via Alpaca with real money.

## Commands

```bash
# Run all tests
pytest

# Run a single test
pytest tests/test_smoke.py::test_backtest_runs_and_returns_metrics

# CLI leaderboard (no UI needed)
python compare.py --symbol AAPL --start 2020-01-01 --end 2024-12-31 --sort sharpe

# Streamlit app (single symbol + portfolio mode)
streamlit run app.py

# Paper/live trading (requires .env)
python -m live.runner --symbol AAPL --strategy sma --poll 60
```

## Environment setup

Copy `.env.example` → `.env` and fill in Alpaca credentials from https://app.alpaca.markets.
`ALPACA_PAPER=true` uses paper trading; set to `false` for real money.
No credentials needed for backtesting or the Streamlit app.

## Recent additions

### `rebalance_to_weights()` on `BasePortfolioStrategy`

`strategies/portfolio_base.py` — call from any `on_portfolio_bar` to translate a target
weight vector into buy/sell orders:

```python
self.rebalance_to_weights(
    weights,          # dict[str, float] or np.ndarray parallel to self.symbols
    bars,             # the `bars` dict from on_portfolio_bar
    min_trade_value=10.0,  # skip trades smaller than this to avoid noise
)
```

Clips negatives (long-only), renormalises to sum=1, sells before buys.
Cross-symbol cash availability within one bar is not guaranteed — the engine
processes symbols in insertion order, so a buy may fill before a sell for a
different symbol clears. Keep a cash buffer or use `equity_fraction < 1`.

### Gymnasium environment — `PortfolioEnv`

`training/envs/portfolio_env.py` — standard `gymnasium.Env` for training DRL agents:

```python
from training.envs.portfolio_env import PortfolioEnv

env = PortfolioEnv(
    data,                  # dict[str, pd.DataFrame] — same format as engine
    initial_cash=100_000,
    lookback=20,           # bars in each observation
    reward_shaping=fn,     # optional (log_return, info) -> float
)
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
```

- **Observation**: `Box(n_symbols, lookback, 5)` — OHLCV min-max normalised to `[0,1]`
  within each lookback window; zero-padded at episode start.
- **Action**: `Box(0, 1, n_symbols)` — softmaxed inside `step()` → portfolio weights
  → `rebalance_to_weights()`.
- **Reward**: `log(equity_t / equity_{t−1})` by default; override via `reward_shaping`.
- **Info**: `{"equity": float, "weights": dict, "date": pd.Timestamp}` every step.
- Inherits `BasePortfolioStrategy` so `rebalance_to_weights` is reused without duplication.

## Architecture

### The broker interface is the key abstraction

Both `SimulatedBroker` (`gym/broker/simulated.py`) and `LiveBroker` (`gym/broker/live_broker.py`)
expose the same interface: `submit_order()` and `process_bar()`. Strategies call `self.buy()` /
`self.sell()` on `BaseStrategy`, which delegates to whichever broker is injected — so the same
strategy class runs in backtesting and live trading with zero changes.

In backtesting, `SimulatedBroker.process_bar()` simulates fills (market orders at open, limit
orders checked against high/low). In live trading, `LiveBroker.process_bar()` is a no-op —
Alpaca handles fills.

### Two strategy base classes

- `BaseStrategy` (`strategies/base.py`) — single symbol. Implement `on_bar(bar, history)`.
- `BasePortfolioStrategy` (`strategies/portfolio_base.py`) — multi-symbol, shared capital.
  Implement `on_portfolio_bar(bars, history)` where both args are `{symbol: ...}` dicts.

### Two backtest engines

- `BacktestEngine` (`backtesting/engine.py`) — feeds one symbol's bars to a `BaseStrategy`.
- `PortfolioBacktestEngine` (`backtesting/portfolio_engine.py`) — aligns multiple symbols to
  common trading days, feeds all bars simultaneously to a `BasePortfolioStrategy`, and records
  `allocation_history` (dollar value per symbol + cash at each timestep) for the allocation chart.

### Data flow

```
yfinance / Alpaca → DataFrame (open/high/low/close/volume, datetime index, no tz)
       ↓
Engine replays bars → strategy.on_bar() → broker.submit_order()
       ↓
broker.process_bar() → Portfolio.apply_fill() → updates cash + positions
       ↓
BacktestResult: equity_curve (pd.Series), fills (list[Fill] with timestamps), metrics
```

`Fill.timestamp` is set in `process_bar` — required for the trade marker overlay in the UI.

### Registering a new strategy

Single-symbol: add to `STRATEGIES` in `compare.py` and `SINGLE_STRATEGIES` in `app.py`.
Portfolio: add to `PORTFOLIO_STRATEGIES` in `app.py`.

## Backtest findings

- **AAPL 2020–2024**: Buy & Hold (Sharpe 0.94, +231%) beats SMA crossover (0.90, +123%).
  RSI mean reversion barely trades — AAPL rarely reaches RSI 30 in a bull trend.
- **AAPL/MSFT/GOOGL portfolio 2020–2024**: Equal Weight monthly rebal (Sharpe 0.95, +197%)
  beats Multi-SMA (0.76, +62%). Trending tech favours staying in over timing entries.
- RSI mean reversion worth testing on range-bound / volatile assets (COIN, MSTR, etc.).

## What's not built yet

- ML strategies — `training/scripts/` is empty
- WebSocket streaming for intraday live trading (runner polls, doesn't stream)
- Walk-forward / out-of-sample validation
- Risk management (position sizing, stop losses)

## Research notes

_(Add notes here as you experiment — papers, ideas, findings from runs)_
