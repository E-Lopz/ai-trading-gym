# AI Trading Gym

A framework for developing, backtesting, and live-trading algorithmic strategies on stocks.
The goal is to test many different methods (rule-based and ML) on real data in simulation,
rank them, train models, and eventually deploy the best ones with real money via Alpaca.

## Stack

- **Data**: yfinance (bulk historical, free) + Alpaca (recent/intraday, same SDK as live trading)
- **Broker**: Alpaca — paper trading (free) and live trading, toggled via `ALPACA_PAPER` in `.env`
- **UI**: Streamlit + Plotly (`streamlit run app.py`)
- **Language**: Python 3.13

## How to run

```bash
# Visual app (single symbol or portfolio mode)
streamlit run app.py

# CLI leaderboard
python compare.py --symbol AAPL --start 2020-01-01 --end 2024-12-31 --sort sharpe

# Paper/live trading (needs .env with Alpaca keys)
python -m live.runner --symbol AAPL --strategy sma --poll 60
```

## Project structure

```
config/               — loads .env (Alpaca keys, paper/live flag, paths)
data/fetchers/
  yfinance_fetcher.py — free historical OHLCV, caches to parquet
  alpaca_fetcher.py   — recent/intraday data via Alpaca SDK
gym/
  portfolio.py        — Portfolio + Position (cash, holdings, fill logic)
  broker/
    simulated.py      — SimulatedBroker: fills market/limit orders against bars
    alpaca_broker.py  — thin Alpaca REST wrapper
    live_broker.py    — LiveBroker: same interface as SimulatedBroker, routes to Alpaca
strategies/
  base.py             — BaseStrategy: implement on_bar(), call buy()/sell()
  portfolio_base.py   — BasePortfolioStrategy: on_portfolio_bar(), buy(symbol, qty)
  buy_and_hold.py
  sma_crossover.py
  rsi.py
  equal_weight.py     — rebalances to equal $ per symbol every N days
  multi_sma.py        — SMA crossover running independently per symbol
backtesting/
  engine.py           — single-symbol bar loop → BacktestResult
  portfolio_engine.py — multi-symbol aligned loop → PortfolioBacktestResult
evaluation/
  metrics.py          — Sharpe, Sortino, max drawdown, Calmar, win rate, ann. return
live/
  runner.py           — polls Alpaca for new bars, drives any BaseStrategy
training/scripts/     — empty, ready for ML training scripts
compare.py            — CLI leaderboard
app.py                — Streamlit UI
```

## How a backtest flows

```
YFinanceFetcher → DataFrame (OHLCV)
       ↓
BacktestEngine.run(strategy, data, symbol)
  for each bar:
    strategy.on_bar(bar, history)   ← strategy logic runs here
    broker.process_bar(bar)         ← fills pending orders
    record equity                   ← close price × positions + cash
       ↓
BacktestResult → equity_curve, fills, metrics
```

Portfolio mode is the same but all symbol bars are fed simultaneously and capital is shared.

## Adding a new strategy

**Single symbol** — subclass `BaseStrategy`, implement `on_bar`:
```python
class MyStrategy(BaseStrategy):
    def on_bar(self, bar, history):
        if some_signal:
            self.buy(100)
        elif other_signal:
            self.sell(100)
```
Register in `compare.py` (`STRATEGIES` dict) and/or `app.py` (`SINGLE_STRATEGIES` dict).

**Portfolio** — subclass `BasePortfolioStrategy`, implement `on_portfolio_bar`:
```python
class MyPortfolio(BasePortfolioStrategy):
    def on_portfolio_bar(self, bars, history):
        for sym, bar in bars.items():
            if some_signal:
                self.buy(sym, 50)
```
Register in `app.py` (`PORTFOLIO_STRATEGIES` dict).

## Backtest findings so far

- **AAPL 2020–2024**: Buy & Hold wins (Sharpe 0.94, +231%). SMA crossover competitive (Sharpe 0.90).
  RSI mean reversion lags — AAPL rarely dips to RSI 30 in a bull run, so few signals fire.
- **AAPL/MSFT/GOOGL portfolio 2020–2024**: Equal Weight (monthly rebal) beats Multi-SMA
  (197% vs 62%). Trending tech stocks favour staying in over in/out timing.
- RSI mean reversion likely works better on range-bound or volatile assets — worth testing on COIN, MSTR, etc.

## What's not built yet

- ML strategies — `training/scripts/` is empty, ready for training scripts
- WebSocket streaming for intraday live trading (runner currently polls, doesn't stream)
- Walk-forward / out-of-sample validation
- Risk management (position sizing, stop losses)

## Research notes

_(Add notes here as you experiment — papers, ideas, findings from runs)_
