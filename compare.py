"""
compare.py — run multiple strategies on real data and print a ranked leaderboard.

Usage:
    python compare.py
    python compare.py --symbol TSLA --start 2022-01-01 --end 2024-01-01
    python compare.py --sort sharpe
"""
import argparse
import pandas as pd
from tabulate import tabulate

from data.fetchers import YFinanceFetcher
from backtesting.engine import BacktestEngine
from backtesting.results import BacktestResult
from strategies.buy_and_hold import BuyAndHold
from strategies.sma_crossover import SMACrossover
from strategies.rsi import RSIMeanReversion

INITIAL_CASH = 100_000.0

STRATEGIES = [
    BuyAndHold(),
    SMACrossover(fast=10, slow=30),
    SMACrossover(fast=20, slow=50),
    RSIMeanReversion(period=14, oversold=30, overbought=70),
    RSIMeanReversion(period=7,  oversold=25, overbought=75),
]

SORT_OPTIONS = ["sharpe", "total_return", "annualized_return", "sortino", "calmar", "max_drawdown"]


def strategy_label(s) -> str:
    name = type(s).__name__
    params = {k: v for k, v in vars(s).items() if not k.startswith("_") and k not in ("broker", "symbol")}
    if params:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{name}({param_str})"
    return name


def run_comparison(symbol: str, start: str, end: str, sort_by: str) -> list[BacktestResult]:
    print(f"\nFetching {symbol} from {start} to {end} via yfinance...")
    fetcher = YFinanceFetcher()
    data = fetcher.fetch(symbol, start=start, end=end, interval="1d")
    print(f"Got {len(data)} bars  ({data.index[0].date()} → {data.index[-1].date()})\n")

    engine = BacktestEngine(initial_cash=INITIAL_CASH)
    results = []

    for strategy in STRATEGIES:
        result = engine.run(strategy, data, symbol)
        results.append((strategy_label(strategy), result))

    rows = []
    for label, r in results:
        m = r.metrics
        rows.append({
            "Strategy": label,
            "Total Return %": m["total_return"],
            "Ann. Return %": m["annualized_return"],
            "Sharpe": m["sharpe"],
            "Sortino": m["sortino"],
            "Max DD %": m["max_drawdown"],
            "Calmar": m["calmar"],
            "Win Rate %": m["win_rate"],
        })

    df = pd.DataFrame(rows)

    sort_col_map = {
        "sharpe": "Sharpe",
        "total_return": "Total Return %",
        "annualized_return": "Ann. Return %",
        "sortino": "Sortino",
        "calmar": "Calmar",
        "max_drawdown": "Max DD %",
    }
    sort_col = sort_col_map.get(sort_by, "Sharpe")
    ascending = sort_by == "max_drawdown"
    df = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    df.index += 1  # rank starts at 1

    print(f"=== Leaderboard: {symbol} ({start} → {end}) | ranked by {sort_col} ===\n")
    print(tabulate(df, headers="keys", tablefmt="simple", floatfmt=".2f"))
    print()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start",  default="2020-01-01")
    parser.add_argument("--end",    default="2024-12-31")
    parser.add_argument("--sort",   default="sharpe", choices=SORT_OPTIONS)
    args = parser.parse_args()

    run_comparison(args.symbol, args.start, args.end, args.sort)


if __name__ == "__main__":
    main()
