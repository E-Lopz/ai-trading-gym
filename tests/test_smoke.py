"""
Smoke test: runs SMA crossover on synthetic data to verify the entire pipeline works
without needing any API keys or internet access.
"""
import numpy as np
import pandas as pd
import pytest
from backtesting.engine import BacktestEngine
from strategies.sma_crossover import SMACrossover


def _make_synthetic_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": rng.integers(100_000, 1_000_000, n),
        },
        index=idx,
    )


def test_backtest_runs_and_returns_metrics():
    data = _make_synthetic_data()
    engine = BacktestEngine(initial_cash=100_000)
    result = engine.run(SMACrossover(fast=10, slow=30), data, symbol="SYNTH")

    assert len(result.equity_curve) == len(data)
    m = result.metrics
    assert "sharpe" in m
    assert "max_drawdown" in m
    assert result.equity_curve.iloc[0] == 100_000.0


def test_equity_curve_is_positive():
    data = _make_synthetic_data()
    engine = BacktestEngine(initial_cash=100_000)
    result = engine.run(SMACrossover(), data, symbol="SYNTH")
    assert result.equity_curve.min() > 0
