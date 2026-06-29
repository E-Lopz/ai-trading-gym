"""
Unit tests for BasePortfolioStrategy.rebalance_to_weights().
All tests drive the method directly with a synthetic broker — no engine needed.
"""
import numpy as np
import pandas as pd
import pytest

from gym.broker.simulated import SimulatedBroker
from gym.portfolio import Position
from strategies.portfolio_base import BasePortfolioStrategy


# ── helpers ──────────────────────────────────────────────────────────────────

class _Strat(BasePortfolioStrategy):
    """Minimal concrete subclass — on_portfolio_bar not used in these tests."""
    def on_portfolio_bar(self, bars, history):
        pass


def _make_bars(*prices: float, symbols=None) -> dict[str, pd.Series]:
    syms = symbols or [chr(65 + i) for i in range(len(prices))]
    return {
        sym: pd.Series({"open": p * 0.999, "high": p * 1.005,
                        "low": p * 0.995, "close": p, "volume": 1_000_000})
        for sym, p in zip(syms, prices)
    }


def _setup(symbols, initial_cash=100_000, positions=None):
    """
    Create a strategy + broker pair.
    positions: dict[sym -> (qty, avg_price)]  pre-fills the portfolio.
    """
    broker = SimulatedBroker(initial_cash)
    for sym, (qty, avg_price) in (positions or {}).items():
        broker.portfolio.positions[sym] = Position(sym, qty, avg_price)
        broker.portfolio.cash -= qty * avg_price

    strat = _Strat()
    strat.broker = broker
    strat.symbols = list(symbols)
    return strat, broker


def _pending_orders(broker):
    return list(broker._pending)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_equal_weights_from_cash_submits_two_buys():
    """Starting from all-cash, equal weights should produce two buy orders."""
    strat, broker = _setup(["A", "B"])
    bars = _make_bars(100.0, 50.0)

    strat.rebalance_to_weights({"A": 0.5, "B": 0.5}, bars)

    orders = _pending_orders(broker)
    assert len(orders) == 2
    assert all(o.qty > 0 for o in orders)
    syms = {o.symbol for o in orders}
    assert syms == {"A", "B"}


def test_zeroing_a_position_submits_sell():
    """Rebalancing A to 0 when we hold A should produce a sell for A and a buy for B."""
    strat, broker = _setup(
        ["A", "B"],
        initial_cash=100_000,
        positions={"A": (300, 100.0)},  # hold 300 shares of A @ $100
    )
    bars = _make_bars(100.0, 100.0)

    strat.rebalance_to_weights({"A": 0.0, "B": 1.0}, bars)

    orders = _pending_orders(broker)
    sell_orders = [o for o in orders if o.qty < 0]
    buy_orders  = [o for o in orders if o.qty > 0]
    assert any(o.symbol == "A" for o in sell_orders), "Expected sell for A"
    assert any(o.symbol == "B" for o in buy_orders),  "Expected buy for B"


def test_weight_normalisation():
    """Unnormalised weights [2, 2] should behave identically to [0.5, 0.5]."""
    strat1, broker1 = _setup(["A", "B"])
    strat2, broker2 = _setup(["A", "B"])
    bars = _make_bars(100.0, 100.0)

    strat1.rebalance_to_weights({"A": 2.0, "B": 2.0}, bars)
    strat2.rebalance_to_weights({"A": 0.5, "B": 0.5}, bars)

    orders1 = sorted(_pending_orders(broker1), key=lambda o: o.symbol)
    orders2 = sorted(_pending_orders(broker2), key=lambda o: o.symbol)
    assert len(orders1) == len(orders2)
    for o1, o2 in zip(orders1, orders2):
        assert o1.symbol == o2.symbol
        assert o1.qty == o2.qty


def test_ndarray_weights_parallel_to_symbols():
    """np.ndarray weights should be matched to self.symbols positionally."""
    strat, broker = _setup(["A", "B"])
    bars = _make_bars(100.0, 100.0)

    strat.rebalance_to_weights(np.array([0.3, 0.7]), bars)

    orders = _pending_orders(broker)
    assert len(orders) == 2
    qty_a = next(o.qty for o in orders if o.symbol == "A")
    qty_b = next(o.qty for o in orders if o.symbol == "B")
    # B should get more shares (70 % vs 30 % of $100k at $100/share)
    assert qty_b > qty_a


def test_min_trade_threshold_suppresses_tiny_rebalance():
    """A position one share away from target should be suppressed when threshold > share price."""
    # equity = $100k; hold 499 shares of A @ $100 → current value $49,900.
    # target A = 50% = $50,000; diff = $100 (one share).
    # With min_trade_value=$200, diff $100 < $200 → no A order.
    # B has no position; diff = $50,000 >> threshold → B buy IS placed (proves threshold
    # only suppresses A, not all orders).
    strat, broker = _setup(
        ["A", "B"],
        initial_cash=100_000,
        positions={"A": (499, 100.0)},
    )
    bars = _make_bars(100.0, 100.0)

    strat.rebalance_to_weights({"A": 0.5, "B": 0.5}, bars, min_trade_value=200.0)

    orders = _pending_orders(broker)
    a_orders = [o for o in orders if o.symbol == "A"]
    b_orders = [o for o in orders if o.symbol == "B"]
    assert len(a_orders) == 0, "A diff ($100) < threshold ($200) → no order expected"
    assert len(b_orders) == 1, "B diff ($50k) >> threshold → buy expected"


def test_all_zero_weights_is_noop():
    """All-zero weight vector should not submit any orders."""
    strat, broker = _setup(["A", "B"])
    bars = _make_bars(100.0, 100.0)

    strat.rebalance_to_weights({"A": 0.0, "B": 0.0}, bars)

    assert len(_pending_orders(broker)) == 0


def test_negative_weights_clipped_to_zero():
    """Negative weights are treated as zero (long-only constraint)."""
    strat, broker = _setup(["A", "B"])
    bars = _make_bars(100.0, 100.0)

    # -1 for A gets clipped to 0 → effectively all weight on B
    strat.rebalance_to_weights({"A": -1.0, "B": 1.0}, bars)

    orders = _pending_orders(broker)
    a_orders = [o for o in orders if o.symbol == "A"]
    assert len(a_orders) == 0  # A clipped, no buy for A


def test_sells_submitted_before_buys():
    """Sell orders must appear before buy orders in _pending."""
    strat, broker = _setup(
        ["A", "B"],
        initial_cash=100_000,
        positions={"A": (500, 100.0)},  # hold A, want to swap into B
    )
    bars = _make_bars(100.0, 100.0)

    strat.rebalance_to_weights({"A": 0.0, "B": 1.0}, bars)

    orders = _pending_orders(broker)
    sell_idx = next(i for i, o in enumerate(orders) if o.qty < 0)
    buy_idx  = next(i for i, o in enumerate(orders) if o.qty > 0)
    assert sell_idx < buy_idx
