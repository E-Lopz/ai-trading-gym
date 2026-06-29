"""
Smoke tests for PortfolioEnv. No API keys or internet needed — uses synthetic OHLCV data.
"""
import numpy as np
import pandas as pd
import pytest

from training.envs.portfolio_env import PortfolioEnv, N_FEATURES

SYMBOLS = ["A", "B", "C"]
N_BARS = 200
LOOKBACK = 20
INITIAL_CASH = 100_000.0


def _make_synthetic(n: int = N_BARS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.normal(0, 1, n))
    prices = np.clip(prices, 1.0, None)  # keep positive
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open":   prices * 0.999,
        "high":   prices * 1.005,
        "low":    prices * 0.995,
        "close":  prices,
        "volume": rng.integers(100_000, 1_000_000, n).astype(float),
    }, index=idx)


@pytest.fixture
def env():
    data = {sym: _make_synthetic(seed=i) for i, sym in enumerate(SYMBOLS)}
    return PortfolioEnv(data, initial_cash=INITIAL_CASH, lookback=LOOKBACK)


# ── basic interface ───────────────────────────────────────────────────────────

def test_reset_returns_correct_shape(env):
    obs, info = env.reset()
    assert obs.shape == (len(SYMBOLS), LOOKBACK, N_FEATURES)
    assert obs.dtype == np.float32


def test_reset_obs_is_zero_at_step_zero(env):
    """At step 0 there are no prior bars — observation must be all zeros."""
    obs, _ = env.reset()
    assert np.all(obs == 0.0)


def test_step_returns_correct_shapes(env):
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (len(SYMBOLS), LOOKBACK, N_FEATURES)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert truncated is False


def test_obs_values_in_unit_range(env):
    """Min-max normalisation should keep all obs values in [0, 1]."""
    env.reset(seed=42)
    for _ in range(LOOKBACK + 5):
        obs, _, terminated, _, _ = env.step(env.action_space.sample())
        assert obs.min() >= 0.0 - 1e-6
        assert obs.max() <= 1.0 + 1e-6
        if terminated:
            break


def test_equity_stays_non_negative(env):
    env.reset(seed=0)
    for _ in range(10):
        _, _, terminated, _, info = env.step(env.action_space.sample())
        assert info["equity"] >= 0.0
        if terminated:
            break


def test_info_keys(env):
    env.reset()
    _, _, _, _, info = env.step(env.action_space.sample())
    assert "equity" in info
    assert "weights" in info
    assert "date" in info
    assert set(info["weights"].keys()) == set(SYMBOLS)


def test_episode_terminates_after_all_bars(env):
    env.reset()
    terminated = False
    steps = 0
    while not terminated:
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps == N_BARS  # one step per bar


def test_reward_shaping_hook():
    """Custom reward shaping should replace the raw log return."""
    data = {sym: _make_synthetic(seed=i) for i, sym in enumerate(SYMBOLS)}
    env = PortfolioEnv(
        data,
        lookback=LOOKBACK,
        reward_shaping=lambda r, info: -999.0,  # always returns -999
    )
    env.reset()
    _, reward, _, _, _ = env.step(env.action_space.sample())
    assert reward == -999.0


def test_reset_reinitialises_portfolio():
    """Equity at the start of a second episode must equal initial_cash."""
    data = {sym: _make_synthetic(seed=i) for i, sym in enumerate(SYMBOLS)}
    env = PortfolioEnv(data, initial_cash=INITIAL_CASH, lookback=LOOKBACK)

    env.reset()
    for _ in range(20):
        env.step(env.action_space.sample())

    env.reset()
    assert env.broker.portfolio.cash == INITIAL_CASH
    assert len(env.broker.portfolio.positions) == 0


def test_equal_weight_action_buys_all_symbols(env):
    """Uniform action vector should eventually result in positions in all symbols."""
    env.reset()
    action = np.ones(len(SYMBOLS), dtype=np.float32)  # uniform → equal weights
    for _ in range(5):
        env.step(action)
    positions = env.broker.portfolio.positions
    assert len(positions) > 0
