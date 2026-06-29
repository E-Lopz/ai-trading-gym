import numpy as np
import pandas as pd


def compute_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Compute standard performance metrics from an equity curve (indexed by date).
    Returns a dict suitable for ranking and display.
    """
    returns = equity_curve.pct_change().dropna()

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    trading_days = len(returns)
    annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0.0

    excess = returns - risk_free_rate / 252
    sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    downside = returns[returns < 0]
    sortino = (excess.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0.0

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    win_rate = (returns > 0).mean()

    return {
        "total_return": round(total_return * 100, 2),
        "annualized_return": round(annualized_return * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown": round(max_drawdown * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate": round(win_rate * 100, 2),
        "n_bars": trading_days,
    }
