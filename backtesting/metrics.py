import numpy as np


def sharpe_ratio(returns, risk_free_rate: float = 0.0, annualization: int = 252) -> float:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    excess = arr - (risk_free_rate / annualization)
    denom = np.std(excess)
    if denom < 1e-12:
        return 0.0
    return float(np.sqrt(annualization) * np.mean(excess) / denom)


def max_drawdown(equity_curve) -> float:
    arr = np.asarray(equity_curve, dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / (peak + 1e-9)
    return float(dd.min())
