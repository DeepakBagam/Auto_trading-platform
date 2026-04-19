import pandas as pd


def direction_strategy(predictions: pd.DataFrame, actual_close_col: str = "actual_close") -> pd.DataFrame:
    df = predictions.copy()
    df["ret"] = df[actual_close_col].pct_change().fillna(0.0)
    df["signal"] = df["direction"].map({"BUY": 1, "SELL": -1, "HOLD": 0}).fillna(0)
    df["strategy_ret"] = df["signal"].shift(1).fillna(0) * df["ret"]
    df["equity_curve"] = (1 + df["strategy_ret"]).cumprod()
    return df
