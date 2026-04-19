import pandas as pd


def build_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("ts")
    out["vol_avg_20"] = out["volume"].rolling(20).mean()
    out["vol_surge_ratio"] = out["volume"] / (out["vol_avg_20"] + 1e-9)
    out["turnover"] = out["close"] * out["volume"]
    out["turnover_avg_20"] = out["turnover"].rolling(20).mean()
    out["turnover_surge_ratio"] = out["turnover"] / (out["turnover_avg_20"] + 1e-9)
    return out
