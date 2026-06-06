import pandas as pd


def resample_candles(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    required = {"ts", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for resampling: {sorted(missing)}")
    work = df.copy()
    work["ts"] = pd.to_datetime(work["ts"], utc=True)
    work = work.sort_values("ts").set_index("ts")

    rule_map = {"30minute": "30min", "day": "1D"}
    if target_interval not in rule_map:
        raise ValueError(f"Unsupported target_interval={target_interval}")

    resample_kwargs = {}
    if target_interval == "30minute":
        # Align bins from the first seen candle (e.g., 09:15 IST) instead of clock-hour boundaries.
        resample_kwargs = {"origin": work.index.min()}

    agg = work.resample(rule_map[target_interval], **resample_kwargs).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    agg = agg.dropna().reset_index()
    return agg
