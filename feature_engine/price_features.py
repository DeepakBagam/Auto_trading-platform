import pandas as pd


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    macd_line = _ema(series, 12) - _ema(series, 26)
    signal = _ema(macd_line, 9)
    return macd_line, signal


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _engulfing_pattern(out: pd.DataFrame) -> pd.Series:
    prev_open = out["open"].shift(1)
    prev_close = out["close"].shift(1)
    prev_body = (prev_close - prev_open).abs()
    cur_body = (out["close"] - out["open"]).abs()
    opposite_direction = (out["close"] >= out["open"]) != (prev_close >= prev_open)
    larger_body = cur_body > prev_body * 1.1
    return (opposite_direction & larger_body).astype(int)


def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("ts")
    volume = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    has_real_volume = bool((volume > 0).any())
    if has_real_volume:
        vol_cum = volume.cumsum().where(lambda s: s > 0)
        vwap = ((out["close"] * volume).cumsum() / vol_cum).replace(
            [float("inf"), float("-inf")], pd.NA
        )
        out["vwap"] = vwap.ffill().bfill().fillna(out["close"])
        volume_sma = volume.replace(0.0, pd.NA).rolling(20, min_periods=1).mean()
        out["volume_sma_20"] = volume_sma.fillna(0.0)
        volume_ratio = (volume / volume_sma.where(volume_sma > 0)).replace(
            [float("inf"), float("-inf")], pd.NA
        )
        out["volume_ratio_20"] = volume_ratio.fillna(1.0)
    else:
        # Index feeds often do not carry usable volume. Keep those features neutral so the
        # live signal stack is not penalized for missing exchange volume.
        out["vwap"] = out["close"]
        out["volume_sma_20"] = 0.0
        out["volume_ratio_20"] = 1.0
    out["ret_1d"] = out["close"].pct_change(1)
    out["ret_5d"] = out["close"].pct_change(5)
    out["gap"] = (out["open"] - out["close"].shift(1)) / (out["close"].shift(1) + 1e-9)
    out["rsi_14"] = _rsi(out["close"], 14)
    macd_line, macd_signal = _macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_line - macd_signal
    out["macd_hist_delta_1"] = out["macd_hist"].diff()
    out["ema_9"] = _ema(out["close"], 9)
    out["ema_21"] = _ema(out["close"], 21)
    out["ema_50"] = _ema(out["close"], 50)
    out["ema_200"] = _ema(out["close"], 200)
    out["ema_21_slope_3"] = out["ema_21"] - out["ema_21"].shift(3)
    rolling_20 = out["close"].rolling(20)
    out["bb_mid"] = rolling_20.mean()
    bb_std = rolling_20.std(ddof=0)
    out["bb_upper"] = out["bb_mid"] + 2 * bb_std
    out["bb_lower"] = out["bb_mid"] - 2 * bb_std
    out["atr_14"] = _atr(out, 14)
    out["kc_mid"] = _ema(out["close"], 20)
    out["kc_upper"] = out["kc_mid"] + 2 * _atr(out, 10)
    out["kc_lower"] = out["kc_mid"] - 2 * _atr(out, 10)
    out["volatility_20d"] = out["ret_1d"].rolling(20).std(ddof=0)
    price_range = (out["high"] - out["low"]).replace(0, 1e-9)
    body = (out["close"] - out["open"]).abs()
    upper_wick = out["high"] - out[["open", "close"]].max(axis=1)
    lower_wick = out[["open", "close"]].min(axis=1) - out["low"]
    out["body_size"] = body
    out["body_pct_range"] = body / price_range
    out["upper_wick_pct"] = upper_wick / price_range
    out["lower_wick_pct"] = lower_wick / price_range
    out["breakout_high_20"] = out["high"].rolling(20).max().shift(1)
    out["breakout_low_20"] = out["low"].rolling(20).min().shift(1)
    out["candle_green"] = (out["close"] > out["open"]).astype(int)
    out["candle_red"] = (out["close"] < out["open"]).astype(int)
    out["pattern_marubozu"] = (out["body_pct_range"] > 0.98).astype(int)
    out["pattern_hanging_man"] = (
        (out["lower_wick_pct"] > 0.75) & (out["body_pct_range"] < 0.15)
    ).astype(int)
    out["pattern_shooting_star"] = (
        (out["upper_wick_pct"] > 0.75) & (out["body_pct_range"] < 0.15)
    ).astype(int)
    out["pattern_spinning_top"] = (
        (out["upper_wick_pct"] < 0.60)
        & (out["lower_wick_pct"] > 0.40)
        & (out["body_pct_range"] < 0.15)
    ).astype(int)
    out["pattern_engulfing"] = _engulfing_pattern(out)
    return out
