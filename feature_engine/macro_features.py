from __future__ import annotations

import numpy as np
import pandas as pd


def _canonical_symbol(symbol: str) -> str:
    cleaned = " ".join(str(symbol or "").strip().upper().split())
    aliases = {
        "NIFTY": "NIFTY 50",
        "NIFTY50": "NIFTY 50",
        "NIFTY 50": "NIFTY 50",
        "SENSEX": "SENSEX",
        "BSE SENSEX": "SENSEX",
        "INDIA VIX": "INDIA VIX",
        "INDIAVIX": "INDIA VIX",
        "VIX": "INDIA VIX",
    }
    return aliases.get(cleaned, cleaned)


def _peer_symbol(symbol: str, available_symbols: set[str]) -> str | None:
    canonical = _canonical_symbol(symbol)
    preferred = {
        "SENSEX": "NIFTY 50",
        "NIFTY 50": "SENSEX",
        "INDIA VIX": "NIFTY 50",
    }
    candidate = preferred.get(canonical)
    if candidate and candidate in available_symbols and candidate != canonical:
        return candidate
    for alt in sorted(available_symbols):
        if alt not in {canonical, "INDIA VIX"}:
            return alt
    return None


def _rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def _symbol_feature_frame(market_df: pd.DataFrame, symbol: str, prefix: str) -> pd.DataFrame:
    if market_df.empty:
        return pd.DataFrame(columns=["session_date"])

    work = market_df[market_df["symbol_canonical"] == symbol].copy()
    if work.empty:
        return pd.DataFrame(columns=["session_date"])

    work = work.sort_values("ts").drop_duplicates(subset=["session_date"], keep="last")
    work[f"{prefix}_ret_1d"] = work["close"].pct_change(1)
    work[f"{prefix}_ret_5d"] = work["close"].pct_change(5)
    work[f"{prefix}_gap"] = (work["open"] - work["close"].shift(1)) / (work["close"].shift(1) + 1e-9)
    work[f"{prefix}_range_pct"] = (work["high"] - work["low"]) / (work["close"] + 1e-9)
    work[f"{prefix}_volatility_20d"] = work["close"].pct_change().rolling(20).std(ddof=0)
    work[f"{prefix}_z_20"] = _rolling_zscore(work["close"], window=20)
    work[f"{prefix}_close_level"] = work["close"].astype(float)
    cols = [
        "session_date",
        f"{prefix}_ret_1d",
        f"{prefix}_ret_5d",
        f"{prefix}_gap",
        f"{prefix}_range_pct",
        f"{prefix}_volatility_20d",
        f"{prefix}_z_20",
        f"{prefix}_close_level",
    ]
    return work[cols]


def build_macro_features(
    df: pd.DataFrame,
    symbol: str,
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if "session_date" not in out:
        out["session_date"] = pd.to_datetime(out["ts"]).dt.date

    market_df = market_df.copy() if market_df is not None else pd.DataFrame()
    if not market_df.empty:
        market_df["session_date"] = pd.to_datetime(market_df["ts"]).dt.date
        market_df["symbol_canonical"] = market_df["symbol"].map(_canonical_symbol)
    else:
        market_df = pd.DataFrame(columns=["session_date", "symbol_canonical"])

    available_symbols = set(market_df.get("symbol_canonical", pd.Series(dtype=str)).dropna().unique().tolist())
    peer_symbol = _peer_symbol(symbol, available_symbols)

    if peer_symbol:
        out = out.merge(_symbol_feature_frame(market_df, peer_symbol, prefix="peer"), on="session_date", how="left")
    else:
        for col in (
            "peer_ret_1d",
            "peer_ret_5d",
            "peer_gap",
            "peer_range_pct",
            "peer_volatility_20d",
            "peer_z_20",
            "peer_close_level",
        ):
            out[col] = 0.0

    if "INDIA VIX" in available_symbols:
        out = out.merge(
            _symbol_feature_frame(market_df, "INDIA VIX", prefix="india_vix"),
            on="session_date",
            how="left",
        )
    else:
        for col in (
            "india_vix_ret_1d",
            "india_vix_ret_5d",
            "india_vix_gap",
            "india_vix_range_pct",
            "india_vix_volatility_20d",
            "india_vix_z_20",
            "india_vix_close_level",
        ):
            out[col] = 0.0

    out["rel_strength_1d"] = out.get("ret_1d", 0.0) - out.get("peer_ret_1d", 0.0)
    out["rel_strength_5d"] = out.get("ret_5d", 0.0) - out.get("peer_ret_5d", 0.0)
    out["risk_off_score"] = (
        0.6 * out.get("india_vix_ret_1d", 0.0).fillna(0.0)
        + 0.4 * out.get("india_vix_z_20", 0.0).fillna(0.0)
    )

    if "peer_ret_1d" in out:
        target_ret = out.get("ret_1d", pd.Series(0.0, index=out.index)).astype(float)
        peer_ret = out["peer_ret_1d"].astype(float)
        corr = target_ret.rolling(20).corr(peer_ret)
        cov = target_ret.rolling(20).cov(peer_ret)
        peer_var = peer_ret.rolling(20).var(ddof=0).replace(0, np.nan)
        out["peer_corr_20"] = corr
        out["peer_beta_20"] = cov / peer_var
    else:
        out["peer_corr_20"] = 0.0
        out["peer_beta_20"] = 0.0

    for col in [
        "peer_ret_1d",
        "peer_ret_5d",
        "peer_gap",
        "peer_range_pct",
        "peer_volatility_20d",
        "peer_z_20",
        "peer_close_level",
        "india_vix_ret_1d",
        "india_vix_ret_5d",
        "india_vix_gap",
        "india_vix_range_pct",
        "india_vix_volatility_20d",
        "india_vix_z_20",
        "india_vix_close_level",
        "rel_strength_1d",
        "rel_strength_5d",
        "risk_off_score",
        "peer_corr_20",
        "peer_beta_20",
    ]:
        if col not in out:
            out[col] = 0.0
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "usd_inr" not in out:
        out["usd_inr"] = 0.0
    if "fii_dii_net" not in out:
        out["fii_dii_net"] = 0.0
    return out
