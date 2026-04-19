from __future__ import annotations

import pandas as pd


def build_stacked_feature_frame(
    label_df: pd.DataFrame,
    xgb_df: pd.DataFrame,
    lstm_df: pd.DataFrame | None,
    gap_df: pd.DataFrame,
    garch_sigma: float,
) -> pd.DataFrame:
    frame = label_df[["session_date", "next_open", "next_high", "next_low", "next_close", "next_direction"]].copy()
    frame = frame.merge(
        xgb_df[["session_date", "xgb_open", "xgb_high", "xgb_low", "xgb_close", "xgb_dir_prob"]],
        on="session_date",
        how="inner",
    )
    if lstm_df is not None and not lstm_df.empty:
        frame = frame.merge(
            lstm_df[
                [
                    "session_date",
                    "lstm_open",
                    "lstm_high",
                    "lstm_low",
                    "lstm_close",
                    "lstm_dir_prob",
                ]
            ],
            on="session_date",
            how="inner",
        )
    else:
        frame["lstm_open"] = frame["xgb_open"]
        frame["lstm_high"] = frame["xgb_high"]
        frame["lstm_low"] = frame["xgb_low"]
        frame["lstm_close"] = frame["xgb_close"]
        frame["lstm_dir_prob"] = frame["xgb_dir_prob"]
    frame = frame.merge(gap_df[["session_date", "gap_pred", "gap_open"]], on="session_date", how="inner")
    frame["garch_sigma"] = float(garch_sigma)
    return frame.dropna().sort_values("session_date")
