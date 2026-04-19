from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time


@dataclass(slots=True)
class TradeIntelligenceScore:
    score: float
    trend_continuation_prob: float
    false_breakout_risk: float
    premium_expansion_prob: float
    tod_profitability_score: float
    reasons: list[str]


def _to_float(value) -> float | None:
    try:
        v = float(value)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tod_edge(now: datetime) -> tuple[float, str]:
    current = now.timetz().replace(tzinfo=None)
    if time(9, 20) <= current <= time(11, 15):
        return 0.85, "Opening trend window is favorable"
    if time(11, 16) <= current <= time(12, 45):
        return 0.45, "Mid-session quality is mixed"
    if time(12, 46) <= current <= time(13, 30):
        return 0.55, "Late entry window still open but weaker"
    if time(13, 31) <= current <= time(15, 15):
        return 0.20, "New entries are outside the preferred strategy window"
    return 0.05, "Outside intraday entry hours"


def score_trade_intelligence(
    *,
    signal_action: str,
    confidence: float,
    expected_return_pct: float,
    technical_context: dict | None,
    now: datetime,
    news_sentiment: float = 0.0,
    recent_prediction_accuracy: float = 0.5,
    vix_level: float | None = None,
) -> TradeIntelligenceScore:
    tc = technical_context or {}
    close = _to_float(tc.get("close"))
    ema9 = _to_float(tc.get("ema_9"))
    ema21 = _to_float(tc.get("ema_21"))
    ema50 = _to_float(tc.get("ema_50"))
    ema21_slope = _to_float(tc.get("ema_21_slope_3"))
    vwap = _to_float(tc.get("vwap")) or close
    rsi = _to_float(tc.get("rsi_14"))
    macd_hist = _to_float(tc.get("macd_hist"))
    atr = _to_float(tc.get("atr_14"))
    mtf_3m_action = str(tc.get("mtf_3m_action") or "").upper()
    mtf_5m_action = str(tc.get("mtf_5m_action") or "").upper()
    mtf_3m_buy_score = _to_float(tc.get("mtf_3m_buy_score"))
    mtf_3m_sell_score = _to_float(tc.get("mtf_3m_sell_score"))
    mtf_5m_buy_score = _to_float(tc.get("mtf_5m_buy_score"))
    mtf_5m_sell_score = _to_float(tc.get("mtf_5m_sell_score"))

    action = str(signal_action or "HOLD").upper()
    confidence = _clip01(confidence)
    recent_prediction_accuracy = _clip01(recent_prediction_accuracy)
    expected_abs = abs(float(expected_return_pct))

    reasons: list[str] = []

    trend_score = confidence * 0.45
    if action not in {"BUY", "SELL"}:
        trend_score = max(trend_score, 0.50)
        reasons.append("Setup is neutral; waiting for directional confirmation")
    else:
        if close is not None and ema9 is not None and ema21 is not None and ema50 is not None:
            if action == "BUY":
                if close > ema21:
                    trend_score += 0.25
                    if close > ema9 > ema21:
                        trend_score += 0.10
                        reasons.append("Price and EMA9 above EMA21 supports BUY")
                    else:
                        reasons.append("Price above EMA21 supports BUY")
                else:
                    trend_score -= 0.10
                    reasons.append("Price below EMA21 weakens BUY setup")
            elif action == "SELL":
                if close < ema21:
                    trend_score += 0.25
                    if close < ema9 < ema21:
                        trend_score += 0.10
                        reasons.append("Price and EMA9 below EMA21 supports SELL")
                    else:
                        reasons.append("Price below EMA21 supports SELL")
                else:
                    trend_score -= 0.10
                    reasons.append("Price above EMA21 weakens SELL setup")
        if close is not None and vwap is not None:
            if action == "BUY" and close >= vwap:
                trend_score += 0.08
            elif action == "SELL" and close <= vwap:
                trend_score += 0.08
            else:
                trend_score -= 0.08
        if macd_hist is not None:
            if action == "BUY" and macd_hist > 0:
                trend_score += 0.07
            elif action == "SELL" and macd_hist < 0:
                trend_score += 0.07
            else:
                trend_score -= 0.05
        if ema21_slope is not None:
            if action == "BUY" and ema21_slope > 0:
                trend_score += 0.04
            elif action == "SELL" and ema21_slope < 0:
                trend_score += 0.04
            else:
                trend_score -= 0.03
        mtf_hits = 0
        mtf_conflicts = 0
        for frame_name, frame_action in (("3m", mtf_3m_action), ("5m", mtf_5m_action)):
            if frame_action == action:
                trend_score += 0.06
                mtf_hits += 1
            elif frame_action in {"BUY", "SELL"} and frame_action != action:
                trend_score -= 0.05
                mtf_conflicts += 1
                reasons.append(f"{frame_name} trend is opposing the trade")
        if mtf_hits:
            reasons.append(f"Short-frame trend confirms on {mtf_hits}/2 frames")
        elif mtf_conflicts == 0:
            if action == "BUY":
                combined_buy_score = sum(value or 0.0 for value in (mtf_3m_buy_score, mtf_5m_buy_score))
                if combined_buy_score >= 1.4:
                    trend_score += 0.04
            elif action == "SELL":
                combined_sell_score = sum(value or 0.0 for value in (mtf_3m_sell_score, mtf_5m_sell_score))
                if combined_sell_score >= 1.4:
                    trend_score += 0.04
    trend_cont_prob = _clip01(trend_score)

    false_breakout = 0.35
    if rsi is not None:
        if action == "BUY" and rsi > 78:
            false_breakout += 0.25
            reasons.append("RSI is overheated for a fresh BUY")
        elif action == "SELL" and rsi < 22:
            false_breakout += 0.25
            reasons.append("RSI is stretched for a fresh SELL")
        else:
            false_breakout -= 0.08
    if atr is not None and close and close > 0:
        atr_ratio = atr / close
        if atr_ratio < 0.0015:
            false_breakout += 0.08
            reasons.append("ATR is compressed; breakout quality is weaker")
        else:
            false_breakout -= min(0.10, atr_ratio * 4.0)
        expected_move_ratio = (expected_abs / atr_ratio) if atr_ratio > 0 else 0.0
        if action in {"BUY", "SELL"}:
            if expected_move_ratio >= 1.2:
                false_breakout -= 0.06
                reasons.append("Forecasted move is large versus current ATR")
            elif expected_move_ratio < 0.35:
                false_breakout += 0.05
    false_breakout = _clip01(false_breakout)

    premium_expansion = 0.22 + min(0.40, expected_abs * 20.0)
    if confidence >= 0.80:
        premium_expansion += 0.12
    elif confidence >= 0.70:
        premium_expansion += 0.06
    if action not in {"BUY", "SELL"}:
        premium_expansion = max(premium_expansion, 0.42)
    if vix_level is not None:
        if vix_level < 13:
            premium_expansion += 0.06
            reasons.append("Low VIX regime supports cleaner premium expansion")
        elif vix_level > 22:
            premium_expansion -= 0.12
            reasons.append("High VIX regime reduces setup quality")
        elif vix_level > 17:
            premium_expansion -= 0.05
            reasons.append("Elevated VIX calls for smaller sizing")
    premium_expansion = _clip01(premium_expansion)

    tod_score, tod_reason = _tod_edge(now)
    reasons.append(tod_reason)

    sentiment_boost = 0.0
    if news_sentiment > 0.3 and action == "BUY":
        sentiment_boost = 0.08
        reasons.append("Positive news sentiment boosts BUY quality")
    elif news_sentiment < -0.3 and action == "SELL":
        sentiment_boost = 0.08
        reasons.append("Negative news sentiment boosts SELL quality")
    elif abs(news_sentiment) > 0.3:
        sentiment_boost = -0.08
        reasons.append("News sentiment is fighting the setup")

    final = (
        trend_cont_prob * 0.34
        + (1.0 - false_breakout) * 0.20
        + premium_expansion * 0.16
        + tod_score * 0.15
        + recent_prediction_accuracy * 0.10
        + (0.50 + sentiment_boost) * 0.05
    )
    final = _clip01(final)

    return TradeIntelligenceScore(
        score=round(final * 100.0, 2),
        trend_continuation_prob=round(trend_cont_prob, 4),
        false_breakout_risk=round(false_breakout, 4),
        premium_expansion_prob=round(premium_expansion, 4),
        tod_profitability_score=round(tod_score, 4),
        reasons=reasons[:6],
    )
