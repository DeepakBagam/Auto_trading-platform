from __future__ import annotations

from dataclasses import dataclass

from utils.pine_strategy import row_setup_flags, strategy_profile_for_symbol
from utils.symbols import normalize_symbol_key


@dataclass(slots=True)
class TradeSignal:
    action: str
    conviction: str
    expected_return_pct: float
    expected_move_points: float
    stop_loss: float | None
    take_profit: float | None
    risk_reward_ratio: float | None
    technical_score: float
    reasons: list[str]


def _to_float(value) -> float | None:
    try:
        out = float(value)
        return out if out == out else None
    except (TypeError, ValueError):
        return None


def _conviction_bucket(confidence: float) -> str:
    if confidence >= 0.80:
        return "high"
    if confidence >= 0.68:
        return "medium"
    return "low"


def _minimum_expected_move_threshold(
    symbol: str | None,
    latest_price: float,
    technical_context: dict | None,
    configured_floor: float,
) -> float:
    tc = technical_context or {}
    atr_14 = _to_float(tc.get("atr_14")) or 0.0
    normalized = normalize_symbol_key(symbol or "")
    if normalized == "INDIAVIX" or "VIX" in normalized:
        dynamic = max(0.12, atr_14 * 0.60, latest_price * 0.0060)
    elif normalized == "BANKNIFTY":
        dynamic = max(60.0, atr_14 * 0.80, latest_price * 0.0018)
    elif normalized == "SENSEX":
        dynamic = max(75.0, atr_14 * 0.80, latest_price * 0.0015)
    else:
        dynamic = max(50.0, atr_14 * 0.80, latest_price * 0.0020)
    if configured_floor > 0:
        return float(min(configured_floor, dynamic))
    return float(dynamic)


def actionable_direction_from_forecast(
    *,
    direction: str,
    latest_price: float,
    predicted_price: float,
    technical_context: dict | None = None,
) -> str:
    normalized = str(direction or "HOLD").upper()
    if normalized in {"BUY", "SELL"}:
        return normalized
    tc = technical_context or {}
    atr_14 = _to_float(tc.get("atr_14")) or 0.0
    move_points = float(predicted_price) - float(latest_price)
    floor_points = max(5.0, atr_14 * 0.35, float(latest_price) * 0.00025)
    if move_points > floor_points:
        return "BUY"
    if move_points < -floor_points:
        return "SELL"
    return "HOLD"


def _strategy_checks(direction: str, technical_context: dict | None, symbol: str | None = None) -> tuple[float, list[str], bool]:
    tc = technical_context or {}
    profile = strategy_profile_for_symbol(symbol)
    setups = row_setup_flags(tc, symbol=symbol)
    reasons: list[str] = []
    dominant_action = str(setups.get("dominant_action") or "NEUTRAL")

    if direction == "BUY":
        if not profile.allow_long:
            reasons.append(f"{profile.name} blocks BUY entries for {symbol or 'this symbol'}")
            return 0.0, reasons, False
        score = float(setups.get("buy_score") or 0.0)
        names = list(setups.get("buy_strategy_names") or [])
        alignment = int(setups.get("buy_alignment_count") or 0)
        if bool(setups.get("buy_setup")):
            reasons.append(f"BUY strategy confirmations: {', '.join(names[:3])}")
            reasons.append(f"Short interval BUY alignment passed on {alignment} of 2 frames")
        elif names and dominant_action != "BUY":
            reasons.append("BUY setups exist, but opposite-side score is stronger")
        elif bool(setups.get("base_buy_setup")) and alignment == 0:
            reasons.append("1m BUY setup exists, but 3m/5m alignment is missing")
        else:
            reasons.append("No Pine-style BUY strategy passed")
        return score + (alignment * 0.20), reasons, bool(setups.get("buy_setup"))

    if direction == "SELL":
        if not profile.allow_short:
            reasons.append(f"{profile.name} blocks SELL entries for {symbol or 'this symbol'}")
            return 0.0, reasons, False
        score = float(setups.get("sell_score") or 0.0)
        names = list(setups.get("sell_strategy_names") or [])
        alignment = int(setups.get("sell_alignment_count") or 0)
        if bool(setups.get("sell_setup")):
            reasons.append(f"SELL strategy confirmations: {', '.join(names[:3])}")
            reasons.append(f"Short interval SELL alignment passed on {alignment} of 2 frames")
        elif names and dominant_action != "SELL":
            reasons.append("SELL setups exist, but opposite-side score is stronger")
        elif bool(setups.get("base_sell_setup")) and alignment == 0:
            reasons.append("1m SELL setup exists, but 3m/5m alignment is missing")
        else:
            reasons.append("No Pine-style SELL strategy passed")
        return score + (alignment * 0.20), reasons, bool(setups.get("sell_setup"))

    reasons.append("Direction is HOLD, strategy confirmation unavailable")
    return 0.0, reasons, False


def build_trade_signal(
    *,
    symbol: str | None = None,
    interval: str,
    direction: str,
    confidence: float,
    latest_price: float,
    predicted_price: float,
    pred_high: float,
    pred_low: float,
    pred_interval_close: dict | None = None,
    technical_context: dict | None = None,
    buy_threshold: float = 0.62,
    sell_threshold: float = 0.62,
    minimum_expected_move: float = 80.0,
    news_sentiment: float = 0.0,
) -> TradeSignal:
    del interval
    ref_price = float(latest_price)
    predicted = float(predicted_price)
    direction = actionable_direction_from_forecast(
        direction=direction,
        latest_price=ref_price,
        predicted_price=predicted,
        technical_context=technical_context,
    )
    expected_return_pct = 0.0 if ref_price == 0 else (predicted - ref_price) / ref_price
    expected_move_points = predicted - ref_price
    confidence = float(confidence)
    conviction = _conviction_bucket(confidence)
    reasons: list[str] = []

    close_band = pred_interval_close or {}
    band_low = _to_float(close_band.get("p10"))
    band_high = _to_float(close_band.get("p90"))

    stop_loss: float | None = None
    take_profit: float | None = None
    if direction == "BUY":
        stop_loss = band_low if band_low is not None else min(ref_price, float(pred_low))
        take_profit = band_high if band_high is not None else max(predicted, float(pred_high))
    elif direction == "SELL":
        stop_loss = band_high if band_high is not None else max(ref_price, float(pred_high))
        take_profit = band_low if band_low is not None else min(predicted, float(pred_low))

    risk_reward_ratio: float | None = None
    if stop_loss is not None and take_profit is not None:
        reward = abs(take_profit - ref_price)
        risk = abs(ref_price - stop_loss)
        risk_reward_ratio = (reward / risk) if risk > 0 else None

    technical_score, technical_reasons, strategy_passed = _strategy_checks(direction, technical_context, symbol=symbol)
    reasons.extend(technical_reasons[:3])
    effective_move_threshold = _minimum_expected_move_threshold(
        symbol=symbol,
        latest_price=ref_price,
        technical_context=technical_context,
        configured_floor=float(minimum_expected_move),
    )
    threshold_discount = min(0.06, max(0.0, technical_score) * 0.025)
    effective_buy_threshold = max(0.45, buy_threshold - threshold_discount)
    effective_sell_threshold = max(0.45, sell_threshold - threshold_discount)

    if abs(news_sentiment) > 0.3:
        if direction == "BUY" and news_sentiment > 0.3:
            technical_score += 0.25
            reasons.append(f"Positive news sentiment supports BUY ({news_sentiment:.2f})")
        elif direction == "SELL" and news_sentiment < -0.3:
            technical_score += 0.25
            reasons.append(f"Negative news sentiment supports SELL ({news_sentiment:.2f})")
        else:
            technical_score -= 0.25
            reasons.append(f"News sentiment is against the {direction} setup ({news_sentiment:.2f})")
    else:
        reasons.append("News sentiment is neutral")

    action = "HOLD"
    if direction == "BUY":
        if confidence < effective_buy_threshold:
            reasons.append(f"ML confidence below BUY threshold ({confidence:.2f} < {effective_buy_threshold:.2f})")
        elif expected_move_points < effective_move_threshold:
            reasons.append(
                f"Expected move too small ({expected_move_points:.2f} < {effective_move_threshold:.2f} points)"
            )
        elif predicted <= ref_price:
            reasons.append("Predicted close does not support BUY direction")
        elif not strategy_passed:
            reasons.append("No Pine-style BUY strategy confirmation passed")
        else:
            action = "BUY"
    elif direction == "SELL":
        if confidence < effective_sell_threshold:
            reasons.append(f"ML confidence below SELL threshold ({confidence:.2f} < {effective_sell_threshold:.2f})")
        elif abs(expected_move_points) < effective_move_threshold:
            reasons.append(
                f"Expected move too small ({abs(expected_move_points):.2f} < {effective_move_threshold:.2f} points)"
            )
        elif predicted >= ref_price:
            reasons.append("Predicted close does not support SELL direction")
        elif not strategy_passed:
            reasons.append("No Pine-style SELL strategy confirmation passed")
        else:
            action = "SELL"
    else:
        reasons.append("Model direction is HOLD")

    if action != "HOLD" and risk_reward_ratio is not None and risk_reward_ratio < 1.20:
        reasons.append(f"Risk/reward too weak ({risk_reward_ratio:.2f} < 1.20)")
        action = "HOLD"

    if action != "HOLD":
        reasons.append(f"Expected move {expected_move_points:.2f} points cleared threshold {effective_move_threshold:.2f}")
        if risk_reward_ratio is not None:
            reasons.append(f"Risk/reward {risk_reward_ratio:.2f}")

    return TradeSignal(
        action=action,
        conviction=conviction,
        expected_return_pct=float(expected_return_pct),
        expected_move_points=float(expected_move_points),
        stop_loss=(float(stop_loss) if stop_loss is not None else None),
        take_profit=(float(take_profit) if take_profit is not None else None),
        risk_reward_ratio=(float(risk_reward_ratio) if risk_reward_ratio is not None else None),
        technical_score=float(technical_score),
        reasons=reasons,
    )
