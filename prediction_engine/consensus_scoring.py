from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time
from functools import lru_cache
from pathlib import Path
from typing import Any

from utils.constants import IST_ZONE
from utils.symbols import normalize_symbol_key


@dataclass(frozen=True, slots=True)
class ConsensusThresholdProfile:
    symbol_key: str
    entry_score_threshold: float
    pine_led_score_threshold: float
    ml_confidence_floor: float
    combined_score_floor: float
    ai_score_floor: float
    pine_led_ai_floor: float
    expected_move_floor: float
    pine_max_age_seconds: int
    trade_drought_lookback_bars: int
    trade_drought_relax_pct: float
    lunch_relax_pct: float
    friday_relax_pct: float
    low_activity_trade_target: int
    low_activity_relax_pct: float
    max_total_relax_pct: float


@dataclass(frozen=True, slots=True)
class EffectiveThresholds:
    entry_score_threshold: float
    pine_led_score_threshold: float
    ml_confidence_floor: float
    combined_score_floor: float
    ai_score_floor: float
    pine_led_ai_floor: float
    expected_move_floor: float
    relaxation_pct: float
    reasons: list[str]


@dataclass(frozen=True, slots=True)
class CandidateScore:
    direction: str
    source: str
    ai_score: float
    weighted_score: float
    combined_score: float
    pine_score: float
    mtf_score: float
    ml_component: bool
    ai_component: bool
    pine_component: bool
    mtf_component: bool
    qualifies: bool
    pine_led_eligible: bool
    reasons: list[str]


def _profile_path() -> Path:
    return Path(__file__).with_name("consensus_profiles.json")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if out == out else default
    except (TypeError, ValueError):
        return default


def _safe_time_in_ist(now: datetime) -> time:
    aware = now if now.tzinfo is not None else now.replace(tzinfo=IST_ZONE)
    return aware.astimezone(IST_ZONE).timetz().replace(tzinfo=None)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=4)
def _load_profiles() -> dict[str, Any]:
    with _profile_path().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_threshold_profile(symbol: str) -> ConsensusThresholdProfile:
    payload = _load_profiles()
    defaults = dict(payload.get("defaults") or {})
    symbol_key = normalize_symbol_key(symbol or "")
    symbol_override = dict((payload.get("symbols") or {}).get(symbol_key, {}) or {})
    merged = _merge(defaults, symbol_override)
    return ConsensusThresholdProfile(
        symbol_key=symbol_key,
        entry_score_threshold=_to_float(merged.get("entry_score_threshold"), 70.0),
        pine_led_score_threshold=_to_float(merged.get("pine_led_score_threshold"), 50.0),
        ml_confidence_floor=_to_float(merged.get("ml_confidence_floor"), 0.55),
        combined_score_floor=_to_float(merged.get("combined_score_floor"), 0.50),
        ai_score_floor=_to_float(merged.get("ai_score_floor"), 35.0),
        pine_led_ai_floor=_to_float(merged.get("pine_led_ai_floor"), 30.0),
        expected_move_floor=_to_float(merged.get("expected_move_floor"), 30.0),
        pine_max_age_seconds=int(merged.get("pine_max_age_seconds") or 180),
        trade_drought_lookback_bars=int(merged.get("trade_drought_lookback_bars") or 20),
        trade_drought_relax_pct=_to_float(merged.get("trade_drought_relax_pct"), 0.15),
        lunch_relax_pct=_to_float(merged.get("lunch_relax_pct"), 0.10),
        friday_relax_pct=_to_float(merged.get("friday_relax_pct"), 0.05),
        low_activity_trade_target=int(merged.get("low_activity_trade_target") or 5),
        low_activity_relax_pct=_to_float(merged.get("low_activity_relax_pct"), 0.20),
        max_total_relax_pct=_to_float(merged.get("max_total_relax_pct"), 0.35),
    )


def compute_effective_thresholds(
    profile: ConsensusThresholdProfile,
    *,
    now: datetime,
    trades_today: int,
    bars_since_trade: int | None,
    recent_trade_count: int | None = None,
) -> EffectiveThresholds:
    relax_pct = 0.0
    reasons: list[str] = []
    current_time = _safe_time_in_ist(now)

    if time(11, 0) <= current_time < time(13, 0):
        relax_pct += profile.lunch_relax_pct
        reasons.append("lunch_relax")
    if now.astimezone(IST_ZONE).weekday() == 4:
        relax_pct += profile.friday_relax_pct
        reasons.append("friday_relax")
    if trades_today < profile.low_activity_trade_target:
        relax_pct += profile.low_activity_relax_pct
        reasons.append("low_activity_relax")
    if recent_trade_count == 0 or (
        bars_since_trade is not None and bars_since_trade >= profile.trade_drought_lookback_bars
    ):
        relax_pct += profile.trade_drought_relax_pct
        reasons.append("trade_drought_relax")

    relax_pct = min(profile.max_total_relax_pct, relax_pct)
    scale = max(0.50, 1.0 - relax_pct)
    min_expected_move = 0.05 if profile.symbol_key == "INDIAVIX" else 10.0

    return EffectiveThresholds(
        entry_score_threshold=round(max(50.0, profile.entry_score_threshold * scale), 2),
        pine_led_score_threshold=round(max(40.0, profile.pine_led_score_threshold * scale), 2),
        ml_confidence_floor=round(max(0.45, profile.ml_confidence_floor * scale), 3),
        combined_score_floor=round(max(0.40, profile.combined_score_floor * scale), 3),
        ai_score_floor=round(max(25.0, profile.ai_score_floor * scale), 2),
        pine_led_ai_floor=round(max(20.0, profile.pine_led_ai_floor * scale), 2),
        expected_move_floor=round(max(min_expected_move, profile.expected_move_floor * scale), 3),
        relaxation_pct=round(relax_pct, 4),
        reasons=reasons,
    )


def derive_ml_direction(*, direction: str, expected_move_points: float) -> str:
    """Convert ML direction to actionable signal. HOLD is converted to directional bias."""
    normalized = str(direction or "HOLD").upper()
    if normalized in {"BUY", "SELL"}:
        return normalized
    # Convert HOLD to directional bias based on expected move
    if expected_move_points > 0:
        return "BUY"
    if expected_move_points < 0:
        return "SELL"
    return "HOLD"


def derive_candidate_directions(*, ml_direction: str, pine_signal: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    ml_direction = str(ml_direction or "HOLD").upper()
    pine_signal = str(pine_signal or "HOLD").upper()
    if ml_direction in {"BUY", "SELL"}:
        candidates.append((ml_direction, "ml"))
    if pine_signal in {"BUY", "SELL"} and pine_signal != ml_direction:
        candidates.append((pine_signal, "pine"))
    return candidates


def technical_confirmation_direction(technical_context: dict[str, Any] | None) -> str:
    tc = technical_context or {}
    close = _to_float(tc.get("close"))
    ema9 = _to_float(tc.get("ema_9"))
    ema21 = _to_float(tc.get("ema_21"))
    ema50 = _to_float(tc.get("ema_50"))
    vwap = _to_float(tc.get("vwap"))
    rsi = _to_float(tc.get("rsi_14"))
    macd_hist = _to_float(tc.get("macd_hist"))
    mtf_3m = str(tc.get("mtf_3m_action") or "NEUTRAL").upper()
    mtf_5m = str(tc.get("mtf_5m_action") or "NEUTRAL").upper()

    buy_votes = 0
    sell_votes = 0
    if close is not None and ema9 is not None and ema21 is not None and ema50 is not None:
        if close >= ema9 >= ema21 >= ema50:
            buy_votes += 2
        elif close <= ema9 <= ema21 <= ema50:
            sell_votes += 2
    if close is not None and vwap is not None:
        if close >= vwap:
            buy_votes += 1
        elif close <= vwap:
            sell_votes += 1
    if rsi is not None:
        if rsi >= 53.0:
            buy_votes += 1
        elif rsi <= 47.0:
            sell_votes += 1
    if macd_hist is not None:
        if macd_hist > 0:
            buy_votes += 1
        elif macd_hist < 0:
            sell_votes += 1
    if mtf_3m == "BUY":
        buy_votes += 1
    elif mtf_3m == "SELL":
        sell_votes += 1
    if mtf_5m == "BUY":
        buy_votes += 1
    elif mtf_5m == "SELL":
        sell_votes += 1

    if buy_votes >= 3 and buy_votes >= sell_votes + 1:
        return "BUY"
    if sell_votes >= 3 and sell_votes >= buy_votes + 1:
        return "SELL"
    return "HOLD"


def mtf_alignment_score(technical_context: dict[str, Any] | None, *, direction: str) -> float:
    tc = technical_context or {}
    actions = [
        str(tc.get("mtf_3m_action") or "NEUTRAL").upper(),
        str(tc.get("mtf_5m_action") or "NEUTRAL").upper(),
    ]
    hits = sum(1 for action in actions if action == direction)
    return round(hits / 2.0, 4)


def pine_confirmation_score(
    *,
    direction: str,
    pine_signal: str,
    pine_age_seconds: int | None,
    max_age_seconds: int,
) -> float:
    if str(direction or "").upper() != str(pine_signal or "").upper():
        return 0.0
    if pine_age_seconds is not None and pine_age_seconds > int(max_age_seconds):
        return 0.0
    return 1.0


def combined_score(
    *,
    ml_confidence: float,
    ai_score: float,
    pine_score: float,
    mtf_score: float,
) -> float:
    """Weighted scoring with reduced ML influence for weak signals."""
    ml_conf = float(ml_confidence)
    
    # Reduce ML weight if confidence < 0.58
    ml_weight = 0.2
    if ml_conf < 0.58:
        ml_weight = 0.1  # Halve ML influence for weak signals
    
    # Redistribute weight to Pine and AI
    pine_weight = 0.4 if ml_conf >= 0.58 else 0.45
    ai_weight = 0.3 if ml_conf >= 0.58 else 0.35
    
    return round(
        float(pine_score) * pine_weight
        + (float(ai_score) / 100.0) * ai_weight
        + ml_conf * ml_weight
        + float(mtf_score) * 0.1,
        4,
    )


def evaluate_candidate(
    *,
    direction: str,
    source: str,
    ml_direction: str,
    ml_trade_action: str,
    ml_confidence: float,
    expected_move_points: float,
    ai_score: float,
    pine_signal: str,
    pine_age_seconds: int | None,
    technical_context: dict[str, Any] | None,
    thresholds: EffectiveThresholds,
    pine_max_age_seconds: int,
) -> CandidateScore:
    normalized = str(direction or "HOLD").upper()
    pine_score = pine_confirmation_score(
        direction=normalized,
        pine_signal=str(pine_signal or "HOLD").upper(),
        pine_age_seconds=pine_age_seconds,
        max_age_seconds=pine_max_age_seconds,
    )
    mtf_score = mtf_alignment_score(technical_context, direction=normalized)
    ml_trade_normalized = str(ml_trade_action or "HOLD").upper()
    # ML is optional confirmation only if confidence >= 0.6
    ml_component = (
        normalized in {"BUY", "SELL"}
        and normalized == str(ml_direction or "HOLD").upper()
        and float(ml_confidence) >= 0.6  # Ignore weak ML signals
        and abs(float(expected_move_points)) >= thresholds.expected_move_floor
    )
    ai_component = float(ai_score) >= thresholds.ai_score_floor
    pine_component = pine_score >= 1.0
    mtf_component = mtf_score > 0.0
    # New hierarchy: Pine (40) -> AI (30) -> ML (20) -> MTF (10)
    weighted_score = (
        (40.0 if pine_component else 0.0)
        + (30.0 if ai_component else 0.0)
        + (20.0 if ml_component else 0.0)
        + (10.0 if mtf_component else 0.0)
    )
    combo = combined_score(
        ml_confidence=float(ml_confidence),
        ai_score=float(ai_score),
        pine_score=pine_score,
        mtf_score=mtf_score,
    )
    combo_pass = combo >= thresholds.combined_score_floor
    pine_led_eligible = (
        normalized in {"BUY", "SELL"}
        and pine_component
        and float(ai_score) >= thresholds.pine_led_ai_floor
        and combo_pass
        and weighted_score >= thresholds.pine_led_score_threshold
    )
    qualifies = normalized in {"BUY", "SELL"} and (
        weighted_score >= thresholds.entry_score_threshold or pine_led_eligible
    )

    reasons = [
        f"score={weighted_score:.0f}",
        f"combined={combo:.2f}",
        f"ml_floor={thresholds.ml_confidence_floor:.2f}",
        f"ai_floor={thresholds.ai_score_floor:.0f}",
    ]
    if pine_led_eligible and weighted_score < thresholds.entry_score_threshold:
        reasons.append("pine_led_override")

    return CandidateScore(
        direction=normalized,
        source=source,
        ai_score=round(float(ai_score), 2),
        weighted_score=round(weighted_score, 2),
        combined_score=combo,
        pine_score=round(pine_score, 4),
        mtf_score=round(mtf_score, 4),
        ml_component=ml_component,
        ai_component=ai_component,
        pine_component=pine_component,
        mtf_component=mtf_component,
        qualifies=qualifies,
        pine_led_eligible=pine_led_eligible,
        reasons=reasons,
    )


def select_best_candidate(candidates: list[CandidateScore]) -> CandidateScore | None:
    if not candidates:
        return None
    qualifying = [candidate for candidate in candidates if candidate.qualifies]
    pool = qualifying or candidates
    return sorted(
        pool,
        key=lambda candidate: (
            1 if candidate.qualifies else 0,
            candidate.weighted_score,
            candidate.combined_score,
            1 if candidate.source == "ml" else 0,
        ),
        reverse=True,
    )[0]
