from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from prediction_engine.options_engine import nearest_strike
from utils.config import get_settings
from utils.symbols import normalize_symbol_key

SignalAction = Literal["BUY", "SELL", "HOLD"]
OptionType = Literal["CE", "PE"]


@dataclass(slots=True)
class StrikeSelectionResult:
    option_type: OptionType
    strike: float
    premium: float
    instrument_key: str | None
    delta: float | None
    oi: float | None
    spread_rupees: float | None
    spread_pct: float | None
    iv: float | None
    volume: float | None
    expected_move: float | None
    reason: str


def _quote_for_row(row: dict, option_type: OptionType) -> dict | None:
    return row.get("ce" if option_type == "CE" else "pe")


def _to_float(value) -> float | None:
    try:
        v = float(value)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def compute_position_lots(
    confidence: float,
    regime: str,
    base_lots: int,
    max_lots: int = 2,
) -> int:
    """Scale position size by confidence and market regime.

    Rules:
    - HIGH_VOLATILITY: always base_lots (never size up into chaos)
    - RANGE_BOUND: always base_lots (range can reverse hard)
    - TRENDING + high confidence (>=0.80): allow up to 2× base
    - TRENDING + medium confidence (>=0.70): allow up to 1.5× base
    - Otherwise: base_lots
    """
    if regime in {"HIGH_VOLATILITY", "RANGE_BOUND"}:
        return base_lots
    if confidence >= 0.80:
        return min(max_lots, base_lots * 2)
    if confidence >= 0.70:
        return min(max_lots, max(base_lots, int(base_lots * 1.5)))
    return base_lots


def _compute_pcr(chain_rows: list[dict]) -> float | None:
    """Put-Call Ratio from chain OI.  >1.5 = heavy put buying; <0.5 = heavy call buying."""
    total_ce = 0.0
    total_pe = 0.0
    for row in chain_rows:
        ce_oi = _to_float((row.get("ce") or {}).get("oi")) or 0.0
        pe_oi = _to_float((row.get("pe") or {}).get("oi")) or 0.0
        total_ce += ce_oi
        total_pe += pe_oi
    if total_ce <= 0:
        return None
    return total_pe / total_ce


def confidence_to_delta_range(confidence: float) -> tuple[float, float]:
    """Map confidence to target delta range for intraday option buying.

    For intraday index options, always stay near ATM (delta 0.40-0.55).
    Deep OTM loses to theta decay and requires a larger move to profit.
    High confidence = slight OTM (max 1 strike); low confidence = ATM.
    """
    score = float(confidence)
    if score >= 0.85:
        return (0.40, 0.52)  # High confidence → 1 strike OTM at most
    if score >= 0.75:
        return (0.43, 0.53)  # Medium-high → near ATM
    return (0.45, 0.55)  # Default → ATM band


def lot_size_for_symbol(symbol: str) -> int:
    normalized = normalize_symbol_key(symbol)
    configured = get_settings().execution_symbol_lot_size_map.get(normalized)
    if configured is not None and configured > 0:
        return configured
    if normalized == "SENSEX":
        return 20
    if normalized == "NIFTY50":
        return 65
    if normalized == "BANKNIFTY":
        return 30
    if "NIFTY" in normalized:
        return 65
    return 1


def select_expiry_date(expiries: list[date], today: date) -> date | None:
    valid = sorted(exp for exp in expiries if (exp - today).days >= 1)
    return valid[0] if valid else None





def _liquidity_check(
    quote: dict | None,
    ltp: float,
    min_oi: float = 1000.0,
    max_spread_pct: float = 0.06,
    min_volume: float = 100.0,
) -> tuple[bool, float | None, float | None, float | None, float | None]:
    """Enhanced liquidity check with relative spread % and volume."""
    if quote is None:
        return False, None, None, None, None
    oi = _to_float(quote.get("oi"))
    volume = _to_float(quote.get("volume"))
    bid = _to_float(quote.get("bid"))
    ask = _to_float(quote.get("ask"))
    
    spread_rupees = None
    spread_pct = None
    if bid is not None and ask is not None and ltp > 0:
        spread_rupees = abs(ask - bid)
        spread_pct = spread_rupees / ltp
    
    oi_ok = oi is None or oi >= min_oi
    spread_ok = spread_pct is None or spread_pct <= max_spread_pct
    volume_ok = volume is None or volume >= min_volume
    
    liquid = bool(oi_ok and spread_ok and volume_ok)
    return liquid, oi, spread_rupees, spread_pct, volume


def calculate_expected_move(
    chain_rows: list[dict],
    atm: float,
    strike_step: int,
) -> float:
    """Calculate expected move from ATM straddle price."""
    atm_row = next((r for r in chain_rows if abs(float(r.get("strike", 0)) - atm) < 1), None)
    if not atm_row:
        return strike_step * 2.0  # Fallback
    
    ce_ltp = _to_float((atm_row.get("ce") or {}).get("ltp")) or 0
    pe_ltp = _to_float((atm_row.get("pe") or {}).get("ltp")) or 0
    straddle_price = ce_ltp + pe_ltp
    
    # Expected move ≈ 85% of straddle price (1 std dev)
    return straddle_price * 0.85 if straddle_price > 0 else strike_step * 2.0


def get_atm_iv(chain_rows: list[dict], atm: float) -> float | None:
    """Get ATM implied volatility."""
    atm_row = next((r for r in chain_rows if abs(float(r.get("strike", 0)) - atm) < 1), None)
    if not atm_row:
        return None
    ce_iv = _to_float((atm_row.get("ce") or {}).get("iv"))
    pe_iv = _to_float((atm_row.get("pe") or {}).get("iv"))
    if ce_iv and pe_iv:
        return (ce_iv + pe_iv) / 2
    return ce_iv or pe_iv


def get_oi_cluster_strikes(chain_rows: list[dict], option_type: OptionType, top_n: int = 3) -> list[float]:
    """Find strikes with highest OI (support/resistance zones)."""
    strikes_oi = []
    for row in chain_rows:
        strike = _to_float(row.get("strike"))
        quote = _quote_for_row(row, option_type)
        oi = _to_float((quote or {}).get("oi"))
        if strike and oi:
            strikes_oi.append((strike, oi))
    
    strikes_oi.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in strikes_oi[:top_n]]


def select_option_contract(
    *,
    signal_action: SignalAction,
    spot_price: float,
    strike_step: int,
    chain_rows: list[dict],
    confidence: float,
    expected_return_pct: float,
    premium_min: float = 1.0,
    premium_max: float = 100000.0,
    days_to_expiry: int = 7,
    capital_per_trade: float = 100000.0,
    iv_rank: float | None = None,
) -> StrikeSelectionResult | None:
    """Strike selection with delta targeting, OI delta, PCR filter, and IV rank."""
    if signal_action not in {"BUY", "SELL"}:
        return None

    option_type: OptionType = "CE" if signal_action == "BUY" else "PE"
    atm = nearest_strike(spot_price, strike_step)
    rows_by_strike = {
        float(row.get("strike")): row
        for row in chain_rows
        if _to_float(row.get("strike")) is not None
    }

    expected_move = calculate_expected_move(chain_rows, atm, strike_step)
    atm_iv = get_atm_iv(chain_rows, atm)

    # PCR sentiment check — penalise crowded trades
    pcr = _compute_pcr(chain_rows)
    # CE trade when everyone is also buying CEs (PCR < 0.5) = crowded long
    # PE trade when everyone is also buying PEs (PCR > 2.0) = crowded short
    pcr_crowded = (
        (option_type == "CE" and pcr is not None and pcr < 0.5)
        or (option_type == "PE" and pcr is not None and pcr > 2.0)
    )

    # IV rank guard: if IV is in top 30% of its range, premium is expensive
    # Reduce premium max to limit overpaying
    iv_rank_expensive = iv_rank is not None and iv_rank > 0.70

    delta_min, delta_max = confidence_to_delta_range(confidence)

    # Tighten toward ATM near expiry to avoid gamma explosion
    if days_to_expiry <= 1:
        delta_min = max(0.45, delta_min)
        delta_max = min(0.55, delta_max)
    elif days_to_expiry <= 5:
        delta_min = max(0.40, delta_min)
        delta_max = min(0.55, delta_max)
    delta_min = max(0.38, delta_min)  # hard floor

    # Cap premium: 2% of capital, reduced to 1% when IV is expensive
    risk_pct = 0.01 if iv_rank_expensive else 0.02
    premium_max = min(premium_max, capital_per_trade * risk_pct)

    oi_clusters = get_oi_cluster_strikes(chain_rows, option_type, top_n=5)

    candidates: list[dict] = []
    relaxed_candidates: list[dict] = []

    for strike, row in rows_by_strike.items():
        quote = _quote_for_row(row, option_type)
        if not quote:
            continue

        premium = _to_float(quote.get("ltp"))
        delta = _to_float(quote.get("delta"))
        iv = _to_float(quote.get("iv"))
        oi = _to_float(quote.get("oi"))
        prev_oi = _to_float(quote.get("prev_oi"))

        if not premium or not delta:
            continue
        if premium < premium_min or premium > premium_max:
            continue

        abs_delta = abs(delta)
        delta_in_band = delta_min <= abs_delta <= delta_max

        distance_from_atm = abs(strike - atm)
        if distance_from_atm > expected_move * 1.5:
            continue

        liquid, _oi, spread_rupees, spread_pct, volume = _liquidity_check(
            quote, ltp=premium, min_oi=1000.0, max_spread_pct=0.06, min_volume=100.0,
        )
        if not liquid and spread_pct is not None and spread_pct > 0.10:
            continue

        score = 0.0

        # 1. Delta closeness (35%)
        delta_target = (delta_min + delta_max) / 2
        delta_score = max(0.0, 1.0 - abs(abs_delta - delta_target) / 0.25)
        score += delta_score * 0.35

        # 2. OI cluster alignment (20%)
        if strike in oi_clusters:
            cluster_rank = oi_clusters.index(strike)
            score += max(0.0, 0.20 - cluster_rank * 0.05)

        # 3. Absolute OI quality (15%)
        if _oi:
            score += min(1.0, _oi / 10000.0) * 0.15

        # 4. Bid-ask spread (15%)
        if spread_pct:
            score += max(0.0, 1.0 - spread_pct / 0.03) * 0.15

        # 5. Volume (10%)
        if volume:
            score += min(1.0, volume / 1000.0) * 0.10

        # 6. IV ratio vs ATM (5%)
        if iv and atm_iv:
            iv_ratio = iv / atm_iv
            if 0.9 <= iv_ratio <= 1.2:
                score += 0.05

        # 7. OI delta bonus/penalty (up to ±0.08)
        # Fresh OI build = participants entering in direction of trade (bullish for CE, bearish for PE)
        if _oi is not None and prev_oi is not None and prev_oi > 0:
            oi_delta_pct = (_oi - prev_oi) / prev_oi
            if oi_delta_pct > 0.02:
                score += min(0.08, oi_delta_pct * 2.0)  # OI expanding: strong participation
            elif oi_delta_pct < -0.05:
                score -= 0.04  # OI unwinding: participants exiting

        # 8. PCR crowding penalty
        if pcr_crowded:
            score *= 0.90  # 10% penalty when trade direction is crowded

        candidate = {
            "strike": strike,
            "premium": premium,
            "delta": delta,
            "oi": _oi,
            "spread_rupees": spread_rupees,
            "spread_pct": spread_pct,
            "volume": volume,
            "iv": iv,
            "quote": quote,
            "score": score,
        }
        if delta_in_band and liquid:
            candidates.append(candidate)
        else:
            # Relaxed pool: penalise for being out of band or illiquid
            relaxed_score = score
            if not delta_in_band:
                relaxed_score -= min(0.20, abs(abs_delta - delta_target) / 0.35)
            if spread_pct is not None and spread_pct > 0.06:
                relaxed_score -= min(0.10, (spread_pct - 0.06) / 0.08)
            candidate["score"] = relaxed_score
            relaxed_candidates.append(candidate)

    if not candidates:
        candidates = [c for c in relaxed_candidates if c["score"] > 0]
    if not candidates:
        return None

    best = max(candidates, key=lambda x: x["score"])

    reason_parts = [
        "regime_adaptive_strike_selection",
        f"target_delta={delta_min:.2f}-{delta_max:.2f}",
        f"actual_delta={abs(best['delta']):.2f}",
        f"expected_move={expected_move:.0f}",
        f"score={best['score']:.2f}",
    ]
    if best["strike"] in oi_clusters:
        reason_parts.append("oi_cluster_zone")
    if atm_iv:
        reason_parts.append(f"atm_iv={atm_iv:.2%}")
    if pcr is not None:
        reason_parts.append(f"pcr={pcr:.2f}{'(crowded)' if pcr_crowded else ''}")
    if iv_rank is not None:
        reason_parts.append(f"iv_rank={iv_rank:.2f}{'(expensive)' if iv_rank_expensive else ''}")

    return StrikeSelectionResult(
        option_type=option_type,
        strike=float(best["strike"]),
        premium=float(best["premium"]),
        instrument_key=str(best["quote"].get("instrument_key")) if best["quote"].get("instrument_key") else None,
        delta=float(best["delta"]),
        oi=float(best["oi"]) if best["oi"] else None,
        spread_rupees=float(best["spread_rupees"]) if best["spread_rupees"] else None,
        spread_pct=float(best["spread_pct"]) if best["spread_pct"] else None,
        iv=float(best["iv"]) if best["iv"] else None,
        volume=float(best["volume"]) if best["volume"] else None,
        expected_move=float(expected_move),
        reason="|".join(reason_parts),
    )
