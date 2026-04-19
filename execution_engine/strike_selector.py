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


def confidence_to_delta_range(confidence: float) -> tuple[float, float]:
    """Map confidence to target delta range (probability-based)."""
    score = float(confidence)
    if score >= 0.85:
        return (0.30, 0.40)  # High confidence → moderate OTM
    if score >= 0.75:
        return (0.40, 0.50)  # Medium-high → slight OTM
    if score >= 0.65:
        return (0.45, 0.55)  # Medium → near ATM
    return (0.48, 0.52)  # Low → ATM


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
) -> StrikeSelectionResult | None:
    """Enhanced strike selection with delta-based, volatility-aware logic."""
    if signal_action not in {"BUY", "SELL"}:
        return None

    option_type: OptionType = "CE" if signal_action == "BUY" else "PE"
    atm = nearest_strike(spot_price, strike_step)
    rows_by_strike = {float(row.get("strike")): row for row in chain_rows if _to_float(row.get("strike")) is not None}
    
    # Calculate expected move and IV context
    expected_move = calculate_expected_move(chain_rows, atm, strike_step)
    atm_iv = get_atm_iv(chain_rows, atm)
    
    # Get target delta range based on confidence
    delta_min, delta_max = confidence_to_delta_range(confidence)
    
    # Adjust for time to expiry (0-1 DTE needs different logic)
    if days_to_expiry <= 1:
        # Weekly expiry: tighter delta range, closer to ATM
        delta_min = max(0.45, delta_min + 0.10)
        delta_max = min(0.55, delta_max + 0.05)
    elif days_to_expiry <= 5:
        # Near expiry: moderate adjustment
        delta_min = max(0.40, delta_min + 0.05)
        delta_max = min(0.55, delta_max + 0.03)
    
    # Dynamic premium cap based on risk (1-2% of capital)
    max_premium_risk = capital_per_trade * 0.02
    premium_max = min(premium_max, max_premium_risk)
    
    # Get OI cluster strikes (support/resistance)
    oi_clusters = get_oi_cluster_strikes(chain_rows, option_type, top_n=5)

    # Score all candidates
    candidates = []
    relaxed_candidates = []
    for strike, row in rows_by_strike.items():
        quote = _quote_for_row(row, option_type)
        if not quote:
            continue
        
        premium = _to_float(quote.get("ltp"))
        delta = _to_float(quote.get("delta"))
        iv = _to_float(quote.get("iv"))
        
        if not premium or not delta:
            continue
        
        # Filter: premium range
        if premium < premium_min or premium > premium_max:
            continue
        
        # Filter: delta range (probability-based)
        abs_delta = abs(delta)
        delta_in_band = delta_min <= abs_delta <= delta_max

        # Filter: within expected move range
        distance_from_atm = abs(strike - atm)
        if distance_from_atm > expected_move * 1.5:
            continue
        
        # Enhanced liquidity check
        liquid, oi, spread_rupees, spread_pct, volume = _liquidity_check(
            quote,
            ltp=premium,
            min_oi=1000.0,
            max_spread_pct=0.06,
            min_volume=100.0,
        )
        if not liquid and spread_pct is not None and spread_pct > 0.10:
            continue

        # Calculate score
        score = 0.0
        
        # Delta score (prefer middle of range)
        delta_target = (delta_min + delta_max) / 2
        delta_score = 1.0 - abs(abs_delta - delta_target) / 0.25
        score += delta_score * 0.35
        
        # OI cluster bonus (support/resistance)
        if strike in oi_clusters:
            cluster_rank = oi_clusters.index(strike)
            score += (0.20 - cluster_rank * 0.05)
        
        # Liquidity score
        if oi:
            oi_score = min(1.0, oi / 10000.0)
            score += oi_score * 0.15
        
        if spread_pct:
            spread_score = max(0, 1.0 - spread_pct / 0.03)
            score += spread_score * 0.15
        
        if volume:
            volume_score = min(1.0, volume / 1000.0)
            score += volume_score * 0.10
        
        # IV score (prefer reasonable IV, not too high)
        if iv and atm_iv:
            iv_ratio = iv / atm_iv
            if 0.9 <= iv_ratio <= 1.2:
                score += 0.05

        candidate = {
            "strike": strike,
            "premium": premium,
            "delta": delta,
            "oi": oi,
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
            relaxed_score = score
            if not delta_in_band:
                delta_target = (delta_min + delta_max) / 2
                relaxed_score -= min(0.20, abs(abs_delta - delta_target) / 0.35)
            if spread_pct is not None and spread_pct > 0.06:
                relaxed_score -= min(0.10, (spread_pct - 0.06) / 0.08)
            candidate["score"] = relaxed_score
            relaxed_candidates.append(candidate)

    if not candidates:
        candidates = [c for c in relaxed_candidates if c["score"] > 0]
    if not candidates:
        return None
    
    # Select best candidate
    best = max(candidates, key=lambda x: x["score"])
    
    reason_parts = [
        "dynamic_heatmap_maxpain_selection",
        "delta_based_selection",
        f"target_delta={delta_min:.2f}-{delta_max:.2f}",
        f"actual_delta={abs(best['delta']):.2f}",
        f"expected_move={expected_move:.0f}",
        f"score={best['score']:.2f}",
    ]
    if best["strike"] in oi_clusters:
        reason_parts.append("oi_cluster_zone")
    if atm_iv:
        reason_parts.append(f"atm_iv={atm_iv:.2%}")
    
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
