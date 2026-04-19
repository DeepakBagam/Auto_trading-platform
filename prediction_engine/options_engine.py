from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

from utils.calendar_utils import is_trading_day
from utils.constants import IST_ZONE

OptionType = Literal["CE", "PE"]
TradeSide = Literal["BUY", "SELL"]
SignalAction = Literal["BUY", "SELL", "HOLD"]
StrategyName = Literal[
    "auto",
    "iron_condor",
    "bull_put_spread",
    "bear_call_spread",
    "long_straddle",
    "long_strangle",
    "trend_vwap_oi",
    "ema_vwap_rsi_atr",
]


@dataclass(slots=True)
class OptionQuoteView:
    instrument_key: str | None
    strike: float
    option_type: OptionType
    ltp: float
    bid: float | None
    ask: float | None
    volume: float | None
    oi: float | None
    close_price: float | None
    bid_qty: float | None
    ask_qty: float | None
    prev_oi: float | None
    iv: float | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    pop: float | None
    pcr: float | None
    underlying_spot_price: float | None
    source: str


def strike_step_for_symbol(symbol: str) -> int:
    normalized = symbol.upper().replace(" ", "")
    if "BANKNIFTY" in normalized:
        return 100
    if "SENSEX" in normalized:
        return 100
    if "FINNIFTY" in normalized:
        return 50
    if "MIDCPNIFTY" in normalized:
        return 25
    return 50


def nearest_strike(price: float, strike_step: int) -> float:
    if strike_step <= 0:
        strike_step = 50
    return round(price / strike_step) * strike_step


def _expiry_weekdays_for_symbol(symbol: str) -> tuple[int, ...]:
    normalized = symbol.upper().replace(" ", "")
    if "SENSEX" in normalized:
        return (1, 4)  # Tuesday and Friday
    return (1,)  # Tuesday


def next_weekly_expiries(symbol: str, count: int = 6, start_dt: datetime | None = None) -> list[date]:
    now = start_dt or datetime.now(IST_ZONE)
    weekdays = _expiry_weekdays_for_symbol(symbol)
    expiries: list[date] = []
    seen: set[date] = set()
    probe = now.date() + timedelta(days=1)
    while len(expiries) < max(1, count):
        if probe.weekday() in weekdays:
            expiry = probe
            while not is_trading_day(expiry):
                expiry -= timedelta(days=1)
            if expiry > now.date() and expiry not in seen:
                seen.add(expiry)
                expiries.append(expiry)
        probe += timedelta(days=1)
    return sorted(expiries)


def synthetic_option_chain(
    *,
    symbol: str,
    underlying_price: float,
    expiry_date: date,
    strike_step: int,
    levels: int = 7,
    atm_iv: float = 0.18,
) -> list[OptionQuoteView]:
    atm = nearest_strike(underlying_price, strike_step)
    days_to_expiry = max((expiry_date - datetime.now(IST_ZONE).date()).days, 1)
    time_factor = (days_to_expiry / 365.0) ** 0.5
    base_time_value = max(underlying_price * atm_iv * time_factor * 0.38, strike_step * 0.35)
    quotes: list[OptionQuoteView] = []
    for idx in range(-levels, levels + 1):
        strike = atm + (idx * strike_step)
        moneyness = (underlying_price - strike) / max(1.0, strike_step * 2)
        decay = max(0.18, 1.0 - (abs(idx) * 0.12))
        ce_intrinsic = max(underlying_price - strike, 0.0)
        pe_intrinsic = max(strike - underlying_price, 0.0)
        ce_time = base_time_value * max(0.12, decay * (1.0 + moneyness * 0.08))
        pe_time = base_time_value * max(0.12, decay * (1.0 - moneyness * 0.08))
        ce_ltp = max(1.0, ce_intrinsic + ce_time)
        pe_ltp = max(1.0, pe_intrinsic + pe_time)
        spread = max(0.05, min(0.35, 0.01 + (abs(idx) * 0.005)))
        ce_delta = max(0.03, min(0.97, 0.5 + (moneyness * 0.26)))
        pe_delta = -max(0.03, min(0.97, 0.5 - (moneyness * 0.26)))
        common_gamma = max(0.002, 0.03 * max(0.22, decay))
        common_theta = -max(0.1, base_time_value / max(2.0, days_to_expiry))
        common_vega = max(0.15, 0.8 * time_factor * max(0.22, decay))
        oi_base = max(1000.0, 25000.0 * max(0.2, decay))
        quotes.append(
            OptionQuoteView(
                instrument_key=f"SYNTH|{symbol}|{int(strike)}|CE",
                strike=float(strike),
                option_type="CE",
                ltp=float(round(ce_ltp, 2)),
                bid=float(round(ce_ltp * (1.0 - spread * 0.5), 2)),
                ask=float(round(ce_ltp * (1.0 + spread * 0.5), 2)),
                volume=float(round(oi_base * (3.2 - min(2.0, abs(idx) * 0.18)), 2)),
                oi=float(round(oi_base)),
                close_price=float(round(max(1.0, ce_ltp * 0.985), 2)),
                bid_qty=float(round(max(25.0, oi_base * 0.04), 2)),
                ask_qty=float(round(max(25.0, oi_base * 0.045), 2)),
                prev_oi=float(round(oi_base * 0.96)),
                iv=float(round(atm_iv + (abs(idx) * 0.01), 4)),
                delta=float(round(ce_delta, 4)),
                gamma=float(round(common_gamma, 5)),
                theta=float(round(common_theta, 4)),
                vega=float(round(common_vega, 4)),
                pop=float(round(max(1.0, min(99.0, 50.0 + (idx * -6.5) + (moneyness * 12.0))), 2)),
                pcr=float(round(max(0.05, min(9.99, (1.15 - (idx * 0.08)))), 2)),
                underlying_spot_price=float(round(underlying_price, 2)),
                source="synthetic",
            )
        )
        quotes.append(
            OptionQuoteView(
                instrument_key=f"SYNTH|{symbol}|{int(strike)}|PE",
                strike=float(strike),
                option_type="PE",
                ltp=float(round(pe_ltp, 2)),
                bid=float(round(pe_ltp * (1.0 - spread * 0.5), 2)),
                ask=float(round(pe_ltp * (1.0 + spread * 0.5), 2)),
                volume=float(round(oi_base * (3.1 - min(2.0, abs(idx) * 0.15)), 2)),
                oi=float(round(oi_base)),
                close_price=float(round(max(1.0, pe_ltp * 0.985), 2)),
                bid_qty=float(round(max(25.0, oi_base * 0.042), 2)),
                ask_qty=float(round(max(25.0, oi_base * 0.047), 2)),
                prev_oi=float(round(oi_base * 0.97)),
                iv=float(round(atm_iv + (abs(idx) * 0.01), 4)),
                delta=float(round(pe_delta, 4)),
                gamma=float(round(common_gamma, 5)),
                theta=float(round(common_theta, 4)),
                vega=float(round(common_vega, 4)),
                pop=float(round(max(1.0, min(99.0, 50.0 + (idx * 6.5) - (moneyness * 12.0))), 2)),
                pcr=float(round(max(0.05, min(9.99, (1.15 - (idx * 0.08)))), 2)),
                underlying_spot_price=float(round(underlying_price, 2)),
                source="synthetic",
            )
        )
    return sorted(quotes, key=lambda x: (x.strike, x.option_type))


def build_chain_rows(quotes: list[OptionQuoteView]) -> list[dict]:
    rows_by_strike: dict[float, dict] = {}
    for quote in quotes:
        row = rows_by_strike.setdefault(
            quote.strike,
            {
                "strike": quote.strike,
                "pcr": quote.pcr,
                "underlying_spot_price": quote.underlying_spot_price,
                "ce": None,
                "pe": None,
            },
        )
        payload = {
            "ltp": quote.ltp,
            "bid": quote.bid,
            "ask": quote.ask,
            "volume": quote.volume,
            "oi": quote.oi,
            "close_price": quote.close_price,
            "bid_qty": quote.bid_qty,
            "ask_qty": quote.ask_qty,
            "prev_oi": quote.prev_oi,
            "iv": quote.iv,
            "delta": quote.delta,
            "gamma": quote.gamma,
            "theta": quote.theta,
            "vega": quote.vega,
            "pop": quote.pop,
            "instrument_key": quote.instrument_key,
            "source": quote.source,
        }
        if row.get("pcr") is None and quote.pcr is not None:
            row["pcr"] = quote.pcr
        if row.get("underlying_spot_price") is None and quote.underlying_spot_price is not None:
            row["underlying_spot_price"] = quote.underlying_spot_price
        if quote.option_type == "CE":
            row["ce"] = payload
        else:
            row["pe"] = payload
    return [rows_by_strike[k] for k in sorted(rows_by_strike.keys())]


def _nearest_available_strike(requested: float, chain_rows: list[dict]) -> float | None:
    strikes = [float(r["strike"]) for r in chain_rows if r.get("strike") is not None]
    if not strikes:
        return None
    return min(strikes, key=lambda s: abs(s - requested))


def max_pain_proxy_for_chain(chain_rows: list[dict]) -> float | None:
    strikes = sorted(_to_float(row.get("strike")) for row in chain_rows)
    strikes = [s for s in strikes if s is not None]
    if not strikes:
        return None
    ce_oi = {
        float(row.get("strike")): float((row.get("ce") or {}).get("oi") or 0.0)
        for row in chain_rows
        if _to_float(row.get("strike")) is not None
    }
    pe_oi = {
        float(row.get("strike")): float((row.get("pe") or {}).get("oi") or 0.0)
        for row in chain_rows
        if _to_float(row.get("strike")) is not None
    }
    best_strike = None
    best_cost = None
    for settle in strikes:
        payout = 0.0
        for strike in strikes:
            payout += max(0.0, settle - strike) * ce_oi.get(strike, 0.0)
            payout += max(0.0, strike - settle) * pe_oi.get(strike, 0.0)
        if best_cost is None or payout < best_cost:
            best_cost = payout
            best_strike = settle
    return best_strike


def _strike_heatmap_score(
    *,
    row: dict,
    option_type: OptionType,
    strike: float,
    target: float,
    atm: float,
    strike_step: int,
    max_pain: float | None,
) -> float:
    quote = row.get("ce" if option_type == "CE" else "pe") or {}
    ltp = _to_float(quote.get("ltp"))
    if ltp is None or ltp <= 0:
        return -10_000.0
    delta = _to_float(quote.get("delta"))
    oi = _to_float(quote.get("oi")) or 0.0
    bid = _to_float(quote.get("bid"))
    ask = _to_float(quote.get("ask"))
    spread = None
    if bid is not None and ask is not None:
        spread = abs(ask - bid) / max(ltp, 1.0)

    ce_oi = _to_float((row.get("ce") or {}).get("oi")) or 0.0
    pe_oi = _to_float((row.get("pe") or {}).get("oi")) or 0.0
    total_oi = ce_oi + pe_oi
    oi_balance = 0.5
    if total_oi > 0:
        raw = (pe_oi - ce_oi) / total_oi if option_type == "CE" else (ce_oi - pe_oi) / total_oi
        oi_balance = max(0.0, min(1.0, (raw + 1.0) * 0.5))

    dist_score = 1.0 - min(abs(strike - target) / max(float(strike_step), abs(target) * 0.03), 1.0)
    delta_score = 0.5
    if delta is not None:
        delta_target = 0.55 if abs(target - atm) >= strike_step else 0.50
        delta_score = 1.0 - min(abs(abs(delta) - delta_target) / 0.25, 1.0)
    liq_oi = min(1.0, oi / 50_000.0)
    liq_spread = 0.6 if spread is None else max(0.0, 1.0 - min(spread / 0.04, 1.0))
    max_pain_score = 0.5
    if max_pain is not None:
        max_pain_score = 1.0 - min(abs(strike - max_pain) / max(float(strike_step) * 6.0, 1.0), 1.0)

    return (
        dist_score * 0.32
        + delta_score * 0.22
        + liq_oi * 0.18
        + liq_spread * 0.15
        + oi_balance * 0.08
        + max_pain_score * 0.05
    )


def auto_select_strike(
    *,
    underlying_price: float,
    strike_step: int,
    option_type: OptionType,
    conviction: str,
    chain_rows: list[dict],
    strike_mode: str = "auto",
    manual_strike: float | None = None,
    expected_return_pct: float = 0.0,
) -> float | None:
    if strike_mode == "manual" and manual_strike is not None:
        return _nearest_available_strike(float(manual_strike), chain_rows)

    atm = nearest_strike(underlying_price, strike_step)
    conviction_weight = {"low": 0.35, "medium": 0.62, "high": 0.82}.get(str(conviction).lower(), 0.55)
    exp_strength = min(1.0, abs(float(expected_return_pct)) / 0.015)
    move_strength = (conviction_weight * 0.7) + (exp_strength * 0.3)
    shift_steps = 0.0
    if move_strength >= 0.84:
        shift_steps = 1.5
    elif move_strength >= 0.68:
        shift_steps = 1.0
    elif move_strength >= 0.56:
        shift_steps = 0.5
    target = float(atm - (strike_step * shift_steps)) if option_type == "CE" else float(atm + (strike_step * shift_steps))

    max_pain = max_pain_proxy_for_chain(chain_rows)
    best_strike = None
    best_score = -10_000.0
    for row in chain_rows:
        strike = _to_float(row.get("strike"))
        if strike is None:
            continue
        if abs(strike - atm) > (4 * max(1, int(strike_step))):
            continue
        score = _strike_heatmap_score(
            row=row,
            option_type=option_type,
            strike=float(strike),
            target=float(target),
            atm=float(atm),
            strike_step=max(1, int(strike_step)),
            max_pain=max_pain,
        )
        if score > best_score:
            best_score = score
            best_strike = float(strike)
    return best_strike if best_strike is not None else _nearest_available_strike(float(target), chain_rows)


def _find_quote(chain_rows: list[dict], strike: float, option_type: OptionType) -> dict | None:
    for row in chain_rows:
        if float(row.get("strike", 0.0)) != float(strike):
            continue
        return row.get("ce" if option_type == "CE" else "pe")
    return None


def _risk_config_from_conviction(conviction: str) -> tuple[float, float, float, float]:
    if conviction == "high":
        return (0.20, 0.40, 0.12, 0.06)  # sl, tp, trigger, step
    if conviction == "medium":
        return (0.24, 0.32, 0.10, 0.07)
    return (0.28, 0.24, 0.08, 0.08)


def _to_float(value) -> float | None:
    try:
        out = float(value)
        return out if out == out else None
    except (TypeError, ValueError):
        return None


def _interval_thresholds(interval: str) -> tuple[float, float]:
    del interval
    sideways = 0.0018
    expansion = 0.0040
    return sideways, expansion


def _oi_bias(chain_rows: list[dict], atm: float, strike_step: int) -> float:
    ce_oi = 0.0
    pe_oi = 0.0
    for row in chain_rows:
        strike = _to_float(row.get("strike"))
        if strike is None or abs(strike - atm) > (2 * strike_step):
            continue
        ce_oi += float((row.get("ce") or {}).get("oi") or 0.0)
        pe_oi += float((row.get("pe") or {}).get("oi") or 0.0)
    total = ce_oi + pe_oi
    if total <= 0:
        return 0.0
    return (ce_oi - pe_oi) / total


def _nearest_chain_strike(chain_rows: list[dict], target: float) -> float | None:
    strikes = [float(r.get("strike")) for r in chain_rows if r.get("strike") is not None]
    if not strikes:
        return None
    return min(strikes, key=lambda x: abs(x - target))


def _single_leg_payload(
    *,
    option_type: OptionType,
    strike: float,
    quote: dict,
    conviction: str,
    atr_ratio: float,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    entry_price = _to_float(quote.get("ltp"))
    if entry_price is None:
        return (None, None, None, None, None)
    sl_pct, tp_pct, trigger_pct, step_pct = _risk_config_from_conviction(conviction)
    # ATR-aware scaling keeps stops wider in high-volatility sessions.
    vol_adj = min(0.12, max(0.0, atr_ratio * 8.0))
    sl_pct = min(0.45, sl_pct + vol_adj)
    tp_pct = min(0.70, tp_pct + (vol_adj * 1.4))
    trigger_pct = min(0.35, trigger_pct + (vol_adj * 0.75))
    stop_loss = round(entry_price * (1.0 - sl_pct), 2)
    take_profit = round(entry_price * (1.0 + tp_pct), 2)
    trail_trigger_price = round(entry_price * (1.0 + trigger_pct), 2)
    return (entry_price, stop_loss, stop_loss, trail_trigger_price, round(step_pct, 4))


def _determine_strategy(
    *,
    strategy_mode: str,
    interval: str,
    underlying_signal_action: SignalAction,
    underlying_confidence: float,
    underlying_expected_return_pct: float,
    allow_option_writing: bool,
    technical_context: dict | None,
    chain_rows: list[dict],
    underlying_price: float,
    strike_step: int,
) -> str:
    if strategy_mode != "auto":
        return strategy_mode

    sideways_thr, expansion_thr = _interval_thresholds(interval)
    move = abs(float(underlying_expected_return_pct))
    tc = technical_context or {}
    close = _to_float(tc.get("close")) or underlying_price
    ema21 = _to_float(tc.get("ema_21"))
    ema50 = _to_float(tc.get("ema_50"))
    rsi = _to_float(tc.get("rsi_14"))
    vwap = _to_float(tc.get("vwap")) or close
    atm = nearest_strike(underlying_price, strike_step)
    oi_skew = _oi_bias(chain_rows, atm, strike_step)

    bullish_trend = (
        ema21 is not None
        and ema50 is not None
        and close >= ema21 >= ema50
        and (rsi is None or rsi >= 50.0)
        and close >= vwap
    )
    bearish_trend = (
        ema21 is not None
        and ema50 is not None
        and close <= ema21 <= ema50
        and (rsi is None or rsi <= 50.0)
        and close <= vwap
    )
    bull_oi_ok = oi_skew >= -0.08
    bear_oi_ok = oi_skew <= 0.08

    if underlying_signal_action == "HOLD":
        if move >= expansion_thr and underlying_confidence >= 0.58:
            return "long_straddle"
        if move >= expansion_thr * 0.8 and underlying_confidence >= 0.52:
            return "long_strangle"
        if allow_option_writing and move <= sideways_thr and underlying_confidence >= 0.50:
            return "iron_condor"
        return "ema_vwap_rsi_atr"

    if underlying_signal_action == "BUY":
        if allow_option_writing and move <= sideways_thr and underlying_confidence >= 0.66:
            return "bull_put_spread"
        if bullish_trend and bull_oi_ok:
            return "trend_vwap_oi"
        return "ema_vwap_rsi_atr"

    if underlying_signal_action == "SELL":
        if allow_option_writing and move <= sideways_thr and underlying_confidence >= 0.66:
            return "bear_call_spread"
        if bearish_trend and bear_oi_ok:
            return "trend_vwap_oi"
        return "ema_vwap_rsi_atr"

    return "ema_vwap_rsi_atr"


def _build_multi_leg(
    *,
    strategy_name: str,
    underlying_signal_action: SignalAction,
    underlying_price: float,
    strike_step: int,
    chain_rows: list[dict],
) -> dict | None:
    atm = nearest_strike(underlying_price, strike_step)
    if strategy_name == "long_straddle":
        ce_k = _nearest_chain_strike(chain_rows, atm)
        pe_k = _nearest_chain_strike(chain_rows, atm)
        if ce_k is None or pe_k is None:
            return None
        ce_q = _find_quote(chain_rows, ce_k, "CE")
        pe_q = _find_quote(chain_rows, pe_k, "PE")
        if ce_q is None or pe_q is None:
            return None
        debit = float(ce_q.get("ltp") or 0.0) + float(pe_q.get("ltp") or 0.0)
        if debit <= 0:
            return None
        return {
            "action": "BUY",
            "side": "BUY",
            "option_type": None,
            "strike": float(atm),
            "entry_price": round(debit, 2),
            "stop_loss": round(debit * 0.65, 2),
            "take_profit": round(debit * 1.50, 2),
            "trailing_stop_loss": None,
            "trail_trigger_price": None,
            "trail_step_pct": None,
            "legs": [
                {"side": "BUY", "option_type": "CE", "strike": float(ce_k), "qty": 1, "price": float(ce_q.get("ltp") or 0.0)},
                {"side": "BUY", "option_type": "PE", "strike": float(pe_k), "qty": 1, "price": float(pe_q.get("ltp") or 0.0)},
            ],
        }

    if strategy_name == "long_strangle":
        ce_k = _nearest_chain_strike(chain_rows, atm + strike_step)
        pe_k = _nearest_chain_strike(chain_rows, atm - strike_step)
        if ce_k is None or pe_k is None:
            return None
        ce_q = _find_quote(chain_rows, ce_k, "CE")
        pe_q = _find_quote(chain_rows, pe_k, "PE")
        if ce_q is None or pe_q is None:
            return None
        debit = float(ce_q.get("ltp") or 0.0) + float(pe_q.get("ltp") or 0.0)
        if debit <= 0:
            return None
        return {
            "action": "BUY",
            "side": "BUY",
            "option_type": None,
            "strike": float(atm),
            "entry_price": round(debit, 2),
            "stop_loss": round(debit * 0.62, 2),
            "take_profit": round(debit * 1.65, 2),
            "trailing_stop_loss": None,
            "trail_trigger_price": None,
            "trail_step_pct": None,
            "legs": [
                {"side": "BUY", "option_type": "CE", "strike": float(ce_k), "qty": 1, "price": float(ce_q.get("ltp") or 0.0)},
                {"side": "BUY", "option_type": "PE", "strike": float(pe_k), "qty": 1, "price": float(pe_q.get("ltp") or 0.0)},
            ],
        }

    if strategy_name == "bull_put_spread":
        short_k = _nearest_chain_strike(chain_rows, atm)
        long_k = _nearest_chain_strike(chain_rows, atm - strike_step)
        if short_k is None or long_k is None or long_k >= short_k:
            return None
        short_q = _find_quote(chain_rows, short_k, "PE")
        long_q = _find_quote(chain_rows, long_k, "PE")
        if short_q is None or long_q is None:
            return None
        credit = float(short_q.get("ltp") or 0.0) - float(long_q.get("ltp") or 0.0)
        if credit <= 0:
            return None
        return {
            "action": "SELL",
            "side": "SELL",
            "option_type": "PE",
            "strike": float(short_k),
            "entry_price": round(credit, 2),
            "stop_loss": round(credit * 1.9, 2),
            "take_profit": round(credit * 0.35, 2),
            "trailing_stop_loss": None,
            "trail_trigger_price": None,
            "trail_step_pct": None,
            "legs": [
                {"side": "SELL", "option_type": "PE", "strike": float(short_k), "qty": 1, "price": float(short_q.get("ltp") or 0.0)},
                {"side": "BUY", "option_type": "PE", "strike": float(long_k), "qty": 1, "price": float(long_q.get("ltp") or 0.0)},
            ],
        }

    if strategy_name == "bear_call_spread":
        short_k = _nearest_chain_strike(chain_rows, atm)
        long_k = _nearest_chain_strike(chain_rows, atm + strike_step)
        if short_k is None or long_k is None or long_k <= short_k:
            return None
        short_q = _find_quote(chain_rows, short_k, "CE")
        long_q = _find_quote(chain_rows, long_k, "CE")
        if short_q is None or long_q is None:
            return None
        credit = float(short_q.get("ltp") or 0.0) - float(long_q.get("ltp") or 0.0)
        if credit <= 0:
            return None
        return {
            "action": "SELL",
            "side": "SELL",
            "option_type": "CE",
            "strike": float(short_k),
            "entry_price": round(credit, 2),
            "stop_loss": round(credit * 1.9, 2),
            "take_profit": round(credit * 0.35, 2),
            "trailing_stop_loss": None,
            "trail_trigger_price": None,
            "trail_step_pct": None,
            "legs": [
                {"side": "SELL", "option_type": "CE", "strike": float(short_k), "qty": 1, "price": float(short_q.get("ltp") or 0.0)},
                {"side": "BUY", "option_type": "CE", "strike": float(long_k), "qty": 1, "price": float(long_q.get("ltp") or 0.0)},
            ],
        }

    if strategy_name == "iron_condor":
        sp = _nearest_chain_strike(chain_rows, atm - strike_step)
        lp = _nearest_chain_strike(chain_rows, atm - (2 * strike_step))
        sc = _nearest_chain_strike(chain_rows, atm + strike_step)
        lc = _nearest_chain_strike(chain_rows, atm + (2 * strike_step))
        if None in {sp, lp, sc, lc}:
            return None
        if not (float(lp) < float(sp) < float(sc) < float(lc)):
            return None
        sp_q = _find_quote(chain_rows, float(sp), "PE")
        lp_q = _find_quote(chain_rows, float(lp), "PE")
        sc_q = _find_quote(chain_rows, float(sc), "CE")
        lc_q = _find_quote(chain_rows, float(lc), "CE")
        if any(x is None for x in (sp_q, lp_q, sc_q, lc_q)):
            return None
        credit = (
            float((sp_q or {}).get("ltp") or 0.0)
            - float((lp_q or {}).get("ltp") or 0.0)
            + float((sc_q or {}).get("ltp") or 0.0)
            - float((lc_q or {}).get("ltp") or 0.0)
        )
        if credit <= 0:
            return None
        return {
            "action": "SELL",
            "side": "SELL",
            "option_type": None,
            "strike": float(atm),
            "entry_price": round(credit, 2),
            "stop_loss": round(credit * 2.2, 2),
            "take_profit": round(credit * 0.30, 2),
            "trailing_stop_loss": None,
            "trail_trigger_price": None,
            "trail_step_pct": None,
            "legs": [
                {"side": "SELL", "option_type": "PE", "strike": float(sp), "qty": 1, "price": float((sp_q or {}).get("ltp") or 0.0)},
                {"side": "BUY", "option_type": "PE", "strike": float(lp), "qty": 1, "price": float((lp_q or {}).get("ltp") or 0.0)},
                {"side": "SELL", "option_type": "CE", "strike": float(sc), "qty": 1, "price": float((sc_q or {}).get("ltp") or 0.0)},
                {"side": "BUY", "option_type": "CE", "strike": float(lc), "qty": 1, "price": float((lc_q or {}).get("ltp") or 0.0)},
            ],
        }
    return None


def build_option_signal(
    *,
    symbol: str,
    interval: str,
    expiry_date: date,
    underlying_price: float,
    underlying_signal_action: SignalAction,
    underlying_conviction: str,
    underlying_confidence: float,
    underlying_expected_return_pct: float,
    chain_rows: list[dict],
    strike_step: int,
    strike_mode: str = "auto",
    manual_strike: float | None = None,
    allow_option_writing: bool = False,
    strategy_mode: StrategyName = "auto",
    technical_context: dict | None = None,
) -> dict:
    reasons: list[str] = []
    option_type: OptionType | None = None
    side: TradeSide = "BUY"
    action: SignalAction = "HOLD"
    tc = technical_context or {}
    close = _to_float(tc.get("close")) or underlying_price
    atr = _to_float(tc.get("atr_14"))
    atr_ratio = (float(atr) / max(1.0, float(close))) if atr is not None else 0.004

    strategy_name = _determine_strategy(
        strategy_mode=str(strategy_mode),
        interval=interval,
        underlying_signal_action=underlying_signal_action,
        underlying_confidence=float(underlying_confidence),
        underlying_expected_return_pct=float(underlying_expected_return_pct),
        allow_option_writing=allow_option_writing,
        technical_context=tc,
        chain_rows=chain_rows,
        underlying_price=underlying_price,
        strike_step=strike_step,
    )
    reasons.append(f"Strategy selected: {strategy_name}")

    multi_leg = strategy_name in {
        "iron_condor",
        "bull_put_spread",
        "bear_call_spread",
        "long_straddle",
        "long_strangle",
    }
    if multi_leg and not allow_option_writing and strategy_name in {
        "iron_condor",
        "bull_put_spread",
        "bear_call_spread",
    }:
        reasons.append("Option writing disabled; switched to directional option-buying strategy")
        strategy_name = "trend_vwap_oi" if strategy_name != "iron_condor" else "ema_vwap_rsi_atr"
        multi_leg = False

    legs: list[dict] = []
    strike = None
    quote = None
    entry_price = None
    stop_loss = None
    trailing_stop_loss = None
    trail_trigger_price = None
    take_profit = None
    trail_step_pct = None

    if multi_leg:
        basket = _build_multi_leg(
            strategy_name=strategy_name,
            underlying_signal_action=underlying_signal_action,
            underlying_price=underlying_price,
            strike_step=strike_step,
            chain_rows=chain_rows,
        )
        if basket is None:
            action = "HOLD"
            reasons.append("Selected multi-leg strategy is not available with current chain liquidity")
        else:
            action = basket["action"]  # type: ignore[assignment]
            side = basket["side"]  # type: ignore[assignment]
            option_type = basket["option_type"]  # type: ignore[assignment]
            strike = basket["strike"]
            entry_price = basket["entry_price"]
            stop_loss = basket["stop_loss"]
            take_profit = basket["take_profit"]
            trailing_stop_loss = basket["trailing_stop_loss"]
            trail_trigger_price = basket["trail_trigger_price"]
            trail_step_pct = basket["trail_step_pct"]
            legs = list(basket.get("legs") or [])
    else:
        if underlying_signal_action == "BUY":
            option_type = "CE"
            action = "BUY"
            reasons.append("Underlying model is bullish, mapped to CE long setup")
        elif underlying_signal_action == "SELL":
            option_type = "PE"
            action = "BUY"
            reasons.append("Underlying model is bearish, mapped to PE long setup")
        else:
            action = "HOLD"
            reasons.append("Underlying signal is HOLD, no directional entry")

        if action == "BUY" and option_type is not None:
            strike = auto_select_strike(
                underlying_price=underlying_price,
                strike_step=strike_step,
                option_type=option_type,
                conviction=underlying_conviction,
                chain_rows=chain_rows,
                strike_mode=strike_mode,
                manual_strike=manual_strike,
                expected_return_pct=float(underlying_expected_return_pct),
            )
            if strike is not None:
                quote = _find_quote(chain_rows, strike, option_type)
            if strike is None or quote is None:
                action = "HOLD"
                reasons.append("No liquid strike available in current chain window")
            else:
                entry_price, stop_loss, trailing_stop_loss, trail_trigger_price, trail_step_pct = _single_leg_payload(
                    option_type=option_type,
                    strike=strike,
                    quote=quote,
                    conviction=underlying_conviction,
                    atr_ratio=atr_ratio,
                )
                if entry_price is None:
                    action = "HOLD"
                    reasons.append("Missing LTP for selected strike")
                else:
                    rr_mult = 1.45 if strategy_name == "trend_vwap_oi" else 1.35
                    take_profit = round(entry_price * rr_mult, 2)
                    tc_rsi = _to_float(tc.get("rsi_14"))
                    tc_macd_hist = _to_float(tc.get("macd_hist"))
                    tc_ema21 = _to_float(tc.get("ema_21"))
                    tc_ema50 = _to_float(tc.get("ema_50"))
                    tc_vwap = _to_float(tc.get("vwap")) or close
                    atm = nearest_strike(underlying_price, strike_step)
                    skew = _oi_bias(chain_rows, atm, strike_step)
                    if strategy_name == "trend_vwap_oi":
                        reasons.append("Trend + VWAP + OI confirmation strategy is active")
                        reasons.append(f"OI skew near ATM: {skew:.2f}")
                    else:
                        reasons.append("EMA + VWAP + RSI + ATR risk strategy is active")
                    if tc_rsi is not None:
                        reasons.append(f"RSI {tc_rsi:.1f}")
                    if tc_macd_hist is not None:
                        reasons.append(f"MACD hist {tc_macd_hist:.2f}")
                    if tc_ema21 is not None and tc_ema50 is not None:
                        reasons.append(f"EMA21 {tc_ema21:.2f} vs EMA50 {tc_ema50:.2f}")
                    reasons.append(f"VWAP reference {tc_vwap:.2f}")
                    reasons.append(
                        f"Risk plan (ATR aware): SL {stop_loss:.2f} | TP {take_profit:.2f}"
                    )
                    legs = [
                        {
                            "side": "BUY",
                            "option_type": option_type,
                            "strike": float(strike),
                            "qty": 1,
                            "price": float(entry_price),
                        }
                    ]

    if strike_mode == "auto" and strike is not None:
        reasons.append("Strike selected automatically via dynamic moneyness + OI heatmap + max pain proxy")
    if strike_mode == "manual" and manual_strike is not None and strike is not None:
        reasons.append("Manual strike mode selected nearest available strike in chain")
    if underlying_expected_return_pct != underlying_expected_return_pct:
        underlying_expected_return_pct = 0.0

    return {
        "symbol": symbol,
        "interval": interval,
        "expiry_date": expiry_date.isoformat(),
        "underlying_price": float(round(underlying_price, 2)),
        "underlying_signal_action": underlying_signal_action,
        "underlying_conviction": underlying_conviction,
        "underlying_confidence": float(underlying_confidence),
        "underlying_expected_return_pct": float(underlying_expected_return_pct),
        "strike_step": int(strike_step),
        "strike_mode": strike_mode,
        "manual_strike": float(manual_strike) if manual_strike is not None else None,
        "option_signal": {
            "action": action,
            "strategy": strategy_name,
            "option_type": option_type,
            "side": side,
            "strike": float(strike) if strike is not None else None,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "trailing_stop_loss": trailing_stop_loss,
            "trail_trigger_price": trail_trigger_price,
            "trail_step_pct": trail_step_pct,
            "take_profit": take_profit,
            "confidence": float(underlying_confidence),
            "legs": legs,
            "reasons": reasons,
        },
    }
