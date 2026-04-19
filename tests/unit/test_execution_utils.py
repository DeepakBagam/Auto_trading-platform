from datetime import datetime
from types import SimpleNamespace

from execution_engine.ai_intelligence import score_trade_intelligence
from execution_engine.broker import PaperBroker
from execution_engine.engine import IntradayOptionsExecutionEngine
from execution_engine.risk_manager import compute_quantity, update_trailing_stop
from execution_engine.strike_selector import lot_size_for_symbol, select_option_contract
from prediction_engine.consensus_engine import ConsensusResult
from prediction_engine.options_engine import build_chain_rows, synthetic_option_chain
from utils.constants import IST_ZONE


def test_strike_selector_maps_buy_to_ce() -> None:
    quotes = synthetic_option_chain(
        symbol="Nifty 50",
        underlying_price=22435.0,
        expiry_date=datetime.now(IST_ZONE).date(),
        strike_step=50,
        levels=4,
    )
    chain = build_chain_rows(quotes)
    out = select_option_contract(
        signal_action="BUY",
        spot_price=22435.0,
        strike_step=50,
        chain_rows=chain,
        confidence=0.72,
        expected_return_pct=0.01,
    )
    assert out is not None
    assert out.option_type == "CE"
    assert "dynamic_heatmap_maxpain_selection" in out.reason


def test_strike_selector_maps_sell_to_pe() -> None:
    quotes = synthetic_option_chain(
        symbol="Nifty 50",
        underlying_price=22435.0,
        expiry_date=datetime.now(IST_ZONE).date(),
        strike_step=50,
        levels=4,
    )
    chain = build_chain_rows(quotes)
    out = select_option_contract(
        signal_action="SELL",
        spot_price=22435.0,
        strike_step=50,
        chain_rows=chain,
        confidence=0.78,
        expected_return_pct=-0.012,
    )
    assert out is not None
    assert out.option_type == "PE"


def test_risk_quantity_and_trailing_logic() -> None:
    sizing = compute_quantity(
        capital=500000.0,
        capital_per_trade_pct=0.02,
        entry_price=100.0,
        lot_size=25,
        fixed_lots=1,
        vix_level=14.0,
    )
    assert sizing.lots == 1
    assert sizing.qty == 25
    tr_buy = update_trailing_stop(
        side="BUY",
        entry_metric=100.0,
        current_metric=130.0,
        existing_trailing_stop=90.0,
        hard_stop=75.0,
    )
    tr_sell = update_trailing_stop(
        side="SELL",
        entry_metric=100.0,
        current_metric=70.0,
        existing_trailing_stop=120.0,
        hard_stop=125.0,
    )
    assert tr_buy >= 100.0
    assert tr_sell <= 100.0


def test_current_default_lot_sizes(monkeypatch) -> None:
    monkeypatch.setattr(
        "execution_engine.strike_selector.get_settings",
        lambda: SimpleNamespace(execution_symbol_lot_size_map={}),
    )
    assert lot_size_for_symbol("Nifty 50") == 65
    assert lot_size_for_symbol("Bank Nifty") == 30
    assert lot_size_for_symbol("SENSEX") == 20


def test_ai_intelligence_score_range() -> None:
    out = score_trade_intelligence(
        signal_action="BUY",
        confidence=0.74,
        expected_return_pct=0.01,
        technical_context={
            "close": 22440.0,
            "ema_21": 22420.0,
            "ema_50": 22390.0,
            "rsi_14": 58.0,
            "macd_hist": 2.1,
            "vwap": 22410.0,
            "atr_14": 52.0,
        },
        now=datetime.now(IST_ZONE),
    )
    assert 0.0 <= out.score <= 100.0


def test_ai_intelligence_rewards_mtf_alignment() -> None:
    aligned = score_trade_intelligence(
        signal_action="BUY",
        confidence=0.78,
        expected_return_pct=0.012,
        technical_context={
            "close": 22440.0,
            "ema_9": 22420.0,
            "ema_21": 22390.0,
            "ema_50": 22340.0,
            "ema_21_slope_3": 11.0,
            "rsi_14": 61.0,
            "macd_hist": 2.1,
            "vwap": 22400.0,
            "atr_14": 58.0,
            "mtf_3m_action": "BUY",
            "mtf_5m_action": "BUY",
        },
        now=datetime.now(IST_ZONE),
    )
    conflicted = score_trade_intelligence(
        signal_action="BUY",
        confidence=0.78,
        expected_return_pct=0.012,
        technical_context={
            "close": 22440.0,
            "ema_9": 22420.0,
            "ema_21": 22390.0,
            "ema_50": 22340.0,
            "ema_21_slope_3": -11.0,
            "rsi_14": 61.0,
            "macd_hist": 2.1,
            "vwap": 22400.0,
            "atr_14": 58.0,
            "mtf_3m_action": "SELL",
            "mtf_5m_action": "SELL",
        },
        now=datetime.now(IST_ZONE),
    )
    assert aligned.score > conflicted.score


def test_execution_engine_disabled_short_circuit() -> None:
    settings = SimpleNamespace(
        execution_enabled=False,
        execution_mode="paper",
        upstox_base_url="https://api.upstox.com",
        upstox_access_token="x",
    )
    engine = IntradayOptionsExecutionEngine(settings=settings, broker=PaperBroker())
    out = engine.run_once(db=SimpleNamespace())
    assert out["status"] == "disabled"


class _FakeDb:
    def add(self, _value) -> None:
        return None

    def flush(self) -> None:
        return None

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


def test_evaluate_symbol_skips_when_pine_signal_missing(monkeypatch) -> None:
    settings = SimpleNamespace(
        execution_enabled=True,
        execution_mode="paper",
        upstox_base_url="https://api.upstox.com",
        upstox_access_token="x",
        execution_symbols="Nifty 50",
        entry_window_start="09:20",
        entry_window_end="13:30",
        force_squareoff_time="15:15",
        execution_capital=500000.0,
        execution_per_trade_risk_pct=0.02,
        execution_premium_min=40.0,
        execution_premium_max=300.0,
        execution_symbol_list=["Nifty 50"],
    )
    engine = IntradayOptionsExecutionEngine(settings=settings, broker=PaperBroker())
    now = datetime.fromisoformat("2026-04-08T10:00:00+05:30")
    candle_ts = datetime.fromisoformat("2026-04-08T09:59:00+05:30")
    monkeypatch.setattr(engine, "_latest_candle_ts", lambda db, symbol, interval: candle_ts)
    monkeypatch.setattr(engine, "_open_positions", lambda db, symbol=None: [])
    monkeypatch.setattr(
        "execution_engine.engine.get_consensus_signal",
        lambda *args, **kwargs: ConsensusResult(
            symbol="Nifty 50",
            interval="1minute",
            timestamp=now,
            ml_signal="BUY",
            ml_confidence=0.82,
            ml_expected_move=110.0,
            ml_reasons=["ML passed"],
            pine_signal="NEUTRAL",
            pine_age_seconds=None,
            ai_score=70.0,
            ai_reasons=["AI passed"],
            news_sentiment=0.0,
            combined_score=0.58,
            consensus="non_trade_signal",
            skip_reason="Pine signal missing",
        ),
    )
    monkeypatch.setattr(
        "execution_engine.engine.log_consensus_result",
        lambda db, result: SimpleNamespace(id=1, skip_reason=result.skip_reason, trade_placed=False, details={}),
    )

    out = engine._evaluate_symbol(_FakeDb(), now, "Nifty 50")
    assert out == "skip:Pine signal missing"


def test_evaluate_symbol_skips_when_pine_signal_mismatch(monkeypatch) -> None:
    settings = SimpleNamespace(
        execution_enabled=True,
        execution_mode="paper",
        upstox_base_url="https://api.upstox.com",
        upstox_access_token="x",
        execution_symbols="Nifty 50",
        entry_window_start="09:20",
        entry_window_end="13:30",
        force_squareoff_time="15:15",
        execution_capital=500000.0,
        execution_per_trade_risk_pct=0.02,
        execution_premium_min=40.0,
        execution_premium_max=300.0,
        execution_symbol_list=["Nifty 50"],
    )
    engine = IntradayOptionsExecutionEngine(settings=settings, broker=PaperBroker())
    now = datetime.fromisoformat("2026-04-08T10:00:00+05:30")
    candle_ts = datetime.fromisoformat("2026-04-08T09:59:00+05:30")
    monkeypatch.setattr(engine, "_latest_candle_ts", lambda db, symbol, interval: candle_ts)
    monkeypatch.setattr(engine, "_open_positions", lambda db, symbol=None: [])
    monkeypatch.setattr(
        "execution_engine.engine.get_consensus_signal",
        lambda *args, **kwargs: ConsensusResult(
            symbol="Nifty 50",
            interval="1minute",
            timestamp=now,
            ml_signal="BUY",
            ml_confidence=0.82,
            ml_expected_move=110.0,
            ml_reasons=["ML passed"],
            pine_signal="SELL",
            pine_age_seconds=15,
            ai_score=70.0,
            ai_reasons=["AI passed"],
            news_sentiment=0.0,
            combined_score=0.54,
            consensus="non_trade_signal",
            skip_reason="Pine signal mismatch (SELL != BUY)",
        ),
    )
    monkeypatch.setattr(
        "execution_engine.engine.log_consensus_result",
        lambda db, result: SimpleNamespace(id=1, skip_reason=result.skip_reason, trade_placed=False, details={}),
    )

    out = engine._evaluate_symbol(_FakeDb(), now, "Nifty 50")
    assert out == "skip:Pine signal mismatch (SELL != BUY)"
