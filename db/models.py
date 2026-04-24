from datetime import datetime

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class RawCandle(Base):
    __tablename__ = "raw_candles"
    __table_args__ = (
        UniqueConstraint("instrument_key", "interval", "ts", name="uq_raw_candle"),
        Index("ix_raw_candles_instrument_interval_ts", "instrument_key", "interval", "ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_key: Mapped[str] = mapped_column(String(128), index=True)
    interval: Mapped[str] = mapped_column(String(32), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    oi: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="upstox")
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RawNews(Base):
    __tablename__ = "raw_news"
    __table_args__ = (UniqueConstraint("source", "url", name="uq_raw_news_source_url"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(2048))
    url: Mapped[str] = mapped_column(String(4096))
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    content: Mapped[str] = mapped_column(Text, default="")
    symbols: Mapped[list[str]] = mapped_column(JSON, default=list)
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0)
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FeaturesDaily(Base):
    __tablename__ = "features_daily"
    __table_args__ = (UniqueConstraint("symbol", "session_date", name="uq_features_daily"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(128), index=True)
    session_date: Mapped[datetime] = mapped_column(Date, index=True)
    features: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LabelsDaily(Base):
    __tablename__ = "labels_daily"
    __table_args__ = (UniqueConstraint("symbol", "session_date", name="uq_labels_daily"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(128), index=True)
    session_date: Mapped[datetime] = mapped_column(Date, index=True)
    next_open: Mapped[float] = mapped_column(Float)
    next_high: Mapped[float] = mapped_column(Float)
    next_low: Mapped[float] = mapped_column(Float)
    next_close: Mapped[float] = mapped_column(Float)
    next_direction: Mapped[int] = mapped_column(Integer)  # -1 sell, 0 hold, 1 buy
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PredictionsDaily(Base):
    __tablename__ = "predictions_daily"
    __table_args__ = (
        UniqueConstraint("symbol", "interval", "target_session_date", "model_version", name="uq_pred_daily"),
        Index("ix_predictions_symbol_interval_target", "symbol", "interval", "target_session_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(128), index=True)
    interval: Mapped[str] = mapped_column(String(32), index=True, default="day")
    target_session_date: Mapped[datetime] = mapped_column(Date, index=True)
    pred_open: Mapped[float] = mapped_column(Float)
    pred_high: Mapped[float] = mapped_column(Float)
    pred_low: Mapped[float] = mapped_column(Float)
    pred_close: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(8))
    confidence: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(128), index=True)
    feature_cutoff_ist: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class PredictionsIntraday(Base):
    __tablename__ = "predictions_intraday"
    __table_args__ = (
        UniqueConstraint("symbol", "interval", "target_ts", "model_version", name="uq_pred_intraday"),
        Index("ix_predictions_intraday_symbol_interval_ts", "symbol", "interval", "target_ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(128), index=True)
    interval: Mapped[str] = mapped_column(String(32), index=True)  # 1minute / 30minute
    target_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    pred_open: Mapped[float] = mapped_column(Float)
    pred_high: Mapped[float] = mapped_column(Float)
    pred_low: Mapped[float] = mapped_column(Float)
    pred_close: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(8))
    confidence: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(128), index=True)
    feature_cutoff_ist: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    __table_args__ = (
        UniqueConstraint("model_name", "model_version", "symbol", name="uq_model_registry_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    model_version: Mapped[str] = mapped_column(String(128), index=True)
    model_type: Mapped[str] = mapped_column(String(64))
    symbol: Mapped[str] = mapped_column(String(128), index=True)
    artifact_path: Mapped[str] = mapped_column(String(1024))
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    trained_from: Mapped[datetime] = mapped_column(Date)
    trained_to: Mapped[datetime] = mapped_column(Date)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class OOFPrediction(Base):
    __tablename__ = "oof_predictions"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "symbol",
            "session_date",
            "model_name",
            name="uq_oof_predictions",
        ),
        Index("ix_oof_run_symbol_date", "run_id", "symbol", "session_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(128), index=True)
    session_date: Mapped[datetime] = mapped_column(Date, index=True)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    fold: Mapped[int] = mapped_column(Integer, default=0)
    pred_open: Mapped[float | None] = mapped_column(Float, nullable=True)
    pred_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    pred_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    pred_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    direction_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_direction: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CalibrationRegistry(Base):
    __tablename__ = "calibration_registry"
    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "model_family",
            "calibration_version",
            name="uq_calibration_registry",
        ),
        Index("ix_calibration_symbol_family_active", "symbol", "model_family", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(128), index=True)
    model_family: Mapped[str] = mapped_column(String(128), index=True)
    calibration_version: Mapped[str] = mapped_column(String(128), index=True)
    method: Mapped[str] = mapped_column(String(64))
    artifact_path: Mapped[str] = mapped_column(String(1024))
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    __table_args__ = (
        UniqueConstraint("run_id", "symbol", "model_family", name="uq_backtest_runs"),
        Index("ix_backtest_symbol_family_created", "symbol", "model_family", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(128), index=True)
    model_family: Mapped[str] = mapped_column(String(128), index=True)
    window_from: Mapped[datetime] = mapped_column(Date, index=True)
    window_to: Mapped[datetime] = mapped_column(Date, index=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    passed_gates: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DriftMetric(Base):
    __tablename__ = "drift_metrics"
    __table_args__ = (
        Index("ix_drift_symbol_family_metric", "symbol", "model_family", "metric_name", "computed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(128), index=True)
    model_family: Mapped[str] = mapped_column(String(128), index=True)
    metric_name: Mapped[str] = mapped_column(String(128), index=True)
    metric_value: Mapped[float] = mapped_column(Float)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DataFreshness(Base):
    __tablename__ = "data_freshness"
    __table_args__ = (UniqueConstraint("source_name", name="uq_source_freshness"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_name: Mapped[str] = mapped_column(String(128), index=True)
    last_success_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(32), default="ok")
    details: Mapped[dict] = mapped_column(JSON, default=dict)


class OptionQuote(Base):
    __tablename__ = "option_quotes"
    __table_args__ = (
        UniqueConstraint(
            "underlying_symbol",
            "expiry_date",
            "strike",
            "option_type",
            "ts",
            name="uq_option_quote_snapshot",
        ),
        Index("ix_option_quotes_underlying_expiry_strike_type_ts",
              "underlying_symbol", "expiry_date", "strike", "option_type", "ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_key: Mapped[str] = mapped_column(String(128), index=True)
    underlying_key: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    underlying_symbol: Mapped[str] = mapped_column(String(64), index=True)
    expiry_date: Mapped[datetime] = mapped_column(Date, index=True)
    strike: Mapped[float] = mapped_column(Float, index=True)
    option_type: Mapped[str] = mapped_column(String(2), index=True)  # CE / PE
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ltp: Mapped[float] = mapped_column(Float)
    bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float] = mapped_column(Float, default=0.0)
    oi: Mapped[float | None] = mapped_column(Float, nullable=True)
    close_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    bid_qty: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask_qty: Mapped[float | None] = mapped_column(Float, nullable=True)
    prev_oi: Mapped[float | None] = mapped_column(Float, nullable=True)
    iv: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    gamma: Mapped[float | None] = mapped_column(Float, nullable=True)
    theta: Mapped[float | None] = mapped_column(Float, nullable=True)
    vega: Mapped[float | None] = mapped_column(Float, nullable=True)
    pop: Mapped[float | None] = mapped_column(Float, nullable=True)
    pcr: Mapped[float | None] = mapped_column(Float, nullable=True)
    underlying_spot_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="model")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class OptionTradeSignal(Base):
    __tablename__ = "option_trade_signals"
    __table_args__ = (
        Index("ix_option_signals_symbol_interval_expiry_generated",
              "symbol", "interval", "expiry_date", "generated_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    interval: Mapped[str] = mapped_column(String(32), index=True)
    expiry_date: Mapped[datetime] = mapped_column(Date, index=True)
    option_type: Mapped[str] = mapped_column(String(2), index=True)  # CE / PE
    side: Mapped[str] = mapped_column(String(4), default="BUY")  # BUY / SELL
    action: Mapped[str] = mapped_column(String(8), index=True)  # BUY / SELL / HOLD
    strike: Mapped[float] = mapped_column(Float, index=True)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    trailing_stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    trail_trigger_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    trail_step_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reasons: Mapped[list[str]] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class ExecutionPosition(Base):
    __tablename__ = "execution_positions"
    __table_args__ = (
        Index("ix_execution_positions_trade_status_symbol", "trade_date", "status", "symbol"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trade_date: Mapped[datetime] = mapped_column(Date, index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    interval: Mapped[str] = mapped_column(String(32), index=True)
    strategy_name: Mapped[str] = mapped_column(String(64), index=True)
    option_type: Mapped[str] = mapped_column(String(2), default="")
    side: Mapped[str] = mapped_column(String(8), default="BUY")  # BUY or SELL premium metric
    expiry_date: Mapped[datetime] = mapped_column(Date, index=True)
    strike: Mapped[float] = mapped_column(Float, default=0.0)
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), index=True, default="OPEN")  # OPEN/CLOSED
    entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    entry_premium: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float] = mapped_column(Float, default=0.0)
    initial_sl: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_sl: Mapped[float | None] = mapped_column(Float, nullable=True)
    trailing_stop: Mapped[float] = mapped_column(Float, default=0.0)
    peak_premium: Mapped[float | None] = mapped_column(Float, nullable=True)
    tsl_active: Mapped[bool] = mapped_column(Boolean, default=False)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_premium: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_premium: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_premium: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_points: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    unrealized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    ml_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ai_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    pine_signal: Mapped[str | None] = mapped_column(String(16), nullable=True)
    consensus_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    entry_order_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, server_default=func.now())
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    legs_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class ExecutionOrder(Base):
    __tablename__ = "execution_orders"
    __table_args__ = (
        Index("ix_execution_orders_position_status", "position_id", "status", "created_at"),
        Index("ix_execution_orders_trade_symbol_kind", "trade_date", "symbol", "order_kind", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    position_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    trade_date: Mapped[datetime] = mapped_column(Date, index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    strike_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    option_type: Mapped[str | None] = mapped_column(String(2), nullable=True)
    expiry_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    order_kind: Mapped[str] = mapped_column(String(16), index=True)  # ENTRY/SL/EXIT/MODIFY/CANCEL
    side: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    trigger_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_premium: Mapped[float | None] = mapped_column(Float, nullable=True)
    initial_sl: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_sl: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_premium: Mapped[float | None] = mapped_column(Float, nullable=True)
    peak_premium: Mapped[float | None] = mapped_column(Float, nullable=True)
    tsl_active: Mapped[bool] = mapped_column(Boolean, default=False)
    exit_premium: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    unrealized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    ml_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ai_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    pine_signal: Mapped[str | None] = mapped_column(String(16), nullable=True)
    consensus_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="NEW")
    broker_name: Mapped[str] = mapped_column(String(32), default="paper")
    broker_order_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    response_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, server_default=func.now())


class DailySummary(Base):
    __tablename__ = "daily_summary"
    __table_args__ = (
        Index("ix_daily_summary_pnl", "date", "total_pnl", "total_trades"),
    )

    date: Mapped[datetime] = mapped_column(Date, primary_key=True)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)
    losing_trades: Mapped[int] = mapped_column(Integer, default=0)
    total_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    max_profit_trade: Mapped[float] = mapped_column(Float, default=0.0)
    max_loss_trade: Mapped[float] = mapped_column(Float, default=0.0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    is_green: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class SignalLog(Base):
    __tablename__ = "signal_log"
    __table_args__ = (
        Index("ix_signal_log_symbol_ts", "symbol", "interval", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, server_default=func.now())
    trade_date: Mapped[datetime] = mapped_column(Date, index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    interval: Mapped[str] = mapped_column(String(32), index=True, default="1minute")
    ml_signal: Mapped[str] = mapped_column(String(8))
    ml_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    ml_expected_move: Mapped[float | None] = mapped_column(Float, nullable=True)
    pine_signal: Mapped[str] = mapped_column(String(16), default="NEUTRAL")
    pine_age_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ai_score: Mapped[float] = mapped_column(Float, default=0.0)
    news_sentiment: Mapped[float | None] = mapped_column(Float, nullable=True)
    combined_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    consensus: Mapped[str] = mapped_column(String(32), default="non_trade_signal")
    skip_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    trade_placed: Mapped[bool] = mapped_column(Boolean, default=False)
    details: Mapped[dict] = mapped_column(JSON, default=dict)


class ExecutionSignalAudit(Base):
    __tablename__ = "execution_signal_audit"
    __table_args__ = (
        UniqueConstraint("symbol", "interval", "candle_ts", name="uq_execution_signal_candle"),
        Index("ix_execution_signal_audit_trade_symbol", "trade_date", "symbol", "interval", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trade_date: Mapped[datetime] = mapped_column(Date, index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    interval: Mapped[str] = mapped_column(String(32), index=True)
    candle_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    signal_action: Mapped[str] = mapped_column(String(8), default="HOLD")
    strategy_name: Mapped[str] = mapped_column(String(64), default="auto")
    executed: Mapped[bool] = mapped_column(default=False)
    skip_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    option_type: Mapped[str | None] = mapped_column(String(2), nullable=True)
    strike: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ExecutionExternalSignal(Base):
    __tablename__ = "execution_external_signals"
    __table_args__ = (
        UniqueConstraint("source", "symbol", "interval", "signal_ts", name="uq_execution_external_signal"),
        Index("ix_execution_external_signal_consume",
              "symbol", "interval", "processed", "signal_ts", "created_at"),
        Index("ix_execution_external_signals_source_processed", "source", "processed", "signal_ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    interval: Mapped[str] = mapped_column(String(32), index=True)
    signal_action: Mapped[str] = mapped_column(String(8), index=True)
    signal_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    processed: Mapped[bool] = mapped_column(default=False, index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class OrderBookSnapshot(Base):
    __tablename__ = "order_book_snapshots"
    __table_args__ = (
        UniqueConstraint("instrument_key", "ts", name="uq_order_book_snapshot"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_key: Mapped[str] = mapped_column(String(128), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    best_bid: Mapped[float] = mapped_column(Float)
    best_ask: Mapped[float] = mapped_column(Float)
    mid_price: Mapped[float] = mapped_column(Float)
    spread_bps: Mapped[float] = mapped_column(Float)
    bid_volume: Mapped[float] = mapped_column(Float)
    ask_volume: Mapped[float] = mapped_column(Float)
    depth_imbalance: Mapped[float] = mapped_column(Float)  # (bid_vol - ask_vol) / total_vol
    liquidity_score: Mapped[float] = mapped_column(Float)  # Combined metric
    depth_data: Mapped[dict] = mapped_column(JSON, default=dict)  # Full depth levels
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
