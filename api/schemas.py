from datetime import date, datetime

from pydantic import BaseModel


class QuantileBand(BaseModel):
    p10: float
    p50: float
    p90: float


class PredictionIntervals(BaseModel):
    open: QuantileBand | None = None
    high: QuantileBand | None = None
    low: QuantileBand | None = None
    close: QuantileBand | None = None


class PredictResponse(BaseModel):
    symbol: str
    interval: str
    prediction_mode: str = "standard"
    source_interval: str | None = None
    target_session_date: date
    target_ts: datetime | None = None
    pred_open: float
    pred_high: float
    pred_low: float
    pred_close: float
    direction: str
    confidence: float
    direction_prob_calibrated: float | None = None
    confidence_score: float | None = None
    confidence_bucket: str | None = None
    pred_interval: PredictionIntervals | None = None
    model_family: str | None = None
    calibration_version: str | None = None
    confidence_components: dict | None = None
    model_version: str
    feature_cutoff_ist: datetime
    generated_at: datetime


class SignalResponse(BaseModel):
    symbol: str
    interval: str
    prediction_mode: str
    action: str
    conviction: str
    latest_price: float
    predicted_price: float
    expected_return_pct: float
    expected_move_points: float
    confidence: float
    confidence_bucket: str | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_reward_ratio: float | None = None
    technical_score: float | None = None
    target_session_date: date
    target_ts: datetime | None = None
    generated_at: datetime
    model_version: str
    reasons: list[str]


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime


class DataFreshnessResponse(BaseModel):
    source_name: str
    last_success_at: datetime
    status: str
    details: dict


class OptionQuotePayload(BaseModel):
    instrument_key: str | None = None
    ltp: float | None = None
    bid: float | None = None
    ask: float | None = None
    volume: float | None = None
    oi: float | None = None
    close_price: float | None = None
    bid_qty: float | None = None
    ask_qty: float | None = None
    prev_oi: float | None = None
    iv: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    pop: float | None = None
    source: str | None = None


class OptionChainRow(BaseModel):
    strike: float
    pcr: float | None = None
    underlying_spot_price: float | None = None
    ce: OptionQuotePayload | None = None
    pe: OptionQuotePayload | None = None


class OptionSignalPayload(BaseModel):
    action: str
    strategy: str | None = None
    option_type: str | None = None
    side: str | None = None
    strike: float | None = None
    entry_price: float | None = None
    stop_loss: float | None = None
    trailing_stop_loss: float | None = None
    trail_trigger_price: float | None = None
    trail_step_pct: float | None = None
    take_profit: float | None = None
    confidence: float | None = None
    legs: list[dict] = []
    reasons: list[str] = []


class TradeIntelligencePayload(BaseModel):
    score: float
    trend_continuation_prob: float
    false_breakout_risk: float
    premium_expansion_prob: float
    tod_profitability_score: float
    reasons: list[str] = []


class OptionsSignalResponse(BaseModel):
    symbol: str
    interval: str
    expiry_date: date
    available_expiries: list[date]
    underlying_price: float
    underlying_signal_action: str
    underlying_conviction: str | None = None
    underlying_confidence: float | None = None
    underlying_expected_return_pct: float | None = None
    strike_step: int
    strike_mode: str
    manual_strike: float | None = None
    auto_selected_strike: float | None = None
    atm_strike: float | None = None
    max_pain: float | None = None
    chain_source: str | None = None
    chain_generated_at: datetime | None = None
    option_signal: OptionSignalPayload
    trade_intelligence: TradeIntelligencePayload | None = None
    chain: list[OptionChainRow]


class ExecutionRunResponse(BaseModel):
    status: str
    at: datetime | None = None
    details: dict = {}


class ExecutionReportResponse(BaseModel):
    trade_date: date
    total_trades: int
    win_rate: float
    max_drawdown: float
    total_profit: float
    missed_signals: int
    executed_signals: int
    total_signal_events: int


class ExternalSignalRequest(BaseModel):
    symbol: str
    signal_action: str
    confidence: float = 0.7
    source: str = "pine"
    metadata_json: dict = {}
