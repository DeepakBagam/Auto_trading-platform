import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.utils.symbols import canonical_symbol_name, is_option_execution_symbol, normalize_symbol_key


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: str = "dev"
    log_level: str = "INFO"
    timezone: str = "Asia/Kolkata"

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "trading"
    db_user: str = "trading_user"
    db_password: str = "trading_pass"
    database_url_override: str = Field(default="", validation_alias="DATABASE_URL")

    upstox_base_url: str = "https://api.upstox.com"
    upstox_access_token: str = ""
    upstox_sandbox_access_token: str = ""
    upstox_analytics_token: str = ""
    upstox_api_key: str = ""
    upstox_api_secret: str = ""
    upstox_redirect_uri: str = ""
    upstox_instrument_keys: str = ""
    upstox_history_api_version: str = "auto"

    newsapi_key: str = ""
    finnhub_api_key: str = ""
    enable_finnhub: bool = True

    news_poll_minutes: int = 10
    candle_poll_minutes: int = 1
    market_data_mode: str = "polling"
    upstox_websocket_mode: str = "full"
    upstox_websocket_reconnect_interval_seconds: int = 5
    upstox_websocket_retry_count: int = 1000
    model_artifacts_dir: str = "models/artifacts"
    feature_schema_version: str = "v3"
    label_schema_version: str = "v3"

    train_window_days: int = 1500
    validation_window_days: int = 250
    retrain_frequency_days: int = 7
    point_in_time_stale_hours: int = 24
    missing_candle_ratio_max: float = 0.03

    promotion_ece_max: float = 0.05
    promotion_coverage_target: float = 0.80
    promotion_coverage_tolerance: float = 0.05

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    smtp_enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from_email: str = ""
    smtp_to_emails: str = ""
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    smtp_timeout_seconds: int = 20

    execution_enabled: bool = False
    execution_mode: str = "sandbox"  # sandbox/live
    execution_symbols: str = ""
    execution_interval: str = "1minute"
    execution_strategy_mode: str = "auto"
    execution_allow_option_writing: bool = False
    execution_poll_seconds: int = 1
    execution_symbol_lot_sizes: str = ""
    option_chain_refresh_seconds: int = 4
    signal_min_score: float = 63.0
    signal_cooldown_minutes: int = 12
    signal_max_per_day: int = 2
    ui_stream_interval_ms: int = 500
    ui_tick_interval_ms: int = 75
    history_retention_years: int = 2
    history_bootstrap_on_start: bool = False
    market_stream_autostart: bool | None = None
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_cache_enabled: bool = True
    redis_chart_cache_ttl_seconds: int = 900

    execution_capital: float = 500000.0
    execution_per_trade_risk_pct: float = Field(
        default=0.02,
        validation_alias=AliasChoices("EXECUTION_PER_TRADE_RISK_PCT", "CAPITAL_PER_TRADE_PERCENT"),
    )
    execution_max_daily_loss_pct: float = Field(
        default=0.05,
        validation_alias=AliasChoices("EXECUTION_MAX_DAILY_LOSS_PCT", "DAILY_LOSS_LIMIT_PERCENT"),
    )
    execution_max_simultaneous_trades: int = Field(
        default=1,
        validation_alias=AliasChoices("EXECUTION_MAX_SIMULTANEOUS_TRADES", "MAX_SIMULTANEOUS_TRADES"),
    )
    execution_max_daily_trades: int = Field(
        default=2,
        validation_alias=AliasChoices("EXECUTION_MAX_DAILY_TRADES", "MAX_DAILY_TRADES"),
    )
    execution_lot_size: int = 1
    execution_stop_loss_pct: float = 0.25
    execution_min_confidence: float = 0.55
    execution_min_ai_score: float = 0.52
    execution_premium_min: float = 10.0
    execution_premium_max: float = 500.0
    execution_accept_external_webhook: bool = True
    pine_webhook_secret: str = ""
    ml_buy_threshold: float = Field(default=0.62, validation_alias="ML_BUY_THRESHOLD")
    ml_sell_threshold: float = Field(default=0.62, validation_alias="ML_SELL_THRESHOLD")
    ml_min_expected_move: float = Field(default=80.0, validation_alias="ML_MIN_EXPECTED_MOVE")
    pine_signal_max_age_seconds: int = Field(default=60, validation_alias="PINE_SIGNAL_MAX_AGE_SECONDS")
    pine_signal_sensitivity: float = Field(default=1.0, validation_alias="PINE_SIGNAL_SENSITIVITY")
    pine_signal_atr_length: int = Field(default=10, validation_alias="PINE_SIGNAL_ATR_LENGTH")
    pine_signal_atr_multiplier: float = Field(default=7.0, validation_alias="PINE_SIGNAL_ATR_MULTIPLIER")
    pine_signal_use_trend_filter: bool = Field(default=True, validation_alias="PINE_SIGNAL_USE_TREND_FILTER")
    pine_signal_ma_length: int = Field(default=20, validation_alias="PINE_SIGNAL_MA_LENGTH")
    pine_signal_use_volume_filter: bool = Field(default=False, validation_alias="PINE_SIGNAL_USE_VOLUME_FILTER")
    pine_signal_volume_threshold: float = Field(default=1.1, validation_alias="PINE_SIGNAL_VOLUME_THRESHOLD")
    pine_signal_show_signals: bool = Field(default=True, validation_alias="PINE_SIGNAL_SHOW_SIGNALS")
    pine_signal_cooldown_bars: int = Field(default=2, validation_alias="PINE_SIGNAL_COOLDOWN_BARS")
    pine_signal_atr_risk: int = Field(default=3, validation_alias="PINE_SIGNAL_ATR_RISK")
    pine_signal_risk_atr_length: int = Field(default=14, validation_alias="PINE_SIGNAL_RISK_ATR_LENGTH")
    pine_signal_percent_stop: float = Field(default=1.0, validation_alias="PINE_SIGNAL_PERCENT_STOP")
    ai_quality_minimum: float = Field(default=65.0, validation_alias="AI_QUALITY_MINIMUM")
    combined_score_threshold: float = Field(default=0.65, validation_alias="COMBINED_SCORE_THRESHOLD")
    tsl_activation_percent: float = Field(default=0.05, validation_alias="TSL_ACTIVATION_PERCENT")
    tsl_trail_percent: float = Field(default=0.03, validation_alias="TSL_TRAIL_PERCENT")
    tsl_immediate: bool = Field(default=True, validation_alias="TSL_IMMEDIATE")
    target_profit_percent: float = Field(default=0.30, validation_alias="TARGET_PROFIT_PERCENT")
    order_retry_attempts: int = Field(default=2, validation_alias="ORDER_RETRY_ATTEMPTS")
    order_retry_backoff_ms: int = Field(default=300, validation_alias="ORDER_RETRY_BACKOFF_MS")
    sandbox_limit_protection_pct: float = Field(default=0.01, validation_alias="SANDBOX_LIMIT_PROTECTION_PCT")
    sandbox_price_tick: float = Field(default=0.05, validation_alias="SANDBOX_PRICE_TICK")
    ui_ws_reconnect_base_ms: int = Field(default=1000, validation_alias="UI_WS_RECONNECT_BASE_MS")
    ui_ws_reconnect_max_ms: int = Field(default=10000, validation_alias="UI_WS_RECONNECT_MAX_MS")
    data_ingestion_symbols: str = Field(
        default="NSE_INDEX|Nifty 50,NSE_INDEX|Nifty Bank,NSE_INDEX|India VIX,BSE_INDEX|SENSEX",
        validation_alias="DATA_INGESTION_SYMBOLS",
    )
    data_ingestion_daily_hour_ist: int = Field(default=18, validation_alias="DATA_INGESTION_DAILY_HOUR_IST")
    data_ingestion_enabled: bool = Field(default=True, validation_alias="DATA_INGESTION_ENABLED")
    
    # Enhanced Risk Management
    enhanced_risk_enabled: bool = Field(default=True, validation_alias="ENHANCED_RISK_ENABLED")
    atr_sl_min_points: float = Field(default=35.0, validation_alias="ATR_SL_MIN_POINTS")
    atr_sl_max_points: float = Field(default=45.0, validation_alias="ATR_SL_MAX_POINTS")
    atr_sl_multiplier: float = Field(default=1.8, validation_alias="ATR_SL_MULTIPLIER")
    target_rr_ratio: float = Field(default=2.2, validation_alias="TARGET_RR_RATIO")
    trailing_breakeven_points: float = Field(default=25.0, validation_alias="TRAILING_BREAKEVEN_POINTS")
    trailing_lock_profit_points: float = Field(default=50.0, validation_alias="TRAILING_LOCK_PROFIT_POINTS")
    trailing_partial_exit_points: float = Field(default=70.0, validation_alias="TRAILING_PARTIAL_EXIT_POINTS")
    trailing_distance_points: float = Field(default=20.0, validation_alias="TRAILING_DISTANCE_POINTS")
    time_exit_max_minutes: int = Field(default=90, validation_alias="TIME_EXIT_MAX_MINUTES")
    time_exit_min_profit: float = Field(default=30.0, validation_alias="TIME_EXIT_MIN_PROFIT")
    regime_min_adx: float = Field(default=22.0, validation_alias="REGIME_MIN_ADX")
    regime_min_ema_separation_pct: float = Field(default=0.3, validation_alias="REGIME_MIN_EMA_SEPARATION_PCT")
    trade_cooldown_minutes: int = Field(default=30, validation_alias="TRADE_COOLDOWN_MINUTES")
    force_squareoff_time: str = Field(default="15:15", validation_alias="FORCE_SQUAREOFF_TIME")
    entry_window_start: str = Field(default="09:20", validation_alias="ENTRY_WINDOW_START")
    entry_window_end: str = Field(default="12:30", validation_alias="ENTRY_WINDOW_END")
    second_trade_entry_end: str = Field(default="11:00", validation_alias="SECOND_TRADE_ENTRY_END")
    live_execution_blocked_symbols: str = "India VIX"
    signal_require_volume_confirmation: bool = Field(default=False, validation_alias="SIGNAL_REQUIRE_VOLUME_CONFIRMATION")
    signal_min_volume_ratio: float = Field(default=1.15, validation_alias="SIGNAL_MIN_VOLUME_RATIO")
    signal_require_breakout: bool = Field(default=True, validation_alias="SIGNAL_REQUIRE_BREAKOUT")
    signal_rsi_buy_min: float = Field(default=52.0, validation_alias="SIGNAL_RSI_BUY_MIN")
    signal_rsi_sell_max: float = Field(default=48.0, validation_alias="SIGNAL_RSI_SELL_MAX")
    signal_vix_min: float = Field(default=11.0, validation_alias="SIGNAL_VIX_MIN")
    signal_vix_max: float = Field(default=20.0, validation_alias="SIGNAL_VIX_MAX")
    signal_atr_min_points: float = Field(default=4.0, validation_alias="SIGNAL_ATR_MIN_POINTS")
    signal_atr_max_points: float = Field(default=80.0, validation_alias="SIGNAL_ATR_MAX_POINTS")
    signal_min_adx: float = Field(default=0.0, validation_alias="SIGNAL_MIN_ADX")
    signal_symbol_profiles: str = Field(default="", validation_alias="SIGNAL_SYMBOL_PROFILES")
    option_min_volume: float = Field(default=500.0, validation_alias="OPTION_MIN_VOLUME")
    option_min_oi: float = Field(default=1000.0, validation_alias="OPTION_MIN_OI")
    option_max_spread_pct: float = Field(default=0.08, validation_alias="OPTION_MAX_SPREAD_PCT")

    # Redis configuration for Celery
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Broker configuration
    broker_name: str = "upstox"  # upstox, zerodha, angelone, etc.
    log_dir: str = "logs"

    @property
    def database_url(self) -> str:
        if self.database_url_override.strip():
            return self.database_url_override.strip()
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def instrument_keys(self) -> List[str]:
        return [x.strip() for x in self.upstox_instrument_keys.split(",") if x.strip()]

    @property
    def data_ingestion_instrument_keys(self) -> List[str]:
        raw = self.data_ingestion_symbols.strip()
        if raw:
            return [x.strip() for x in raw.split(",") if x.strip()]
        return self.instrument_keys

    @property
    def market_data_access_token(self) -> str:
        access_token = self.upstox_access_token.strip()
        if access_token:
            return access_token
        return self.upstox_analytics_token.strip()

    @property
    def has_market_data_access(self) -> bool:
        return bool(self.market_data_access_token)

    @property
    def execution_symbol_list(self) -> List[str]:
        if self.execution_symbols.strip():
            symbols = [x.strip() for x in self.execution_symbols.split(",") if x.strip()]
        else:
            symbols = [x.split("|", 1)[1] if "|" in x else x for x in self.instrument_keys]
        return [symbol for symbol in symbols if is_option_execution_symbol(symbol)]

    @property
    def smtp_recipients(self) -> List[str]:
        return [x.strip() for x in self.smtp_to_emails.split(",") if x.strip()]

    @property
    def execution_symbol_lot_size_map(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for item in self.execution_symbol_lot_sizes.split(","):
            chunk = item.strip()
            if not chunk:
                continue
            if ":" in chunk:
                symbol, qty = chunk.split(":", 1)
            elif "=" in chunk:
                symbol, qty = chunk.split("=", 1)
            else:
                continue
            try:
                out[normalize_symbol_key(symbol)] = int(qty)
            except ValueError:
                continue
        return out

    @property
    def live_execution_blocked_symbol_list(self) -> List[str]:
        return [normalize_symbol_key(x) for x in self.live_execution_blocked_symbols.split(",") if x.strip()]

    def signal_profile_for_symbol(self, symbol: str) -> dict:
        raw = str(self.signal_symbol_profiles or "").strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        if not isinstance(payload, dict):
            return {}
        target = normalize_symbol_key(canonical_symbol_name(symbol))
        for key, value in payload.items():
            if (
                normalize_symbol_key(canonical_symbol_name(key)) == target
                and isinstance(value, dict)
            ):
                return dict(value)
        return {}

    @property
    def should_autostart_market_stream(self) -> bool:
        if self.market_stream_autostart is not None:
            return bool(self.market_stream_autostart)
        return self.market_data_mode.strip().lower() == "websocket"

    @model_validator(mode="after")
    def validate_live_execution_safety(self) -> "Settings":
        mode = self.execution_mode.strip().lower()
        if mode == "paper":
            self.execution_mode = "sandbox"
            self.execution_enabled = False
            mode = "sandbox"
        if mode not in {"sandbox", "live"}:
            raise ValueError("EXECUTION_MODE must be 'sandbox' or 'live'")
        if mode != "live" or not self.execution_enabled:
            return self

        missing: list[str] = []
        if not self.upstox_access_token.strip():
            missing.append("UPSTOX_ACCESS_TOKEN")
        if not self.execution_symbols.strip():
            missing.append("EXECUTION_SYMBOLS")
        if missing:
            raise ValueError(
                "Live execution is enabled but required settings are missing: "
                + ", ".join(missing)
            )
        if self.execution_accept_external_webhook and not self.pine_webhook_secret.strip():
            raise ValueError(
                "Live execution with external webhooks requires PINE_WEBHOOK_SECRET."
            )
        if self.execution_allow_option_writing:
            raise ValueError("Live option writing is blocked by default safety validation.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def read_runtime_upstox_access_token(settings: Settings | None = None) -> str:
    """Read the freshest Upstox token available to the running process."""
    token_file = os.environ.get("UPSTOX_TOKEN_FILE", "").strip()
    if token_file:
        try:
            token = Path(token_file).read_text(encoding="utf-8").strip()
            if token:
                return token
        except OSError:
            pass
    token = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip()
    if token:
        return token
    env_path = Path(".env")
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("UPSTOX_ACCESS_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    if settings and settings.upstox_access_token.strip():
        return settings.upstox_access_token.strip()
    return ""


def read_runtime_upstox_sandbox_access_token(settings: Settings | None = None) -> str:
    """Read the sandbox token without falling back to the live token."""
    token = os.environ.get("UPSTOX_SANDBOX_ACCESS_TOKEN", "").strip()
    if token:
        return token
    env_path = Path(".env")
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("UPSTOX_SANDBOX_ACCESS_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    if settings and settings.upstox_sandbox_access_token.strip():
        return settings.upstox_sandbox_access_token.strip()
    return ""
