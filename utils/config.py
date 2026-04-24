from functools import lru_cache
from typing import List

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils.symbols import normalize_symbol_key


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
    execution_mode: str = "paper"  # paper/live
    execution_symbols: str = ""
    execution_interval: str = "1minute"
    execution_strategy_mode: str = "auto"
    execution_allow_option_writing: bool = False
    execution_poll_seconds: int = 1
    execution_symbol_lot_sizes: str = ""
    option_chain_refresh_seconds: int = 4
    signal_min_score: float = 63.0
    signal_cooldown_minutes: int = 12
    signal_max_per_day: int = 3
    ui_stream_interval_ms: int = 500
    ui_tick_interval_ms: int = 75
    history_retention_years: int = 2
    history_bootstrap_on_start: bool = False
    market_stream_autostart: bool | None = None

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
        default=5,
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
    ai_quality_minimum: float = Field(default=65.0, validation_alias="AI_QUALITY_MINIMUM")
    combined_score_threshold: float = Field(default=0.65, validation_alias="COMBINED_SCORE_THRESHOLD")
    tsl_activation_percent: float = Field(default=0.05, validation_alias="TSL_ACTIVATION_PERCENT")
    tsl_trail_percent: float = Field(default=0.03, validation_alias="TSL_TRAIL_PERCENT")
    target_profit_percent: float = Field(default=0.30, validation_alias="TARGET_PROFIT_PERCENT")
    force_squareoff_time: str = Field(default="15:15", validation_alias="FORCE_SQUAREOFF_TIME")
    entry_window_start: str = Field(default="09:20", validation_alias="ENTRY_WINDOW_START")
    entry_window_end: str = Field(default="13:30", validation_alias="ENTRY_WINDOW_END")
    live_execution_blocked_symbols: str = "India VIX"

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
    def market_data_access_token(self) -> str:
        analytics = self.upstox_analytics_token.strip()
        if analytics:
            return analytics
        return self.upstox_access_token.strip()

    @property
    def has_market_data_access(self) -> bool:
        return bool(self.market_data_access_token)

    @property
    def execution_symbol_list(self) -> List[str]:
        if self.execution_symbols.strip():
            return [x.strip() for x in self.execution_symbols.split(",") if x.strip()]
        return [x.split("|", 1)[1] if "|" in x else x for x in self.instrument_keys]

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

    @property
    def should_autostart_market_stream(self) -> bool:
        if self.market_stream_autostart is not None:
            return bool(self.market_stream_autostart)
        return self.market_data_mode.strip().lower() == "websocket" and self.database_url.startswith("sqlite")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
