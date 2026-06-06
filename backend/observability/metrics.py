"""
Prometheus metrics for observability.
Tracks API requests, model performance, execution metrics, and system health.
"""
from prometheus_client import Counter, Gauge, Histogram, Info

# API Metrics
api_requests_total = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

api_request_duration_seconds = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Data Ingestion Metrics
candles_ingested_total = Counter(
    "candles_ingested_total",
    "Total candles ingested",
    ["symbol", "interval", "source"],
)

news_articles_ingested_total = Counter(
    "news_articles_ingested_total",
    "Total news articles ingested",
    ["source"],
)

data_ingestion_errors_total = Counter(
    "data_ingestion_errors_total",
    "Total data ingestion errors",
    ["source", "error_type"],
)

data_freshness_seconds = Gauge(
    "data_freshness_seconds",
    "Seconds since last successful data ingestion",
    ["source"],
)

# Model Metrics
model_training_duration_seconds = Histogram(
    "model_training_duration_seconds",
    "Model training duration in seconds",
    ["symbol", "model_type"],
    buckets=[10, 30, 60, 300, 600, 1800, 3600],
)

model_inference_duration_seconds = Histogram(
    "model_inference_duration_seconds",
    "Model inference duration in seconds",
    ["symbol", "model_type"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

predictions_generated_total = Counter(
    "predictions_generated_total",
    "Total predictions generated",
    ["symbol", "interval", "model_version"],
)

model_accuracy = Gauge(
    "model_accuracy",
    "Model accuracy score",
    ["symbol", "model_type", "metric"],
)

model_confidence = Histogram(
    "model_confidence",
    "Model confidence distribution",
    ["symbol", "model_type"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Execution Metrics
trades_executed_total = Counter(
    "trades_executed_total",
    "Total trades executed",
    ["symbol", "action", "strategy", "status"],
)

positions_open = Gauge(
    "positions_open",
    "Number of open positions",
    ["symbol", "strategy"],
)

position_pnl = Histogram(
    "position_pnl",
    "Position PnL distribution",
    ["symbol", "strategy"],
    buckets=[-10000, -5000, -1000, -500, 0, 500, 1000, 5000, 10000, 50000],
)

daily_pnl = Gauge(
    "daily_pnl",
    "Daily PnL",
    ["date"],
)

win_rate = Gauge(
    "win_rate",
    "Win rate percentage",
    ["symbol", "strategy"],
)

execution_latency_seconds = Histogram(
    "execution_latency_seconds",
    "Order execution latency in seconds",
    ["broker", "order_type"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)

slippage_bps = Histogram(
    "slippage_bps",
    "Slippage in basis points",
    ["symbol", "broker"],
    buckets=[0, 1, 5, 10, 20, 50, 100],
)

# Risk Metrics
capital_utilization = Gauge(
    "capital_utilization",
    "Capital utilization percentage",
    ["strategy"],
)

max_drawdown = Gauge(
    "max_drawdown",
    "Maximum drawdown",
    ["symbol", "strategy"],
)

risk_limit_breaches_total = Counter(
    "risk_limit_breaches_total",
    "Total risk limit breaches",
    ["limit_type"],
)

# System Metrics
websocket_connections = Gauge(
    "websocket_connections",
    "Active WebSocket connections",
    ["broker", "status"],
)

websocket_reconnections_total = Counter(
    "websocket_reconnections_total",
    "Total WebSocket reconnections",
    ["broker", "reason"],
)

database_query_duration_seconds = Histogram(
    "database_query_duration_seconds",
    "Database query duration in seconds",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)

celery_task_duration_seconds = Histogram(
    "celery_task_duration_seconds",
    "Celery task duration in seconds",
    ["task_name", "status"],
    buckets=[1, 5, 10, 30, 60, 300, 600, 1800],
)

celery_queue_length = Gauge(
    "celery_queue_length",
    "Number of tasks in Celery queue",
    ["queue_name"],
)

# Application Info
app_info = Info("app_info", "Application information")
app_info.info(
    {
        "version": "0.1.0",
        "environment": "production",
        "platform": "automated_ai_trading",
    }
)
