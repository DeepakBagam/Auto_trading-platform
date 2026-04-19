# Automated AI Trading Platform

Production-first greenfield platform for next-day Indian market candle prediction.

## What This V1 Includes
- Upstox historical/live candle ingestion scaffolding
- India-focused news ingestion with 3-source blend:
  - Economic Times RSS + Moneycontrol RSS
  - NewsAPI
  - Finnhub (global macro events)
- Raw + processed data model in Postgres/Timescale
- Canonical candle/prediction tables plus per-symbol/per-interval SQL views:
  - `candles_nifty50_1m`, `candles_nifty50_30m`, `candles_nifty50_1d`
  - `candles_indiavix_1m`, `candles_indiavix_30m`, `candles_indiavix_1d`
  - `candles_sensex_1m`, `candles_sensex_30m`, `candles_sensex_1d`
  - matching `predictions_*` views
- Feature engineering and next-day label generation
- XGBoost stack for OHLC regression + direction classification
- V2 stack modules:
  - `lstm_gru_v2` (sequence model)
  - `garch_v2` (EWMA-GARCH volatility proxy)
  - `gap_v2` (next-open gap regressor)
  - `meta_model_v2` (stacked ridge/logistic combiner)
- Confidence scoring, API endpoints, scheduler, and basic backtesting utilities

## Quick Start
1. Copy env:
```bash
cp .env.example .env
```
2. Install deps:
```bash
pip install -e ".[dev]"
```
3. Initialize DB schema (SQLite default):
```bash
python -m db.init_db
```

### Optional: PostgreSQL/Timescale Later
1. Start DB:
```bash
docker compose up -d
```
2. In `.env`, either set `DATABASE_URL=postgresql+psycopg://...` or use `DB_*` values.
3. Re-run:
```bash
python -m db.init_db
```

## Run Commands
- Generate Upstox access token (OAuth code flow): `python scripts/generate_upstox_access_token.py`
- Ingest news once: `python scripts/ingest_news.py`
- Ingest candles once: `python scripts/ingest_candles.py`
- Start live market stream over WebSocket: `python scripts/start_market_stream.py`
- Build features+labels: `python scripts/build_features.py`
- Train models: `python scripts/train_models.py`
- Run inference: `python scripts/run_inference.py`
- Start API: `python scripts/start_api.py`
- Start scheduler: `python scripts/start_scheduler.py`
- Phase-1 runner: `python scripts/run_phase1_upstox_nodocker.py`
- Generate a one-session trade audit: `python scripts/generate_session_trade_audit.py --date 2026-04-17`

## Notes
- FinBERT scoring is supported through optional dependency group `.[nlp]`.
- Celery, Redis, Flower, and Prometheus extras are available through `.[ops]`.
- `lstm_gru_v2` also uses `torch` from `.[nlp]`.
- Without NLP extras, the system falls back to lightweight heuristic sentiment.
- Upstox response shapes can vary by endpoint version; adjust parser in `data_layer/collectors/upstox_collector.py` as needed.
- Set `MARKET_DATA_MODE=websocket` to disable candle polling in the scheduler and use `python scripts/start_market_stream.py` for live 1-minute ingestion.
- Default news polling cadence is every 10 minutes (`NEWS_POLL_MINUTES=10`).
- Finnhub is enabled by default (`ENABLE_FINNHUB=true`) when `FINNHUB_API_KEY` is set.
- Real live-data paper-trading instructions are in [docs/REAL_MARKET_PAPER_TRADING.md](docs/REAL_MARKET_PAPER_TRADING.md).
- Trade/CE-PE/strike/deployment notes are in [docs/TRADE_FLOW_AND_DEPLOYMENT.md](docs/TRADE_FLOW_AND_DEPLOYMENT.md).
- `docker-compose.yml` is the minimal DB stack. `docker-compose-full.yml` is the full local stack and now builds from the repo `Dockerfile`.
