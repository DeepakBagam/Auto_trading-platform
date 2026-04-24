# Realtime Options Trading Desk

FastAPI backend + React frontend for low-latency index options monitoring and paper/live execution.

## What is in this repo now
- Upstox websocket market stream for live 1-minute candles
- Lightweight technical signal engine with strict breakout and trend confirmation
- Fast option contract selection for CE/PE entries
- Streaming UI served directly from the API
- Paper or live execution loop with stop, target, and trailing-stop management

## Quick start
1. Copy `.env.example` to `.env`
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Initialize the database:
```bash
python -m db.init_db
```
4. Start the API:
```bash
python scripts/start_api.py
```
5. Start the execution worker:
```bash
python scripts/start_execution_loop.py
```
6. Start the market stream:
```bash
python scripts/start_market_stream.py
```

## Main runtime paths
- UI: `/`
- Live snapshot API: `/api/live/snapshot`
- Live stream API: `/api/live/stream`
- Execution status: `/execution/status`

## Notes
- The old ML, Pine, backtest, and training stack has been removed from the live runtime.
- `EXECUTION_ENABLED`, `EXECUTION_MODE`, `UPSTOX_ACCESS_TOKEN`, and `UPSTOX_INSTRUMENT_KEYS` are the key env vars for deployment.
- The UI is static and served from the Python app, so no separate Node build step is required.
