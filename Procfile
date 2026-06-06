web: uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT
worker: python backend/scripts/start_execution_loop.py
stream: python backend/scripts/start_market_stream.py
