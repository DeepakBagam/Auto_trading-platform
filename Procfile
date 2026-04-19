web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
worker: python scripts/start_execution_loop.py
stream: python scripts/start_market_stream.py
