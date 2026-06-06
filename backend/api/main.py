import asyncio
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

from backend.api.market_stream_runtime import ensure_market_stream_started, stop_market_stream_background
from backend.api.routes.candles import router as candles_router
from backend.api.routes.execution import router as execution_router
from backend.api.routes.health import router as health_router
from backend.api.routes.live import router as live_router
from backend.data_layer.collectors.upstox_collector import UpstoxCollector
from backend.db.connection import SessionLocal
from backend.db.init_db import init_db
from backend.observability.middleware import PrometheusMiddleware
from backend.utils.config import get_settings
from backend.utils.constants import IST_ZONE
from backend.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="Realtime Options Trading Desk", version="1.0.0")

app.add_middleware(GZipMiddleware, minimum_size=1024)

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)

# ── Simple in-memory rate limiter for execution endpoints ──────────────────────
_RATE_LIMIT_WINDOW = 30   # seconds
_RATE_LIMIT_MAX = 6       # max guarded execution mutations per window per IP
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_rate_limit_lock = threading.Lock()
_RATE_LIMITED_EXECUTION_PATHS = {
    "/execution/run-once",
    "/execution/emergency-exit",
    "/execution/paper/reset",
    "/execution/update-sl-target",
}


def _check_rate_limit(client_ip: str) -> bool:
    now = time.monotonic()
    with _rate_limit_lock:
        calls = _rate_limit_store[client_ip]
        calls[:] = [t for t in calls if now - t < _RATE_LIMIT_WINDOW]
        if len(calls) >= _RATE_LIMIT_MAX:
            return False
        calls.append(now)
        return True


def _should_rate_limit_execution_request(method: str, path: str) -> bool:
    normalized_path = str(path or "").rstrip("/")
    is_position_mutation = (
        normalized_path.startswith("/execution/positions/")
        and method.upper() in {"POST", "DELETE"}
    )
    is_guarded_endpoint = normalized_path in _RATE_LIMITED_EXECUTION_PATHS or is_position_mutation
    return method.upper() != "GET" and is_guarded_endpoint


@app.middleware("http")
async def execution_rate_limit(request: Request, call_next):
    if _should_rate_limit_execution_request(request.method, request.url.path):
        client_ip = request.client.host if request.client else "unknown"
        if not _check_rate_limit(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded: max {_RATE_LIMIT_MAX} execution calls per {_RATE_LIMIT_WINDOW}s"},
            )
    return await call_next(request)


# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(health_router)
app.include_router(candles_router)
app.include_router(live_router)
app.include_router(execution_router)


# ── Option chain periodic refresh ─────────────────────────────────────────────
_option_chain_refresh_task: asyncio.Task | None = None
_daily_ingestion_task: asyncio.Task | None = None


async def _refresh_option_chains_loop() -> None:
    from backend.db.connection import SessionLocal
    from backend.data_layer.collectors.upstox_option_chain import UpstoxOptionChainCollector
    from backend.execution_engine.live_service import resolve_underlying_key, _resolve_expiry

    settings = get_settings()
    refresh_interval = int(getattr(settings, "option_chain_refresh_seconds", 300))

    while True:
        await asyncio.sleep(refresh_interval)
        now = datetime.now(IST_ZONE)
        # Only refresh during market hours (9:00 – 15:35 IST)
        if not (9 <= now.hour < 15 or (now.hour == 15 and now.minute <= 35)):
            continue
        try:
            collector = UpstoxOptionChainCollector(settings=settings)
            db = SessionLocal()
            try:
                for symbol in settings.execution_symbol_list:
                    underlying_key = resolve_underlying_key(db, symbol, settings=settings)
                    expiry_date, _ = _resolve_expiry(
                        symbol=symbol,
                        underlying_key=underlying_key,
                        settings=settings,
                    )
                    collector.sync_option_chain(
                        db,
                        underlying_key=underlying_key,
                        underlying_symbol=symbol,
                        expiry_date=expiry_date,
                    )
                    db.commit()
            finally:
                db.close()
        except Exception:
            logger.exception("Option chain refresh failed; next scheduled cycle will retry.")


async def _daily_ingestion_loop() -> None:
    settings = get_settings()
    if not bool(settings.data_ingestion_enabled):
        return
    target_hour = max(0, min(23, int(settings.data_ingestion_daily_hour_ist)))
    while True:
        now = datetime.now(IST_ZONE)
        next_run = now.replace(hour=target_hour, minute=5, second=0, microsecond=0)
        if next_run <= now:
            next_run = next_run + timedelta(days=1)
        await asyncio.sleep(max(60.0, (next_run - now).total_seconds()))
        db = SessionLocal()
        try:
            collector = UpstoxCollector()
            collector.ingest_historical_batch(db, days_back=7)
        except Exception:
            logger.exception("Daily historical ingestion failed; next scheduled cycle will retry.")
        finally:
            db.close()


@app.on_event("startup")
def startup_market_stream() -> None:
    init_db()
    ensure_market_stream_started(get_settings())


@app.on_event("startup")
async def startup_option_chain_refresh() -> None:
    global _option_chain_refresh_task, _daily_ingestion_task
    _option_chain_refresh_task = asyncio.create_task(_refresh_option_chains_loop())
    _daily_ingestion_task = asyncio.create_task(_daily_ingestion_loop())


@app.on_event("shutdown")
def shutdown_market_stream() -> None:
    stop_market_stream_background()
    if _option_chain_refresh_task is not None:
        _option_chain_refresh_task.cancel()
    if _daily_ingestion_task is not None:
        _daily_ingestion_task.cancel()

WEB_DIR = Path(__file__).resolve().parents[2] / "frontend" / "web"
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/dashboard", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        # Serve a favicon if available; otherwise return 204 to avoid noisy 404s
        favicon_path = WEB_DIR / "favicon.ico"
        if favicon_path.exists():
            return FileResponse(favicon_path)
        from fastapi.responses import Response

        return Response(status_code=204)
