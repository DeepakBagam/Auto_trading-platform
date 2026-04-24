import asyncio
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

from api.market_stream_runtime import ensure_market_stream_started, stop_market_stream_background
from api.routes.execution import router as execution_router
from api.routes.health import router as health_router
from api.routes.live import router as live_router
from observability.middleware import PrometheusMiddleware
from utils.config import get_settings
from utils.constants import IST_ZONE
from utils.logger import setup_logging

setup_logging()

app = FastAPI(title="Realtime Options Trading Desk", version="1.0.0")

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)

# ── Simple in-memory rate limiter for execution endpoints ──────────────────────
_RATE_LIMIT_WINDOW = 30   # seconds
_RATE_LIMIT_MAX = 3       # max calls per window per IP
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_rate_limit_lock = threading.Lock()


def _check_rate_limit(client_ip: str) -> bool:
    now = time.monotonic()
    with _rate_limit_lock:
        calls = _rate_limit_store[client_ip]
        calls[:] = [t for t in calls if now - t < _RATE_LIMIT_WINDOW]
        if len(calls) >= _RATE_LIMIT_MAX:
            return False
        calls.append(now)
        return True


@app.middleware("http")
async def execution_rate_limit(request: Request, call_next):
    if request.url.path.startswith("/execution/"):
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
app.include_router(live_router)
app.include_router(execution_router)


# ── Option chain periodic refresh ─────────────────────────────────────────────
_option_chain_refresh_task: asyncio.Task | None = None


async def _refresh_option_chains_loop() -> None:
    from db.connection import SessionLocal
    from data_layer.collectors.upstox_option_chain import UpstoxOptionChainCollector
    from execution_engine.live_service import resolve_underlying_key, _resolve_expiry

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
            pass  # Non-critical; next cycle will retry


@app.on_event("startup")
def startup_market_stream() -> None:
    ensure_market_stream_started(get_settings())


@app.on_event("startup")
async def startup_option_chain_refresh() -> None:
    global _option_chain_refresh_task
    _option_chain_refresh_task = asyncio.create_task(_refresh_option_chains_loop())


@app.on_event("shutdown")
def shutdown_market_stream() -> None:
    stop_market_stream_background()
    if _option_chain_refresh_task is not None:
        _option_chain_refresh_task.cancel()

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
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
