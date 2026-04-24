from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.market_stream_runtime import get_market_stream_runtime_status
from api.schemas import HealthResponse
from db.connection import get_db_session
from db.models import DataFreshness, RawCandle
from utils.config import get_settings
from utils.constants import IST_ZONE

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=datetime.now(IST_ZONE))


@router.get("/detailed")
def health_detailed(db: Session = Depends(get_db_session)) -> dict:
    now = datetime.now(IST_ZONE)
    checks: dict[str, dict] = {}

    # 1. Database connectivity
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = {"status": "ok"}
    except Exception as exc:
        checks["database"] = {"status": "error", "detail": str(exc)}

    # 2. Data freshness — last candle timestamp
    try:
        latest = db.query(RawCandle).filter(RawCandle.interval == "1minute").order_by(RawCandle.ts.desc()).first()
        if latest is None:
            checks["data_freshness"] = {"status": "warn", "detail": "no candles found"}
        else:
            age_seconds = (now - latest.ts.replace(tzinfo=IST_ZONE) if latest.ts.tzinfo is None else now - latest.ts).total_seconds()
            stale = age_seconds > 120  # stale if >2 min during market hours
            checks["data_freshness"] = {
                "status": "warn" if stale else "ok",
                "last_candle_ts": latest.ts.isoformat(),
                "age_seconds": round(age_seconds, 1),
                "instrument_key": latest.instrument_key,
            }
    except Exception as exc:
        checks["data_freshness"] = {"status": "error", "detail": str(exc)}

    # 3. Market stream status
    try:
        stream_status = get_market_stream_runtime_status()
        checks["market_stream"] = {
            "status": "ok" if stream_status.get("running") else "warn",
            **stream_status,
        }
    except Exception as exc:
        checks["market_stream"] = {"status": "error", "detail": str(exc)}

    # 4. Upstox token configured
    settings = get_settings()
    token_ok = bool(settings.upstox_access_token.strip())
    checks["broker_token"] = {
        "status": "ok" if token_ok else "error",
        "detail": "token configured" if token_ok else "UPSTOX_ACCESS_TOKEN not set",
    }

    # 5. Execution state
    checks["execution"] = {
        "status": "ok",
        "enabled": settings.execution_enabled,
        "mode": settings.execution_mode,
        "symbols": settings.execution_symbol_list,
    }

    overall = "ok" if all(c["status"] == "ok" for c in checks.values()) else (
        "error" if any(c["status"] == "error" for c in checks.values()) else "warn"
    )
    return {
        "status": overall,
        "timestamp": now.isoformat(),
        "checks": checks,
    }
