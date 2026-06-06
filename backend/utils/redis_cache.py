from __future__ import annotations

import json
from typing import Any

from backend.utils.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_CLIENT: Any | None = None
_CLIENT_FAILED = False


def _client() -> Any | None:
    global _CLIENT, _CLIENT_FAILED
    settings = get_settings()
    if not bool(getattr(settings, "redis_cache_enabled", True)):
        return None
    if _CLIENT_FAILED:
        return None
    if _CLIENT is not None:
        return _CLIENT
    try:
        import redis

        client = redis.Redis.from_url(
            str(getattr(settings, "redis_url", "redis://127.0.0.1:6379/0")),
            socket_connect_timeout=0.03,
            socket_timeout=0.05,
            decode_responses=True,
        )
        client.ping()
        _CLIENT = client
        return _CLIENT
    except Exception as exc:
        _CLIENT_FAILED = True
        logger.warning("Redis cache unavailable; using in-process cache only: %s", exc)
        return None


def get_json(key: str) -> dict[str, Any] | None:
    global _CLIENT_FAILED
    client = _client()
    if client is None:
        return None
    try:
        raw = client.get(key)
        if not raw:
            return None
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        _CLIENT_FAILED = True
        logger.warning("Redis cache read failed for %s: %s", key, exc)
        return None


def set_json(key: str, payload: dict[str, Any], *, ttl_seconds: int) -> None:
    global _CLIENT_FAILED
    client = _client()
    if client is None:
        return
    try:
        client.setex(key, max(1, int(ttl_seconds)), json.dumps(payload, separators=(",", ":")))
    except Exception as exc:
        _CLIENT_FAILED = True
        logger.warning("Redis cache write failed for %s: %s", key, exc)
