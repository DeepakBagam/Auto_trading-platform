from __future__ import annotations

from datetime import datetime
from threading import Lock, Thread
from typing import Any

from backend.data_layer.streamers.upstox_market_stream import UpstoxMarketStream
from backend.utils.config import Settings, get_settings, read_runtime_upstox_access_token
from backend.utils.constants import IST_ZONE
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_runtime_lock = Lock()
_runtime_thread: Thread | None = None
_runtime_stream: UpstoxMarketStream | None = None
_runtime_status: dict[str, Any] = {
    "owner": "api_process",
    "autostart_enabled": False,
    "running": False,
    "start_attempted_at": None,
    "last_started_at": None,
    "last_error": None,
    "token_marker": None,
}


def _token_marker(settings: Settings) -> str:
    token = read_runtime_upstox_access_token(settings)
    if not token:
        return ""
    return f"{len(token)}:{token[:4]}:{token[-4:]}"


def ensure_market_stream_started(settings: Settings | None = None) -> bool:
    cfg = settings or get_settings()
    enabled = bool(cfg.should_autostart_market_stream)
    marker = _token_marker(cfg)
    stale_stream: UpstoxMarketStream | None = None
    stale_thread: Thread | None = None

    with _runtime_lock:
        _runtime_status["autostart_enabled"] = enabled
        thread = _runtime_thread
        if thread is not None and thread.is_alive():
            if marker and _runtime_status.get("token_marker") != marker:
                logger.warning("Market stream token changed; restarting websocket stream")
                stale_stream = _runtime_stream
                stale_thread = _runtime_thread
                globals()["_runtime_stream"] = None
                globals()["_runtime_thread"] = None
                _runtime_status["running"] = False
            else:
                _runtime_status["running"] = True
                return False

    if stale_stream is not None:
        try:
            stale_stream.stop()
        except Exception:
            logger.exception("Failed to stop stale market stream before restart")
    if stale_thread is not None and stale_thread.is_alive():
        stale_thread.join(timeout=5.0)

    with _runtime_lock:
        thread = _runtime_thread
        if thread is not None and thread.is_alive():
            _runtime_status["running"] = True
            return False
        if not enabled:
            _runtime_status["running"] = False
            return False
        _runtime_status["start_attempted_at"] = datetime.now(IST_ZONE).isoformat()
        _runtime_status["last_error"] = None
        try:
            stream = UpstoxMarketStream(settings=cfg)
        except Exception as exc:
            _runtime_status["running"] = False
            _runtime_status["last_error"] = str(exc)
            logger.exception("Failed to initialize API-managed market stream")
            return False

        def _runner() -> None:
            try:
                stream.run_forever()
            except Exception as exc:
                logger.exception("API-managed market stream stopped unexpectedly")
                with _runtime_lock:
                    _runtime_status["last_error"] = str(exc)
            finally:
                with _runtime_lock:
                    _runtime_status["running"] = False

        thread = Thread(target=_runner, name="api-market-stream", daemon=True)
        _runtime_status["running"] = True
        _runtime_status["last_started_at"] = datetime.now(IST_ZONE).isoformat()
        _runtime_status["token_marker"] = marker
        globals()["_runtime_stream"] = stream
        globals()["_runtime_thread"] = thread
        thread.start()
        logger.info("Started API-managed market stream background thread")
        return True


def stop_market_stream_background(join_timeout: float = 5.0) -> None:
    with _runtime_lock:
        stream = _runtime_stream
        thread = _runtime_thread

    if stream is not None:
        try:
            stream.stop()
        except Exception:
            logger.exception("Failed to stop API-managed market stream cleanly")

    if thread is not None and thread.is_alive():
        thread.join(timeout=join_timeout)

    with _runtime_lock:
        globals()["_runtime_stream"] = None
        globals()["_runtime_thread"] = None
        _runtime_status["running"] = False


def get_market_stream_runtime_status(settings: Settings | None = None) -> dict[str, Any]:
    cfg = settings or get_settings()
    with _runtime_lock:
        status = dict(_runtime_status)
        status["thread_alive"] = bool(_runtime_thread is not None and _runtime_thread.is_alive())
    status["autostart_expected"] = bool(cfg.should_autostart_market_stream)
    return status
