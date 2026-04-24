import json
import logging
import re
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo

from utils.config import get_settings

# ---------------------------------------------------------------------------
# Regex patterns used by SanitizingFilter
# ---------------------------------------------------------------------------
_BEARER_RE = re.compile(
    r"(Bearer\s+)[A-Za-z0-9._\-+/=]{20,}",
    re.IGNORECASE,
)
_SENSITIVE_KEYS_RE = re.compile(
    r"(?i)(access_token|api_key|api_secret|client_secret|password)"
    r"(\s*[=:]\s*)\S+",
)


def _scrub(text: str) -> str:
    text = _BEARER_RE.sub(r"\1***", text)
    text = _SENSITIVE_KEYS_RE.sub(r"\1\2***", text)
    return text


class SanitizingFilter(logging.Filter):
    """Remove sensitive tokens and credentials from log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            msg = record.getMessage()
            sanitized = _scrub(msg)
            record.msg = sanitized
            record.args = None

            if record.exc_text:
                record.exc_text = _scrub(record.exc_text)
        except Exception:
            # Never let a filter crash the logging machinery.
            pass
        return True


class JsonFormatter(logging.Formatter):
    def __init__(self, tz: ZoneInfo) -> None:
        super().__init__()
        self.tz = tz

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(self.tz).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def _sanitize_log_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned or "platform"


def setup_logging(log_name: str | None = None) -> None:
    settings = get_settings()
    root = logging.getLogger()
    if getattr(root, "_aatp_logging_configured", False):
        return
    root.setLevel(settings.log_level.upper())
    formatter = JsonFormatter(ZoneInfo(settings.timezone))

    sanitizing_filter = SanitizingFilter()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(sanitizing_filter)

    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{_sanitize_log_name(log_name or 'platform')}.log"
    file_handler = TimedRotatingFileHandler(
        log_dir / file_name,
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(sanitizing_filter)

    root.handlers = [console_handler, file_handler]
    root._aatp_logging_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
