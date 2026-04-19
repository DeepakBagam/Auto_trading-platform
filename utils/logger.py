import json
import logging
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo

from utils.config import get_settings


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

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

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

    root.handlers = [console_handler, file_handler]
    root._aatp_logging_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
