from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import AppSetting, AuditLog, ExecutionPosition
from backend.utils.config import Settings, get_settings
from backend.utils.constants import IST_ZONE

RUNTIME_SETTINGS_KEY = "runtime_execution_settings"
SENSITIVE_RUNTIME_KEYS = {"upstox_access_token", "upstox_sandbox_access_token", "smtp_password"}
RUNTIME_SETTING_KEYS = {
    "execution_enabled",
    "execution_symbols",
    "execution_capital",
    "execution_per_trade_risk_pct",
    "execution_max_daily_loss_pct",
    "execution_max_simultaneous_trades",
    "execution_max_daily_trades",
    "execution_lot_size",
    "execution_premium_min",
    "execution_premium_max",
    "execution_stop_loss_pct",
    "tsl_activation_percent",
    "tsl_trail_percent",
    "tsl_immediate",
    "target_profit_percent",
    "entry_window_start",
    "entry_window_end",
    "force_squareoff_time",
    "signal_min_score",
    "signal_cooldown_minutes",
    "signal_max_per_day",
    "signal_require_volume_confirmation",
    "signal_min_volume_ratio",
    "signal_require_breakout",
    "signal_rsi_buy_min",
    "signal_rsi_sell_max",
    "signal_vix_min",
    "signal_vix_max",
    "signal_atr_min_points",
    "signal_atr_max_points",
    "signal_min_adx",
    "signal_symbol_profiles",
    "option_min_volume",
    "option_min_oi",
    "option_max_spread_pct",
    "upstox_access_token",
    "upstox_sandbox_access_token",
    "sandbox_limit_protection_pct",
    "sandbox_price_tick",
    "smtp_enabled",
    "smtp_host",
    "smtp_port",
    "smtp_username",
    "smtp_password",
    "smtp_from_email",
    "smtp_to_emails",
    "smtp_use_tls",
    "smtp_use_ssl",
}


def _ensure_setting(db: Session, key: str) -> AppSetting:
    row = db.get(AppSetting, key)
    if row is None:
        row = AppSetting(key=key, value_json={})
        db.add(row)
        db.flush()
    return row


def get_setting_value(db: Session, key: str, default: Any = None) -> Any:
    row = db.get(AppSetting, key)
    if row is None:
        return default
    value = row.value_json or {}
    if isinstance(value, dict) and "value" in value:
        return value.get("value", default)
    return value if value is not None else default


def set_setting_value(db: Session, key: str, value: Any) -> None:
    row = _ensure_setting(db, key)
    row.value_json = {"value": value}


def mask_secret(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}...{text[-4:]}"


def get_runtime_execution_settings(db: Session) -> dict[str, Any]:
    raw = get_setting_value(db, RUNTIME_SETTINGS_KEY, {})
    return dict(raw) if isinstance(raw, dict) else {}


def _set_env_file_key(key: str, value: str, env_path: Path | None = None) -> None:
    path = env_path or Path(".env")
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    found = False
    updated: list[str] = []
    for line in lines:
        if line.lstrip().startswith(f"{key}="):
            updated.append(f"{key}={value}")
            found = True
        else:
            updated.append(line)
    if not found:
        updated.append(f"{key}={value}")
    path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def set_runtime_execution_settings(db: Session, values: dict[str, Any]) -> dict[str, Any]:
    current = get_runtime_execution_settings(db)
    for key, value in values.items():
        if key not in RUNTIME_SETTING_KEYS:
            continue
        if key in SENSITIVE_RUNTIME_KEYS and (value is None or str(value).strip() == ""):
            continue
        if key == "upstox_access_token" and value:
            token = str(value).strip()
            current[key] = token
            os.environ["UPSTOX_ACCESS_TOKEN"] = token
            _set_env_file_key("UPSTOX_ACCESS_TOKEN", token)
        elif key == "upstox_sandbox_access_token" and value:
            token = str(value).strip()
            current[key] = token
            os.environ["UPSTOX_SANDBOX_ACCESS_TOKEN"] = token
            _set_env_file_key("UPSTOX_SANDBOX_ACCESS_TOKEN", token)
        else:
            current[key] = value
    set_setting_value(db, RUNTIME_SETTINGS_KEY, current)
    return current


def apply_runtime_execution_settings(db: Session, settings: Settings) -> Settings:
    runtime = get_runtime_execution_settings(db)
    for key, value in runtime.items():
        if key not in RUNTIME_SETTING_KEYS or not hasattr(settings, key):
            continue
        if key in SENSITIVE_RUNTIME_KEYS and (value is None or str(value).strip() == ""):
            continue
        setattr(settings, key, value)
    token = runtime.get("upstox_access_token")
    if token:
        os.environ["UPSTOX_ACCESS_TOKEN"] = str(token).strip()
    sandbox_token = runtime.get("upstox_sandbox_access_token")
    if sandbox_token:
        os.environ["UPSTOX_SANDBOX_ACCESS_TOKEN"] = str(sandbox_token).strip()
    return settings


def get_runtime_trading_mode(db: Session, settings: Settings | None = None) -> str:
    cfg = settings or get_settings()
    runtime_mode = str(get_setting_value(db, "runtime_trading_mode", cfg.execution_mode) or cfg.execution_mode)
    normalized = runtime_mode.strip().lower()
    if normalized == "paper":
        set_setting_value(db, "runtime_trading_mode", "sandbox")
        runtime = get_runtime_execution_settings(db)
        runtime["execution_enabled"] = False
        set_setting_value(db, RUNTIME_SETTINGS_KEY, runtime)
        db.commit()
        return "sandbox"
    return normalized if normalized in {"sandbox", "live"} else "sandbox"


def set_runtime_trading_mode(db: Session, mode: str) -> str:
    normalized = str(mode or "sandbox").strip().lower()
    if normalized not in {"sandbox", "live"}:
        raise ValueError("Trading mode must be 'sandbox' or 'live'")
    set_setting_value(db, "runtime_trading_mode", normalized)
    return normalized


def get_sandbox_reset_at(db: Session) -> datetime | None:
    raw = get_setting_value(db, "sandbox_reset_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


def get_sandbox_starting_balance(db: Session, settings: Settings | None = None) -> float:
    cfg = settings or get_settings()
    raw = get_setting_value(db, "sandbox_starting_balance", cfg.execution_capital)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(cfg.execution_capital)


def reset_sandbox_account(
    db: Session,
    *,
    starting_balance: float,
    clear_open_positions: bool = False,
) -> dict[str, Any]:
    now = datetime.now(IST_ZONE)
    set_setting_value(db, "sandbox_starting_balance", float(starting_balance))
    set_setting_value(db, "sandbox_reset_at", now.isoformat())
    deleted_positions = 0
    if clear_open_positions:
        rows = (
            db.execute(select(ExecutionPosition).where(ExecutionPosition.status == "OPEN"))
            .scalars()
            .all()
        )
        for row in rows:
            metadata = row.metadata_json or {}
            if str(metadata.get("execution_mode") or "").lower() != "sandbox":
                continue
            db.delete(row)
            deleted_positions += 1
    return {
        "sandbox_starting_balance": float(starting_balance),
        "sandbox_reset_at": now.isoformat(),
        "deleted_open_positions": deleted_positions,
    }


def create_audit_log(
    db: Session,
    *,
    action: str,
    resource: str,
    status: str = "INFO",
    message: str = "",
    resource_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> AuditLog:
    row = AuditLog(
        action=str(action),
        resource=str(resource),
        resource_id=str(resource_id) if resource_id is not None else None,
        status=str(status),
        message=str(message or ""),
        details=details or {},
    )
    db.add(row)
    db.flush()
    return row
