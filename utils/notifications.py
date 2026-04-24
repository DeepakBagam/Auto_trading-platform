from __future__ import annotations

import smtplib
from collections.abc import Mapping
from datetime import date, datetime
from email.message import EmailMessage
from typing import Any

from utils.config import Settings, get_settings
from utils.constants import IST_ZONE
from utils.logger import get_logger

logger = get_logger(__name__)


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if parsed == parsed else None
    except (TypeError, ValueError):
        return None


def _format_money(value: Any) -> str:
    parsed = _float_or_none(value)
    return f"{parsed:.2f}" if parsed is not None else "-"


def _format_signed_money(value: Any) -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return "-"
    return f"{parsed:+.2f}"


def _format_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(IST_ZONE).strftime("%Y-%m-%d %H:%M:%S %Z")
    return "-"


def _format_date(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return "-"


def _format_bool(value: Any) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "-"


def _format_contract(strike: Any, option_type: Any) -> str:
    strike_value = _float_or_none(strike)
    if strike_value is None and not option_type:
        return "Spot / synthetic"
    if strike_value is None:
        return str(option_type or "-")
    strike_text = f"{strike_value:.2f}"
    suffix = str(option_type or "").strip()
    return f"{strike_text} {suffix}".strip()


def smtp_ready(settings: Settings | None = None) -> bool:
    cfg = settings or get_settings()
    return bool(
        cfg.smtp_enabled
        and cfg.smtp_host.strip()
        and cfg.smtp_from_email.strip()
        and cfg.smtp_recipients
    )


def build_order_notification_message(
    payload: Mapping[str, Any],
    settings: Settings | None = None,
) -> EmailMessage | None:
    cfg = settings or get_settings()
    if not smtp_ready(cfg):
        return None

    symbol = str(payload.get("symbol") or "-")
    order_kind = str(payload.get("order_kind") or "ORDER")
    side = str(payload.get("side") or "-")
    status = str(payload.get("status") or "-")
    contract = _format_contract(payload.get("strike_price"), payload.get("option_type"))
    pnl_value = payload.get("realized_pnl")
    if _float_or_none(pnl_value) is None:
        pnl_value = payload.get("unrealized_pnl")
    pnl_label = _format_signed_money(pnl_value)
    invested_amount = _float_or_none(payload.get("capital_invested"))
    entry_price = payload.get("entry_premium") if payload.get("entry_premium") is not None else payload.get("price")
    stop_loss = payload.get("current_sl") if payload.get("current_sl") is not None else payload.get("initial_sl")
    target = payload.get("target_premium")
    tsl_label = "ON" if payload.get("tsl_active") else "OFF"
    strategy_line = (
        f"{symbol} {contract} @ {_format_money(entry_price)}"
        if contract != "Spot / synthetic"
        else f"{symbol} spot/synthetic @ {_format_money(entry_price)}"
    )
    risk_line = (
        f"SL {_format_money(stop_loss)} | TSL {tsl_label} | TGT {_format_money(target)}"
    )

    message = EmailMessage()
    message["Subject"] = f"[{str(cfg.env).upper()}] {symbol} {order_kind} {side} {status} | P&L {pnl_label}"
    message["From"] = cfg.smtp_from_email
    message["To"] = ", ".join(cfg.smtp_recipients)
    text_lines = [
        "Realtime options trading desk order update",
        "",
        strategy_line,
        risk_line,
        "TSL = trailing stop loss that moves with the position when activated.",
        "",
        f"Symbol: {symbol}",
        f"Contract: {contract}",
        f"Order Kind: {order_kind}",
        f"Side: {side}",
        f"Quantity: {payload.get('quantity', '-')}",
        f"Broker: {payload.get('broker_name', '-')}",
        f"Broker Order ID: {payload.get('broker_order_id', '-') or '-'}",
        f"Status: {status}",
        f"Order Time (IST): {_format_timestamp(payload.get('created_at'))}",
        f"Trade Date: {_format_date(payload.get('trade_date'))}",
        f"Expiry Date: {_format_date(payload.get('expiry_date'))}",
        f"Price: {_format_money(payload.get('price'))}",
        f"Trigger Price: {_format_money(payload.get('trigger_price'))}",
        f"Entry Premium: {_format_money(payload.get('entry_premium'))}",
        f"Exit Premium: {_format_money(payload.get('exit_premium'))}",
        f"Initial Stop Loss: {_format_money(payload.get('initial_sl'))}",
        f"Current Stop Loss: {_format_money(payload.get('current_sl'))}",
        f"Target Premium: {_format_money(payload.get('target_premium'))}",
        f"Trailing Stop Active: {_format_bool(payload.get('tsl_active'))}",
        f"Capital Invested: {_format_money(invested_amount)}",
        f"Realized P&L: {_format_signed_money(payload.get('realized_pnl'))}",
        f"Open P&L: {_format_signed_money(payload.get('unrealized_pnl'))}",
        f"Position Status: {payload.get('position_status', '-') or '-'}",
        f"Position Opened (IST): {_format_timestamp(payload.get('position_opened_at'))}",
        f"Position Closed (IST): {_format_timestamp(payload.get('position_closed_at'))}",
        f"Exit Reason: {payload.get('exit_reason', '-') or '-'}",
        f"Consensus Reason: {payload.get('consensus_reason', '-') or '-'}",
    ]
    message.set_content("\n".join(text_lines))
    html_rows = [
        ("Symbol", symbol),
        ("Contract", contract),
        ("Order Kind", order_kind),
        ("Side", side),
        ("Quantity", payload.get("quantity", "-")),
        ("Status", status),
        ("Order Time (IST)", _format_timestamp(payload.get("created_at"))),
        ("Trade Date", _format_date(payload.get("trade_date"))),
        ("Expiry Date", _format_date(payload.get("expiry_date"))),
        ("Entry Premium", _format_money(payload.get("entry_premium"))),
        ("Exit Premium", _format_money(payload.get("exit_premium"))),
        ("Initial SL", _format_money(payload.get("initial_sl"))),
        ("Current SL", _format_money(payload.get("current_sl"))),
        ("Target", _format_money(payload.get("target_premium"))),
        ("TSL Active", _format_bool(payload.get("tsl_active"))),
        ("Capital Invested", _format_money(invested_amount)),
        ("Realized P&L", _format_signed_money(payload.get("realized_pnl"))),
        ("Open P&L", _format_signed_money(payload.get("unrealized_pnl"))),
        ("Exit Reason", payload.get("exit_reason", "-") or "-"),
        ("Consensus Reason", payload.get("consensus_reason", "-") or "-"),
    ]
    html_table = "".join(
        f"<tr><th align='left' style='padding:6px 10px;border:1px solid #d7d7d7;background:#f4f4f4'>{label}</th>"
        f"<td style='padding:6px 10px;border:1px solid #d7d7d7'>{value}</td></tr>"
        for label, value in html_rows
    )
    message.add_alternative(
        (
            "<html><body>"
            "<h3>Realtime Options Trading Desk Order Update</h3>"
            f"<p><strong>{symbol}</strong> {contract} | {order_kind} {side} | P&amp;L {pnl_label}</p>"
            f"<div style='margin:12px 0;padding:12px 14px;border:1px solid #d7d7d7;border-radius:10px;background:#f7f9fc'>"
            f"<div style='font-size:16px;font-weight:700'>{strategy_line}</div>"
            f"<div style='margin-top:6px'>{risk_line}</div>"
            "<div style='margin-top:6px;font-size:12px;color:#5b6470'>TSL = trailing stop loss that moves with the position after activation.</div>"
            "</div>"
            f"<table style='border-collapse:collapse;font-family:Arial,sans-serif;font-size:13px'>{html_table}</table>"
            "</body></html>"
        ),
        subtype="html",
    )
    return message


def send_email_message(message: EmailMessage, settings: Settings | None = None) -> bool:
    cfg = settings or get_settings()
    smtp_class = smtplib.SMTP_SSL if cfg.smtp_use_ssl else smtplib.SMTP
    try:
        with smtp_class(cfg.smtp_host, cfg.smtp_port, timeout=cfg.smtp_timeout_seconds) as client:
            client.ehlo()
            if not cfg.smtp_use_ssl and cfg.smtp_use_tls:
                client.starttls()
                client.ehlo()
            if cfg.smtp_username.strip():
                client.login(cfg.smtp_username, cfg.smtp_password)
            client.send_message(message)
        return True
    except Exception:
        logger.exception("SMTP notification failed")
        return False


def send_order_notification(
    payload: Mapping[str, Any],
    settings: Settings | None = None,
) -> bool:
    cfg = settings or get_settings()
    message = build_order_notification_message(payload, settings=cfg)
    if message is None:
        return False
    return send_email_message(message, settings=cfg)
