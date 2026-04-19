from __future__ import annotations

from datetime import date, datetime, time
from email.message import EmailMessage
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from db.models import DailySummary, ExecutionOrder, ExecutionPosition, RawCandle, SignalLog
from utils.calendar_utils import is_trading_day, previous_trading_day
from utils.config import Settings, get_settings
from utils.constants import IST_ZONE
from utils.notifications import send_email_message, smtp_ready
from utils.symbols import instrument_key_filter


def _format_money(value: Any) -> str:
    try:
        parsed = float(value)
        return f"{parsed:.2f}"
    except (TypeError, ValueError):
        return "-"


def _format_signed(value: Any) -> str:
    try:
        parsed = float(value)
        return f"{parsed:+.2f}"
    except (TypeError, ValueError):
        return "-"


def _format_ts(value: datetime | None) -> str:
    if value is None:
        return "-"
    aware = value if value.tzinfo is not None else value.replace(tzinfo=IST_ZONE)
    return aware.astimezone(IST_ZONE).strftime("%Y-%m-%d %H:%M:%S %Z")


def _format_contract(symbol: str, strike: Any, option_type: Any) -> str:
    strike_value = None
    try:
        strike_value = float(strike)
    except (TypeError, ValueError):
        strike_value = None
    if strike_value is None:
        return f"{symbol} {option_type or '-'}".strip()
    strike_text = str(int(round(strike_value))) if abs(strike_value - round(strike_value)) < 1e-6 else f"{strike_value:.2f}"
    return f"{symbol} {strike_text} {option_type or '-'}".strip()


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(
        f"<th style='padding:6px 8px;border:1px solid #d7d7d7;background:#f4f4f4'>{header}</th>"
        for header in headers
    )
    body = "".join(
        "<tr>"
        + "".join(f"<td style='padding:6px 8px;border:1px solid #d7d7d7'>{cell}</td>" for cell in row)
        + "</tr>"
        for row in rows
    )
    return f"<table style='border-collapse:collapse;font-family:Arial,sans-serif;font-size:13px'><tr>{head}</tr>{body}</table>"


def _latest_close_before(db: Session, symbol: str, boundary: datetime) -> float | None:
    value = db.scalar(
        select(RawCandle.close)
        .where(
            and_(
                instrument_key_filter(RawCandle.instrument_key, symbol),
                RawCandle.interval == "1minute",
                RawCandle.ts < boundary,
            )
        )
        .order_by(RawCandle.ts.desc())
        .limit(1)
    )
    return float(value) if value is not None else None


def _day_open_and_last(db: Session, symbol: str, trade_date: date) -> tuple[float | None, float | None]:
    rows = (
        db.execute(
            select(RawCandle)
            .where(
                and_(
                    instrument_key_filter(RawCandle.instrument_key, symbol),
                    RawCandle.interval == "1minute",
                    func.date(RawCandle.ts) == trade_date,
                )
            )
            .order_by(RawCandle.ts.asc())
        )
        .scalars()
        .all()
    )
    if not rows:
        return None, None
    return float(rows[0].open), float(rows[-1].close)


def build_gap_report_message(
    db: Session,
    report_date: date | None = None,
    settings: Settings | None = None,
) -> EmailMessage | None:
    cfg = settings or get_settings()
    if not smtp_ready(cfg):
        return None

    report_date = report_date or datetime.now(IST_ZONE).date()
    if not is_trading_day(report_date):
        return None

    midnight = datetime.combine(report_date, time.min, tzinfo=IST_ZONE)
    symbols = cfg.execution_symbol_list or [key.split("|", 1)[-1] for key in cfg.instrument_keys]
    rows: list[list[str]] = []
    text_lines = [
        f"Gap report for {report_date.isoformat()}",
        "",
    ]
    for symbol in symbols:
        prev_close = _latest_close_before(db, symbol, midnight)
        open_price, latest_price = _day_open_and_last(db, symbol, report_date)
        gap_points = ((open_price or 0.0) - prev_close) if prev_close is not None and open_price is not None else None
        gap_pct = ((gap_points / prev_close) * 100.0) if prev_close and gap_points is not None else None
        rows.append(
            [
                symbol,
                _format_money(prev_close),
                _format_money(open_price),
                _format_signed(gap_points),
                f"{gap_pct:+.2f}%" if gap_pct is not None else "-",
                _format_money(latest_price),
            ]
        )
        text_lines.append(
            f"{symbol}: prev_close={_format_money(prev_close)} open={_format_money(open_price)} gap={_format_signed(gap_points)} latest={_format_money(latest_price)}"
        )

    message = EmailMessage()
    message["Subject"] = f"[{str(cfg.env).upper()}] Market Gap Report | {report_date.isoformat()}"
    message["From"] = cfg.smtp_from_email
    message["To"] = ", ".join(cfg.smtp_recipients)
    message.set_content("\n".join(text_lines))
    message.add_alternative(
        (
            "<html><body>"
            f"<h3>Market Gap Report - {report_date.isoformat()}</h3>"
            + _html_table(
                ["Symbol", "Previous Close", "Open", "Gap Points", "Gap %", "Latest"],
                rows,
            )
            + "</body></html>"
        ),
        subtype="html",
    )
    return message


def send_gap_report(
    db: Session,
    report_date: date | None = None,
    settings: Settings | None = None,
) -> bool:
    cfg = settings or get_settings()
    message = build_gap_report_message(db, report_date=report_date, settings=cfg)
    if message is None:
        return False
    return send_email_message(message, settings=cfg)


def build_daily_summary_message(
    db: Session,
    trade_date: date | None = None,
    settings: Settings | None = None,
) -> EmailMessage | None:
    cfg = settings or get_settings()
    if not smtp_ready(cfg):
        return None

    trade_date = trade_date or datetime.now(IST_ZONE).date()
    summary = db.get(DailySummary, trade_date)
    if summary is None:
        summary = DailySummary(date=trade_date)

    positions = (
        db.execute(
            select(ExecutionPosition)
            .where(ExecutionPosition.trade_date == trade_date)
            .order_by(ExecutionPosition.opened_at.asc())
        )
        .scalars()
        .all()
    )
    orders_count = db.scalar(select(func.count(ExecutionOrder.id)).where(ExecutionOrder.trade_date == trade_date)) or 0
    signal_count = db.scalar(select(func.count(SignalLog.id)).where(SignalLog.trade_date == trade_date)) or 0
    orders = (
        db.execute(
            select(ExecutionOrder)
            .where(ExecutionOrder.trade_date == trade_date)
            .order_by(ExecutionOrder.created_at.asc())
        )
        .scalars()
        .all()
    )
    signal_rows = (
        db.execute(
            select(SignalLog)
            .where(SignalLog.trade_date == trade_date)
            .order_by(SignalLog.timestamp.asc())
            .limit(80)
        )
        .scalars()
        .all()
    )

    trade_rows: list[list[str]] = []
    order_rows: list[list[str]] = []
    signal_table_rows: list[list[str]] = []
    text_lines = [
        f"Daily trading summary for {trade_date.isoformat()}",
        "",
        f"Total Trades: {int(summary.total_trades or 0)}",
        f"Wins: {int(summary.winning_trades or 0)}",
        f"Losses: {int(summary.losing_trades or 0)}",
        f"Win Rate: {float(summary.win_rate or 0.0):.2f}%",
        f"Total P&L: {_format_signed(summary.total_pnl or 0.0)}",
        f"Orders Logged: {int(orders_count)}",
        f"Signals Logged: {int(signal_count)}",
        "",
    ]
    for row in positions:
        contract = _format_contract(row.symbol, row.strike, row.option_type)
        invested = float(row.entry_premium or row.entry_price or 0.0) * int(row.quantity or 0)
        trade_rows.append(
            [
                _format_ts(row.opened_at),
                contract,
                str(int(row.quantity or 0)),
                _format_money(row.entry_premium or row.entry_price),
                _format_money(row.exit_premium or row.current_premium or row.current_price),
                _format_money(row.current_sl or row.initial_sl or row.stop_loss),
                "Yes" if row.tsl_active else "No",
                _format_money(row.target_premium or row.take_profit),
                _format_money(invested),
                _format_signed(row.realized_pnl or row.unrealized_pnl or row.pnl_value),
                str(row.exit_reason or row.status),
            ]
        )
        text_lines.append(
            f"{contract} qty={int(row.quantity or 0)} entry={_format_money(row.entry_premium or row.entry_price)} exit={_format_money(row.exit_premium or row.current_premium or row.current_price)} pnl={_format_signed(row.realized_pnl or row.unrealized_pnl or row.pnl_value)} tsl={'Yes' if row.tsl_active else 'No'} invested={_format_money(invested)} reason={row.exit_reason or row.status}"
        )
    for row in orders:
        order_rows.append(
            [
                _format_ts(row.created_at),
                row.symbol,
                _format_contract(row.symbol, row.strike_price, row.option_type),
                row.order_kind,
                row.side,
                str(int(row.quantity or 0)),
                _format_money(row.price),
                _format_money(row.current_sl),
                _format_money(row.target_premium),
                _format_signed(row.realized_pnl or row.unrealized_pnl),
                str(row.status),
            ]
        )
    for row in signal_rows:
        signal_table_rows.append(
            [
                _format_ts(row.timestamp),
                row.symbol,
                row.ml_signal,
                row.pine_signal,
                _format_money(row.ai_score),
                _format_money(row.combined_score),
                row.consensus,
                row.skip_reason or ("TRADE PLACED" if row.trade_placed else "-"),
            ]
        )

    message = EmailMessage()
    message["Subject"] = f"[{str(cfg.env).upper()}] Daily Trading Summary | {trade_date.isoformat()}"
    message["From"] = cfg.smtp_from_email
    message["To"] = ", ".join(cfg.smtp_recipients)
    message.set_content("\n".join(text_lines))
    message.add_alternative(
        (
            "<html><body>"
            f"<h3>Daily Trading Summary - {trade_date.isoformat()}</h3>"
            f"<p>Total trades: {int(summary.total_trades or 0)} | Wins: {int(summary.winning_trades or 0)} | Losses: {int(summary.losing_trades or 0)} | Win rate: {float(summary.win_rate or 0.0):.2f}% | P&amp;L: {_format_signed(summary.total_pnl or 0.0)}</p>"
            "<h4>Trades</h4>"
            + _html_table(
                ["Opened", "Contract", "Qty", "Entry", "Exit", "SL", "TSL", "Target", "Invested", "P&L", "Status"],
                trade_rows or [["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]],
            )
            + "<h4 style='margin-top:18px'>Orders</h4>"
            + _html_table(
                ["Time", "Symbol", "Contract", "Kind", "Side", "Qty", "Price", "SL", "Target", "P&L", "Status"],
                order_rows or [["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]],
            )
            + "<h4 style='margin-top:18px'>Signal Logs</h4>"
            + _html_table(
                ["Time", "Symbol", "ML", "Pine", "AI", "Combined", "Consensus", "Reason"],
                signal_table_rows or [["-", "-", "-", "-", "-", "-", "-", "-"]],
            )
            + f"<p>Orders logged: {int(orders_count)} | Signals logged: {int(signal_count)} | TSL = trailing stop loss.</p>"
            + "</body></html>"
        ),
        subtype="html",
    )
    return message


def send_daily_summary_report(
    db: Session,
    trade_date: date | None = None,
    settings: Settings | None = None,
) -> bool:
    cfg = settings or get_settings()
    message = build_daily_summary_message(db, trade_date=trade_date, settings=cfg)
    if message is None:
        return False
    return send_email_message(message, settings=cfg)
