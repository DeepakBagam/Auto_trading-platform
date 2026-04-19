try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

import argparse
import json
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import select

from db.connection import SessionLocal
from db.models import DailySummary, ExecutionOrder, ExecutionPosition, SignalLog
from utils.constants import IST_ZONE


def _serialize_dt(value):
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one trading day's logs to JSON.")
    parser.add_argument("--date", default=datetime.now(IST_ZONE).date().isoformat(), help="Trade date in YYYY-MM-DD")
    parser.add_argument(
        "--output",
        default="logs/daily_exports",
        help="Directory where the JSON file should be written",
    )
    args = parser.parse_args()
    trade_date = date.fromisoformat(str(args.date))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        summary = db.get(DailySummary, trade_date)
        positions = db.execute(
            select(ExecutionPosition)
            .where(ExecutionPosition.trade_date == trade_date)
            .order_by(ExecutionPosition.opened_at.asc())
        ).scalars().all()
        orders = db.execute(
            select(ExecutionOrder)
            .where(ExecutionOrder.trade_date == trade_date)
            .order_by(ExecutionOrder.created_at.asc())
        ).scalars().all()
        signals = db.execute(
            select(SignalLog)
            .where(SignalLog.trade_date == trade_date)
            .order_by(SignalLog.timestamp.asc())
        ).scalars().all()

        payload = {
            "trade_date": trade_date.isoformat(),
            "exported_at": datetime.now(IST_ZONE).isoformat(),
            "daily_summary": None
            if summary is None
            else {
                "date": summary.date.isoformat(),
                "total_trades": int(summary.total_trades or 0),
                "winning_trades": int(summary.winning_trades or 0),
                "losing_trades": int(summary.losing_trades or 0),
                "total_pnl": float(summary.total_pnl or 0.0),
                "max_profit_trade": float(summary.max_profit_trade or 0.0),
                "max_loss_trade": float(summary.max_loss_trade or 0.0),
                "win_rate": float(summary.win_rate or 0.0),
                "is_green": bool(summary.is_green),
                "updated_at": _serialize_dt(summary.updated_at),
            },
            "positions": [
                {
                    "id": row.id,
                    "symbol": row.symbol,
                    "interval": row.interval,
                    "strategy_name": row.strategy_name,
                    "option_type": row.option_type,
                    "expiry_date": _serialize_dt(row.expiry_date),
                    "strike": float(row.strike or 0.0),
                    "quantity": int(row.quantity or 0),
                    "status": row.status,
                    "entry_premium": row.entry_premium,
                    "exit_premium": row.exit_premium,
                    "current_sl": row.current_sl,
                    "target_premium": row.target_premium,
                    "tsl_active": bool(row.tsl_active),
                    "realized_pnl": row.realized_pnl,
                    "unrealized_pnl": row.unrealized_pnl,
                    "ai_score": row.ai_score,
                    "pine_signal": row.pine_signal,
                    "consensus_reason": row.consensus_reason,
                    "opened_at": _serialize_dt(row.opened_at),
                    "closed_at": _serialize_dt(row.closed_at),
                    "exit_reason": row.exit_reason,
                    "metadata_json": row.metadata_json or {},
                }
                for row in positions
            ],
            "orders": [
                {
                    "id": row.id,
                    "position_id": row.position_id,
                    "symbol": row.symbol,
                    "strike_price": row.strike_price,
                    "option_type": row.option_type,
                    "order_kind": row.order_kind,
                    "side": row.side,
                    "quantity": row.quantity,
                    "price": row.price,
                    "trigger_price": row.trigger_price,
                    "entry_premium": row.entry_premium,
                    "current_sl": row.current_sl,
                    "target_premium": row.target_premium,
                    "tsl_active": bool(row.tsl_active),
                    "exit_premium": row.exit_premium,
                    "realized_pnl": row.realized_pnl,
                    "unrealized_pnl": row.unrealized_pnl,
                    "status": row.status,
                    "broker_name": row.broker_name,
                    "broker_order_id": row.broker_order_id,
                    "created_at": _serialize_dt(row.created_at),
                    "response_json": row.response_json or {},
                }
                for row in orders
            ],
            "signal_log": [
                {
                    "id": row.id,
                    "timestamp": _serialize_dt(row.timestamp),
                    "symbol": row.symbol,
                    "interval": row.interval,
                    "ml_signal": row.ml_signal,
                    "ml_confidence": row.ml_confidence,
                    "ml_expected_move": row.ml_expected_move,
                    "pine_signal": row.pine_signal,
                    "pine_age_seconds": row.pine_age_seconds,
                    "ai_score": row.ai_score,
                    "news_sentiment": row.news_sentiment,
                    "combined_score": row.combined_score,
                    "consensus": row.consensus,
                    "skip_reason": row.skip_reason,
                    "trade_placed": bool(row.trade_placed),
                    "details": row.details or {},
                }
                for row in signals
            ],
        }

        output_path = out_dir / f"daily_log_{trade_date.isoformat()}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(output_path)
    finally:
        db.close()


if __name__ == "__main__":
    main()
