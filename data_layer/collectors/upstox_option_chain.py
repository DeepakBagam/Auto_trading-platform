from __future__ import annotations

from datetime import date, datetime
from typing import Any

import requests
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from db.models import DataFreshness, OptionQuote
from utils.config import get_settings
from utils.constants import IST_ZONE
from utils.logger import get_logger

logger = get_logger(__name__)


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
        return out if out == out else None
    except (TypeError, ValueError):
        return None


class UpstoxOptionChainCollector:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.upstox_base_url.rstrip("/")
        access_token = getattr(self.settings, "market_data_access_token", "") or getattr(
            self.settings, "upstox_access_token", ""
        )
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def fetch_option_contracts(self, underlying_key: str, expiry_date: date | None = None) -> list[dict[str, Any]]:
        params: dict[str, str] = {"instrument_key": underlying_key}
        if expiry_date is not None:
            params["expiry_date"] = expiry_date.isoformat()
        response = requests.get(
            f"{self.base_url}/v2/option/contract",
            params=params,
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return list(payload.get("data") or [])

    def list_expiries(self, underlying_key: str, *, max_items: int = 6) -> list[date]:
        rows = self.fetch_option_contracts(underlying_key)
        expiries = sorted(
            {
                date.fromisoformat(str(row.get("expiry")))
                for row in rows
                if row.get("expiry")
            }
        )
        return expiries[: max(1, int(max_items))]

    def fetch_option_chain(self, underlying_key: str, expiry_date: date) -> list[dict[str, Any]]:
        response = requests.get(
            f"{self.base_url}/v2/option/chain",
            params={"instrument_key": underlying_key, "expiry_date": expiry_date.isoformat()},
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return list(payload.get("data") or [])

    def persist_option_chain(
        self,
        db: Session,
        *,
        underlying_key: str,
        underlying_symbol: str,
        expiry_date: date,
        chain_rows: list[dict[str, Any]],
        fetched_at: datetime | None = None,
    ) -> dict[str, int]:
        snapshot_ts = (fetched_at or datetime.now(IST_ZONE)).astimezone(IST_ZONE).replace(second=0, microsecond=0)
        inserted = 0
        updated = 0

        for row in chain_rows:
            strike = _to_float(row.get("strike_price"))
            pcr = _to_float(row.get("pcr"))
            spot = _to_float(row.get("underlying_spot_price"))
            for option_type, node_key in (("CE", "call_options"), ("PE", "put_options")):
                node = row.get(node_key) or {}
                market = node.get("market_data") or {}
                greeks = node.get("option_greeks") or {}
                instrument_key = str(node.get("instrument_key") or "").strip()
                if strike is None or not instrument_key:
                    continue

                existing = db.scalar(
                    select(OptionQuote).where(
                        and_(
                            OptionQuote.underlying_symbol == underlying_symbol,
                            OptionQuote.expiry_date == expiry_date,
                            OptionQuote.strike == strike,
                            OptionQuote.option_type == option_type,
                            OptionQuote.ts == snapshot_ts,
                        )
                    )
                )
                if existing is None:
                    existing = OptionQuote(
                        instrument_key=instrument_key,
                        underlying_key=underlying_key,
                        underlying_symbol=underlying_symbol,
                        expiry_date=expiry_date,
                        strike=strike,
                        option_type=option_type,
                        ts=snapshot_ts,
                        ltp=float(_to_float(market.get("ltp")) or 0.0),
                        bid=_to_float(market.get("bid_price")),
                        ask=_to_float(market.get("ask_price")),
                        volume=float(_to_float(market.get("volume")) or 0.0),
                        oi=_to_float(market.get("oi")),
                        close_price=_to_float(market.get("close_price")),
                        bid_qty=_to_float(market.get("bid_qty")),
                        ask_qty=_to_float(market.get("ask_qty")),
                        prev_oi=_to_float(market.get("prev_oi")),
                        iv=_to_float(greeks.get("iv")),
                        delta=_to_float(greeks.get("delta")),
                        gamma=_to_float(greeks.get("gamma")),
                        theta=_to_float(greeks.get("theta")),
                        vega=_to_float(greeks.get("vega")),
                        pop=_to_float(greeks.get("pop")),
                        pcr=pcr,
                        underlying_spot_price=spot,
                        source="upstox_option_chain",
                    )
                    db.add(existing)
                    inserted += 1
                    continue

                existing.instrument_key = instrument_key
                existing.underlying_key = underlying_key
                existing.ltp = float(_to_float(market.get("ltp")) or 0.0)
                existing.bid = _to_float(market.get("bid_price"))
                existing.ask = _to_float(market.get("ask_price"))
                existing.volume = float(_to_float(market.get("volume")) or 0.0)
                existing.oi = _to_float(market.get("oi"))
                existing.close_price = _to_float(market.get("close_price"))
                existing.bid_qty = _to_float(market.get("bid_qty"))
                existing.ask_qty = _to_float(market.get("ask_qty"))
                existing.prev_oi = _to_float(market.get("prev_oi"))
                existing.iv = _to_float(greeks.get("iv"))
                existing.delta = _to_float(greeks.get("delta"))
                existing.gamma = _to_float(greeks.get("gamma"))
                existing.theta = _to_float(greeks.get("theta"))
                existing.vega = _to_float(greeks.get("vega"))
                existing.pop = _to_float(greeks.get("pop"))
                existing.pcr = pcr
                existing.underlying_spot_price = spot
                existing.source = "upstox_option_chain"
                updated += 1

        self._mark_freshness(
            db,
            source_name=f"upstox_option_chain:{underlying_symbol}",
            status="ok",
            details={
                "underlying_key": underlying_key,
                "underlying_symbol": underlying_symbol,
                "expiry_date": expiry_date.isoformat(),
                "rows": len(chain_rows),
                "inserted": inserted,
                "updated": updated,
                "snapshot_ts": snapshot_ts.isoformat(),
            },
        )
        db.commit()
        return {"inserted": inserted, "updated": updated, "rows": len(chain_rows)}

    def sync_option_chain(
        self,
        db: Session,
        *,
        underlying_key: str,
        underlying_symbol: str,
        expiry_date: date,
    ) -> dict[str, int]:
        rows = self.fetch_option_chain(underlying_key, expiry_date)
        return self.persist_option_chain(
            db,
            underlying_key=underlying_key,
            underlying_symbol=underlying_symbol,
            expiry_date=expiry_date,
            chain_rows=rows,
        )

    def _mark_freshness(self, db: Session, source_name: str, status: str, details: dict[str, Any]) -> None:
        row = db.scalar(select(DataFreshness).where(DataFreshness.source_name == source_name))
        if row is None:
            row = DataFreshness(source_name=source_name, last_success_at=datetime.now(IST_ZONE))
            db.add(row)
        row.last_success_at = datetime.now(IST_ZONE)
        row.status = status
        row.details = details
