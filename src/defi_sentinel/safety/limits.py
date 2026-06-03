"""Spending limits — budget enforcement for agent actions."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SpendingRecord:
    """A single spending entry."""
    amount_usd: float
    timestamp: float
    action: str
    chain: str


class SpendingLimits:
    """Enforce spending limits on agent actions.

    Limits:
    - Per-transaction max
    - Per-hour max
    - Per-day max
    - Total budget

    Usage:
        limits = SpendingLimits(max_usd=100, max_per_tx=10, max_per_hour=25)
        if limits.check(amount_usd=5.0):
            limits.record(5.0, action="swap", chain="ethereum")
        else:
            print("Blocked by spending limit")
    """

    def __init__(
        self,
        max_usd: float = 100.0,
        max_per_tx: float = 50.0,
        max_per_hour: float = 25.0,
        max_per_day: float = 75.0,
    ):
        self.max_usd = max_usd
        self.max_per_tx = max_per_tx
        self.max_per_hour = max_per_hour
        self.max_per_day = max_per_day
        self._records: list[SpendingRecord] = []

    def check(self, amount_usd: float) -> bool:
        """Check if a spend is within all limits.

        Returns True if allowed, False if any limit exceeded.
        """
        now = time.time()

        # Per-tx limit
        if amount_usd > self.max_per_tx:
            logger.warning(
                "Per-tx limit exceeded: $%.2f > $%.2f",
                amount_usd, self.max_per_tx,
            )
            return False

        # Total budget
        total = sum(r.amount_usd for r in self._records)
        if total + amount_usd > self.max_usd:
            logger.warning(
                "Total budget exceeded: $%.2f + $%.2f > $%.2f",
                total, amount_usd, self.max_usd,
            )
            return False

        # Per-hour limit
        hour_ago = now - 3600
        hour_total = sum(r.amount_usd for r in self._records if r.timestamp > hour_ago)
        if hour_total + amount_usd > self.max_per_hour:
            logger.warning(
                "Hourly limit exceeded: $%.2f + $%.2f > $%.2f",
                hour_total, amount_usd, self.max_per_hour,
            )
            return False

        # Per-day limit
        day_ago = now - 86400
        day_total = sum(r.amount_usd for r in self._records if r.timestamp > day_ago)
        if day_total + amount_usd > self.max_per_day:
            logger.warning(
                "Daily limit exceeded: $%.2f + $%.2f > $%.2f",
                day_total, amount_usd, self.max_per_day,
            )
            return False

        return True

    def record(self, amount_usd: float, action: str = "", chain: str = "") -> None:
        """Record a spending event."""
        self._records.append(SpendingRecord(
            amount_usd=amount_usd,
            timestamp=time.time(),
            action=action,
            chain=chain,
        ))
        total = sum(r.amount_usd for r in self._records)
        logger.info("Recorded spend: $%.2f (%s on %s). Total: $%.2f/$%.2f",
                     amount_usd, action, chain, total, self.max_usd)

    def summary(self) -> dict[str, float]:
        """Return spending summary."""
        now = time.time()
        total = sum(r.amount_usd for r in self._records)
        hour = sum(r.amount_usd for r in self._records if r.timestamp > now - 3600)
        day = sum(r.amount_usd for r in self._records if r.timestamp > now - 86400)

        return {
            "total_usd": round(total, 2),
            "hour_usd": round(hour, 2),
            "day_usd": round(day, 2),
            "budget_remaining": round(self.max_usd - total, 2),
            "records": len(self._records),
        }
