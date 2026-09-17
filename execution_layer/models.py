"""Modelos de datos de la capa de ejecución."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from signal_layer.base_provider import SignalDirection


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass
class Position:
    symbol: str
    direction: SignalDirection
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_order_id: str
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: PositionStatus = PositionStatus.OPEN
    exit_price: float | None = None
    closed_at: datetime | None = None
    close_reason: str | None = None
    # Trailing stop
    atr_at_entry: float | None = None
    trail_atr_multiplier: float | None = None
    best_price_since_entry: float | None = None

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        sign = 1 if self.direction == SignalDirection.LONG else -1
        return sign * (self.exit_price - self.entry_price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        if self.exit_price is None or self.entry_price == 0:
            return 0.0
        sign = 1 if self.direction == SignalDirection.LONG else -1
        return sign * (self.exit_price - self.entry_price) / self.entry_price
