"""Modelos de datos compartidos por la capa de datos."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd


@dataclass
class MarketData:
    """Snapshot de mercado para un símbolo, con velas en uno o más timeframes."""

    symbol: str
    ohlcv: dict[str, pd.DataFrame]   # {"1h": df, "15m": df, ...}
    primary_timeframe: str
    order_book: dict | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def candles(self, timeframe: str) -> pd.DataFrame:
        return self.ohlcv[timeframe]

    @property
    def last_price(self) -> float:
        return float(self.ohlcv[self.primary_timeframe]["close"].iloc[-1])
