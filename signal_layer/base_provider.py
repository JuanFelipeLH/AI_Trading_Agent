"""Definición base del sistema de señales: el contrato que debe cumplir
cualquier proveedor de señales (análisis técnico, API externa, modelo de ML,
sentiment analysis, ...) para poder conectarse al bot sin tocar el resto de
capas (risk_layer / execution_layer sólo dependen de `Signal`)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from data_layer.models import MarketData


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class Signal:
    symbol: str
    direction: SignalDirection
    confidence: float                # 0.0 - 1.0
    price: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provider: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.direction != SignalDirection.NONE


class BaseSignalProvider(ABC):
    """Contrato para cualquier generador de señales.

    Para añadir un nuevo proveedor (API externa, modelo de ML, sentiment
    analysis, etc.) basta con heredar de esta clase e implementar
    `generate_signal`. El resto del sistema no necesita cambios: main.py
    sólo conoce la interfaz `BaseSignalProvider`, nunca la implementación
    concreta.
    """

    name: str = "base"

    @abstractmethod
    async def generate_signal(self, market_data: MarketData) -> Signal:
        """Analiza `market_data` y devuelve una Signal (puede ser NONE)."""
        raise NotImplementedError

    def on_trade_closed(self, symbol: str, pnl: float, metadata: dict[str, Any] | None = None) -> None:
        """Hook opcional: se llama cuando se cierra una posición originada por
        este proveedor. Por defecto no hace nada; un proveedor con estado
        (p.ej. uno que le da contexto histórico a un LLM) puede sobreescribirlo
        para registrar el resultado. Usa tipos primitivos (no `Position`) para
        no acoplar signal_layer a execution_layer."""
        return None
