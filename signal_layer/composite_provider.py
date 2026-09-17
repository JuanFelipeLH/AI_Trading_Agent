"""Proveedor de señales adaptativo: en lugar de esperar a que UNA condición
fija se cumpla, evalúa el mercado en cada ciclo y elige la estrategia según
el régimen detectado (signal_layer/regime.py):

- mercado en TENDENCIA  -> Donchian breakout (rupturas, fiable en tendencias)
- mercado en RANGO      -> scoring técnico clásico (RSI/MACD/EMA/S-R)

Siempre devuelve un Signal con metadata enriquecida (`regime`, `adx`,
`volatility`, `size_multiplier`, `strategy`) para que RiskManager pueda
dimensionar según la calidad de la señal y el riesgo del momento.

Como todos los proveedores, solo aporta dirección y confianza; el stop loss,
take profit, trailing y tamaño los calcula RiskManager.
"""
from __future__ import annotations

from data_layer.models import MarketData
from signal_layer.base_provider import BaseSignalProvider, Signal, SignalDirection
from signal_layer.donchian_provider import DonchianProvider
from signal_layer.regime import classify
from signal_layer.technical_provider import TechnicalAnalysisProvider


class CompositeSignalProvider(BaseSignalProvider):
    name = "composite_analysis"

    def __init__(self, config: dict):
        self.technical = TechnicalAnalysisProvider(config)
        self.donchian = DonchianProvider(config)
        self.min_confidence = config["signal"]["min_confidence"]
        self.adx_threshold = config["regime"]["adx_threshold"]
        self.confidence_cap = config["risk"].get("confidence_cap", 1.5)
        self.primary_tf = config["timeframes"]["primary"]

    async def generate_signal(self, market_data: MarketData) -> Signal:
        df = market_data.candles(self.primary_tf)
        regime = classify(df, adx_threshold=self.adx_threshold)

        signal_t = await self.technical.generate_signal(market_data)
        signal_d = await self.donchian.generate_signal(market_data)

        if regime.trending:
            primary, secondary = signal_d, signal_t
            strategy = "donchian_breakout"
        else:
            primary, secondary = signal_t, signal_d
            strategy = "technical_scoring"

        chosen = primary if primary.is_actionable else secondary
        if not chosen.is_actionable:
            chosen = self._neutral(market_data, regime, signal_d, signal_t)

        chosen.metadata["strategy"] = strategy
        chosen.metadata["regime"] = regime.regime.value
        chosen.metadata["adx"] = regime.adx
        chosen.metadata["atr_pct"] = regime.atr_pct
        chosen.metadata["high_volatility"] = regime.high_volatility
        chosen.metadata["momentum_pct"] = regime.momentum_pct
        chosen.metadata["size_multiplier"] = self._size_multiplier(chosen, regime)
        chosen.metadata["composite_reasons"] = (
            chosen.metadata.get("reasons", [])
            + [f"régimen: {regime.regime.value} (ADX {regime.adx:.1f}), estrategia: {strategy}"]
        )
        return chosen

    def _neutral(self, market_data: MarketData, regime, signal_d: Signal, signal_t: Signal) -> Signal:
        return Signal(
            symbol=market_data.symbol,
            direction=SignalDirection.NONE,
            confidence=0.0,
            price=float(market_data.candles(self.primary_tf)["close"].iloc[-1]),
            provider=self.name,
            metadata={
                "reasons": [
                    f"sin señal accionable en régimen {regime.regime.value} "
                    "(donchian sin ruptura y scoring técnico con banda neutra)",
                ],
            },
        )

    def _size_multiplier(self, signal: Signal, regime) -> float:
        """Escala el tamaño de la posición según confianza y régimen:
        en tendencia más tamaño, en rango menos (mean reversion es más ruidoso).
        Siempre dentro de [0.25, confidence_cap]."""
        base = signal.confidence / self.min_confidence if self.min_confidence else 1.0
        base = min(base, self.confidence_cap)
        if not regime.trending:
            base *= 0.5
        return round(max(0.25, base), 3)