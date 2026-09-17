"""Estrategia de ruptura Donchian validada con walk-forward en backtests
largos (Quattro: +1,107% / +83% APY en BTC 4H en el paper más reciente y
DonchianBreakout +174.6% en la comparativa freqtrade).

Reglas:
- `entry_lookback` velas (p.ej. 20): el precio rompe el máximo (LONG) o el
  mínimo (SHORT) de esas N velas ANTERIORES (desplazadas 1 vela para evitar
  lookahead bias).
- Filtro de tendencia: solo LONG si el precio está por encima de la EMA larga
  `ema_trend` y esa EMA está subiendo (comparando `ema_rise_bars` velas);
  simétrico para SHORT.
- Se usa en mercados en TENDENCIA (ver signal_layer/regime.py); en rango el
  bot usa el scoring técnico tradicional o se mantiene en efectivo.

Como todos los proveedores, solo aporta dirección y confianza: el stop loss,
take profit, trailing y el dimensionado los calcula siempre RiskManager.
"""
from __future__ import annotations

from signal_layer.base_provider import BaseSignalProvider, Signal, SignalDirection
from signal_layer.regime import classify, MarketRegime
from signal_layer.technical_provider import ema


class DonchianProvider(BaseSignalProvider):
    name = "donchian_breakout"

    def __init__(self, config: dict):
        dc = config.get("donchian", {})
        self.enabled = dc.get("enabled", True)
        self.entry_lookback = dc.get("entry_lookback", 20)
        self.ema_trend = dc.get("ema_trend", 200)
        self.ema_rise_bars = dc.get("ema_rise_bars", 20)
        self.confidence_floor = dc.get("confidence_floor", 0.6)
        self.min_confidence = config["signal"]["min_confidence"]
        self.primary_tf = config["timeframes"]["primary"]
        self.confirmation_tf = config["timeframes"]["confirmation"]
        self.adx_threshold = config["regime"]["adx_threshold"]

    async def generate_signal(self, market_data) -> Signal:
        df = market_data.candles(self.primary_tf)
        price = float(df["close"].iloc[-1])
        direction, confidence, reasons = self._decide(df)
        regime = classify(df, adx_threshold=self.adx_threshold)

        if confidence < self.min_confidence:
            direction = SignalDirection.NONE

        metadata = {
            "atr": float(self._atr(df)),
            "regime": regime.regime.value,
            "adx": regime.adx,
            "high_volatility": regime.high_volatility,
            "momentum_pct": regime.momentum_pct,
            "reasons": reasons,
            "strategy": self.name,
        }
        return Signal(
            symbol=market_data.symbol,
            direction=direction,
            confidence=confidence,
            price=price,
            provider=self.name,
            metadata=metadata,
        )

    def _decide(self, df) -> tuple[SignalDirection, float, list[str]]:
        from signal_layer.technical_provider import atr

        reasons: list[str] = []
        if not self.enabled or len(df) < self.ema_trend + 2:
            return SignalDirection.NONE, 0.0, ["donchian deshabilitado o sin datos suficientes"]

        close = df["close"]
        ema_long = ema(close, self.ema_trend)

        # Extremos de las N velas anteriores (excluye la vela actual)
        prior = df.iloc[-self.entry_lookback - 1:-1]
        upper = float(prior["high"].max())
        lower = float(prior["low"].min())
        price = float(close.iloc[-1])

        ema_now = float(ema_long.iloc[-1])
        ema_prev = float(ema_long.iloc[-1 - self.ema_rise_bars])
        ema_rising = ema_now > ema_prev
        price_above_ema = price > ema_now
        price_below_ema = price < ema_now

        atr_val = float(atr(df).iloc[-1]) or price * 0.01
        atr_pct = atr_val / price if price else 1.0

        long_breakout = price > upper and price_above_ema and ema_rising
        short_breakout = price < lower and price_below_ema and not ema_rising

        if long_breakout:
            strength = min(1.0, (price - upper) / atr_val)
            confidence = round(0.6 + 0.4 * strength, 2)
            reasons.append(f"ruptura alcista de Donchian {self.entry_lookback}v ({price:.2f} > {upper:.2f})")
            reasons.append("EMA larga subiendo y precio por encima")
            return SignalDirection.LONG, confidence, reasons

        if short_breakout:
            strength = min(1.0, (lower - price) / atr_val)
            confidence = round(0.6 + 0.4 * strength, 2)
            reasons.append(f"ruptura bajista de Donchian {self.entry_lookback}v ({price:.2f} < {lower:.2f})")
            reasons.append("EMA larga bajando y precio por debajo")
            return SignalDirection.SHORT, confidence, reasons

        reasons.append(f"sin ruptura (rango {lower:.2f}-{upper:.2f}, precio {price:.2f})")
        return SignalDirection.NONE, 0.0, reasons

    def _atr(self, df) -> float:
        from signal_layer.technical_provider import atr
        return float(atr(df).iloc[-1])