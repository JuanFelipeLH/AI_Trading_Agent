"""Proveedor de señales basado en análisis técnico clásico: tendencia (EMA),
momentum (RSI + MACD) y niveles de soporte/resistencia, confirmado con un
timeframe menor.

Sirve como implementación de referencia de BaseSignalProvider. Para añadir
otro proveedor (p.ej. una API de sentiment o un modelo de ML) sólo hay que
crear una nueva clase que herede de BaseSignalProvider e implemente
`generate_signal` siguiendo el mismo patrón; no requiere cambios en
risk_layer ni execution_layer.
"""
from __future__ import annotations

import pandas as pd

from data_layer.models import MarketData
from signal_layer.base_provider import BaseSignalProvider, Signal, SignalDirection


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-12)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


class TechnicalAnalysisProvider(BaseSignalProvider):
    name = "technical_analysis"

    def __init__(self, config: dict):
        cfg = config["signal"]
        self.rsi_period = cfg["rsi_period"]
        self.rsi_oversold = cfg["rsi_oversold"]
        self.rsi_overbought = cfg["rsi_overbought"]
        self.macd_fast = cfg["macd_fast"]
        self.macd_slow = cfg["macd_slow"]
        self.macd_signal = cfg["macd_signal"]
        self.ema_fast = cfg["ema_fast"]
        self.ema_slow = cfg["ema_slow"]
        self.atr_period = cfg["atr_period"]
        self.sr_lookback = cfg["support_resistance_lookback"]
        self.min_confidence = cfg["min_confidence"]
        self.neutral_band = cfg.get("neutral_band", 0.10)
        self.primary_tf = config["timeframes"]["primary"]
        self.confirmation_tf = config["timeframes"]["confirmation"]

    async def generate_signal(self, market_data: MarketData) -> Signal:
        df_primary = market_data.candles(self.primary_tf)
        df_confirm = market_data.candles(self.confirmation_tf)

        primary = self.analyze_timeframe(df_primary)
        confirm = self.analyze_timeframe(df_confirm)

        price = float(df_primary["close"].iloc[-1])
        direction, confidence, reasons = self._decide(primary, confirm)

        if confidence < self.min_confidence:
            direction = SignalDirection.NONE

        metadata = {
            "atr": float(primary["atr"]),
            "trend": primary["trend"],
            "rsi_primary": float(primary["rsi"]),
            "rsi_confirmation": float(confirm["rsi"]),
            "macd_hist_primary": float(primary["macd_hist"]),
            "macd_hist_confirmation": float(confirm["macd_hist"]),
            "support": float(primary["support"]),
            "resistance": float(primary["resistance"]),
            "reasons": reasons,
        }

        return Signal(
            symbol=market_data.symbol,
            direction=direction,
            confidence=confidence,
            price=price,
            provider=self.name,
            metadata=metadata,
        )

    def analyze_timeframe(self, df: pd.DataFrame) -> dict:
        """Calcula tendencia/momentum/S-R para un timeframe. Público para que
        otros proveedores (p.ej. VeniceSignalProvider) puedan reusar estos
        indicadores como contexto sin duplicar el cálculo."""
        close = df["close"]
        ema_fast = ema(close, self.ema_fast)
        ema_slow = ema(close, self.ema_slow)
        rsi_series = rsi(close, self.rsi_period)
        _, _, hist = macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
        atr_series = atr(df, self.atr_period)

        window = df.tail(self.sr_lookback)

        return {
            "close": close.iloc[-1],
            "rsi": rsi_series.iloc[-1],
            "macd_hist": hist.iloc[-1],
            "macd_hist_prev": hist.iloc[-2],
            "trend": "bullish" if ema_fast.iloc[-1] > ema_slow.iloc[-1] else "bearish",
            "support": window["low"].min(),
            "resistance": window["high"].max(),
            "atr": atr_series.iloc[-1],
        }

    def _decide(self, primary: dict, confirm: dict) -> tuple[SignalDirection, float, list[str]]:
        reasons: list[str] = []
        long_score = 0.0
        short_score = 0.0

        # Tendencia (1h)
        if primary["trend"] == "bullish":
            long_score += 0.3
            reasons.append("tendencia 1h alcista (EMA rápida > EMA lenta)")
        else:
            short_score += 0.3
            reasons.append("tendencia 1h bajista (EMA rápida < EMA lenta)")

        # Momentum RSI (1h): zona sana, ni sobrecomprado ni sobrevendido
        if self.rsi_oversold < primary["rsi"] < 60:
            long_score += 0.2
            reasons.append(f"RSI 1h en zona alcista sana ({primary['rsi']:.1f})")
        elif 40 < primary["rsi"] < self.rsi_overbought:
            short_score += 0.2
            reasons.append(f"RSI 1h en zona bajista sana ({primary['rsi']:.1f})")

        # Momentum MACD (1h): histograma positivo/creciente o negativo/decreciente
        if primary["macd_hist"] > 0 and primary["macd_hist"] > primary["macd_hist_prev"]:
            long_score += 0.25
            reasons.append("MACD 1h con histograma positivo y creciente")
        elif primary["macd_hist"] < 0 and primary["macd_hist"] < primary["macd_hist_prev"]:
            short_score += 0.25
            reasons.append("MACD 1h con histograma negativo y decreciente")

        # Confirmación 15m: no debe ir en contra del lado evaluado
        if confirm["macd_hist"] > 0 and confirm["rsi"] < self.rsi_overbought:
            long_score += 0.25
            reasons.append("confirmación 15m alineada al alza")
        if confirm["macd_hist"] < 0 and confirm["rsi"] > self.rsi_oversold:
            short_score += 0.25
            reasons.append("confirmación 15m alineada a la baja")

        if abs(long_score - short_score) < self.neutral_band and long_score > 0 and short_score > 0:
            reasons.append(f"señales contradictorias (long {long_score:.2f} vs short {short_score:.2f}): posición neutral")
            return SignalDirection.NONE, 0.0, reasons

        if long_score >= short_score:
            return SignalDirection.LONG, round(long_score, 2), reasons
        return SignalDirection.SHORT, round(short_score, 2), reasons
