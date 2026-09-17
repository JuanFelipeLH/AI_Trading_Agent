"""Detección del régimen de mercado actual (tendencia vs rango) usando ADX
(índice direccional medio de Wilder) y nivel de volatilidad relativa (ATR como
% del precio).

El resto del sistema (composite_provider, risk, execution) usa este módulo
para adaptar la estrategia en cada ciclo en lugar de esperar una condición
fija: en mercado con tendencia se opera con rupturas (Donchian), en rango se
opera con el scoring técnico tradicional o se mantiene en efectivo.

ADX se calcula con pandas puro (sin TA-Lib) para no añadir dependencias.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from signal_layer.technical_provider import atr


class MarketRegime(str, Enum):
    TRENDING = "trending"
    RANGING = "ranging"


@dataclass
class RegimeSnapshot:
    regime: MarketRegime
    adx: float
    atr_pct: float            # ATR(period)/precio * 100
    high_volatility: bool
    momentum_pct: float       # % de cambio en las últimas `momentum_period` velas

    @property
    def trending(self) -> bool:
        return self.regime == MarketRegime.TRENDING


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Índice direccional medio (Wilder). Devuelve serie con el ADX."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        [(u if (u > d and u > 0) else 0.0) for u, d in zip(up_move, down_move)],
        index=df.index,
    )
    minus_dm = pd.Series(
        [(d if (d > u and d > 0) else 0.0) for u, d in zip(up_move, down_move)],
        index=df.index,
    )

    atr_series = atr(df, period)
    atr_safe = atr_series.replace(0, pd.NA).ffill().fillna(1e-12)

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_safe
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-12)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def classify(df: pd.DataFrame, adx_threshold: float = 25.0,
             atr_pct_high: float = 1.5, adx_period: int = 14,
             momentum_period: int = 12) -> RegimeSnapshot:
    """Clasifica el régimen de las últimas velas de `df` (debe estar en orden
    cronológico). Comportamiento conservador: ante datos insuficientes se
    asume RANGING y volatilidad normal."""
    close = df["close"]
    adx_series = adx(df, adx_period)

    adx_val = float(adx_series.iloc[-1] if len(adx_series) else 0.0)
    atr_val = float(atr(df, adx_period).iloc[-1] if len(df) else 0.0)
    atr_pct = (atr_val / float(close.iloc[-1]) * 100) if len(close) and float(close.iloc[-1]) else 0.0

    if len(close) > momentum_period and float(close.iloc[-1 - momentum_period]) > 0:
        momentum_pct = (float(close.iloc[-1]) / float(close.iloc[-1 - momentum_period]) - 1) * 100
    else:
        momentum_pct = 0.0

    regime = MarketRegime.TRENDING if adx_val >= adx_threshold else MarketRegime.RANGING

    return RegimeSnapshot(
        regime=regime,
        adx=adx_val,
        atr_pct=atr_pct,
        high_volatility=atr_pct >= atr_pct_high,
        momentum_pct=momentum_pct,
    )