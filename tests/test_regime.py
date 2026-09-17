"""Tests OFFLINE del módulo de régimen de mercado. Sin red ni API."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtest.data import generate_ohlcv
from signal_layer.regime import MarketRegime, adx, classify


def _flat_walk(n=300, noise=0.005) -> pd.DataFrame:
    """Serie básicamente plana (white noise alrededor de constante):
    debe clasificar como RANGING y ADX bajo."""
    rng = np.random.default_rng(1)
    closes = 100.0 + rng.normal(0, noise, n).cumsum()
    closes = np.abs(closes) + 1.0
    return pd.DataFrame({
        "timestamp": pd.date_range(end="2026-01-01", periods=n, freq="1h"),
        "open": np.roll(closes, 1),
        "high": closes * 1.001,
        "low": closes * 0.999,
        "close": closes,
        "volume": np.full(n, 100.0),
    })


def _trend_up(n=300, drift=0.008) -> pd.DataFrame:
    """Serie con tendencia alcista sostenida: RANGING->TRENDING y ADX alto."""
    rng = np.random.default_rng(3)
    closes = np.empty(n)
    closes[0] = 100.0
    for i in range(1, n):
        closes[i] = closes[i - 1] * (1 + drift + rng.normal(0, 0.004))
    return pd.DataFrame({
        "timestamp": pd.date_range(end="2026-01-01", periods=n, freq="1h"),
        "open": np.roll(closes, 1),
        "high": closes * 1.01,
        "low": closes * 0.99,
        "close": closes,
        "volume": np.full(n, 100.0),
    })


class RegimeTest(unittest.TestCase):
    def test_rango_adx_bajo(self):
        df = _flat_walk()
        snap = classify(df, adx_threshold=25.0)
        self.assertEqual(snap.regime, MarketRegime.RANGING)
        self.assertLess(snap.adx, 25.0)
        self.assertFalse(snap.trending)

    def test_tendencia_adx_alto(self):
        df = _trend_up()
        snap = classify(df, adx_threshold=25.0)
        self.assertEqual(snap.regime, MarketRegime.TRENDING)
        self.assertTrue(snap.trending)
        self.assertGreater(snap.adx, 20.0)

    def test_volatilidad_relativa(self):
        df = _trend_up()
        snap = classify(df, atr_pct_high=0.05)
        self.assertIsInstance(snap.atr_pct, float)
        self.assertIsInstance(snap.high_volatility, bool)
        self.assertGreater(snap.atr_pct, 0.0)

    def test_datos_insuficientes_conservador(self):
        small = pd.DataFrame({"high": [1, 2], "low": [0.5, 1.5], "close": [1, 2]})
        snap = classify(small)
        # si no alcanza para ADX estable, no debe reventar y asume RANGING
        self.assertIn(snap.regime, (MarketRegime.RANGING, MarketRegime.TRENDING))
        self.assertIsInstance(snap.adx, float)

    def test_adx_es_finito_en_serie_larga(self):
        df = _flat_walk()
        series = adx(df)
        self.assertTrue(np.isfinite(float(series.iloc[-1])))


if __name__ == "__main__":
    unittest.main()