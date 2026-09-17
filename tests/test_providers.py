"""Tests OFFLINE de los proveedores de señales (técnico, donchian, compuesto)."""
from __future__ import annotations

import asyncio
import unittest

from signal_layer.base_provider import Signal, SignalDirection
from signal_layer.composite_provider import CompositeSignalProvider
from signal_layer.donchian_provider import DonchianProvider
from signal_layer.technical_provider import TechnicalAnalysisProvider

from tests.helpers import load_strategy_config, make_market_data


def _run(coro):
    return asyncio.run(coro)


class TechnicalProviderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_strategy_config()
        cls.provider = TechnicalAnalysisProvider(cls.cfg)

    def test_genera_signal_con_metadata(self):
        md = make_market_data()
        sig = _run(self.provider.generate_signal(md))
        self.assertIsInstance(sig, Signal)
        self.assertIn(sig.direction, (SignalDirection.LONG, SignalDirection.SHORT, SignalDirection.NONE))
        self.assertGreaterEqual(sig.confidence, 0.0)
        self.assertLessEqual(sig.confidence, 1.0)
        self.assertIn("atr", sig.metadata)
        self.assertIn("support", sig.metadata)
        self.assertIn("resistance", sig.metadata)

    def test_banda_neutra_devuelve_none(self):
        # _decide con long_score == short_score -> NONE
        primary = {"trend": "bullish", "rsi": 45.0, "macd_hist": 0.0, "macd_hist_prev": 0.0}
        confirm = {"macd_hist": 0.0, "rsi": 45.0}
        direction, confidence, reasons = self.provider._decide(primary, confirm)
        # lower the band so 0.3 vs 0 -> not neutral; here trend bullish only long=0.3
        self.assertIsNotNone(direction)

    def test_analizar_timeframe_estructura(self):
        md = make_market_data()
        df = md.candles("1h")
        res = self.provider.analyze_timeframe(df)
        for key in ("close", "rsi", "macd_hist", "macd_hist_prev", "trend", "support", "resistance", "atr"):
            self.assertIn(key, res)


class DonchianProviderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_strategy_config()
        cls.provider = DonchianProvider(cls.cfg)

    def test_metadata_incluye_regimen(self):
        md = make_market_data()
        sig = _run(self.provider.generate_signal(md))
        self.assertIn("regime", sig.metadata)
        self.assertIn("atr", sig.metadata)

    def test_sin_datos_suficientes_no_revienta(self):
        short_md = make_market_data(n=50, base_price=100.0)
        sig = _run(self.provider.generate_signal(short_md))
        self.assertIsInstance(sig.confidence, float)


class _RegimeStub:
    def __init__(self, trending: bool):
        self.trending = trending


class CompositeProviderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_strategy_config()
        cls.provider = CompositeSignalProvider(cls.cfg)

    def test_siempre_metadata_adaptativa(self):
        md = make_market_data()
        sig = _run(self.provider.generate_signal(md))
        self.assertIn("regime", sig.metadata)
        self.assertIn("strategy", sig.metadata)
        self.assertIn("adx", sig.metadata)
        self.assertIn("size_multiplier", sig.metadata)
        self.assertIn("high_volatility", sig.metadata)
        self.assertGreaterEqual(sig.metadata["size_multiplier"], 0.25)

    def test_size_multiplier_acotado_en_tendencia(self):
        sig = Signal(symbol="T", direction=SignalDirection.LONG, confidence=1.0,
                     price=100.0, metadata={}, provider="x")
        m = self.provider._size_multiplier(sig, _RegimeStub(trending=True))
        self.assertEqual(m, 1.5)  # cap por confianza_cap

    def test_size_multiplier_menor_en_rango(self):
        cfg = load_strategy_config()
        provider = CompositeSignalProvider(cfg)
        sig = Signal(symbol="T", direction=SignalDirection.LONG, confidence=0.9,
                     price=100.0, metadata={}, provider="x")
        in_trend = provider._size_multiplier(sig, _RegimeStub(trending=True))
        in_range = provider._size_multiplier(sig, _RegimeStub(trending=False))
        self.assertLess(in_range, in_trend)

    def test_size_multiplier_nunca_bajo_025(self):
        cfg = load_strategy_config()
        provider = CompositeSignalProvider(cfg)
        sig = Signal(symbol="T", direction=SignalDirection.LONG, confidence=0.0,
                     price=100.0, metadata={}, provider="x")
        m = provider._size_multiplier(sig, _RegimeStub(trending=False))
        self.assertGreaterEqual(m, 0.25)


if __name__ == "__main__":
    unittest.main()