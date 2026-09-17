"""Tests OFFLINE del backtester con datos sintéticos (sin red ni créditos)."""
from __future__ import annotations

import unittest

from backtest.data import generate_ohlcv
from backtest.engine import BacktestMetrics, run_backtest
from signal_layer.composite_provider import CompositeSignalProvider

from tests.helpers import load_strategy_config


class BacktestDataTest(unittest.TestCase):
    def test_genera_datos_validos(self):
        df = generate_ohlcv(n=300, seed=1, base_price=100.0)
        self.assertEqual(len(df), 300)
        self.assertTrue((df["high"] >= df[["open", "close"]].max(axis=1)).all())
        self.assertTrue((df["low"] <= df[["open", "close"]].min(axis=1)).all())
        self.assertTrue((df["close"] > 0).all())

    def test_mismo_seed_misma_serie(self):
        a = generate_ohlcv(seed=99)
        b = generate_ohlcv(seed=99)
        self.assertTrue((a["close"] == b["close"]).all())


class BacktestEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_strategy_config()
        cls.provider = CompositeSignalProvider(cls.cfg)

    def test_corrida_completa_devuelve_metricas(self):
        df = generate_ohlcv(n=600, seed=42, base_price=10000.0)
        m = run_backtest(self.provider, df, self.cfg, capital=10000.0)
        self.assertIsInstance(m, BacktestMetrics)
        self.assertGreaterEqual(m.num_trades, 0)
        self.assertGreaterEqual(len(m.equity_curve), 0)
        self.assertGreaterEqual(m.max_drawdown_pct, 0.0)
        self.assertIsInstance(m.total_return_pct, float)
        self.assertIsInstance(m.win_rate_pct, float)

    def test_reproductible_con_mismo_seed(self):
        df = generate_ohlcv(n=500, seed=5, base_price=10000.0)
        a = run_backtest(self.provider, df, self.cfg, capital=10000.0)
        b = run_backtest(self.provider, df, self.cfg, capital=10000.0)
        self.assertEqual(a.total_return_pct, b.total_return_pct)
        self.assertEqual(a.num_trades, b.num_trades)

    def test_summary_no_vacio(self):
        df = generate_ohlcv(n=500, seed=3, base_price=10000.0)
        m = run_backtest(self.provider, df, self.cfg, capital=10000.0)
        self.assertGreater(len(m.summary()), 10)


if __name__ == "__main__":
    unittest.main()