"""Tests OFFLINE de RiskManager: SL/TP, sizing, circuit breaker y trailing stop."""
from __future__ import annotations

import unittest

from risk_layer.risk_manager import RiskManager
from signal_layer.base_provider import Signal, SignalDirection

from tests.helpers import load_strategy_config


def _mk_signal(direction=SignalDirection.LONG, confidence=0.8, price=100.0,
               atr=None, size_multiplier=None) -> Signal:
    metadata = {"atr": atr} if atr else {}
    if size_multiplier is not None:
        metadata["size_multiplier"] = size_multiplier
    return Signal(symbol="TEST/USDT", direction=direction, confidence=confidence,
                  price=price, provider="test", metadata=metadata)


class RiskManagerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_strategy_config()
        cls.risk = RiskManager(cls.cfg)

    def test_sl_atr_long(self):
        sl = self.risk.calculate_stop_loss(100.0, SignalDirection.LONG, atr=2.0)
        self.assertEqual(sl, 100.0 - 2.0 * self.risk.atr_multiplier)

    def test_sl_atr_short(self):
        sl = self.risk.calculate_stop_loss(100.0, SignalDirection.SHORT, atr=2.0)
        self.assertEqual(sl, 100.0 + 2.0 * self.risk.atr_multiplier)

    def test_tp_ratio(self):
        entry, sl = 100.0, 98.0
        tp = self.risk.calculate_take_profit(entry, sl, SignalDirection.LONG)
        self.assertAlmostEqual(tp, entry + 2.0 * self.risk.risk_reward_ratio)

    def test_sizing_respeta_riesgo(self):
        qty = self.risk.calculate_position_size(10000.0, entry_price=100.0, stop_loss=50.0)
        # riesgo 2% del capital = 200 ; por unidad 50 -> 200/50 = 4 unidades
        self.assertAlmostEqual(qty, 4.0, places=6)

    def test_sizing_tope_de_exposicion(self):
        # SL muy cercano: el sizing por riesgo (200/1=200) supera el tope del 50%
        # del capital (5000/100=50) y debe quedar clavado en el tope.
        qty = self.risk.calculate_position_size(10000.0, entry_price=100.0, stop_loss=99.0)
        self.assertAlmostEqual(qty, 50.0, places=6)
        self.assertLessEqual(qty * 100.0, 10000.0 * self.risk.max_position_pct_of_capital)

    def test_validate_none(self):
        dec = self.risk.validate_and_size(_mk_signal(direction=SignalDirection.NONE), 10000.0, 0, 2)
        self.assertFalse(dec.approved)

    def test_validate_limite_posiciones(self):
        dec = self.risk.validate_and_size(_mk_signal(), 10000.0, 2, 2)
        self.assertFalse(dec.approved)

    def test_validate_incluye_trailing_y_atr(self):
        dec = self.risk.validate_and_size(_mk_signal(atr=2.0), 10000.0, 0, 2)
        self.assertTrue(dec.approved)
        self.assertEqual(dec.atr, 2.0)
        self.assertTrue(dec.trail_atr_multiplier is not None)

    def test_circuit_breaker(self):
        r = RiskManager(self.cfg)  # config por defecto max_consecutive_losses
        for _ in range(r.max_consecutive_losses):
            r.record_trade_result(-10.0)
        self.assertFalse(r.trading_halted)
        r.record_trade_result(-10.0)  # 1 más => supera
        self.assertTrue(r.trading_halted)
        dec = r.validate_and_size(_mk_signal(), 10000.0, 0, 2)
        self.assertFalse(dec.approved)

    def test_reset_circuit_breaker(self):
        r = RiskManager(self.cfg)
        for _ in range(r.max_consecutive_losses + 2):
            r.record_trade_result(-10.0)
        self.assertTrue(r.trading_halted)
        r.reset_circuit_breaker()
        self.assertFalse(r.trading_halted)

    def test_confidence_multiplier_con_hint(self):
        sig = _mk_signal(size_multiplier=1.5)
        self.assertEqual(self.risk.confidence_multiplier(sig), 1.5)

    def test_confidence_multiplier_sin_hint(self):
        self.risk.confidence_sizing = True
        sig = _mk_signal(confidence=0.75)
        self.assertAlmostEqual(self.risk.confidence_multiplier(sig), 1.0, places=6)

    # --- trailing stop (validación empírica: salidas por trailing ganan) ---
    def test_trailing_long_avanza(self):
        stop = self.risk.update_trailing_stop(
            current_price=110.0, entry_price=100.0, direction=SignalDirection.LONG,
            stop_loss=98.0, atr=2.0, trail_multiplier=2.0)
        self.assertGreater(stop, 98.0)  # 110 - 4 = 106

    def test_trailing_long_no_se_aleja(self):
        stop = self.risk.update_trailing_stop(
            current_price=101.0, entry_price=100.0, direction=SignalDirection.LONG,
            stop_loss=98.0, atr=2.0, trail_multiplier=2.0)
        self.assertEqual(stop, 98.0)  # no alcanzó 1x distancia, no se mueve

    def test_trailing_short_avanza(self):
        stop = self.risk.update_trailing_stop(
            current_price=90.0, entry_price=100.0, direction=SignalDirection.SHORT,
            stop_loss=102.0, atr=2.0, trail_multiplier=2.0)
        self.assertLess(stop, 102.0)  # 90 + 4 = 94

    def test_trailing_deshabilitado_no_mueve(self):
        self.risk.trail_enabled = False
        stop = self.risk.update_trailing_stop(
            current_price=150.0, entry_price=100.0, direction=SignalDirection.LONG,
            stop_loss=98.0, atr=2.0, trail_multiplier=2.0)
        self.risk.trail_enabled = True
        self.assertEqual(stop, 98.0)


if __name__ == "__main__":
    unittest.main()