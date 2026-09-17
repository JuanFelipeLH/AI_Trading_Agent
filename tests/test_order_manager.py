"""Tests OFFLINE de OrderManager/Position: apertura, cierre SL/TP y trailing.
Sin red: el client es un fake, no ccxt/binance."""
from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from execution_layer.models import Position, PositionStatus
from execution_layer.order_manager import OrderManager
from risk_layer.risk_manager import RiskDecision
from signal_layer.base_provider import Signal, SignalDirection
from utils.logger import TradeLogger

from tests.helpers import load_strategy_config, temp_log_dir


class FakeClient:
    """Sustituto del BinanceClient: deja 'verificar' órdenes sin red."""

    def market(self, symbol: str) -> dict:
        return {
            "limits": {"amount": {"min": 0.0001}, "cost": {"min": 0.001}},
        }

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        return round(amount, 6)


def _mk_decision(**kwargs) -> RiskDecision:
    d = dict(approved=True, reason="ok", quantity=1.0, stop_loss=90.0,
             take_profit=110.0, risk_amount=200.0, atr=2.0, trail_atr_multiplier=2.0)
    d.update(kwargs)
    return RiskDecision(**d)


def _mk_signal(direction=SignalDirection.LONG, price=100.0) -> Signal:
    return Signal(symbol="TEST/USDT", direction=direction, confidence=0.8,
                  price=price, provider="test", metadata={})


class OrderManagerTest(unittest.TestCase):
    def setUp(self):
        cfg = load_strategy_config()
        from risk_layer.risk_manager import RiskManager
        self.risk = RiskManager(cfg)
        self.logger = TradeLogger(temp_log_dir(), "trades.jsonl")
        self.mgr = OrderManager(FakeClient(), self.logger, "paper_trading", risk_manager=self.risk)

    def test_apertura_rellena_campos_trailing(self):
        pos = asyncio.run(self.mgr.open_position(_mk_signal(), _mk_decision()))
        self.assertIn("TEST/USDT", self.mgr.positions)
        self.assertEqual(pos.trail_atr_multiplier, 2.0)
        self.assertEqual(pos.atr_at_entry, 2.0)
        self.assertEqual(pos.best_price_since_entry, pos.entry_price)
        self.assertEqual(pos.status, PositionStatus.OPEN)

    def test_cierre_por_stop_loss(self):
        pos = asyncio.run(self.mgr.open_position(_mk_signal(), _mk_decision()))
        closed = asyncio.run(self.mgr.check_exit_conditions("TEST/USDT", 89.0))
        self.assertIsNotNone(closed)
        self.assertEqual(closed.close_reason, "stop_loss")
        self.assertEqual(closed.exit_price, 89.0)
        self.assertNotIn("TEST/USDT", self.mgr.positions)

    def test_cierre_por_take_profit(self):
        asyncio.run(self.mgr.open_position(_mk_signal(), _mk_decision()))
        closed = asyncio.run(self.mgr.check_exit_conditions("TEST/USDT", 111.0))
        self.assertIsNotNone(closed)
        self.assertEqual(closed.close_reason, "take_profit")

    def test_sin_condicion_no_cierra(self):
        asyncio.run(self.mgr.open_position(_mk_signal(), _mk_decision()))
        closed = asyncio.run(self.mgr.check_exit_conditions("TEST/USDT", 100.0))
        self.assertIsNone(closed)

    def test_trailing_actualiza_stop(self):
        asyncio.run(self.mgr.open_position(_mk_signal(), _mk_decision()))
        # el best-price sube a 106, dist 4 => nuevos stop 106-4=102
        asyncio.run(self.mgr.check_exit_conditions("TEST/USDT", 106.0))
        self.assertGreater(self.mgr.positions["TEST/USDT"].stop_loss, 90.0)

    def test_cierre_con_callback_pnl(self):
        holder = {}
        asyncio.run(self.mgr.open_position(_mk_signal(), _mk_decision()))
        asyncio.run(self.mgr.check_exit_conditions(
            "TEST/USDT", 110.0, on_close=lambda p: holder.update(pnl=p.pnl)))
        self.assertGreater(holder["pnl"], 0.0)

    def test_position_pnl_positivo_y_negativo(self):
        pos = Position(symbol="T", direction=SignalDirection.LONG, entry_price=100.0,
                       quantity=1.0, stop_loss=90.0, take_profit=110.0, entry_order_id="x")
        pos.exit_price = 110.0
        self.assertAlmostEqual(pos.pnl, 10.0)
        pos.exit_price = 90.0
        self.assertAlmostEqual(pos.pnl, -10.0)
        self.assertAlmostEqual(pos.pnl_pct, -0.10)


if __name__ == "__main__":
    unittest.main()