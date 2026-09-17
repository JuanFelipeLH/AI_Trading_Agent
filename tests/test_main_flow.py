"""Tests OFFLINE del flujo completo (main.py) con cliente fake: sin red, sin
API de Binance, sin créditos de LLM. Verifica que el orquestador no revienta
y que el polling dinámico funciona."""
from __future__ import annotations

import asyncio
import unittest

import pandas as pd

from config.settings import Settings
from backtest.data import generate_ohlcv
from data_layer.models import MarketData
from main import TradingBot

from tests.helpers import load_strategy_config, temp_log_dir


class FakeBinanceClient:
    """Reemplaza a BinanceClient: entrega datos sintéticos, no hace red."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.closed = False

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True

    async def fetch_market_data(self, symbol, primary_tf, confirmation_tf,
                                primary_candles, confirmation_candles) -> MarketData:
        primary = self.df.copy()
        confirm = primary.tail(100).reset_index(drop=True)
        order_book = {
            "bids": [[float(self.df["close"].iloc[-1]), 1]],
            "asks": [[float(self.df["close"].iloc[-1]), 1]],
        }
        return MarketData(symbol=symbol, ohlcv={primary_tf: primary, confirmation_tf: confirm},
                          primary_timeframe=primary_tf, order_book=order_book)

    def market(self, symbol: str) -> dict:
        return {"limits": {"amount": {"min": 0.0001}, "cost": {"min": 0.001}}}

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        return round(amount, 6)

    async def fetch_last_price(self, symbol: str) -> float:
        return float(self.df["close"].iloc[-1])


def _build_bot() -> tuple[TradingBot, FakeBinanceClient]:
    cfg = load_strategy_config()
    cfg["logging"]["log_dir"] = temp_log_dir()
    cfg["venice"]["enabled"] = False
    settings = Settings(
        api_key="", api_secret="", use_testnet=True, trading_mode="paper_trading",
        initial_capital=10000.0, log_level="WARNING", venice_api_key="", strategy=cfg,
    )
    bot = TradingBot(settings)
    fake = FakeBinanceClient(generate_ohlcv(n=600, seed=11, base_price=10000.0))
    bot.client = fake
    bot.order_manager.client = fake
    return bot, fake


class MainFlowTest(unittest.TestCase):
    def test_ciclo_completo_sin_red(self):
        bot, fake = _build_bot()
        asyncio.run(bot._run_cycle())
        self.assertIsInstance(bot._next_poll_seconds(), (int, float))

    def test_varios_ciclos_no_revienta(self):
        bot, fake = _build_bot()
        for _ in range(5):
            asyncio.run(bot._run_cycle())
        # si abrió posiciones, deben tener SL/TP consistentes
        for pos in bot.order_manager.positions.values():
            self.assertLess(pos.stop_loss, pos.entry_price) if pos.direction.value == "LONG" else self.assertGreater(pos.stop_loss, pos.entry_price)

    def test_poll_dinamico_se_mueve_con_el_mercado(self):
        bot, _ = _build_bot()
        bot._market_state = {"high_volatility": True, "trending": True}
        fast = bot._next_poll_seconds()
        bot._market_state = {"high_volatility": False, "trending": False}
        slow = bot._next_poll_seconds()
        self.assertLessEqual(fast, slow)
        self.assertGreaterEqual(fast, bot.strategy["execution"]["min_poll_seconds"])
        self.assertLessEqual(slow, bot.strategy["execution"]["max_poll_seconds"])

    def test_poll_fijo_si_dynamic_off(self):
        bot, _ = _build_bot()
        bot.strategy["execution"]["dynamic_poll"] = False
        bot.strategy["execution"]["poll_interval_seconds"] = 60
        self.assertEqual(bot._next_poll_seconds(), 60.0)

    def test_cierre_de_campana(self):
        bot, fake = _build_bot()
        for _ in range(3):
            asyncio.run(bot._process_symbol("BTC/USDT"))
        self.assertIsInstance(bot.order_manager.open_positions_count, int)


if __name__ == "__main__":
    unittest.main()