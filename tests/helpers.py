"""Helpers compartidos para tests OFFLINE: carga de configuración real
(directo del YAML, sin depender de session_config.json) y construcción de
MarketData sintético."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import yaml

from backtest.data import generate_ohlcv
from data_layer.models import MarketData

BASE_DIR = Path(__file__).resolve().parent.parent
STRATEGY_PATH = BASE_DIR / "config" / "strategy_config.yaml"


def load_strategy_config(path: Path = STRATEGY_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # los tests corren siempre OFFLINE: nunca activar venice
    cfg.setdefault("venice", {})["enabled"] = False
    return cfg


def make_market_data(symbol: str = "TEST/USDT", seed: int = 7, n: int = 400,
                     base_price: float = 30000.0, tf: str = "1h",
                     confirm_tf: str = "15m") -> MarketData:
    df = generate_ohlcv(n=n, seed=seed, base_price=base_price)
    return MarketData(
        symbol=symbol,
        ohlcv={tf: df, confirm_tf: df.tail(100).reset_index(drop=True)},
        primary_timeframe=tf,
        order_book={"bids": [[float(df["close"].iloc[-1]), 1]], "asks": [[float(df["close"].iloc[-1]), 1]]},
    )


def temp_log_dir():
    tmp = tempfile.mkdtemp(prefix="td_logs_")
    return tmp