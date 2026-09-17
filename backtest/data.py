"""Generador de datos sintéticos OHLCV para validación OFFLINE del bot.

Requisito del proyecto: probar y validar sin gastar créditos ni llamar a la
API real. Este módulo produce series con regímenes alternados (tendencia
alcista/bajista y rangos) y ráfagas de volatilidad, para que el detector de
régimen y las estrategias puedan ejercitarse sin red.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ohlcv(n: int = 800, seed: int = 42, base_price: float = 30000.0,
                   atr_pct: float = 0.015, timeframe_minutes: int = 60,
                   seed_price: bool = True) -> pd.DataFrame:
    """Serie sintética con cambio de régimen (drift) y volatilidad variable.

    Regímenes: tendencia alcista (drift +0.4%), bajista (-0.4%), rango (0%),
    alternados cada `segment` velas, con ocasionales ráfagas de volatilidad 3x.
    """
    rng = np.random.default_rng(seed)

    segment = 80
    regimes = []
    for s in range(0, n, segment):
        r = rng.choice(["bull", "bear", "range"])
        # ocasionalmente frag a volatilidad alta
        vol_mult = 3.0 if rng.random() < 0.15 else 1.0
        regimes.extend([{"drift": {"bull": 0.004, "bear": -0.004, "range": 0.0}[r],
                         "vol": vol_mult}] * segment)

    ts = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor(f"{timeframe_minutes}min"),
                       periods=n, freq=f"{timeframe_minutes}min")

    closes = np.empty(n)
    opens_arr = np.empty(n)
    closes[0] = base_price
    opens_arr[0] = base_price
    hi = np.empty(n)
    lo = np.empty(n)
    hi[0] = base_price * (1 + atr_pct)
    lo[0] = base_price * (1 - atr_pct)
    for i in range(1, n):
        vol = atr_pct * regimes[i]["vol"]
        ret = regimes[i]["drift"] + rng.normal(0, vol)
        closes[i] = max(closes[i - 1] * (1 + ret), 1.0)
        opens_arr[i] = closes[i - 1]
        bar_hi = max(opens_arr[i], closes[i]) * (1 + abs(rng.normal(0, atr_pct / 3)))
        bar_lo = min(opens_arr[i], closes[i]) * (1 - abs(rng.normal(0, atr_pct / 3)))
        hi[i] = max(bar_hi, opens_arr[i], closes[i])
        lo[i] = min(bar_lo, opens_arr[i], closes[i])

    volumes = rng.uniform(100, 1000, n) * (3 if seed_price else 1)

    return pd.DataFrame({
        "timestamp": ts,
        "open": opens_arr,
        "high": hi,
        "low": lo,
        "close": closes,
        "volume": volumes,
    })