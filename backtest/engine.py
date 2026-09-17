"""Motor de backtest OFFLINE: ejecuta el pipeline real del bot
(data -> señal -> riesgo -> SL/TP/trailing) sobre velas sintéticas o históricas
sin llamar a la API ni a ningún modelo externo.

Con esto se puede validar de forma walk-forward-ligera que una configuración
es robusta (rentabilidad, ratio beneficio/riesgo, drawdown) ANTES de activarla
en testnet o producción. Cero créditos, cero red.

Uso programático:
    from backtest.engine import run_backtest
    from backtest.data import generate_ohlcv
    from signal_layer.composite_provider import CompositeSignalProvider
    metrics = run_backtest(provider, generate_ohlcv(), strategy_config, capital=10000)

CLI:
    python main.py --backtest
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np

from backtest.data import generate_ohlcv
from data_layer.models import MarketData
from risk_layer.risk_manager import RiskManager
from signal_layer.base_provider import BaseSignalProvider, SignalDirection

logger = logging.getLogger("trading_bot")

WARMUP_BARS = 200  # velas necesarias para EMA200 y ADX estables


@dataclass
class OpenTrade:
    symbol: str
    direction: SignalDirection
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    atr: float
    trail_multiplier: float | None
    best_price: float
    opened_at: int


@dataclass
class ClosedTrade:
    symbol: str
    direction: SignalDirection
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    reason: str
    opened_at: int
    closed_at: int


@dataclass
class BacktestMetrics:
    total_return_pct: float
    num_trades: int
    win_rate_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_pnl_per_trade: float
    equity_curve: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Return: {self.total_return_pct:+.2f}% | Trades: {self.num_trades} | "
            f"Win rate: {self.win_rate_pct:.1f}% | Profit factor: {self.profit_factor:.2f} | "
            f"Max DD: {self.max_drawdown_pct:.2f}% | Sharpe: {self.sharpe_ratio:.2f} | "
            f"PnL/trade: {self.avg_pnl_per_trade:.2f}"
        )


def _signal_at(df, symbol: str, index: int, primary_tf: str, confirm_tf: str) -> MarketData:
    """Snapshot de mercado en el bar `index` (solo velas hasta ese punto, sin
    lookahead)."""
    primary = df.iloc[:index + 1].copy()
    confirm = primary.tail(100) if len(primary) >= 100 else primary
    return MarketData(
        symbol=symbol,
        ohlcv={primary_tf: primary, confirm_tf: confirm},
        primary_timeframe=primary_tf,
        order_book={"bids": [[float(primary["close"].iloc[-1]), 1]], "asks": [[float(primary["close"].iloc[-1]), 1]]},
    )


def run_backtest(provider: BaseSignalProvider, df, strategy_config: dict,
                 capital: float = 10000.0, symbol: str = "TEST/USDT",
                 primary_tf: str | None = None, max_open_positions: int = 1) -> BacktestMetrics:
    """Recorre `df` bar a barra aplicando el pipeline real del bot. Devuelve
    métricas de performance. Sin red ni API."""
    primary_tf = primary_tf or strategy_config["timeframes"]["primary"]
    confirm_tf = strategy_config["timeframes"]["confirmation"]
    risk = RiskManager(strategy_config)
    open_trades: list[OpenTrade] = []
    closed_trades: list[ClosedTrade] = []
    equity = capital
    equity_curve: list[float] = []
    peak = capital
    max_dd = 0.0
    returns: list[float] = []

    max_open = min(max_open_positions, 2)

    for i in range(WARMUP_BARS, len(df)):
        bar = df.iloc[i]
        high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])

        for pos in list(open_trades):
            stop = risk.update_trailing_stop(
                current_price=(high if pos.direction == SignalDirection.LONG else low),
                entry_price=pos.entry_price,
                direction=pos.direction,
                stop_loss=pos.stop_loss,
                atr=pos.atr,
                trail_multiplier=pos.trail_multiplier,
            )
            pos.stop_loss = stop
            if pos.direction == SignalDirection.LONG:
                pos.best_price = max(pos.best_price, high)
            else:
                pos.best_price = min(pos.best_price, low)

            hit_sl = (pos.direction == SignalDirection.LONG and low <= pos.stop_loss) or \
                     (pos.direction == SignalDirection.SHORT and high >= pos.stop_loss)
            hit_tp = (pos.direction == SignalDirection.LONG and high >= pos.take_profit) or \
                     (pos.direction == SignalDirection.SHORT and low <= pos.take_profit)

            if hit_sl and hit_tp:
                # conservador: el stop se toca primero
                exit_price, reason = pos.stop_loss, "stop_loss"
            elif hit_sl:
                exit_price, reason = pos.stop_loss, "stop_loss"
            elif hit_tp:
                exit_price, reason = pos.take_profit, "take_profit"
            else:
                continue

            sign = 1 if pos.direction == SignalDirection.LONG else -1
            pnl = sign * (exit_price - pos.entry_price) * pos.quantity
            pnl_pct = sign * (exit_price - pos.entry_price) / pos.entry_price
            equity += pnl
            closed_trades.append(ClosedTrade(
                symbol=symbol, direction=pos.direction, entry_price=pos.entry_price,
                exit_price=exit_price, quantity=pos.quantity, pnl=pnl, pnl_pct=pnl_pct,
                reason=reason, opened_at=pos.opened_at, closed_at=i,
            ))
            risk.record_trade_result(pnl)
            open_trades.remove(pos)

        equity_curve.append(equity)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)

        if risk.trading_halted:
            continue  # circuit breaker: no abrir más posiciones

        if len(open_trades) >= max_open:
            continue

        md = _signal_at(df, symbol, i, primary_tf, confirm_tf)
        try:
            signal = asyncio.run(provider.generate_signal(md))
        except Exception:
            logger.exception("señal falló en bar %s", i)
            continue

        if not signal.is_actionable:
            continue

        decision = risk.validate_and_size(signal, equity, len(open_trades), max_open)
        if not decision.approved:
            continue

        open_trades.append(OpenTrade(
            symbol=symbol, direction=signal.direction, entry_price=float(close),
            quantity=decision.quantity, stop_loss=decision.stop_loss,
            take_profit=decision.take_profit, atr=decision.atr or float(close) * 0.01,
            trail_multiplier=decision.trail_atr_multiplier, best_price=float(close),
            opened_at=i,
        ))

    # cierre forzado al final de la serie
    final_close = float(df["close"].iloc[-1])
    for pos in list(open_trades):
        sign = 1 if pos.direction == SignalDirection.LONG else -1
        pnl = sign * (final_close - pos.entry_price) * pos.quantity
        pnl_pct = sign * (final_close - pos.entry_price) / pos.entry_price
        equity += pnl
        closed_trades.append(ClosedTrade(
            symbol=symbol, direction=pos.direction, entry_price=pos.entry_price,
            exit_price=final_close, quantity=pos.quantity, pnl=pnl, pnl_pct=pnl_pct,
            reason="end_of_backtest", opened_at=pos.opened_at, closed_at=len(df) - 1,
        ))
        open_trades.remove(pos)

    total_return = (equity - capital) / capital * 100
    if closed_trades:
        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl <= 0]
        win_rate = len(wins) / len(closed_trades) * 100
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        pf = gross_win / gross_loss if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)
        avg_trade = sum(t.pnl for t in closed_trades) / len(closed_trades)

        per_bar = np.diff(equity_curve) / np.array(equity_curve[:-1], dtype=float) if len(equity_curve) > 1 else np.array([])
        std = float(np.std(per_bar)) if len(per_bar) else 0.0
        if std > 0:
            sharpe = float(np.mean(per_bar)) / std * np.sqrt(365 * 24)
            sharpe = sharpe if np.isfinite(sharpe) else 0.0
        else:
            sharpe = 0.0
    else:
        win_rate = pf = avg_trade = sharpe = 0.0

    return BacktestMetrics(
        total_return_pct=total_return,
        num_trades=len(closed_trades),
        win_rate_pct=win_rate,
        profit_factor=pf,
        max_drawdown_pct=max_dd * 100,
        sharpe_ratio=sharpe,
        avg_pnl_per_trade=avg_trade,
        equity_curve=equity_curve,
    )


def quick_validation(strategy_config: dict, seeds=(7, 2026), capital: float = 10000.0) -> BacktestMetrics:
    """Valida la configuración actual del bot contra varios datasets sintéticos
    (distintos seeds) y devuelve el promedio. Útil para un smoke test robusto."""
    from signal_layer.composite_provider import CompositeSignalProvider

    provider = CompositeSignalProvider(strategy_config)
    total = 0.0
    collected = []
    for seed in seeds:
        df = generate_ohlcv(seed=seed, base_price=30000.0 + seed * 100)
        m = run_backtest(provider, df, strategy_config, capital=capital)
        collected.append(m)
        total += m.total_return_pct
    avg = total / len(collected)
    aggregate = collected[0]
    aggregate.total_return_pct = avg
    aggregate.num_trades = sum(c.num_trades for c in collected)
    aggregate.win_rate_pct = sum(c.win_rate_pct for c in collected) / len(collected)
    aggregate.profit_factor = sum(c.profit_factor for c in collected) / len(collected)
    aggregate.max_drawdown_pct = max(c.max_drawdown_pct for c in collected)
    aggregate.sharpe_ratio = sum(c.sharpe_ratio for c in collected) / len(collected)
    aggregate.avg_pnl_per_trade = sum(c.avg_pnl_per_trade for c in collected) / len(collected)
    return aggregate