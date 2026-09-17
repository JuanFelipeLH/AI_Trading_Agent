"""Ejecución de órdenes y gestión del ciclo de vida de las posiciones.

Soporta dos modos (`trading_mode`):
- paper_trading: simula fills usando el precio de mercado actual, sin enviar
  órdenes reales al exchange (útil incluso contra testnet, para validar la
  lógica sin arriesgar ni siquiera fondos de prueba).
- live_trading: envía órdenes reales a través de BinanceClient (que a su vez
  puede apuntar a testnet o a producción según USE_TESTNET).

El stop loss / take profit se gestionan por polling (comparando el precio
actual contra los niveles calculados por RiskManager) en vez de una OCO
nativa del exchange, para mantener el código simple y portable entre
exchanges. Para producción de alto volumen se recomendaría migrar a
órdenes OCO/bracket nativas de Binance.
"""
from __future__ import annotations

import uuid
from typing import Callable

from data_layer.binance_client import BinanceClient
from execution_layer.models import Position, PositionStatus
from risk_layer.risk_manager import RiskDecision
from signal_layer.base_provider import Signal, SignalDirection
from utils.logger import TradeLogger


class OrderValidationError(Exception):
    pass


class OrderManager:
    def __init__(self, client: BinanceClient, logger: TradeLogger, trading_mode: str, risk_manager=None):
        self.client = client
        self.logger = logger
        self.trading_mode = trading_mode  # "paper_trading" | "live_trading"
        self.risk_manager = risk_manager
        self.positions: dict[str, Position] = {}  # symbol -> Position (una posición abierta por símbolo)

    @property
    def open_positions_count(self) -> int:
        return len(self.positions)

    async def validate_order(self, symbol: str, quantity: float, price: float) -> None:
        market = self.client.market(symbol)
        min_amount = market["limits"]["amount"]["min"] or 0
        min_notional = (market["limits"].get("cost") or {}).get("min") or 0

        if quantity < min_amount:
            raise OrderValidationError(f"cantidad {quantity} por debajo del mínimo {min_amount} para {symbol}")
        if quantity * price < min_notional:
            raise OrderValidationError(
                f"notional {quantity * price:.2f} por debajo del mínimo {min_notional} para {symbol}"
            )

    async def open_position(self, signal: Signal, decision: RiskDecision) -> Position:
        symbol = signal.symbol
        side = "buy" if signal.direction == SignalDirection.LONG else "sell"
        quantity = self.client.amount_to_precision(symbol, decision.quantity)

        await self.validate_order(symbol, quantity, signal.price)

        if self.trading_mode == "live_trading":
            order = await self.client.create_market_order(symbol, side, quantity)
            entry_price = float(order.get("average") or order.get("price") or signal.price)
            order_id = str(order.get("id"))
        else:
            entry_price = signal.price
            order_id = f"paper-{uuid.uuid4().hex[:10]}"

        position = Position(
            symbol=symbol,
            direction=signal.direction,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            entry_order_id=order_id,
            atr_at_entry=decision.atr,
            trail_atr_multiplier=decision.trail_atr_multiplier,
            best_price_since_entry=entry_price,
        )
        self.positions[symbol] = position

        self.logger.log_event("position_opened", {
            "symbol": symbol,
            "direction": signal.direction.value,
            "entry_price": entry_price,
            "quantity": quantity,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
            "order_id": order_id,
            "mode": self.trading_mode,
        })
        return position

    async def close_position(self, symbol: str, exit_price: float, reason: str) -> Position:
        position = self.positions[symbol]
        side = "sell" if position.direction == SignalDirection.LONG else "buy"

        if self.trading_mode == "live_trading":
            order = await self.client.create_market_order(symbol, side, position.quantity)
            exit_price = float(order.get("average") or order.get("price") or exit_price)

        position.exit_price = exit_price
        position.status = PositionStatus.CLOSED
        position.close_reason = reason

        self.logger.log_event("position_closed", {
            "symbol": symbol,
            "direction": position.direction.value,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "quantity": position.quantity,
            "pnl": position.pnl,
            "pnl_pct": position.pnl_pct,
            "reason": reason,
            "mode": self.trading_mode,
        })

        del self.positions[symbol]
        return position

    async def check_exit_conditions(
        self, symbol: str, current_price: float, on_close: Callable[[Position], None] | None = None
    ) -> Position | None:
        """Comprueba si el precio actual dispara el stop loss o el take profit
        de la posición abierta en `symbol`, y la cierra si corresponde."""
        position = self.positions.get(symbol)
        if position is None:
            return None

        trailing_moved = self._update_trailing_stop(position, current_price)

        hit_stop = (
            (position.direction == SignalDirection.LONG and current_price <= position.stop_loss)
            or (position.direction == SignalDirection.SHORT and current_price >= position.stop_loss)
        )
        hit_target = (
            (position.direction == SignalDirection.LONG and current_price >= position.take_profit)
            or (position.direction == SignalDirection.SHORT and current_price <= position.take_profit)
        )

        if not (hit_stop or hit_target):
            return None

        reason = "stop_loss" if hit_stop else "take_profit"
        closed = await self.close_position(symbol, current_price, reason)
        if on_close:
            on_close(closed)
        return closed

    def _update_trailing_stop(self, position: Position, current_price: float) -> bool:
        """Ratchet del stop loss a favor de la posición usando ATR de entrada.
        Devuelve True si el stop se movió. No dispara ninguna orden: solo
        prepara el nivel; el disparo lo decide check_exit_conditions."""
        if not position.trail_atr_multiplier or position.atr_at_entry is None:
            return False
        if self.risk_manager is None:
            return False

        if position.direction == SignalDirection.LONG:
            position.best_price_since_entry = max(position.best_price_since_entry or position.entry_price, current_price)
        else:
            position.best_price_since_entry = min(position.best_price_since_entry or position.entry_price, current_price)

        new_stop = self.risk_manager.update_trailing_stop(
            current_price=position.best_price_since_entry,
            entry_price=position.entry_price,
            direction=position.direction,
            stop_loss=position.stop_loss,
            atr=position.atr_at_entry,
            trail_multiplier=position.trail_atr_multiplier,
        )

        if abs(new_stop - position.stop_loss) > 1e-9:
            self.logger.log_event("trailing_stop_updated", {
                "symbol": position.symbol,
                "stop_loss_from": position.stop_loss,
                "stop_loss_to": new_stop,
                "best_price": position.best_price_since_entry,
            })
            position.stop_loss = new_stop
            return True
        return False
