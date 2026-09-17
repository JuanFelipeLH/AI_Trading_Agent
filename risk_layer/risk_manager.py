"""Gestión de riesgo: dimensionamiento de posición, cálculo de stop
loss/take profit y circuit breaker por pérdidas consecutivas. Ninguna orden
debería llegar al exchange sin pasar antes por `RiskManager.validate_and_size`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from signal_layer.base_provider import Signal, SignalDirection


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    quantity: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_amount: float = 0.0
    atr: float | None = None
    trail_atr_multiplier: float | None = None


class RiskManager:
    def __init__(self, config: dict):
        cfg = config["risk"]
        self.risk_per_trade_pct = cfg["risk_per_trade_pct"]
        self.stop_loss_method = cfg["stop_loss_method"]           # atr | fixed_pct
        self.atr_multiplier = cfg["atr_multiplier"]
        self.fixed_stop_loss_pct = cfg["fixed_stop_loss_pct"]
        self.risk_reward_ratio = cfg["risk_reward_ratio"]
        self.max_position_pct_of_capital = cfg["max_position_pct_of_capital"]
        self.max_consecutive_losses = cfg["max_consecutive_losses"]
        self.trail_enabled = cfg.get("trail_enabled", True)
        self.trail_atr_multiplier = cfg.get("trail_atr_multiplier", 2.0)
        self.confidence_sizing = cfg.get("confidence_sizing", True)
        self.confidence_cap = cfg.get("confidence_cap", 1.5)

        self._recent_results: deque[str] = deque(maxlen=50)  # "win" | "loss"
        self.trading_halted = False

    # --- circuit breaker ----------------------------------------------------
    def record_trade_result(self, pnl: float) -> None:
        self._recent_results.append("win" if pnl > 0 else "loss")

        consecutive_losses = 0
        for result in reversed(self._recent_results):
            if result == "loss":
                consecutive_losses += 1
            else:
                break

        if consecutive_losses > self.max_consecutive_losses:
            self.trading_halted = True

    def reset_circuit_breaker(self) -> None:
        self.trading_halted = False
        self._recent_results.clear()

    # --- stop loss / take profit --------------------------------------------
    def calculate_stop_loss(self, entry_price: float, direction: SignalDirection, atr: float | None) -> float:
        if self.stop_loss_method == "atr" and atr:
            distance = atr * self.atr_multiplier
        else:
            distance = entry_price * self.fixed_stop_loss_pct

        if direction == SignalDirection.LONG:
            return entry_price - distance
        return entry_price + distance

    def calculate_take_profit(self, entry_price: float, stop_loss: float, direction: SignalDirection) -> float:
        risk = abs(entry_price - stop_loss)
        reward = risk * self.risk_reward_ratio
        if direction == SignalDirection.LONG:
            return entry_price + reward
        return entry_price - reward

    # --- trailing stop --------------------------------------------------------
    def update_trailing_stop(self, current_price: float, entry_price: float,
                             direction: SignalDirection, stop_loss: float,
                             atr: float | None, trail_multiplier: float | None) -> float:
        """Avanza el stop loss solo a favor de la posición (sin alejarlo nunca
        de la entrada). Devuelve el stop actualizado o el mismo si no aplica.
        Validación empírica: el trailing stop es una de las pocas salidas con
        win-rate ~100% en los postmortems de estrategias de ruptura."""
        if not self.trail_enabled or not trail_multiplier or trail_multiplier <= 0 or not atr or atr <= 0:
            return stop_loss

        distance = atr * trail_multiplier
        if direction == SignalDirection.LONG:
            # solo agarra ganancia si el precio subió al menos `distance` desde la entrada
            if current_price < entry_price + distance:
                return stop_loss
            candidate = current_price - distance
            return max(stop_loss, candidate)

        # SHORT: el precio debe bajar al menos `distance` para armar el trailing
        if current_price > entry_price - distance:
            return stop_loss
        candidate = current_price + distance
        return min(stop_loss, candidate)

    # --- position sizing -------------------------------------------------------
    def calculate_position_size(self, capital: float, entry_price: float, stop_loss: float) -> float:
        risk_amount = capital * self.risk_per_trade_pct
        per_unit_risk = abs(entry_price - stop_loss)
        if per_unit_risk <= 0:
            return 0.0
        quantity = risk_amount / per_unit_risk

        max_notional = capital * self.max_position_pct_of_capital
        if quantity * entry_price > max_notional:
            quantity = max_notional / entry_price

        return quantity

    def confidence_multiplier(self, signal: Signal) -> float:
        """Multiplicador de tamaño según la confianza de la señal. Usa el
        `size_multiplier` que el proveedor adaptativo ya calculó por régimen
        si existe; si no, deriva 1.0 de confianza máxima. Siempre en [0.25, cap]."""
        if not self.confidence_sizing:
            return 1.0
        hint = signal.metadata.get("size_multiplier")
        if hint is not None:
            return max(0.25, min(float(hint), self.confidence_cap))
        base = signal.confidence / 0.75 if signal.confidence > 0 else 0.0  # conf 0.75 -> 1.0
        return max(0.25, min(base, self.confidence_cap))

    # --- punto de entrada usado por execution_layer -----------------------------
    def validate_and_size(self, signal: Signal, capital: float, open_positions: int,
                           max_open_positions: int) -> RiskDecision:
        if self.trading_halted:
            return RiskDecision(False, "circuit breaker activo: demasiadas pérdidas consecutivas")

        if not signal.is_actionable:
            return RiskDecision(False, "señal no accionable (NONE)")

        if open_positions >= max_open_positions:
            return RiskDecision(False, "límite de posiciones abiertas alcanzado")

        atr = signal.metadata.get("atr")
        stop_loss = self.calculate_stop_loss(signal.price, signal.direction, atr)
        take_profit = self.calculate_take_profit(signal.price, stop_loss, signal.direction)
        quantity = self.calculate_position_size(capital, signal.price, stop_loss)
        quantity *= self.confidence_multiplier(signal)

        max_notional = capital * self.max_position_pct_of_capital
        if quantity * signal.price > max_notional:
            quantity = max_notional / signal.price

        if quantity <= 0:
            return RiskDecision(False, "tamaño de posición calculado es 0")

        risk_amount = capital * self.risk_per_trade_pct
        trail_mult = self.trail_atr_multiplier if self.trail_enabled else None

        return RiskDecision(
            approved=True,
            reason="aprobado",
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
            atr=float(atr) if atr else None,
            trail_atr_multiplier=trail_mult,
        )
