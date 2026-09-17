"""Circuit breaker a nivel de campaña completa: detiene el bot cuando el
capital acumulado alcanza la meta de rentabilidad o el drawdown máximo
definidos al iniciar la sesión (ver start_bot.py / config/session_config.json).

Es independiente del circuit breaker por pérdidas consecutivas de
RiskManager — este opera sobre el equity total de la campaña, no sobre una
racha de trades."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class CampaignStatus:
    halted: bool
    reason: str | None
    equity: float
    pnl_pct: float
    target_reached: bool
    drawdown_breached: bool
    deadline_exceeded: bool


class CampaignTracker:
    def __init__(self, name: str, initial_capital: float, target_pct: float,
                 max_drawdown_pct: float, deadline_days: int, started_at: str):
        self.name = name
        self.initial_capital = initial_capital
        self.target_pct = target_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.deadline_days = deadline_days
        self.started_at = datetime.fromisoformat(started_at)

        self.halted = False
        self.reason: str | None = None

    def evaluate(self, current_equity: float) -> CampaignStatus:
        pnl_pct = (current_equity - self.initial_capital) / self.initial_capital * 100
        target_reached = pnl_pct >= self.target_pct
        drawdown_breached = pnl_pct <= -self.max_drawdown_pct
        days_elapsed = (datetime.now(timezone.utc) - self.started_at).total_seconds() / 86400
        deadline_exceeded = days_elapsed >= self.deadline_days

        if not self.halted:
            if target_reached:
                self.halted = True
                self.reason = f"meta de campaña alcanzada: {pnl_pct:.2f}% >= {self.target_pct}%"
            elif drawdown_breached:
                self.halted = True
                self.reason = f"drawdown máximo de campaña alcanzado: {pnl_pct:.2f}% <= -{self.max_drawdown_pct}%"

        return CampaignStatus(
            halted=self.halted,
            reason=self.reason,
            equity=current_equity,
            pnl_pct=pnl_pct,
            target_reached=target_reached,
            drawdown_breached=drawdown_breached,
            deadline_exceeded=deadline_exceeded,
        )
