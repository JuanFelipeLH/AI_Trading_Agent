"""Proveedor de señales que delega la decisión LONG/SHORT/HOLD a un modelo de
lenguaje servido por Venice.ai, usando TechnicalAnalysisProvider como fuente
de indicadores (contexto del prompt) y como respaldo automático si Venice
falla, da timeout o responde algo no parseable.

IMPORTANTE: Venice sólo aporta dirección y confianza. El stop loss, take
profit y tamaño de posición siguen calculándose siempre en RiskManager (2%
de riesgo, ATR, ratio 1:2) exactamente igual que con cualquier otro
proveedor — así el circuit breaker y los límites de riesgo no dependen de
que el modelo responda algo sensato.

El modelo/endpoint de Venice se toman de config (`venice.model`,
`venice.api_url`); no están verificados aquí, confírmalos contra tu cuenta
antes de poner `venice.enabled: true`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import aiohttp
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from data_layer.models import MarketData
from signal_layer.base_provider import BaseSignalProvider, Signal, SignalDirection
from signal_layer.technical_provider import TechnicalAnalysisProvider

logger = logging.getLogger("trading_bot")

RETRYABLE_ERRORS = (aiohttp.ClientError, TimeoutError)

ACTION_TO_DIRECTION = {
    "LONG": SignalDirection.LONG,
    "SHORT": SignalDirection.SHORT,
    "HOLD": SignalDirection.NONE,
}


class VeniceSignalProvider(BaseSignalProvider):
    name = "venice"

    def __init__(self, config: dict, api_key: str, technical_fallback: TechnicalAnalysisProvider):
        cfg = config["venice"]
        self.api_key = api_key
        self.api_url = cfg["api_url"]
        self.model = cfg["model"]
        self.temperature = cfg.get("temperature", 0.2)
        self.max_tokens = cfg.get("max_tokens", 500)
        self.timeout_seconds = cfg.get("timeout_seconds", 30)
        self.max_retries = cfg.get("max_retries", 2)
        self.history_window = cfg.get("history_window", 10)
        self.min_confidence = config["signal"]["min_confidence"]

        self.technical = technical_fallback
        self.primary_tf = config["timeframes"]["primary"]
        self.confirmation_tf = config["timeframes"]["confirmation"]

        self.context_path = Path(cfg.get("context_file", "logs/venice_context.json"))
        self.context_path.parent.mkdir(parents=True, exist_ok=True)
        self._context = self._load_context()

    # --- contexto persistido (para compensar que el modelo no tiene memoria) ---
    def _load_context(self) -> dict:
        if self.context_path.exists():
            with self.context_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {"total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                "current_streak": 0, "recent_trades": []}

    def _save_context(self) -> None:
        with self.context_path.open("w", encoding="utf-8") as f:
            json.dump(self._context, f, indent=2, default=str)

    def on_trade_closed(self, symbol: str, pnl: float, metadata: dict[str, Any] | None = None) -> None:
        metadata = metadata or {}
        self._context["total_trades"] += 1
        if pnl > 0:
            self._context["winning_trades"] += 1
            self._context["current_streak"] = max(1, self._context["current_streak"] + 1)
        else:
            self._context["losing_trades"] += 1
            self._context["current_streak"] = min(-1, self._context["current_streak"] - 1)

        self._context["recent_trades"].append({"symbol": symbol, "pnl": pnl, **metadata})
        self._context["recent_trades"] = self._context["recent_trades"][-self.history_window:]
        self._save_context()

    # --- generación de señal -------------------------------------------------
    async def generate_signal(self, market_data: MarketData) -> Signal:
        try:
            return await self._generate_via_venice(market_data)
        except Exception:
            logger.exception(
                "Venice falló generando señal para %s, usando TechnicalAnalysisProvider como respaldo",
                market_data.symbol,
            )
            fallback = await self.technical.generate_signal(market_data)
            fallback.metadata["source"] = "fallback_technical"
            return fallback

    async def _generate_via_venice(self, market_data: MarketData) -> Signal:
        df_primary = market_data.candles(self.primary_tf)
        df_confirm = market_data.candles(self.confirmation_tf)
        primary = self.technical.analyze_timeframe(df_primary)
        confirm = self.technical.analyze_timeframe(df_confirm)
        price = float(df_primary["close"].iloc[-1])

        prompt = self._build_prompt(market_data.symbol, price, primary, confirm)
        decision = await self._call_venice(prompt)

        direction = ACTION_TO_DIRECTION.get(str(decision.get("action", "HOLD")).upper(), SignalDirection.NONE)
        confidence = float(decision.get("confidence", 0.0))
        if confidence < self.min_confidence:
            direction = SignalDirection.NONE

        metadata = {
            "atr": float(primary["atr"]),
            "trend": primary["trend"],
            "rsi_primary": float(primary["rsi"]),
            "rsi_confirmation": float(confirm["rsi"]),
            "macd_hist_primary": float(primary["macd_hist"]),
            "macd_hist_confirmation": float(confirm["macd_hist"]),
            "support": float(primary["support"]),
            "resistance": float(primary["resistance"]),
            "reasons": [decision.get("reasoning", "")],
            "source": "venice",
            "venice_raw": decision,
        }

        return Signal(
            symbol=market_data.symbol,
            direction=direction,
            confidence=confidence,
            price=price,
            provider=self.name,
            metadata=metadata,
        )

    def _build_prompt(self, symbol: str, price: float, primary: dict, confirm: dict) -> str:
        total = self._context["total_trades"]
        wins = self._context["winning_trades"]
        win_rate = (wins / total * 100) if total else 0.0
        recent = self._context["recent_trades"][-5:]
        recent_lines = "\n".join(
            f"  - {t['symbol']}: pnl={t.get('pnl', 0):.2f} razon_cierre={t.get('reason', 'n/a')}"
            for t in recent
        ) or "  (sin operaciones previas)"

        return f"""Eres el módulo de decisión de un bot de swing trading en {symbol}.
No tienes memoria entre llamadas: todo el contexto relevante está en este mensaje.

HISTORIAL (resumen):
- Trades totales: {total} | Ganadores: {wins} | Win rate: {win_rate:.1f}%
- Racha actual: {self._context['current_streak']}
- Últimos cierres:
{recent_lines}

MERCADO — {symbol}
Precio actual: {price:.2f}

Timeframe principal (1h):
- Tendencia: {primary['trend']}
- RSI(14): {primary['rsi']:.1f}
- MACD histograma: {primary['macd_hist']:.4f} (anterior: {primary['macd_hist_prev']:.4f})
- Soporte: {primary['support']:.2f} | Resistencia: {primary['resistance']:.2f}
- ATR(14): {primary['atr']:.2f}

Timeframe de confirmación (15m):
- RSI(14): {confirm['rsi']:.1f}
- MACD histograma: {confirm['macd_hist']:.4f}

No hay posición abierta en este símbolo (si la hubiera, no se te consultaría: las
salidas las gestiona automáticamente el sistema de riesgo vía stop loss/take profit).
El tamaño de posición, stop loss y take profit los calcula un módulo de riesgo aparte
en base al ATR — tú NO los definas, sólo evalúa dirección y confianza.

Responde ÚNICAMENTE con este JSON, sin texto adicional ni markdown:
{{"action": "LONG" | "SHORT" | "HOLD", "confidence": 0.0-1.0, "reasoning": "explicación breve"}}
"""

    async def _call_venice(self, prompt: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Eres un trader cuantitativo experto. Responde ÚNICAMENTE con el JSON solicitado."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        decorated = retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=2, min=2, max=15),
            retry=retry_if_exception_type(RETRYABLE_ERRORS),
        )(self._post)
        content = await decorated(payload, headers)

        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)

    async def _post(self, payload: dict, headers: dict) -> str:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.api_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    body = await response.text()
                    raise RuntimeError(f"Venice API error {response.status}: {body}")
                result = await response.json()
                return result["choices"][0]["message"]["content"]
