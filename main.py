"""Punto de entrada del bot. Orquesta las 4 capas en un loop asíncrono:

    data_layer -> signal_layer -> risk_layer -> execution_layer

con logging completo de cada decisión (señal, validación de riesgo,
ejecución, resultado) en logs/trades.jsonl.

Uso:
    python main.py            # arranca el bot (modo definido en .env)
    python main.py --report   # genera report.csv / report.html a partir
                               # de los logs existentes y termina
"""
from __future__ import annotations

import argparse
import asyncio
import signal as os_signal

from config.settings import Settings
from data_layer.binance_client import BinanceClient
from execution_layer.models import Position
from execution_layer.order_manager import OrderManager, OrderValidationError
from risk_layer.campaign_tracker import CampaignTracker
from risk_layer.risk_manager import RiskManager
from signal_layer.base_provider import BaseSignalProvider, SignalDirection
from signal_layer.technical_provider import TechnicalAnalysisProvider
from utils.logger import TradeLogger, setup_logging


class TradingBot:
    """Orquesta el ciclo data -> signal -> risk -> execution para cada símbolo
    configurado, en un loop continuo."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.strategy = settings.strategy
        self.logger = setup_logging(settings.log_level)

        self.client = BinanceClient(
            api_key=settings.api_key,
            api_secret=settings.api_secret,
            use_testnet=settings.use_testnet,
            market_type=self.strategy["exchange"]["market_type"],
            max_retries=self.strategy["api"]["max_retries"],
            retry_backoff_seconds=self.strategy["api"]["retry_backoff_seconds"],
        )
        self.signal_provider: BaseSignalProvider = self._build_signal_provider()
        self.risk_manager = RiskManager(self.strategy)
        self.trade_logger = TradeLogger(
            log_dir=self.strategy["logging"]["log_dir"],
            filename=self.strategy["logging"]["trades_log_file"],
        )
        self.order_manager = OrderManager(self.client, self.trade_logger, settings.trading_mode)

        self.capital = settings.initial_capital
        self.campaign = self._build_campaign_tracker()
        self._running = False

    def _build_campaign_tracker(self) -> CampaignTracker | None:
        """Si el bot se arrancó vía start_bot.py (config/session_config.json
        presente), esto detiene el bot al alcanzar la meta de rentabilidad o
        el drawdown máximo de la campaña. Sin sesión interactiva, no hay
        campaña y el bot corre indefinidamente como antes."""
        if not self.settings.campaign:
            return None
        c = self.settings.campaign
        return CampaignTracker(
            name=c["name"],
            initial_capital=self.capital,
            target_pct=c["target_pct"],
            max_drawdown_pct=c["max_drawdown_pct"],
            deadline_days=c["deadline_days"],
            started_at=c["started_at"],
        )

    def _build_signal_provider(self) -> BaseSignalProvider:
        """Instancia el proveedor de señales activo. TechnicalAnalysisProvider
        es siempre la implementación por defecto; si `venice.enabled` está en
        true en la config, se usa VeniceSignalProvider (que a su vez usa
        TechnicalAnalysisProvider como fuente de indicadores y como respaldo
        automático si la API de Venice falla)."""
        technical = TechnicalAnalysisProvider(self.strategy)
        if not self.strategy.get("venice", {}).get("enabled", False):
            return technical

        from signal_layer.venice_provider import VeniceSignalProvider  # import diferido: opcional

        if not self.settings.venice_api_key:
            raise ValueError("venice.enabled=true pero falta VENICE_API_KEY en .env")
        return VeniceSignalProvider(self.strategy, self.settings.venice_api_key, technical)

    async def start(self) -> None:
        await self.client.connect()
        self._running = True
        self.logger.info(
            "Bot iniciado | modo=%s | testnet=%s | símbolos=%s",
            self.settings.trading_mode, self.settings.use_testnet, self.strategy["symbols"],
        )
        if self.campaign is not None:
            self.logger.info(
                "Campaña '%s' | capital inicial=%.2f | meta=+%.1f%% | drawdown máx=-%.1f%% | plazo=%d días",
                self.campaign.name, self.campaign.initial_capital, self.campaign.target_pct,
                self.campaign.max_drawdown_pct, self.campaign.deadline_days,
            )

        try:
            while self._running:
                await self._run_cycle()
                await asyncio.sleep(self.strategy["execution"]["poll_interval_seconds"])
        finally:
            await self.client.close()

    def stop(self) -> None:
        self.logger.info("Deteniendo bot...")
        self._running = False

    def _on_position_closed(self, position: Position) -> None:
        self.capital += position.pnl
        self.risk_manager.record_trade_result(position.pnl)
        self.signal_provider.on_trade_closed(
            position.symbol, position.pnl,
            {"reason": position.close_reason, "direction": position.direction.value, "pnl_pct": position.pnl_pct},
        )

        if self.campaign is not None:
            status = self.campaign.evaluate(self.capital)
            self.trade_logger.log_event("campaign_status", {
                "campaign": self.campaign.name,
                "equity": status.equity,
                "pnl_pct": status.pnl_pct,
                "target_reached": status.target_reached,
                "drawdown_breached": status.drawdown_breached,
                "deadline_exceeded": status.deadline_exceeded,
                "halted": status.halted,
            })
            if status.halted:
                self.logger.info("Campaña '%s' detenida: %s", self.campaign.name, status.reason)
                self.stop()

    async def _run_cycle(self) -> None:
        for symbol in self.strategy["symbols"]:
            try:
                await self._process_symbol(symbol)
            except Exception:
                self.logger.exception("Error procesando %s", symbol)

        self.trade_logger.print_dashboard()

    async def _process_symbol(self, symbol: str) -> None:
        tf = self.strategy["timeframes"]
        market_data = await self.client.fetch_market_data(
            symbol, tf["primary"], tf["confirmation"], tf["primary_candles"], tf["confirmation_candles"],
        )
        current_price = market_data.last_price

        # 1. Gestionar salidas de posiciones abiertas antes de buscar nuevas entradas
        closed = await self.order_manager.check_exit_conditions(
            symbol, current_price, on_close=self._on_position_closed,
        )
        if closed is not None:
            return

        if symbol in self.order_manager.positions:
            return  # ya hay una posición abierta en este símbolo

        # 2. Generar señal
        signal = await self.signal_provider.generate_signal(market_data)
        self.trade_logger.log_event("signal_generated", {
            "symbol": symbol,
            "direction": signal.direction.value,
            "confidence": signal.confidence,
            "price": signal.price,
            "provider": signal.provider,
            "metadata": signal.metadata,
        })

        if signal.direction == SignalDirection.NONE:
            return

        # 3. Validar riesgo y dimensionar la posición
        decision = self.risk_manager.validate_and_size(
            signal, self.capital, self.order_manager.open_positions_count,
            self.strategy["execution"]["max_open_positions"],
        )
        self.trade_logger.log_event("risk_validation", {
            "symbol": symbol,
            "approved": decision.approved,
            "reason": decision.reason,
            "quantity": decision.quantity,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
        })

        if not decision.approved:
            return

        # 4. Ejecutar
        try:
            await self.order_manager.open_position(signal, decision)
        except OrderValidationError as exc:
            self.trade_logger.log_event("order_rejected", {"symbol": symbol, "reason": str(exc)})


def _generate_report(settings: Settings) -> None:
    trade_logger = TradeLogger(
        log_dir=settings.strategy["logging"]["log_dir"],
        filename=settings.strategy["logging"]["trades_log_file"],
    )
    trade_logger.print_dashboard()
    trade_logger.export_csv("report.csv")
    trade_logger.export_html("report.html")
    print("Reporte generado: report.csv, report.html")


async def _run_bot(settings: Settings) -> None:
    bot = TradingBot(settings)

    loop = asyncio.get_running_loop()
    for sig in (os_signal.SIGINT, os_signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, bot.stop)
        except NotImplementedError:
            pass  # no soportado en algunas plataformas (p.ej. Windows)

    await bot.start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading bot BTC/USDT & ETH/USDT sobre Binance")
    parser.add_argument("--report", action="store_true",
                         help="Genera report.csv/report.html a partir de logs existentes y termina")
    args = parser.parse_args()

    settings = Settings.load()

    if args.report:
        _generate_report(settings)
        return

    asyncio.run(_run_bot(settings))


if __name__ == "__main__":
    main()
