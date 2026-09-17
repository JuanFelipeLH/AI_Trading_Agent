"""Carga y valida la configuración del bot combinando variables de entorno
(.env: credenciales y modo de ejecución), la configuración de estrategia
(YAML: parámetros de indicadores, riesgo y ejecución) y, si existe, una
sesión de campaña generada interactivamente por start_bot.py
(config/session_config.json), que sobreescribe símbolos/timeframe/riesgo/
frecuencia de consulta y añade la meta de rentabilidad/drawdown de la
campaña."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
SESSION_CONFIG_PATH = BASE_DIR / "config" / "session_config.json"

# Al cambiar el timeframe principal desde la configuración interactiva, la
# confirmación se reescala junto con él para conservar la relación
# "tendencia en timeframe grande, timing en uno más chico" del requisito
# original (1h principal / 15m confirmación).
CONFIRMATION_TF_MAP = {"1d": "4h", "4h": "1h", "1h": "15m", "30m": "5m", "15m": "5m"}

REQUIRED_SESSION_KEYS = (
    "campaign_name", "capital_inicial", "rentabilidad_target", "plazo_dias",
    "max_drawdown", "riesgo_por_trade", "activos", "timeframe_principal",
    "consulta_minutos", "started_at",
)


@dataclass
class Settings:
    api_key: str
    api_secret: str
    use_testnet: bool
    trading_mode: str          # paper_trading | live_trading
    initial_capital: float
    log_level: str
    venice_api_key: str
    strategy: dict[str, Any]
    campaign: dict[str, Any] | None = None

    @classmethod
    def load(cls, env_path: str | None = None, strategy_path: str | None = None,
              session_path: str | None = None) -> "Settings":
        load_dotenv(env_path or BASE_DIR / ".env")

        strategy_file = Path(strategy_path or BASE_DIR / "config" / "strategy_config.yaml")
        with strategy_file.open("r", encoding="utf-8") as f:
            strategy = yaml.safe_load(f)

        trading_mode = os.getenv("TRADING_MODE", "paper_trading")
        if trading_mode not in ("paper_trading", "live_trading"):
            raise ValueError(f"TRADING_MODE inválido: {trading_mode!r} (usa paper_trading o live_trading)")

        initial_capital = float(os.getenv("INITIAL_CAPITAL", "10000"))
        campaign: dict[str, Any] | None = None

        session_file = Path(session_path or SESSION_CONFIG_PATH)
        if session_file.exists():
            with session_file.open("r", encoding="utf-8") as f:
                session = json.load(f)
            missing = [k for k in REQUIRED_SESSION_KEYS if k not in session]
            if missing:
                raise ValueError(f"{session_file} incompleto, faltan claves: {missing}")

            _apply_session_overrides(strategy, session)
            initial_capital = float(session["capital_inicial"])
            campaign = {
                "name": session["campaign_name"],
                "target_pct": float(session["rentabilidad_target"]),
                "max_drawdown_pct": float(session["max_drawdown"]),
                "deadline_days": int(session["plazo_dias"]),
                "started_at": session["started_at"],
            }

        return cls(
            api_key=os.getenv("API_KEY", ""),
            api_secret=os.getenv("API_SECRET", ""),
            use_testnet=os.getenv("USE_TESTNET", "true").lower() == "true",
            trading_mode=trading_mode,
            initial_capital=initial_capital,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            venice_api_key=os.getenv("VENICE_API_KEY", ""),
            strategy=strategy,
            campaign=campaign,
        )


def _apply_session_overrides(strategy: dict[str, Any], session: dict[str, Any]) -> None:
    strategy["symbols"] = session["activos"]

    primary = session["timeframe_principal"]
    strategy["timeframes"]["primary"] = primary
    strategy["timeframes"]["confirmation"] = CONFIRMATION_TF_MAP.get(
        primary, strategy["timeframes"]["confirmation"]
    )

    strategy["risk"]["risk_per_trade_pct"] = float(session["riesgo_por_trade"]) / 100
    if "perdidas_consecutivas_max" in session:
        strategy["risk"]["max_consecutive_losses"] = int(session["perdidas_consecutivas_max"])

    strategy["execution"]["poll_interval_seconds"] = int(session["consulta_minutos"]) * 60
