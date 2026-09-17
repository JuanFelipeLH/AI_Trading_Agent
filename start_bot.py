#!/usr/bin/env python3
"""Configuración interactiva de una campaña de trading: pregunta capital,
meta de rentabilidad y perfil de riesgo, guarda config/session_config.json
y arranca el bot con esa configuración (Settings.load() la recoge
automáticamente — ver config/settings.py).

El modo de ejecución es siempre automático: el proveedor de señales activo
decide, RiskManager valida y dimensiona (2% de riesgo, SL/TP), el bot
ejecuta. Modos semi-automático (aprobar cada trade a mano) o manual (solo
opinión, sin ejecutar) no están implementados.

Uso:
    python start_bot.py
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import questionary

from config.settings import BASE_DIR, Settings
from main import _run_bot

SESSION_CONFIG_PATH = BASE_DIR / "config" / "session_config.json"

RISK_PROFILES = {
    "conservador": {"meta": 10, "stop": 8, "riesgo": 1.0, "plazo": 30},
    "moderado": {"meta": 20, "stop": 15, "riesgo": 2.0, "plazo": 30},
    "agresivo": {"meta": 30, "stop": 20, "riesgo": 3.0, "plazo": 14},
}


def _preguntar_modo() -> str:
    return questionary.select(
        "¿Qué modo de configuración prefieres?",
        choices=[
            questionary.Choice("⚡ Rápido — perfil de riesgo predefinido", value="rapido"),
            questionary.Choice("🔧 Personalizado — configurar cada parámetro", value="personalizado"),
        ],
    ).ask()


def _configuracion_rapida() -> dict:
    print("\n--- Configuración rápida ---\n")
    capital = questionary.text(
        "¿Capital inicial en USDT?", default="1000",
        validate=lambda x: x.replace(".", "", 1).isdigit() and float(x) >= 100,
    ).ask()

    perfil = questionary.select(
        "Perfil de riesgo:",
        choices=[
            questionary.Choice("🟢 Conservador — meta +10%, drawdown máx -8%", value="conservador"),
            questionary.Choice("🟡 Moderado — meta +20%, drawdown máx -15%", value="moderado"),
            questionary.Choice("🔴 Agresivo — meta +30%, drawdown máx -20%", value="agresivo"),
        ],
    ).ask()

    p = RISK_PROFILES[perfil]
    return {
        "campaign_name": f"{perfil}_{datetime.now().strftime('%Y%m%d_%H%M')}",
        "capital_inicial": float(capital),
        "rentabilidad_target": p["meta"],
        "plazo_dias": p["plazo"],
        "max_drawdown": p["stop"],
        "riesgo_por_trade": p["riesgo"],
        "perdidas_consecutivas_max": 5,
        "activos": ["BTC/USDT", "ETH/USDT"],
        "timeframe_principal": "1h",
        "consulta_minutos": 60,
    }


def _configuracion_completa() -> dict:
    print("\n--- Configuración personalizada ---\n")
    config: dict = {}

    config["campaign_name"] = questionary.text(
        "Nombre de esta campaña:",
        default=f"campaign_{datetime.now().strftime('%Y%m%d_%H%M')}",
    ).ask()

    print("\n💰 CAPITAL")
    config["capital_inicial"] = float(questionary.text(
        "Capital inicial (USDT):", default="1000", validate=lambda x: float(x) >= 50,
    ).ask())

    print("\n🎯 META")
    config["rentabilidad_target"] = float(questionary.text(
        "Rentabilidad objetivo (%):", default="20", validate=lambda x: 0 < float(x) <= 200,
    ).ask())
    config["plazo_dias"] = int(questionary.text(
        "Plazo orientativo (días):", default="30", validate=lambda x: int(x) >= 1,
    ).ask())

    print("\n🛡️  RIESGO")
    config["max_drawdown"] = float(questionary.text(
        "Drawdown máximo de la campaña antes de detener el bot (%):",
        default="15", validate=lambda x: 0 < float(x) <= 80,
    ).ask())
    config["riesgo_por_trade"] = float(questionary.select(
        "Riesgo por trade:", choices=["0.5", "1.0", "2.0", "3.0", "5.0"], default="2.0",
    ).ask())
    config["perdidas_consecutivas_max"] = int(questionary.select(
        "Detener tras cuántas pérdidas consecutivas (circuit breaker por racha):",
        choices=["3", "4", "5", "7", "10"], default="5",
    ).ask())

    print("\n📊 MERCADO")
    config["activos"] = questionary.checkbox(
        "Activos a operar:",
        choices=[
            questionary.Choice("BTC/USDT", checked=True),
            questionary.Choice("ETH/USDT", checked=True),
            questionary.Choice("SOL/USDT"),
            questionary.Choice("BNB/USDT"),
        ],
        validate=lambda sel: len(sel) > 0 or "Selecciona al menos un activo",
    ).ask()
    config["timeframe_principal"] = questionary.select(
        "Timeframe principal (la confirmación se ajusta automáticamente, p.ej. 1h -> 15m):",
        choices=["15m", "30m", "1h", "4h", "1d"], default="1h",
    ).ask()

    print("\n🤖 CONSULTA")
    config["consulta_minutos"] = questionary.select(
        "¿Cada cuántos minutos evaluar el mercado?",
        choices=[
            questionary.Choice("15 min", value=15),
            questionary.Choice("30 min", value=30),
            questionary.Choice("60 min (recomendado)", value=60),
            questionary.Choice("4 horas", value=240),
        ],
    ).ask()

    return config


def _mostrar_resumen_y_confirmar(config: dict) -> bool:
    objetivo = config["capital_inicial"] * (1 + config["rentabilidad_target"] / 100)
    print("\n" + "=" * 60)
    print("  RESUMEN DE CONFIGURACIÓN")
    print("=" * 60)
    print(f"""
Campaña: {config['campaign_name']}

CAPITAL: {config['capital_inicial']} USDT
META: +{config['rentabilidad_target']}% (objetivo: {objetivo:.2f} USDT), plazo orientativo {config['plazo_dias']} días
PROTECCIÓN:
  - el bot se detiene solo si el equity cae -{config['max_drawdown']}% desde el capital inicial
  - riesgo {config['riesgo_por_trade']}% del capital por trade (RiskManager, ATR-based)
  - pausa tras {config.get('perdidas_consecutivas_max', 5)} pérdidas consecutivas seguidas
MERCADO: {', '.join(config['activos'])} en {config['timeframe_principal']}
CONSULTA: cada {config['consulta_minutos']} minutos
""")
    return questionary.confirm("¿Todo correcto? ¿Iniciar el bot ahora?", default=True).ask()


def _guardar_configuracion(config: dict) -> None:
    config["started_at"] = datetime.now(timezone.utc).isoformat()
    SESSION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SESSION_CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguración guardada en {SESSION_CONFIG_PATH}")


def run_interactive_setup() -> dict:
    print("\n" + "=" * 60)
    print("  TRADING BOT — CONFIGURACIÓN DE CAMPAÑA")
    print("=" * 60)
    print("\nEsto define capital, meta de rentabilidad y riesgo para esta sesión.")
    print("El modo de ejecución es siempre automático: el proveedor de señales decide,")
    print("RiskManager valida y dimensiona cada orden (2% de riesgo, SL/TP), el bot ejecuta.\n")

    modo = _preguntar_modo()
    config = _configuracion_rapida() if modo == "rapido" else _configuracion_completa()

    if not _mostrar_resumen_y_confirmar(config):
        print("Configuración cancelada.")
        raise SystemExit(0)

    _guardar_configuracion(config)
    return config


def main() -> None:
    run_interactive_setup()
    settings = Settings.load()
    print("🚀 Iniciando bot...\n")
    asyncio.run(_run_bot(settings))


if __name__ == "__main__":
    main()
