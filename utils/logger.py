"""Logging estructurado de todas las decisiones del bot en JSONL, más
utilidades para generar un dashboard de consola y un reporte HTML/CSV."""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger("trading_bot")


class TradeLogger:
    """Escribe cada evento relevante (señal recibida, validación de riesgo,
    ejecución, resultado) como una línea JSON en un archivo .jsonl, y ofrece
    utilidades para convertir ese histórico en un dashboard o reporte.
    """

    def __init__(self, log_dir: str, filename: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / filename
        self._logger = logging.getLogger("trading_bot")

    def log_event(self, event_type: str, data: dict) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **data,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        summary = {k: v for k, v in data.items() if k in ("symbol", "direction", "reason", "pnl")}
        self._logger.info("%s: %s", event_type, summary)

    def read_events(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        with self.log_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def print_dashboard(self) -> None:
        events = self.read_events()
        closes = [e for e in events if e["event"] == "position_closed"]
        wins = [e for e in closes if e.get("pnl", 0) > 0]
        losses = [e for e in closes if e.get("pnl", 0) <= 0]
        total_pnl = sum(e.get("pnl", 0) for e in closes)
        win_rate = (len(wins) / len(closes) * 100) if closes else 0.0

        print("\n" + "=" * 50)
        print(" DASHBOARD - TRADING BOT")
        print("=" * 50)
        print(f" Trades cerrados:   {len(closes)}")
        print(f" Ganadores:         {len(wins)}")
        print(f" Perdedores:        {len(losses)}")
        print(f" Win rate:          {win_rate:.1f}%")
        print(f" PnL total:         {total_pnl:.2f} USDT")
        print("=" * 50 + "\n")

    def export_csv(self, output_path: str) -> None:
        events = [e for e in self.read_events() if e["event"] == "position_closed"]
        if not events:
            return
        fieldnames = sorted({key for e in events for key in e.keys()})
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events)

    def export_html(self, output_path: str) -> None:
        events = [e for e in self.read_events() if e["event"] == "position_closed"]
        rows = "".join(
            f"<tr><td>{e.get('timestamp', '')}</td><td>{e.get('symbol', '')}</td>"
            f"<td>{e.get('direction', '')}</td><td>{e.get('entry_price', '')}</td>"
            f"<td>{e.get('exit_price', '')}</td><td>{e.get('pnl', 0):.2f}</td>"
            f"<td>{e.get('reason', '')}</td></tr>"
            for e in events
        )
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Trading Bot Report</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
th {{ background: #222; color: #fff; }}
</style></head><body>
<h1>Trading Bot - Reporte de Operaciones</h1>
<table>
<tr><th>Fecha</th><th>Símbolo</th><th>Dirección</th><th>Entrada</th><th>Salida</th><th>PnL</th><th>Razón</th></tr>
{rows}
</table>
</body></html>"""
        Path(output_path).write_text(html, encoding="utf-8")
