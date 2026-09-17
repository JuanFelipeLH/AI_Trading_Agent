"""Tests OFFLINE de VeniceSignalProvider: parsing robusto del JSON del LLM y
fallback automático a otro proveedor. NO se usa la red ni créditos: se
monkeypatchea _call_venice."""
from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from signal_layer.base_provider import Signal, SignalDirection
from signal_layer.venice_provider import (
    VeniceSignalProvider,
    _extract_json_object,
    parse_decision,
)
from signal_layer.technical_provider import TechnicalAnalysisProvider

from tests.helpers import load_strategy_config, make_market_data


class ParseDecisionTest(unittest.TestCase):
    def test_json_limpio(self):
        self.assertEqual(parse_decision('{"action": "LONG", "confidence": 0.8, "reasoning": "x"}'),
                         {"action": "LONG", "confidence": 0.8, "reasoning": "x"})

    def test_con_markdown_fences(self):
        d = parse_decision('```json\n{"action": "SHORT", "confidence": 0.6, "reasoning": "down"}\n```')
        self.assertEqual(d["action"], "SHORT")
        self.assertEqual(d["confidence"], 0.6)

    def test_con_texto_alrededor(self):
        raw = 'Claro, aquí va mi análisis:\n{"action": "HOLD", "confidence": 0.2, "reasoning": "sin claridad"} eso es todo.'
        d = parse_decision(raw)
        self.assertEqual(d["action"], "HOLD")
        self.assertEqual(d["confidence"], 0.2)

    def test_confianza_fuera_de_rango_se_recorta(self):
        d = parse_decision('{"action": "LONG", "confidence": 5.0, "reasoning": ""}')
        self.assertEqual(d["confidence"], 1.0)

    def test_action_invalida_por_defecto_hold(self):
        d = parse_decision('{"action": "SIDEWAYS", "confidence": 0.9, "reasoning": ""}')
        self.assertEqual(d["action"], "HOLD")
        self.assertEqual(d["confidence"], 0.9)

    def test_confianza_no_numerica_cero(self):
        d = parse_decision('{"action": "LONG", "confidence": "alta", "reasoning": ""}')
        self.assertEqual(d["confidence"], 0.0)

    def test_sin_json_lanza_error(self):
        with self.assertRaises(ValueError):
            parse_decision("no hay objeto json aquí")

    def test_extract_json_anidado(self):
        content = 'texto {"action": "LONG", "reasoning": "a {b}"} resto'
        obj = _extract_json_object(content)
        self.assertTrue(obj.startswith("{"))
        self.assertTrue(obj.endswith("}"))


class VeniceFallbackTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_strategy_config()
        self.cfg["venice"]["context_file"] = str(Path(self.cfg["venice"]["context_file"])) or "logs/venice_context.json"
        self.technical = TechnicalAnalysisProvider(self.cfg)
        self.fallback_path = self.cfg["venice"]["context_file"]

    def test_fallback_si_api_falla(self):
        provider = VeniceSignalProvider(self.cfg, api_key="fake", technical_fallback=self.technical)
        calls = {"n": 0}

        async def _explota(prompt):
            calls["n"] += 1
            raise RuntimeError("API 402")

        provider._call_venice = _explota
        sig = asyncio.run(provider.generate_signal(make_market_data()))
        self.assertIsInstance(sig, Signal)
        self.assertEqual(sig.metadata.get("source"), "fallback_technical")
        self.assertGreater(calls["n"], 0)

    def test_fallback_personalizado_se_usa(self):
        class CustomFake:
            async def generate_signal(self, market_data):
                s = await self.technical.generate_signal(market_data)
                s.metadata["source"] = "custom_fallback"
                return s

        fake = CustomFake()
        fake.technical = self.technical
        provider = VeniceSignalProvider(self.cfg, api_key="fake", technical_fallback=self.technical,
                                        fallback_provider=fake)

        async def _explota(prompt):
            raise TimeoutError

        provider._call_venice = _explota
        sig = asyncio.run(provider.generate_signal(make_market_data()))
        self.assertEqual(sig.metadata.get("source"), "custom_fallback")

    def test_decision_respetada_sin_red(self):
        provider = VeniceSignalProvider(self.cfg, api_key="fake", technical_fallback=self.technical)

        async def _falsa(prompt):
            return '{"action": "LONG", "confidence": 0.9, "reasoning": "momento fuerte"}'

        provider._call_venice = _falsa
        sig = asyncio.run(provider.generate_signal(make_market_data()))
        self.assertEqual(sig.direction, SignalDirection.LONG)
        self.assertEqual(sig.metadata["source"], "venice")
        self.assertIn("regime", sig.metadata)
        self.assertIn("best_bid", sig.metadata)


if __name__ == "__main__":
    unittest.main()