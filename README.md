# Trading Bot — BTC/USDT & ETH/USDT (Binance)

Bot de trading algorítmico con arquitectura por capas (datos, señales, riesgo,
ejecución), pensado para correr contra **Binance Spot Testnet** y, mediante
configuración, también contra producción.

## Arquitectura

```
trading_bot/
├── main.py                          # orquestador: data -> signal -> risk -> execution
├── config/
│   ├── settings.py                  # carga .env + strategy_config.yaml
│   └── strategy_config.yaml         # símbolos, timeframes, indicadores, riesgo, ejecución
├── data_layer/
│   ├── binance_client.py            # ccxt async, testnet/live, reintentos
│   └── models.py                    # MarketData
├── signal_layer/
│   ├── base_provider.py             # BaseSignalProvider (contrato) + Signal
│   ├── regime.py                    # detección de régimen de mercado (ADX/ATR)
│   ├── technical_provider.py        # TechnicalAnalysisProvider (RSI+MACD+EMA+S/R)
│   ├── donchian_provider.py         # ruptura Donchian + filtro EMA (validado)
│   ├── composite_provider.py        # elige estrategia según el régimen en cada ciclo
│   └── venice_provider.py           # decisión LLM con contexto real-time + fallback
├── risk_layer/
│   ├── risk_manager.py              # position sizing, SL/TP, trailing, circuit breaker
│   └── campaign_tracker.py          # circuit breaker a nivel de campaña
├── execution_layer/
│   ├── order_manager.py             # apertura/cierre de posiciones + trailing stop
│   └── models.py                    # Position
├── backtest/
│   ├── data.py                      # generador de datos sintéticos OFFLINE
│   └── engine.py                    # backtester sin red ni API
└── utils/
    └── logger.py                    # log JSONL + dashboard + reporte CSV/HTML
```

**Extensibilidad de señales**: cualquier estrategia nueva (API externa,
modelo de ML, sentiment, etc.) se añade creando una clase que hereda de
`BaseSignalProvider` e implementa `generate_signal(market_data) -> Signal`.
Ni `risk_layer` ni `execution_layer` necesitan cambios.

## Estrategia (adaptativa por régimen)

El bot NO espera pasivamente a que "todas las condiciones de una señal se
cumplan": en cada ciclo detecta el **régimen de mercado** (ADX/ATR) y adapta
la decisión en ese momento:

- **Mercado en tendencia** → estrategia Donchian breakout (ruptura de
  extremos de N velas + filtro EMA200, validada con walk-forward: +1,107% /
  +83% APY en BTC 4H) — primada por el compuesto.
- **Mercado en rango** → scoring técnico clásico (tendencia EMA + RSI/MACD +
  soporte/resistencia) con banda neutra (no fuerza dirección si las señales
  se contradicen).
- **Ritmo dinámico**: `execution.dynamic_poll` re-evalúa más rápido cuando el
  mercado está volátil o en tendencia y más lento cuando está tranquilo.
- Stop loss configurable: **ATR** (`atr_multiplier x ATR14`) o **porcentaje fijo**.
- Take profit con ratio riesgo:beneficio (**1:2** por defecto).
- **Trailing stop** (`risk.trail_enabled`): avanza el SL a favor de la
  posición tras superar `trail_atr_multiplier x ATR` — validación empírica:
  las salidas por trailing/TP son de las pocas con win-rate alto en los
  postmortems públicos; salir por "señal opuesta" es el patrón que más dinero
  pierde.
- **Sizing dinámico**: el tamaño de la posición se escala con la confianza de
  la señal (`risk.confidence_sizing`), mayor en tendencia, menor en rango,
  siempre dentro del tope de exposición.
- **Circuit breaker por racha**: el bot deja de operar tras superar
  `risk.max_consecutive_losses` pérdidas seguidas.
- Todos los parámetros están en [config/strategy_config.yaml](config/strategy_config.yaml).

## Instalación

```bash
cd trading_bot
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edita .env con tus credenciales de Binance Testnet:
# https://testnet.binance.vision/ -> genera API_KEY / API_SECRET
```

`.env`:

```
API_KEY=tu_api_key_de_testnet
API_SECRET=tu_api_secret_de_testnet
USE_TESTNET=true
TRADING_MODE=paper_trading   # paper_trading (simula) | live_trading (envía órdenes reales)
INITIAL_CAPITAL=10000
LOG_LEVEL=INFO
```

## Uso

### Opción A: configuración interactiva por campaña (recomendado)

```bash
python start_bot.py
```

Pregunta capital inicial, meta de rentabilidad, perfil de riesgo, activos y
timeframe; guarda la respuesta en `config/session_config.json` y arranca el
bot con esa configuración. Esa sesión:

- Sobreescribe símbolos, timeframe principal (la confirmación se ajusta sola,
  p.ej. `1h` -> `15m`), riesgo por trade y frecuencia de consulta de
  `strategy_config.yaml` para esta campaña.
- Activa un `CampaignTracker` (`risk_layer/campaign_tracker.py`): un circuit
  breaker a nivel de campaña completa, independiente del circuit breaker por
  pérdidas consecutivas de `RiskManager`. El bot **deja de abrir posiciones
  automáticamente** en cuanto el equity (capital inicial + PnL acumulado)
  alcanza la meta de rentabilidad o cae por debajo del drawdown máximo
  definido.
- El modo de ejecución es siempre automático: el proveedor de señales
  decide, `RiskManager` valida/dimensiona cada orden, el bot ejecuta. No hay
  modos semi-automático (aprobar cada trade a mano) ni manual (solo opinión)
  — quedan fuera de alcance por ahora.
- Las "notificaciones" son las que ya existen: `logs/trades.jsonl` +
  dashboard en consola en cada ciclo; no hay integración con Telegram/email.

Para repetir la misma campaña en un reinicio, vuelve a correr `python main.py`
directamente: si `config/session_config.json` existe, se reutiliza
automáticamente. Bórralo (o corre `start_bot.py` de nuevo) para volver al
comportamiento sin campaña (corre indefinidamente, capital fijo desde `.env`).

### Opción B: arranque directo (sin meta de campaña)

```bash
python main.py
```

El bot:
1. Descarga velas 1h/15m de BTC/USDT y ETH/USDT en cada ciclo (`execution.poll_interval_seconds`).
2. Genera una señal con `TechnicalAnalysisProvider`.
3. La valida y dimensiona con `RiskManager` (2% de riesgo, SL/TP calculados).
4. Si se aprueba, abre la posición (simulada en `paper_trading`, real en `live_trading`).
5. En cada ciclo revisa si el precio tocó el stop loss o el take profit de las posiciones abiertas.
6. Al cerrar una posición, suma el PnL al capital (`self.capital`) — el position sizing del
   siguiente trade ya refleja ganancias/pérdidas acumuladas, no queda fijo en `INITIAL_CAPITAL`.
7. Registra cada paso en `logs/trades.jsonl` y muestra un dashboard en consola.

Generar un reporte CSV/HTML a partir del histórico de operaciones (sin arrancar el bot):

```bash
python main.py --report
```

Esto produce `report.csv` y `report.html` con todas las operaciones cerradas.

Validar la configuración actual del bot con datos sintéticos (sin red, sin
API, sin créditos de LLM):

```bash
python main.py --backtest
```

Ejecuta el pipeline real (señal adaptativa -> riesgo -> SL/TP/trailing) sobre
varias series sintéticas y muestra rentabilidad, win rate, profit factor, max
drawdown y Sharpe. Sirve de *smoke test* de robustez ANTES de activar
testnet/producción.

## Tests (100% offline)

No usan la red de Binance ni la API de Venice (ni créditos):

```bash
venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

Cubren: detección de régimen, los tres proveedores (técnico, Donchian,
compuesto), RiskManager (sizing, circuit breaker, trailing), OrderManager
(cierre SL/TP y trailing), parsing robusto del JSON del LLM y su fallback
(monkeypatch de la llamada), el backtester y el flujo completo de `main.py`
con un cliente fake.

## Proveedor de señales opcional: Venice.ai (LLM)

Además de `TechnicalAnalysisProvider`, existe `signal_layer/venice_provider.py`:
un `VeniceSignalProvider` que le pide a un modelo servido por Venice.ai que
decida LONG/SHORT/HOLD, usando los indicadores técnicos (RSI, MACD, EMA,
soporte/resistencia) como contexto del prompt.

- **Desactivado por defecto** (`venice.enabled: false` en `strategy_config.yaml`).
- Venice solo aporta **dirección y confianza**; el stop loss, take profit,
  trailing y tamaño de posición los sigue calculando siempre `RiskManager`
  (riesgo configurado, ATR, ratio, trailing) — igual que con cualquier otro
  proveedor.
- Venice recibe contexto del **momento exacto**: régimen, ADX, volatilidad
  (ATR% del precio), momentum y order book (bid/ask/spread) en el prompt de
  cada ciclo.
- Si la llamada a Venice falla, da timeout o responde algo no parseable, el
  bot cae automáticamente al **proveedor compuesto adaptativo** para ese
  ciclo.
- Mantiene un contexto persistido (`venice.context_file`, por defecto
  `logs/venice_context.json`) con el historial de resultados, ya que el
  modelo no tiene memoria entre llamadas: cada prompt incluye win rate,
  racha actual y los últimos cierres.

Para activarlo:

```yaml
# config/strategy_config.yaml
venice:
  enabled: true
  model: openai-gpt-6-astra   # ya verificado contra GET /v1/models con esta cuenta
```

```bash
# .env
VENICE_API_KEY=tu_api_key_de_venice
```

Con la cadencia de `start_bot.py` ("60 min", ~48 llamadas/día entre BTC/USDT
y ETH/USDT sin posición abierta) `openai-gpt-6-astra` cuesta ~$11.6/mes
($10/$50 por millón de tokens entrada/salida) — asumible dado el propósito.
Si en algún momento quieres bajar el gasto sin perder razonamiento, cambia
`model` a `kimi-k2-5` (~$0.56/$3.50, ~$0.72/mes a esa misma cadencia) o
`deepseek-v4-flash` (~$0.14/$0.275, aún más barato).

**Usa siempre `start_bot.py` para arrancar el bot con Venice activo.** Si
en cambio corrés `python main.py` directamente con `venice.enabled: true`
pero sin haber generado antes `config/session_config.json`, se aplica el
default genérico `execution.poll_interval_seconds: 60` (60 **segundos**, no
minutos) — a esa cadencia serían ~2,880 llamadas/día y el costo se dispara a
cientos de dólares al mes con cualquier modelo.

**Importante**: la cuenta usada en las pruebas devolvió `402 Insufficient
USD or Diem balance` al llamar al modelo — la key es válida y la conexión
funciona, pero necesita crédito cargado en
[venice.ai/settings/api](https://venice.ai/settings/api) antes de que
cualquier modelo responda. Hasta entonces, `VeniceSignalProvider` seguirá
cayendo automáticamente al proveedor compuesto adaptativo en cada ciclo (así
lo verás en `logs/trades.jsonl` con el proveedor de respaldo, o
`"source": "fallback_technical"` en caso de que el respaldo sea el técnico).

## Pasar a producción

1. Genera API keys reales en Binance (con permisos de trading, sin retiros).
2. En `.env`: `USE_TESTNET=false`.
3. Cuando estés seguro de la estrategia: `TRADING_MODE=live_trading`.
4. Revisa `config/strategy_config.yaml` (símbolos, riesgo, `max_open_positions`, etc.) antes de operar con fondos reales.

## Notas de diseño

- El stop loss / take profit / trailing stop se gestionan por *polling*
  (comparando el precio de cada ciclo contra los niveles calculados), no con
  una OCO nativa del exchange, para mantener el código simple y portable. Para
  uso intensivo en producción se recomienda migrar a órdenes OCO/bracket
  nativas de Binance.
- El polling es **dinámico**: el bot acelera la consulta en mercados volátiles
  o con tendencia y la ralentiza cuando está tranquilo (tiempo muerto en
  mercado vivo, no "esperamos a que pase algo").
- Los reintentos ante errores de red/API usan backoff exponencial (`tenacity`),
  configurables en `api.max_retries` / `api.retry_backoff_seconds`.
# AI_Trading_Agent
