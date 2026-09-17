"""Cliente de datos y órdenes contra Binance, con soporte para Testnet/Live
mediante configuración (`use_testnet`). Encapsula ccxt.async_support.binance
y añade reintentos automáticos ante fallos transitorios de red/API."""
from __future__ import annotations

import pandas as pd
import ccxt.async_support as ccxt
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from data_layer.models import MarketData

RETRYABLE_ERRORS = (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout)


def with_retries(max_retries: int, backoff_seconds: int):
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=backoff_seconds, min=backoff_seconds, max=30),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
    )


class BinanceClient:
    """Wrapper async sobre ccxt para Binance. Un único flag (`use_testnet`)
    decide si se opera contra Binance Spot Testnet o contra producción."""

    def __init__(self, api_key: str, api_secret: str, use_testnet: bool, market_type: str = "spot",
                 max_retries: int = 3, retry_backoff_seconds: int = 2):
        self.exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": market_type},
        })
        self.exchange.set_sandbox_mode(use_testnet)
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    async def connect(self) -> None:
        await self._call(self.exchange.load_markets)

    async def close(self) -> None:
        await self.exchange.close()

    async def _call(self, func, *args, **kwargs):
        decorated = with_retries(self.max_retries, self.retry_backoff_seconds)(func)
        return await decorated(*args, **kwargs)

    def market(self, symbol: str) -> dict:
        return self.exchange.markets[symbol]

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        return float(self.exchange.amount_to_precision(symbol, amount))

    def price_to_precision(self, symbol: str, price: float) -> float:
        return float(self.exchange.price_to_precision(symbol, price))

    async def fetch_ohlcv_df(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        raw = await self._call(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        return await self._call(self.exchange.fetch_order_book, symbol, limit)

    async def fetch_market_data(self, symbol: str, primary_tf: str, confirmation_tf: str,
                                 primary_candles: int, confirmation_candles: int) -> MarketData:
        primary_df = await self.fetch_ohlcv_df(symbol, primary_tf, primary_candles)
        confirm_df = await self.fetch_ohlcv_df(symbol, confirmation_tf, confirmation_candles)
        order_book = await self.fetch_order_book(symbol)
        return MarketData(
            symbol=symbol,
            ohlcv={primary_tf: primary_df, confirmation_tf: confirm_df},
            primary_timeframe=primary_tf,
            order_book=order_book,
        )

    async def fetch_last_price(self, symbol: str) -> float:
        ticker = await self._call(self.exchange.fetch_ticker, symbol)
        return float(ticker["last"])

    async def fetch_balance(self) -> dict:
        return await self._call(self.exchange.fetch_balance)

    async def create_market_order(self, symbol: str, side: str, amount: float) -> dict:
        return await self._call(self.exchange.create_order, symbol, "market", side, amount)

    async def create_limit_order(self, symbol: str, side: str, amount: float, price: float) -> dict:
        return await self._call(self.exchange.create_order, symbol, "limit", side, amount, price)

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        return await self._call(self.exchange.cancel_order, order_id, symbol)
