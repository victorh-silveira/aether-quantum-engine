"""Contratos Protocol das ports outbound."""

from src.application.ports.outbound import MarketCandlePort, ModelArtifactPort, SettlementQueuePort
from src.domain.models.market_data import Candle


def test_runtime_checkable_ports():
    class FakeMarket:
        async def get_latest_candle(self, symbol: str, granularity: int) -> Candle | None:
            _ = (symbol, granularity)
            return None

        async def stream_candles(self, symbol: str, granularity: int):
            _ = (symbol, granularity)
            if False:
                yield None

    class FakeSettle:
        async def enqueue(self, contract_id: str, score: float, payload):
            _ = (contract_id, score, payload)
            return True

        async def pop_due(self, score: float, limit: int = 32):
            _ = (score, limit)
            return []

    class FakeModel:
        async def load_bytes(self, key: str):
            _ = key

        async def put_bytes(self, key: str, data: bytes) -> None:
            _ = (key, data)

    assert isinstance(FakeMarket(), MarketCandlePort)
    assert isinstance(FakeSettle(), SettlementQueuePort)
    assert isinstance(FakeModel(), ModelArtifactPort)
