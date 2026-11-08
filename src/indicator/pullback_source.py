"""Pool pequeno de conexoes publicas para observar todos os indices a tempo."""

from __future__ import annotations

from indicator.deriv import PublicDeriv


class PullbackSource:
    """Distribui consultas entre quatro WebSockets sem compartilhar recv concorrente."""

    def __init__(self, url: str, size: int = 4) -> None:
        self.clients = [PublicDeriv(url) for _ in range(max(1, size))]
        self.next_client = 0

    async def symbols(self) -> list[dict[str, object]]:
        """Busca catalogo na primeira conexao."""
        return await self.clients[0].symbols()

    async def candles(
        self, symbol: str, count: int = 1000, end: str | int = "latest", period: int = 300
    ) -> list[dict[str, object]]:
        """Distribui consulta de OHLC em rotacao deterministica no ciclo."""
        client = self.clients[self.next_client % len(self.clients)]
        self.next_client += 1
        return await client.candles(symbol, count=count, end=end, period=period)

    async def close(self) -> None:
        """Fecha todas as conexoes publicas."""
        for client in self.clients:
            await client.close()
