"""Gerencia conexões WebSocket assíncronas e ciclos de vida de subscrição."""

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

import websockets


class WebSocketManager:
    """Orquestra a conexão WebSocket com a API da Deriv.

    Fornece uma interface robusta para enviar solicitações, gerenciar subscrições,
    ping automático e lidar com respostas assíncronas via futures.
    """

    def __init__(self, uri: str, ping_interval: int = 15, request_timeout: int = 30):
        """Inicializa o gerenciador com a URI e os parâmetros de conexão.

        Args:
            uri (str): A URI completa do WebSocket.
            ping_interval (int): Segundos entre pings de manutenção de conexão.
            request_timeout (int): Tempo limite padrão para solicitações em segundos.
        """
        self.uri = uri
        self.ping_interval = ping_interval
        self.request_timeout = request_timeout
        self.ws: websockets.WebSocketClientProtocol | None = None
        self.req_id_counter = 0
        self.callbacks: dict[int, asyncio.Future] = {}
        self.subscriptions: dict[str, Callable] = {}
        self.is_running = False
        self.logger = logging.getLogger("AETH")

    async def connect(self, uri: str | None = None):
        """Estabelece a conexão WebSocket e inicia as tarefas de segundo plano."""
        if uri:
            self.uri = uri
        if not self.uri:
            raise ConnectionError("WSS: URI nao definida.")
        self.ws = await websockets.connect(self.uri)
        self.is_running = True
        self.logger.debug("WSS: Conexão estabelecida com sucesso.")
        asyncio.create_task(self._listen())
        asyncio.create_task(self._ping_loop())

    async def close(self):
        """Encerra a conexão WebSocket de forma graciosa."""
        self.is_running = False
        if self.ws:
            await self.ws.close()
            self.logger.debug("WSS: Conexão encerrada.")

    async def _listen(self):
        """Loop interno que escuta as mensagens WebSocket recebidas e as roteia."""
        try:
            async for message in self.ws:
                data = json.loads(message)
                msg_type = data.get("msg_type")
                if not msg_type:
                    if "ohlc" in data:
                        msg_type = "ohlc"
                    if "proposal_open_contract" in data:
                        msg_type = "proposal_open_contract"

                if msg_type != "ping":
                    self.logger.debug(f"WSS: RECV: {msg_type} | {message[:100]}...")

                req_id = data.get("req_id")

                if req_id in self.callbacks:
                    future = self.callbacks.pop(req_id)
                    if not future.done():
                        future.set_result(data)

                if msg_type in self.subscriptions:

                    async def safe_callback(d, m=msg_type):
                        """Executa o callback de forma segura capturando exceções."""
                        try:
                            await self.subscriptions[m](d)
                        except Exception as e:
                            self.logger.error(f"WSS: Erro no callback {m}: {e}")

                    asyncio.create_task(safe_callback(data))
        except websockets.ConnectionClosed:
            self.logger.debug("WSS: Conexão fechada pelo broker.")
            self.is_running = False
        except Exception as e:
            self.logger.error(f"WSS: Erro inesperado: {e}")
            self.is_running = False

    async def _ping_loop(self):
        """Loop interno que envia pings periódicos para manter a conexão ativa com timeout estrito."""
        try:
            while self.is_running:
                await asyncio.sleep(self.ping_interval)
                if self.is_running and self.ws:
                    try:
                        await self.send({"ping": 1}, timeout=5)
                    except Exception:
                        self.logger.debug("WSS: Ping falhou silenciosamente. Tentando reconectar...")
                        self.is_running = False
                        if self.ws:
                            await self.ws.close()
        except Exception as e:
            self.logger.error(f"WSS: Falha crítica no loop de ping: {e}")
            self.is_running = False
            if self.ws:
                asyncio.create_task(self.ws.close())

    async def send(self, request: dict[str, Any], timeout: int | None = None) -> dict[str, Any]:
        """Envia uma solicitação JSON e aguarda sua resposta correspondente.

        Args:
            request (Dict[str, Any]): O payload a ser enviado.
            timeout (int): Tempo máximo para aguardar uma resposta (substitui o padrão).

        Returns:
            Dict[str, Any]: A resposta do servidor.

        Raises:
            asyncio.TimeoutError: Se o servidor não responder a tempo.
        """
        if not self.ws or not self.is_running:
            raise ConnectionError("WSS: Impossível enviar comando. WebSocket não está conectado.")

        self.req_id_counter += 1
        request["req_id"] = self.req_id_counter
        actual_timeout = timeout or self.request_timeout
        cmd = next((k for k in request if k != "req_id"), "unknown")
        future = asyncio.get_event_loop().create_future()
        self.callbacks[self.req_id_counter] = future

        await self.ws.send(json.dumps(request))
        try:
            return await asyncio.wait_for(future, timeout=actual_timeout)
        except TimeoutError:
            if cmd != "ping":
                self.logger.debug(f"WSS: Timeout na requisição {self.req_id_counter} ({cmd}) após {actual_timeout}s")
            self.callbacks.pop(self.req_id_counter, None)
            raise

    def subscribe(self, msg_type: str, callback: Callable):
        """Registra um callback para tipos específicos de mensagens do fluxo.

        Args:
            msg_type (str): O valor do campo 'msg_type' para filtrar.
            callback (Callable): Função assíncrona a ser chamada com os dados da mensagem.
        """
        self.subscriptions[msg_type] = callback
