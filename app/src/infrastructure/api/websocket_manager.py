"""Gerencia conexoes WebSocket assincronas e ciclos de vida de subscricao."""

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import websockets

from src.infrastructure.api.websocket_connect import connect_wss_with_ip_failover


class WebSocketManager:
    """Orquestra a conexao WebSocket com a API da Deriv.

    Fornece uma interface robusta para enviar solicitacoes, gerenciar subscricoes,
    ping automatico e lidar com respostas assincronas via futures.
    """

    def __init__(self, uri: str, ping_interval: int = 15, request_timeout: int = 30):
        """Inicializa o gerenciador com a URI e os parametros de conexao.

        Args:
            uri (str): A URI completa do WebSocket.
            ping_interval (int): Segundos entre pings de manutencao de conexao.
            request_timeout (int): Tempo limite padrao para solicitacoes em segundos.
        """
        self.uri = uri
        self.ping_interval = ping_interval
        self.request_timeout = request_timeout
        self.ws: websockets.WebSocketClientProtocol | None = None
        self.req_id_counter = 0
        self.callbacks: dict[int, asyncio.Future] = {}
        self.subscriptions: dict[str, Callable] = {}
        self.is_running = False
        self.last_rtt_seconds = 0.0
        self.logger = logging.getLogger("AETH")
        self._connect_in_progress = False

    async def connect(
        self,
        uri: str | None = None,
        *,
        max_attempts: int = 1,
        open_timeout: float = 20.0,
        retry_delay: float = 3.0,
        retry_backoff: float = 1.5,
        uri_factory=None,
    ):
        """Estabelece a conexao WebSocket com retentativas e inicia tarefas de segundo plano."""
        if self._connect_in_progress:
            self.logger.debug("WSS: Conexao ja em andamento (SingleFlight). Retornando.")
            return
        self._connect_in_progress = True
        try:
            if uri:
                self.uri = uri
            if not self.uri:
                raise ConnectionError("WSS: URI nao definida.")
            attempts = max(1, int(max_attempts))

            delay = max(0.5, float(retry_delay))
            last_err: BaseException | None = None
            for attempt in range(1, attempts + 1):
                self.is_running = False
                try:
                    scheme = (urlparse(self.uri).scheme or "").lower()
                    if scheme == "wss":
                        self.ws = await connect_wss_with_ip_failover(
                            self.uri,
                            open_timeout=float(open_timeout),
                            close_timeout=10.0,
                            uri_factory=uri_factory,
                        )
                    else:
                        self.ws = await websockets.connect(
                            self.uri,
                            open_timeout=float(open_timeout),
                            close_timeout=10.0,
                        )
                    self.is_running = True
                    self.logger.debug("WSS: Conexao estabelecida com sucesso.")
                    asyncio.create_task(self._listen())
                    asyncio.create_task(self._ping_loop())
                    return
                except websockets.InvalidStatus as exc:
                    response = getattr(exc, "response", None)
                    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
                    if status is None and response is not None:
                        status = getattr(response, "status_code", None) or getattr(response, "status", None)
                    if status == 401:
                        self.uri = ""
                        raise ConnectionError(
                            "WSS: OTP expirado ou reutilizado (HTTP 401). Renove via REST POST /otp."
                        ) from exc
                    last_err = exc
                    if attempt >= attempts:
                        break
                    await asyncio.sleep(delay)
                    delay = min(delay * float(retry_backoff), 60.0)
                except (TimeoutError, ConnectionError, OSError, websockets.WebSocketException) as exc:
                    last_err = exc
                    if attempt >= attempts:
                        break
                    self.logger.warning(
                        "WSS: conexao falhou (%d/%d): %s",
                        attempt,
                        attempts,
                        exc,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * float(retry_backoff), 60.0)
            detail = str(last_err).strip() if last_err else "erro desconhecido"
            raise ConnectionError(f"WSS: conexao esgotada apos {attempts} tentativas: {detail}") from last_err
        finally:
            self._connect_in_progress = False

    async def close(self):
        """Encerra a conexao WebSocket de forma graciosa."""
        self.is_running = False
        if self.ws:
            await self.ws.close()
            self.uri = ""
            self.logger.debug("WSS: Conexao encerrada.")

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
                        """Executa o callback de forma segura capturando excecoes."""
                        try:
                            await self.subscriptions[m](d)
                        except Exception as e:
                            self.logger.error(f"WSS: Erro no callback {m}: {e}")

                    asyncio.create_task(safe_callback(data))
        except websockets.ConnectionClosed:
            self.logger.debug("WSS: Conexao fechada pelo broker.")
            self.is_running = False
        except Exception as e:
            self.logger.error(f"WSS: Erro inesperado: {e}")
            self.is_running = False

    async def _ping_loop(self):
        """Loop interno que envia pings periodicos para manter a conexao ativa com timeout estrito."""
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
            self.logger.error(f"WSS: Falha critica no loop de ping: {e}")
            self.is_running = False
            if self.ws:
                asyncio.create_task(self.ws.close())

    async def send(self, request: dict[str, Any], timeout: int | None = None) -> dict[str, Any]:
        """Envia uma solicitacao JSON e aguarda sua resposta correspondente.

        Args:
            request (Dict[str, Any]): O payload a ser enviado.
            timeout (int): Tempo maximo para aguardar uma resposta (substitui o padrao).

        Returns:
            Dict[str, Any]: A resposta do servidor.

        Raises:
            asyncio.TimeoutError: Se o servidor nao responder a tempo.
        """
        if not self.ws or not self.is_running:
            raise ConnectionError("WSS: Impossivel enviar comando. WebSocket nao esta conectado.")

        self.req_id_counter += 1
        request["req_id"] = self.req_id_counter
        actual_timeout = timeout or self.request_timeout
        cmd = next((k for k in request if k != "req_id"), "unknown")
        future = asyncio.get_event_loop().create_future()
        self.callbacks[self.req_id_counter] = future

        started = asyncio.get_event_loop().time()
        await self.ws.send(json.dumps(request))
        try:
            result = await asyncio.wait_for(future, timeout=actual_timeout)
            self.last_rtt_seconds = max(0.001, asyncio.get_event_loop().time() - started)
            return result
        except TimeoutError:
            if cmd != "ping":
                self.logger.debug(f"WSS: Timeout na requisicao {self.req_id_counter} ({cmd}) apos {actual_timeout}s")
            self.callbacks.pop(self.req_id_counter, None)
            raise

    def subscribe(self, msg_type: str, callback: Callable):
        """Registra um callback para tipos especificos de mensagens do fluxo.

        Args:
            msg_type (str): O valor do campo 'msg_type' para filtrar.
            callback (Callable): Funcao assincrona a ser chamada com os dados da mensagem.
        """
        self.subscriptions[msg_type] = callback
