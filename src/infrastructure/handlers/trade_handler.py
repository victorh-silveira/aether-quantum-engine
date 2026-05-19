"""Lida com solicitações de propostas de trade e compra de contratos."""

import logging
import time

from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from src.infrastructure.api.websocket_manager import WebSocketManager


class TradeHandler:
    """Gerencia o ciclo de vida das operações de trading, incluindo propostas e execução.

    Comunica-se com a API WebSocket para obter propostas de preços e executar
    compras de opções binárias.
    """

    def __init__(self, ws_manager: WebSocketManager, config: dict):
        """Inicializa o manipulador com um gerenciador de conexão e configuração.

        Args:
            ws_manager (WebSocketManager): O gerenciador de conexão WebSocket.
            config (dict): Configuração da API e estratégia.
        """
        self.ws = ws_manager
        self.config = config
        self.logger = logging.getLogger("AETH")

    async def buy_with_parameters(
        self, symbol: str, direction: TradeDirection, stake: float, params: dict | None = None
    ) -> Contract:
        """Compra um contrato diretamente usando parâmetros, sem solicitar proposta antes."""
        p_cfg = params if params is not None else self.config["risk_management"]["params"]
        is_multiplier = p_cfg.get("contract_type") == "MULTIPLIER"
        c_type = direction.value
        if is_multiplier:
            c_type = "MULTUP" if direction == TradeDirection.CALL else "MULTDOWN"

        parameters = {
            "amount": stake,
            "basis": "stake",
            "contract_type": c_type,
            "currency": "USD",
            "symbol": symbol,
        }

        if is_multiplier:
            parameters["multiplier"] = p_cfg.get("multiplier", 100)
            if "cancellation" in p_cfg:
                parameters["cancellation"] = p_cfg["cancellation"]
            if "limit_order" in p_cfg:
                parameters["limit_order"] = p_cfg["limit_order"]
        else:
            parameters["duration"] = p_cfg.get("duration", 5)
            parameters["duration_unit"] = p_cfg.get("duration_unit", "m")

        if "barrier" in p_cfg:
            parameters["barrier"] = p_cfg["barrier"]

        request = {"buy": 1, "price": stake, "subscribe": 1, "parameters": parameters}
        response = await self.ws.send(request)

        if "error" in response:
            msg = response["error"].get("message", "Erro desconhecido")
            raise RuntimeError(f"Erro na compra direta: {msg}")

        b = response["buy"]
        return Contract(
            contract_id=int(b["contract_id"]),
            proposal_id="",
            status=TradeStatus.OPEN,
            buy_price=b["buy_price"],
            payout=float(b.get("payout", 0.0)),
            symbol=symbol,
            direction=direction,
            stake=stake,
            expiry_time=int(time.time() + 900),
            longcode=b.get("longcode", ""),
        )
