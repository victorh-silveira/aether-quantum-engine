"""Lida com solicitações de propostas de trade e compra de contratos."""

import logging
import time

from src.domain.models.trade import Contract, Proposal, TradeDirection, TradeStatus
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

    async def get_proposal(
        self, symbol: str, direction: TradeDirection, stake: float, params: dict | None = None
    ) -> Proposal:
        """Solicita uma proposta de contrato Rise/Fall (CALL/PUT) à corretora.

        Args:
            symbol (str): O símbolo do ativo.
            direction (TradeDirection): A direção do trade.
            stake (float): O valor a ser investido.
            params (dict, optional): Parâmetros customizados (ex: duration).

        Returns:
            Proposal: A proposta validada.

        Raises:
            RuntimeError: Caso a corretora rejeite a proposta.
        """
        p_cfg = params if params is not None else self.config["risk_management"]["params"]
        is_multiplier = p_cfg.get("contract_type") == "MULTIPLIER"
        c_type = direction.value
        if is_multiplier:
            c_type = "MULTUP" if direction == TradeDirection.CALL else "MULTDOWN"

        request = {
            "proposal": 1,
            "amount": stake,
            "basis": "stake",
            "contract_type": c_type,
            "currency": "USD",
            "symbol": symbol,
        }

        if is_multiplier:
            request["multiplier"] = p_cfg.get("multiplier", 100)
            if "cancellation" in p_cfg:
                request["cancellation"] = p_cfg["cancellation"]
            if "limit_order" in p_cfg:
                request["limit_order"] = p_cfg["limit_order"]
        else:
            request["duration"] = p_cfg.get("duration", 5)
            request["duration_unit"] = p_cfg.get("duration_unit", "m")

        if "barrier" in p_cfg:
            request["barrier"] = p_cfg["barrier"]

        response = await self.ws.send(request)

        if "error" in response:
            msg = response["error"].get("message", "Erro desconhecido")
            self.logger.error(f"EXEC: Rejeição da corretora. Request: {request}")
            raise RuntimeError(f"Erro na proposta: {msg}")

        p = response["proposal"]
        return Proposal(
            proposal_id=p["id"],
            symbol=symbol,
            direction=direction,
            stake=stake,
            payout=float(p.get("payout", 0.0)),
            spot=float(p.get("spot", 0.0)),
            expiry_time=int(p.get("date_expiry") or p.get("expiry_time") or (time.time() + 60)),
        )

    async def buy_contract(self, proposal: Proposal) -> Contract:
        """Executa a compra de um contrato usando um ID de proposta obtido anteriormente.

        Args:
            proposal (Proposal): A proposta a ser executada.

        Returns:
            Contract: Os detalhes do contrato recém-aberto.
        """
        request = {"buy": proposal.proposal_id, "price": proposal.stake, "subscribe": 1}
        response = await self.ws.send(request)

        if "error" in response:
            msg = response["error"].get("message", "Erro desconhecido")
            raise RuntimeError(f"Erro na compra: {msg}")

        b = response["buy"]
        return Contract(
            contract_id=int(b["contract_id"]),
            proposal_id=proposal.proposal_id,
            status=TradeStatus.OPEN,
            buy_price=b["buy_price"],
            payout=b["payout"],
            symbol=proposal.symbol,
            direction=proposal.direction,
            stake=proposal.stake,
            expiry_time=proposal.expiry_time,
            longcode=b["longcode"],
        )
