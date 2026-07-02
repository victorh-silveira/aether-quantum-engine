"""Lida com solicitações de propostas de trade e compra de contratos."""

import logging
import time
from typing import Any

from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from src.infrastructure.api.websocket_manager import WebSocketManager
from src.infrastructure.handlers.stream_reconnect_profit_audit import schedule_profit_table_audit


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

    def schedule_profit_table_audit(self, orch: Any, *, reason: str = "broker_unavailable") -> None:
        """Agenda auditoria profit_table em background com backoff exponencial."""
        schedule_profit_table_audit(orch, reason=reason)

    async def buy_with_parameters(
        self, symbol: str, direction: TradeDirection, stake: float, params: dict | None = None
    ) -> Contract:
        """Compra um contrato via proposal e buy (API Deriv atual)."""
        p_cfg = params if params is not None else self.config["risk_management"]["params"]
        proposal_req = build_proposal_request(symbol, direction, stake, p_cfg)
        timeout = int(self.ws.request_timeout)
        proposal_resp = await self.ws.send(proposal_req, timeout=timeout)
        if "error" in proposal_resp:
            msg = proposal_resp["error"].get("message", "Erro desconhecido")
            raise RuntimeError(f"Erro na proposta: {msg}")

        proposal = proposal_resp.get("proposal")
        if not isinstance(proposal, dict):
            raise RuntimeError("Erro na proposta: resposta sem proposal")

        prop_id = proposal.get("id")
        if not prop_id:
            raise RuntimeError("Erro na proposta: id ausente")

        ask_price = float(proposal.get("ask_price") or stake)
        buy_resp = await self.ws.send({"buy": str(prop_id), "price": ask_price}, timeout=timeout)
        if "error" in buy_resp:
            msg = buy_resp["error"].get("message", "Erro desconhecido")
            raise RuntimeError(f"Erro na compra direta: {msg}")

        b = buy_resp["buy"]
        expiry = int(proposal.get("date_expiry") or b.get("date_expiry") or 0)
        if expiry <= 0:
            expiry = int(time.time()) + _contract_duration_seconds(proposal_req)
        return Contract(
            contract_id=int(b["contract_id"]),
            proposal_id=str(prop_id),
            status=TradeStatus.OPEN,
            buy_price=float(b.get("buy_price") or ask_price),
            payout=float(b.get("payout") or proposal.get("payout") or 0.0),
            symbol=symbol,
            direction=direction,
            stake=stake,
            expiry_time=expiry,
            longcode=str(b.get("longcode") or proposal.get("longcode") or ""),
        )


def resolve_api_contract_type(direction: TradeDirection, p_cfg: dict[str, Any]) -> str:
    """Mapeia direcao do motor para contract_type aceito na API Deriv."""
    if p_cfg.get("contract_type") == "MULTIPLIER":
        return "MULTUP" if direction == TradeDirection.CALL else "MULTDOWN"
    return direction.value


def build_proposal_request(
    symbol: str, direction: TradeDirection, stake: float, p_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Monta payload proposal com underlying_symbol (API Deriv atual)."""
    is_multiplier = p_cfg.get("contract_type") == "MULTIPLIER"
    request: dict[str, Any] = {
        "proposal": 1,
        "amount": stake,
        "basis": "stake",
        "contract_type": resolve_api_contract_type(direction, p_cfg),
        "currency": "USD",
        "underlying_symbol": symbol,
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

    return request


def _contract_duration_seconds(parameters: dict) -> int:
    """Converte duration e duration_unit em segundos para expiry estimado."""
    dur = max(1, int(parameters.get("duration", 5)))
    unit = str(parameters.get("duration_unit", "m")).lower().strip()
    if unit == "m":
        return dur * 60
    if unit == "s":
        return dur
    if unit == "t":
        return dur * 2
    if unit == "d":
        return dur * 86400
    return dur * 60
