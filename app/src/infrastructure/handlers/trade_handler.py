"""Lida com solicitações de propostas de trade e compra de contratos."""

import logging
import math
import time
from typing import Any

from src.application.services.rise_fall_quote_guard import unit_ev
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from src.domain.risk.payout_observation import contract_profit_rate
from src.infrastructure.api.websocket_manager import WebSocketManager
from src.infrastructure.handlers.stream_reconnect_profit_audit import schedule_profit_table_audit
from src.infrastructure.handlers.trade_handler_buy_recovery import recover_unknown_contract_proposal
from src.infrastructure.handlers.trade_handler_errors import PurchaseOutcomeUnknownError
from src.infrastructure.handlers.trade_handler_proposal_quote import (
    fetch_proposal_snapshot,
    resolve_order_proposal,
)
from src.infrastructure.handlers.trade_handler_purchase_audit import record_purchase_audit
from src.infrastructure.handlers.trade_handler_quote_guard import validate_quote_guard_params
from src.infrastructure.handlers.trade_handler_quote_observation import record_indicator_quote
from src.infrastructure.handlers.trade_handler_requests import (
    _contract_duration_seconds,
    build_proposal_request,
    require_current_signal_candle,
    resolve_api_contract_type,
)


__all__ = ("TradeHandler", "build_proposal_request", "resolve_api_contract_type", "_contract_duration_seconds")


class TradeHandler:
    """Gerencia propostas e compras via WebSocket e fallback REST."""

    def __init__(self, ws_manager: WebSocketManager, config: dict, auth: Any | None = None, market_writer=None):
        """Inicializa o manipulador com um gerenciador de conexão e configuração.

        Args:
            ws_manager (WebSocketManager): O gerenciador de conexão WebSocket.
            config (dict): Configuração da API e estratégia.
            auth: AuthManager opcional para compras REST (bulk-purchase).
            market_writer: Captura Timescale opcional para auditoria de contratos.
        """
        self.ws = ws_manager
        self.config = config
        self.auth = auth
        self.market_writer = market_writer
        self.trading_transport = "ws"
        self.deriv_account_id = ""
        self.latest_payout_rate: float | None = None
        self.logger = logging.getLogger("AETH")

    def schedule_profit_table_audit(self, orch: Any, *, reason: str = "broker_unavailable") -> None:
        """Agenda auditoria profit_table em background com backoff exponencial."""
        schedule_profit_table_audit(orch, reason=reason)

    async def fetch_proposal_payout(
        self, symbol: str, direction: TradeDirection, stake: float = 1.0, params: dict | None = None
    ) -> float | None:
        """Consulta cotacao real de payout via proposal na API Deriv sem executar ordem."""
        snapshot = await fetch_proposal_snapshot(self, symbol, direction, stake, params)
        return float(snapshot["rate"]) if snapshot is not None else None

    async def fetch_proposal_snapshot(
        self, symbol: str, direction: TradeDirection, stake: float = 1.0, params: dict | None = None
    ) -> dict | None:
        """Exibe a cotacao integral para ordenar candidatos sem enviar compra."""
        return await fetch_proposal_snapshot(self, symbol, direction, stake, params)

    async def buy_with_parameters(
        self, symbol: str, direction: TradeDirection, stake: float, params: dict | None = None
    ) -> Contract:
        """Compra um contrato via proposal/buy (WS) ou bulk-purchase (REST)."""
        if str(self.trading_transport).lower() == "rest":
            exec_cfg = self.config.get("orchestrator", {}).get("execution", {})
            if bool(exec_cfg.get("require_quote_edge", False)) or (
                isinstance(params, dict) and "_quote_guard_side_probability" in params
            ):
                raise RuntimeError("Rise/Fall REST sem cotacao final verificavel; compra bloqueada")
            return await self._buy_via_bulk_purchase(symbol, direction, stake, params)
        return await self._buy_via_websocket(symbol, direction, stake, params)

    async def _buy_via_websocket(
        self, symbol: str, direction: TradeDirection, stake: float, params: dict | None
    ) -> Contract:
        """Compra um contrato via proposal e buy (API Deriv WebSocket autenticada)."""
        p_cfg = params if params is not None else self.config["risk_management"]["params"]
        exec_cfg = self.config.get("orchestrator", {}).get("execution", {})
        if (
            p_cfg.get("_indicator_demo_pilot")
            and str(p_cfg.get("_indicator_model_version", "")).startswith("pullback-1-")
            and str(self.config.get("trading", {}).get("mode", "real")).lower() != "demo"
        ):
            raise RuntimeError("Piloto de setup permitido somente em DEMO")
        validate_quote_guard_params(p_cfg, exec_cfg)
        if float(p_cfg.get("_quote_guard_min_edge", 0.0) or 0.0) < 0.0 and not (
            str(self.config.get("trading", {}).get("mode", "real")).lower() == "demo"
            and p_cfg.get("_indicator_demo_pilot") is True
        ):
            raise RuntimeError("EV negativo permitido somente no piloto DEMO; compra bloqueada")
        proposal_req = build_proposal_request(symbol, direction, stake, p_cfg)
        timeout = int(self.ws.request_timeout)
        prop_id, ask_price, payout_val, proposal = await resolve_order_proposal(
            self, symbol, direction, stake, p_cfg, proposal_req, timeout
        )
        is_multiplier = str(p_cfg.get("contract_type", "")).upper() == "MULTIPLIER"
        if not math.isfinite(ask_price) or ask_price <= 0:
            raise RuntimeError("Cotacao final Rise/Fall sem preco ou payout valido; compra bloqueada")
        if "_pullback_stake_cap" in p_cfg and ask_price > float(p_cfg["_pullback_stake_cap"]) + 1e-9:
            raise RuntimeError("Cotacao final do pullback excede o teto de stake")
        rate: float | None = None
        if payout_val is not None:
            if not math.isfinite(payout_val) or payout_val < 0:
                raise RuntimeError("Cotacao final Rise/Fall sem preco ou payout valido; compra bloqueada")
            if not is_multiplier and payout_val == 0 and "_quote_guard_side_probability" not in p_cfg:
                raise RuntimeError("Cotacao final Rise/Fall sem preco ou payout valido; compra bloqueada")
            rate = contract_profit_rate(payout_val, ask_price)
            if rate is not None:
                self.latest_payout_rate = rate
        if (
            p_cfg.get("_indicator_demo_pilot")
            and str(p_cfg.get("_indicator_model_version", "")).startswith("pullback-1-")
            and (payout_val is None or rate is None or rate <= 0.0)
        ):
            raise RuntimeError("Cotacao final do piloto DEMO sem payout valido")
        if "_pullback_stake_cap" in p_cfg:
            quoted_rate = float(p_cfg.get("_pullback_quote_rate") or 0.0)
            if rate is None or rate <= 0.0:
                raise RuntimeError("Cotacao final do pullback sem payout valido")
            if quoted_rate > 0 and rate + 0.02 < quoted_rate:
                raise RuntimeError("Payout final do pullback inferior ao usado para dimensionar a stake")
        await record_indicator_quote(
            self.market_writer,
            self.config,
            symbol,
            direction.name,
            p_cfg,
            str(prop_id),
            ask_price,
            payout_val,
            rate,
        )
        min_payout = float(p_cfg.get("min_payout_rate") or 0.0)
        if rate is not None and min_payout > 0.0 and rate + 1e-9 < min_payout:
            raise RuntimeError(
                f"Cotacao final Rise/Fall perdeu vantagem: payout_rate={rate:.4f} "
                f"piso_payout={min_payout:.4f}; compra bloqueada"
            )
        quote_ev: float | None = None
        if "_quote_guard_side_probability" in p_cfg:
            p_side = float(p_cfg["_quote_guard_side_probability"])
            min_edge = float(p_cfg.get("_quote_guard_min_edge", 0.0))
            margin = float(p_cfg.get("_quote_guard_safety_margin", 0.0) or 0.0)
            if rate is None or rate <= 0.0:
                raise RuntimeError("Cotacao final Rise/Fall perdeu vantagem: payout liquido invalido; compra bloqueada")
            quote_ev = unit_ev(p_side, rate)
            if quote_ev <= min_edge:
                required_probability = max((1.0 + min_edge) / (1.0 + rate), 1.0 / (1.0 + rate) + margin)
                raise RuntimeError(
                    f"Cotacao final Rise/Fall perdeu vantagem: quote_ev={quote_ev:+.4f} "
                    f"min_edge={min_edge:+.4f} p_side={p_side:.4f} "
                    f"p_minima={required_probability:.4f} payout_rate={rate:.4f}; compra bloqueada"
                )
            if margin > 0.0 and p_side < (1.0 / (1.0 + rate) + margin):
                raise RuntimeError(
                    f"Cotacao final Rise/Fall perdeu margem de seguranca sobre break-even: "
                    f"p_side={p_side:.4f} minimo={1.0 / (1.0 + rate) + margin:.4f} "
                    f"payout_rate={rate:.4f}; compra bloqueada"
                )

        require_current_signal_candle(p_cfg)
        request_epoch_ms = time.time_ns() // 1_000_000
        try:
            buy_resp = await self.ws.send({"buy": str(prop_id), "price": ask_price}, timeout=timeout)
        except (TimeoutError, ConnectionError, OSError) as exc:
            raise PurchaseOutcomeUnknownError(
                "Resposta da compra indisponivel; resultado no broker desconhecido"
            ) from exc
        ack_epoch_ms = time.time_ns() // 1_000_000
        buy_resp, direct_parameters = await recover_unknown_contract_proposal(
            self.ws,
            buy_resp,
            proposal_req,
            stake,
            timeout=timeout,
            enabled=bool(p_cfg.get("_indicator_direct_buy_fallback")),
            expected_candle_epoch=int(p_cfg.get("_indicator_candle_epoch") or 0),
            period_seconds=int(p_cfg.get("_indicator_period_seconds") or 300),
            logger=self.logger,
        )
        if direct_parameters:
            ack_epoch_ms = time.time_ns() // 1_000_000
        if "error" in buy_resp:
            msg = buy_resp["error"].get("message", "Erro desconhecido")
            raise RuntimeError(f"Erro na compra direta: {msg}")

        b = buy_resp.get("buy")
        if not isinstance(b, dict):
            raise RuntimeError("Erro na compra direta: resposta sem contrato")
        await self._record_purchase_audit(b, symbol, direction, request_epoch_ms, ack_epoch_ms)
        expiry = int(proposal.get("date_expiry") or b.get("date_expiry") or 0)
        if expiry <= 0:
            expiry = int(time.time()) + _contract_duration_seconds(proposal_req)
        return Contract(
            contract_id=int(b["contract_id"]),
            proposal_id="direct_parameters" if direct_parameters else str(prop_id),
            status=TradeStatus.OPEN,
            buy_price=float(b.get("buy_price") or ask_price),
            payout=float(b.get("payout") or proposal.get("payout") or 0.0),
            symbol=symbol,
            direction=direction,
            stake=stake,
            expiry_time=expiry,
            longcode=str(b.get("longcode") or proposal.get("longcode") or ""),
            entry_spot=b.get("entry_tick") or b.get("entry_spot"),
            entry_time=b.get("entry_tick_time") or b.get("entry_spot_time"),
            quote_price=None if direct_parameters else ask_price,
            quote_payout=None if direct_parameters else payout_val,
            quote_ev=None if direct_parameters else quote_ev,
        )

    async def _buy_via_bulk_purchase(
        self, symbol: str, direction: TradeDirection, stake: float, params: dict | None
    ) -> Contract:
        """Compra via REST bulk-purchase quando o WSS OTP esta indisponivel."""
        if self.auth is None:
            raise RuntimeError("Compra REST exige AuthManager")
        p_cfg = params if params is not None else self.config["risk_management"]["params"]
        proposal_req = build_proposal_request(symbol, direction, stake, p_cfg)
        contract_parameters = {k: v for k, v in proposal_req.items() if k != "proposal"}
        pat = self.auth.get_pat()
        if not pat:
            raise RuntimeError("AETHER_DERIV_PAT ausente para bulk-purchase")
        account_id = str(self.deriv_account_id or self.auth.account_id_override or "").strip()
        if not account_id:
            raise RuntimeError("deriv_account_id ausente para bulk-purchase")
        client = self.auth.rest_client()
        request_epoch_ms = time.time_ns() // 1_000_000
        tx = await client.bulk_purchase(
            mode=str(self.auth.mode),
            account_id=account_id,
            pat_token=pat,
            contract_parameters=contract_parameters,
        )
        ack_epoch_ms = time.time_ns() // 1_000_000
        await self._record_purchase_audit(tx, symbol, direction, request_epoch_ms, ack_epoch_ms)
        buy_price = float(tx.get("buy_price") or stake)
        payout = float(tx.get("payout") or 0.0)
        purchase_time = int(tx.get("purchase_time") or time.time())
        start_time = int(tx.get("start_time") or purchase_time)
        expiry = start_time + _contract_duration_seconds(proposal_req)
        shortcode = str(tx.get("shortcode") or "")
        return Contract(
            contract_id=int(tx["contract_id"]),
            proposal_id=str(tx.get("transaction_id") or tx["contract_id"]),
            status=TradeStatus.OPEN,
            buy_price=buy_price,
            payout=payout,
            symbol=symbol,
            direction=direction,
            stake=stake,
            expiry_time=expiry,
            longcode=shortcode,
            entry_spot=tx.get("entry_tick") or tx.get("entry_spot"),
            entry_time=tx.get("entry_tick_time") or tx.get("entry_spot_time"),
        )

    async def _record_purchase_audit(
        self,
        payload: dict,
        symbol: str,
        direction: TradeDirection,
        request_ms: int,
        ack_ms: int,
    ) -> None:
        """Auditoria nao pode transformar compra confirmada em erro/recompra."""
        await record_purchase_audit(
            self.market_writer, self.logger, self.config, payload, symbol, direction.value, request_ms, ack_ms
        )
