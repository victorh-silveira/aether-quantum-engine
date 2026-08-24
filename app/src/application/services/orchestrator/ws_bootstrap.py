"""Bootstrap WebSocket PAT (OTP) e stream de velas do orquestrador."""

from __future__ import annotations

import asyncio
import time
import urllib.error
from typing import TYPE_CHECKING

from src.application.services.deep_learning.dl_model_artifacts import bootstrap_and_validate_models
from src.application.services.deep_learning.dl_startup import resolve_startup_fetch_bars
from src.application.services.infra_timing_config import resolve_orchestrator_timing_config
from src.application.services.orchestrator.engine_mode import training_enabled
from src.application.services.orchestrator.orchestrator_state_restore import restore_orchestrator_state
from src.application.services.orchestrator.session_target_bootstrap import bootstrap_active_session_targets
from src.infrastructure.api.deriv_rest_client import DerivRestError, select_account
from src.infrastructure.factories.infra_factory import validate_infra_services
from src.infrastructure.inference.meta_classifier_client import meta_classifier_enabled
from src.infrastructure.inference.meta_classifier_pool import bootstrap_meta_classifier_client


if TYPE_CHECKING:
    from src.application.services.orchestrator import Orchestrator
    from src.infrastructure.api.deriv_rest_client import DerivTradingSession


_BROKER_HANDSHAKE_TIMEOUT_MESSAGE = (
    "[AETHER] HANDSHAKE_TIMEOUT: WebSocket/Deriv estagnou (rede ou firewall). "
    "TCP silent drop ou barreira local bloqueou o aperto de mao seguro."
)
_DEFAULT_PUBLIC_WS_URL = "wss://api.derivws.com/trading/v1/options/ws/public"


def resolve_public_ws_url(config: dict | None) -> str:
    """URL do gateway publico de market data (ticks_history sem OTP)."""
    api = (config or {}).get("api_config") if isinstance((config or {}).get("api_config"), dict) else {}
    url = str((api or {}).get("public_ws_url") or "").strip()
    return url or _DEFAULT_PUBLIC_WS_URL


def ws_connect_options(orch: Orchestrator) -> dict[str, float | int]:
    """Parametros de reconexao WebSocket a partir da configuracao."""
    timing = resolve_orchestrator_timing_config((orch.config or {}).get("orchestrator"))
    ws = timing["ws_connect"]
    return {
        "max_attempts": int(ws["max_attempts"]),
        "open_timeout": float(ws["open_timeout_seconds"]),
        "retry_delay": float(ws["retry_delay_seconds"]),
        "retry_backoff": float(ws["retry_backoff"]),
    }


async def _broker_pat_websocket_handshake(orch: Orchestrator) -> DerivTradingSession:
    """Abre sessao PAT/OTP e conecta o WebSocketManager da corretora."""
    session = await orch.auth.open_trading_session()
    if orch.auth.mode == "demo" and session.balance <= 0.0:
        orch.logger.warning("AUTH: Saldo da conta demo e 0.0. Tentando resetar saldo...")
        try:
            client = orch.auth.rest_client()
            path = f"/trading/v1/options/accounts/{session.account_id}/reset-demo-balance"
            res = await asyncio.to_thread(client._request, "POST", path)
            new_bal = float(res.get("data", {}).get("balance", 0.0))
            if new_bal > 0.0:
                orch.logger.info("AUTH: Saldo demo resetado com sucesso para $%.2f", new_bal)
                session = await orch.auth.open_trading_session()
        except Exception as reset_err:
            orch.logger.error("AUTH: Falha ao resetar saldo demo: %s", reset_err)
    opts = ws_connect_options(orch)
    holder: dict[str, DerivTradingSession] = {"session": session}

    async def _fresh_otp_uri() -> str:
        """Emite OTP novo quando o failover troca de IP Cloudflare."""
        refreshed = await orch.auth.open_trading_session()
        holder["session"] = refreshed
        return refreshed.ws_url

    await orch.ws.connect(session.ws_url, **opts, uri_factory=_fresh_otp_uri)
    return holder["session"]


async def open_broker_handshake(orch: Orchestrator) -> DerivTradingSession:
    """Handshake broker com timeout fail-fast (PAT/OTP + WebSocket multi-IP)."""
    try:
        return await asyncio.wait_for(
            _broker_pat_websocket_handshake(orch),
            timeout=float(
                resolve_orchestrator_timing_config((orch.config or {}).get("orchestrator"))[
                    "broker_handshake_timeout_seconds"
                ]
            ),
        )
    except TimeoutError as exc:
        raise RuntimeError(_BROKER_HANDSHAKE_TIMEOUT_MESSAGE) from exc


async def _resolve_rest_account_balance(orch: Orchestrator) -> tuple[str, float]:
    """Saldo via REST accounts (sem emitir OTP)."""
    client = orch.auth.rest_client()
    accounts = await client.list_accounts()
    account = select_account(accounts, orch.auth.mode, orch.auth.account_id_override)
    return str(account.account_id), float(account.balance)


async def open_public_market_handshake(orch: Orchestrator) -> None:
    """Conecta ao WSS publico (treino/historico); sem OTP."""
    url = resolve_public_ws_url(orch.config)
    opts = ws_connect_options(orch)
    try:
        await asyncio.wait_for(
            orch.ws.connect(url, **opts),
            timeout=float(
                resolve_orchestrator_timing_config((orch.config or {}).get("orchestrator"))[
                    "broker_handshake_timeout_seconds"
                ]
            ),
        )
    except TimeoutError as exc:
        raise RuntimeError(_BROKER_HANDSHAKE_TIMEOUT_MESSAGE) from exc
    orch.logger.info("AUTH: WSS publico (market data) ok | %s", url)


async def subscribe_account_transactions(orch: Orchestrator) -> None:
    """Inscreve no stream de transacoes da conta para liquidacao."""
    try:
        await orch.ws.send(
            {"transaction": 1, "subscribe": 1},
            timeout=float(
                resolve_orchestrator_timing_config((orch.config or {}).get("orchestrator"))["ws_connect"][
                    "subscribe_transaction_timeout_seconds"
                ]
            ),
        )
        orch.ws.subscribe("transaction", orch._on_transaction)
    except Exception as e:
        orch.logger.warning("SETTLE.settle_subscribe: subscribe transaction falhou: %s", e)


def _setup_trading_session_failure(orch: Orchestrator, exc: BaseException) -> bool:
    """Registra falha categorizada do bootstrap e retorna False."""
    if isinstance(exc, DerivRestError):
        orch.logger.error("INIT: Deriv REST falhou: %s", exc)
    elif isinstance(exc, urllib.error.HTTPError):
        orch.logger.error("INIT: HTTP %s em %s (infra)", exc.code, exc.url)
    elif isinstance(exc, RuntimeError) and "HANDSHAKE_TIMEOUT" in str(exc):
        orch.logger.error("%s", exc)
    elif isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        detalhe = str(exc).strip() or repr(exc)
        orch.logger.warning("INIT: broker indisponivel [%s]: %s", type(exc).__name__, detalhe)
    elif isinstance(exc, RuntimeError):
        orch.logger.error("INIT: sanity TorchScript falhou: %s", exc)
    else:
        detalhe = str(exc).strip() or repr(exc)
        orch.logger.error("INIT: Erro no setup [%s]: %s", type(exc).__name__, detalhe, exc_info=True)
    return False


async def _try_optional_otp_trading_ws(orch: Orchestrator) -> bool:
    """Tenta WSS OTP autenticado; False se a rede bloquear o gateway demo/real."""
    try:
        session = await asyncio.wait_for(
            _broker_pat_websocket_handshake(orch),
            timeout=min(
                25.0,
                float(
                    resolve_orchestrator_timing_config((orch.config or {}).get("orchestrator"))[
                        "broker_handshake_timeout_seconds"
                    ]
                ),
            ),
        )
        orch.state.balance = float(session.balance)
        orch.deriv_account_id = str(session.account_id)
        orch.trade_handler.deriv_account_id = orch.deriv_account_id
        await subscribe_account_transactions(orch)
        orch.trading_transport = "ws"
        orch.trade_handler.trading_transport = "ws"
        auth_log = orch.logger.debug if bool(getattr(orch, "_streams_ever_started", False)) else orch.logger.info
        auth_log(
            "AUTH: PAT+OTP ok conta=%s saldo=%.2f (trading via WSS)",
            session.account_id,
            orch.state.balance,
        )
        return True
    except Exception as exc:
        orch.logger.warning(
            "AUTH: WSS OTP indisponivel (%s); market data publico + trading REST bulk-purchase",
            type(exc).__name__,
        )
        return False


async def setup_trading_session(orch: Orchestrator) -> bool:
    """Conecta market data publico; OTP se disponivel, senao REST bulk-purchase."""
    initial_boot = bool(getattr(orch, "_is_initial_boot", True))
    try:
        await validate_infra_services(orch.infra, orch.config)
        if meta_classifier_enabled(orch.config):
            await bootstrap_meta_classifier_client(orch.config)
        await bootstrap_and_validate_models(orch, is_initial_boot=initial_boot)
        await restore_orchestrator_state(orch)
        if orch.ws.ws:
            await orch.ws.close()
        account_id, balance = await _resolve_rest_account_balance(orch)
        orch.deriv_account_id = account_id
        orch.trade_handler.deriv_account_id = account_id
        orch.state.balance = balance
        if training_enabled(orch):
            await open_public_market_handshake(orch)
            await bootstrap_active_session_targets(orch, float(orch.state.balance))
            if orch.risk_manager.initial_bankroll <= 0.0:
                orch.risk_manager.set_initial_bankroll(orch.state.balance)
            orch.trading_transport = "rest"
            orch.trade_handler.trading_transport = "rest"
            orch.logger.info(
                "AUTH: treino via WSS publico | conta=%s saldo=%.2f (sem OTP)",
                account_id,
                orch.state.balance,
            )
        else:
            otp_ok = await _try_optional_otp_trading_ws(orch)
            if not otp_ok:
                if orch.ws.ws:
                    await orch.ws.close()
                await open_public_market_handshake(orch)
                orch.trading_transport = "rest"
                orch.trade_handler.trading_transport = "rest"
                orch.logger.info(
                    "AUTH: execucao hibrida | WSS publico + bulk-purchase REST | conta=%s saldo=%.2f",
                    account_id,
                    orch.state.balance,
                )
            await bootstrap_active_session_targets(orch, float(orch.state.balance))
            if orch.risk_manager.initial_bankroll <= 0.0:
                orch.risk_manager.set_initial_bankroll(orch.state.balance)
        orch._is_initial_boot = False
        return True
    except Exception as exc:
        return _setup_trading_session_failure(orch, exc)


async def start_orchestrator_streams(orch: Orchestrator) -> bool:
    """Inicia stream de velas e contratos abertos do orquestrador."""
    orch.ws.subscribe("proposal_open_contract", orch._on_contract_update)
    retries = 2
    delay = 1.0
    try:
        for attempt in range(1, retries + 1):
            orch.logger.debug("STRM: sincronizando velas...")
            try:
                bars, mode = resolve_startup_fetch_bars(orch.config, orch.symbols)
                orch.stream.config["_startup_fetch_count"] = bars
                reconnect_quiet = bool(getattr(orch, "_streams_ever_started", False))
                is_train = training_enabled(orch)
                if is_train:
                    orch.stream.config["_startup_train_lean"] = True
                boot_log = orch.logger.debug if reconnect_quiet else orch.logger.info
                boot_log(
                    "DATA: Startup %s | %d simbolos | alvo %d velas micro%s",
                    mode,
                    len(orch.symbols),
                    bars,
                    " (lean)" if is_train else "",
                )
                await orch.stream.start_candle_stream(orch._on_candle)
                orch.stream.config.pop("_startup_fetch_count", None)
                orch.stream.config.pop("_startup_quiet", None)
                orch.stream.config.pop("_startup_train_lean", None)
                orch._stream_ready_at = time.time()
                orch._streams_ever_started = True
                return True
            except ConnectionError as e:
                if attempt >= retries:
                    raise e
                orch.logger.debug(f"STRM: reconexao durante startup ({attempt}/{retries}): {e}")
                await asyncio.sleep(delay)
                if not orch.ws.is_running:
                    if training_enabled(orch) or str(getattr(orch, "trading_transport", "ws")).lower() == "rest":
                        await open_public_market_handshake(orch)
                    else:
                        session = await open_broker_handshake(orch)
                        orch.state.balance = session.balance
    except Exception as e:
        detalhe = str(e).strip() or repr(e)
        orch.logger.error("STRM: Falha [%s]: %s", type(e).__name__, detalhe, exc_info=True)
        return False
