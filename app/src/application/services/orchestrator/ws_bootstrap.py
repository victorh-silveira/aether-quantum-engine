"""Bootstrap WebSocket PAT (OTP) e stream de velas do orquestrador."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from src.application.services.deep_learning.dl_model_artifacts import bootstrap_and_validate_models
from src.application.services.deep_learning.dl_startup import resolve_startup_fetch_bars
from src.application.services.orchestrator.orchestrator_state_restore import restore_orchestrator_state
from src.infrastructure.api.deriv_rest_client import DerivRestError
from src.infrastructure.factories.infra_factory import validate_infra_services


if TYPE_CHECKING:
    from src.application.services.orchestrator import Orchestrator


def ws_connect_options(orch: Orchestrator) -> dict[str, float | int]:
    """Parametros de reconexao WebSocket a partir da configuracao."""
    api = orch.config.get("api_config") or {}
    return {
        "max_attempts": int(api.get("ws_connect_max_attempts", 5)),
        "open_timeout": float(api.get("ws_connect_open_timeout_seconds", 25)),
        "retry_delay": float(api.get("ws_connect_retry_delay_seconds", 4.0)),
        "retry_backoff": float(api.get("ws_connect_retry_backoff", 1.5)),
    }


async def subscribe_account_transactions(orch: Orchestrator) -> None:
    """Inscreve no stream de transacoes da conta para liquidacao."""
    try:
        await orch.ws.send({"transaction": 1, "subscribe": 1}, timeout=10)
        orch.ws.subscribe("transaction", orch._on_transaction)
    except Exception as e:
        orch.logger.warning("SETTLE: subscribe transaction falhou: %s", e)


async def setup_trading_session(orch: Orchestrator) -> bool:
    """Conecta WebSocket via OTP PAT e prepara saldo e transacoes."""
    try:
        await validate_infra_services(orch.infra, orch.config)
        await bootstrap_and_validate_models(orch)
        await restore_orchestrator_state(orch)
        if orch.ws.ws:
            await orch.ws.close()
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

        await orch.ws.connect(session.ws_url, **ws_connect_options(orch))
        orch.state.balance = session.balance
        orch._maybe_reset_daily_risk_session(int(time.time()))
        if orch.risk_manager.initial_bankroll <= 0.0:
            orch.risk_manager.set_initial_bankroll(orch.state.balance)
        await subscribe_account_transactions(orch)
        orch.logger.debug(
            "AUTH: PAT+OTP ok conta=%s saldo=%.2f",
            session.account_id,
            orch.state.balance,
        )
        return True
    except DerivRestError as e:
        orch.logger.error("INIT: Deriv REST falhou: %s", e)
        return False
    except (ConnectionError, TimeoutError, OSError) as e:
        detalhe = str(e).strip() or repr(e)
        orch.logger.warning("INIT: broker indisponivel [%s]: %s", type(e).__name__, detalhe)
        return False
    except RuntimeError as e:
        orch.logger.error("INIT: sanity TorchScript falhou: %s", e)
        return False
    except Exception as e:
        detalhe = str(e).strip() or repr(e)
        orch.logger.error("INIT: Erro no setup [%s]: %s", type(e).__name__, detalhe, exc_info=True)
        return False


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
                if mode != "inferencia":
                    orch.logger.info(
                        "DATA: Startup treino | %d simbolos | alvo %d velas",
                        len(orch.symbols),
                        bars,
                    )
                await orch.stream.start_candle_stream(orch._on_candle)
                orch.stream.config.pop("_startup_fetch_count", None)
                orch._stream_ready_at = time.time()
                return True
            except ConnectionError as e:
                if attempt >= retries:
                    raise e
                orch.logger.debug(f"STRM: reconexao durante startup ({attempt}/{retries}): {e}")
                await asyncio.sleep(delay)
                if not orch.ws.is_running:
                    await orch.ws.connect(**ws_connect_options(orch))
    except Exception as e:
        detalhe = str(e).strip() or repr(e)
        orch.logger.error("STRM: Falha [%s]: %s", type(e).__name__, detalhe, exc_info=True)
        return False
