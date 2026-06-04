"""Bootstrap WebSocket PAT (OTP) e stream de velas do orquestrador."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from src.infrastructure.api.deriv_rest_client import DerivRestError


if TYPE_CHECKING:
    from src.application.services.orchestrator import Orchestrator


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
        if orch.ws.ws:
            await orch.ws.close()
        session = await orch.auth.open_trading_session()
        await orch.ws.connect(session.ws_url)
        orch.state.balance = session.balance
        orch.risk_manager.set_initial_bankroll(orch.state.balance)
        orch._maybe_reset_daily_risk_session(int(time.time()))
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
                await orch.stream.start_candle_stream(orch._on_candle)
                orch._stream_ready_at = time.time()
                return True
            except ConnectionError as e:
                if attempt >= retries:
                    raise e
                orch.logger.debug(f"STRM: reconexao durante startup ({attempt}/{retries}): {e}")
                await asyncio.sleep(delay)
                if not orch.ws.is_running:
                    await orch.ws.connect()
    except Exception as e:
        detalhe = str(e).strip() or repr(e)
        orch.logger.error("STRM: Falha [%s]: %s", type(e).__name__, detalhe, exc_info=True)
        return False
