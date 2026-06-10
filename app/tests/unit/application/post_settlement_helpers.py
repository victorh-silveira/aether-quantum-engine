import asyncio
from unittest.mock import patch


POST_SETTLEMENT_MODULE = "src.application.services.orchestrator.post_settlement_cycle"
SETTLEMENT_MODULE = "src.application.services.orchestrator.execution_settlement"


async def instant_poll_delay(_seconds: float) -> None:
    await asyncio.sleep(0)


def poll_delay_stop_after(orch, max_calls: int):
    state = {"n": 0}

    async def delay(_seconds: float) -> None:
        state["n"] += 1
        if state["n"] >= max_calls:
            orch.running = False
        await asyncio.sleep(0)

    return delay


def settlement_poll_clear_after(max_calls: int, risk_manager):
    state = {"n": 0}

    async def delay(_seconds: float) -> None:
        state["n"] += 1
        await asyncio.sleep(0)
        if state["n"] >= max_calls:
            risk_manager.active_contract_ids = []

    return delay


def patch_instant_post_settlement_poll():
    return patch(f"{POST_SETTLEMENT_MODULE}._poll_delay", side_effect=instant_poll_delay)


def patch_post_settlement_poll_stop_after(orch, max_calls: int):
    return patch(f"{POST_SETTLEMENT_MODULE}._poll_delay", side_effect=poll_delay_stop_after(orch, max_calls))


def patch_instant_settlement_poll():
    return patch(f"{SETTLEMENT_MODULE}._settlement_poll_delay", side_effect=instant_poll_delay)


def patch_settlement_poll_clear_after(risk_manager, max_calls: int):
    return patch(
        f"{SETTLEMENT_MODULE}._settlement_poll_delay",
        side_effect=settlement_poll_clear_after(max_calls, risk_manager),
    )
