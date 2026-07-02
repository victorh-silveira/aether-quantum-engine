import asyncio
from unittest.mock import patch


POST_SETTLEMENT_MODULE = "src.application.services.orchestrator.post_settlement_cycle"
SETTLEMENT_MODULE = "src.application.services.orchestrator.execution_settlement"


async def _yield_to_event_loop() -> None:
    loop = asyncio.get_running_loop()
    done = loop.create_future()
    done.set_result(None)
    await done


async def instant_poll_delay(_seconds: float) -> None:
    await _yield_to_event_loop()


def poll_delay_stop_after(orch, max_calls: int):
    state = {"n": 0}

    async def delay(_seconds: float) -> None:
        state["n"] += 1
        if state["n"] >= max_calls:
            orch.running = False
        await _yield_to_event_loop()

    return delay


def settlement_poll_clear_after(max_calls: int, risk_manager):
    state = {"n": 0}

    async def delay(_seconds: float) -> None:
        state["n"] += 1
        await _yield_to_event_loop()
        if state["n"] >= max_calls:
            risk_manager.active_contract_ids = []

    return delay


def patch_incrementing_monotonic(step: float = 0.02):
    state = {"t": 0.0}

    def monotonic() -> float:
        value = state["t"]
        state["t"] += step
        return value

    return patch(f"{POST_SETTLEMENT_MODULE}.time.monotonic", side_effect=monotonic)


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
