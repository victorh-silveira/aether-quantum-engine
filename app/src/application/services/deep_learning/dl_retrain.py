"""Agenda retreino walk-forward por vela, loss ou janela rolling."""


def mark_force_retrain(orch, symbol: str) -> None:
    """Marca simbolo para retreino forcado no proximo ciclo."""
    forced = getattr(orch, "_dl_force_retrain", None)
    if forced is None:
        orch._dl_force_retrain = {}
        forced = orch._dl_force_retrain
    forced[str(symbol)] = True


def clear_force_retrain(orch, symbol: str) -> None:
    """Remove flag de retreino forcado apos treino concluido."""
    forced = getattr(orch, "_dl_force_retrain", None)
    if forced and str(symbol) in forced:
        del forced[str(symbol)]


def tick_bars_since_train(orch, symbols: list[str]) -> None:
    """Incrementa contador de barras desde ultimo treino por simbolo."""
    state = getattr(orch, "_dl_bars_since_train", None)
    if state is None:
        orch._dl_bars_since_train = {str(s): 0 for s in symbols}
        return
    for sym in symbols:
        key = str(sym)
        state[key] = int(state.get(key, 0)) + 1


def reset_bars_since_train(orch, symbol: str) -> None:
    """Zera contador apos treino bem-sucedido."""
    state = getattr(orch, "_dl_bars_since_train", None)
    if state is not None:
        state[str(symbol)] = 0


def should_retrain_symbol(
    orch,
    symbol: str,
    runtime: dict,
    params: dict,
    candle_epoch: int,
) -> tuple[bool, str]:
    """Indica se o simbolo deve treinar neste ciclo e o motivo."""
    forced = getattr(orch, "_dl_force_retrain", None) or {}
    if forced.get(str(symbol)):
        return True, "loss_retrain"
    last_epoch = int(runtime.get("last_candle_epoch", 0))
    if not params.get("train_on_new_candle", True) or candle_epoch != last_epoch:
        return True, "new_candle"
    rolling = int(params.get("rolling_retrain_bars", 0))
    if rolling > 0:
        state = getattr(orch, "_dl_bars_since_train", None) or {}
        if int(state.get(str(symbol), 0)) >= rolling:
            return True, "rolling"
    return False, ""
