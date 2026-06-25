"""Peso de amostras de treino a partir de resultados reais de trades."""


def _symbol_history(orch, symbol: str) -> list[bool]:
    """Retorna historico de wins/losses registrados para o simbolo."""
    flags = getattr(orch, "_dl_outcome_flags", {})
    return list(flags.get(str(symbol), []))


def live_win_rate(orch, symbol: str, *, window: int = 12) -> float | None:
    """Taxa de acerto recente em trades reais do simbolo."""
    history = _symbol_history(orch, symbol)
    if len(history) < 4:
        return None
    tail = history[-min(window, len(history)) :]
    return sum(1 for x in tail if x) / float(len(tail))


def blended_val_accuracy(
    orch,
    symbol: str,
    val_accuracy: float,
    *,
    live_weight: float = 0.55,
    min_live_samples: int = 4,
) -> float:
    """Combina val_acc offline com win rate live (pessimista para gating)."""
    history = _symbol_history(orch, symbol)
    if len(history) < int(min_live_samples):
        return float(val_accuracy)
    live = live_win_rate(orch, symbol, window=max(int(min_live_samples), 4))
    weight = max(0.0, min(1.0, float(live_weight)))
    losses = sum(1 for x in history[-5:] if not x)
    if losses >= 2:
        weight = min(weight, 0.25)
    if losses >= 5:
        weight = min(weight, 0.45)
    blended = (1.0 - weight) * float(val_accuracy) + weight * float(live)
    out = min(float(val_accuracy), blended)
    if losses >= 5 and live is not None:
        out = min(out, float(val_accuracy) * 0.65)
    return out


def tick_dl_session_pauses(orch) -> None:
    """Decrementa pausas de sessao por simbolo (desativado)."""
    pass


def is_symbol_session_paused(orch, symbol: str) -> bool:
    """Indica pausa longa apos sequencia de losses no simbolo (sempre False)."""
    _ = orch
    _ = symbol
    return False


def maybe_pause_symbol_session(
    orch,
    symbol: str,
    *,
    max_losses_in_window: int,
    window_trades: int,
    pause_cycles: int,
) -> None:
    """Ativa pausa quando losses no tail excedem limiar (desativado)."""
    _ = orch
    _ = symbol
    _ = max_losses_in_window
    _ = window_trades
    _ = pause_cycles


def record_symbol_outcome(orch, symbol: str, *, won: bool, candle_epoch: int | None = None) -> None:
    """Registra resultado recente por simbolo para ponderar proximo treino."""
    if not hasattr(orch, "_dl_outcome_flags"):
        orch._dl_outcome_flags = {}
    if not hasattr(orch, "_dl_outcome_epochs"):
        orch._dl_outcome_epochs = {}
    sym = str(symbol)
    history = orch._dl_outcome_flags.setdefault(sym, [])
    history.append(bool(won))
    if len(history) > 80:
        del history[: len(history) - 80]
    if candle_epoch is not None:
        epochs = orch._dl_outcome_epochs.setdefault(sym, [])
        epochs.append(int(candle_epoch))
        if len(epochs) > 80:
            del epochs[: len(epochs) - 80]
    if not won:
        dl_cfg = getattr(orch, "config", {}).get("deep_learning", {})
        maybe_pause_symbol_session(
            orch,
            sym,
            max_losses_in_window=int(dl_cfg.get("session_max_losses_in_window", 3)),
            window_trades=int(dl_cfg.get("session_window_trades", 5)),
            pause_cycles=int(dl_cfg.get("session_pause_cycles", 8)),
        )


def sample_weights_for_symbol(
    orch,
    symbol: str,
    sample_count: int,
    targets: list[float] | None = None,
) -> list[float]:
    """Gera pesos alinhados ao tail de treino com boost apos losses recentes."""
    if sample_count <= 0:
        return []
    flags = getattr(orch, "_dl_outcome_flags", {}).get(str(symbol), [])
    weights = [1.0] * sample_count
    if not flags:
        return weights
    tail = flags[-min(16, len(flags)) :]
    loss_ratio = 1.0 - (sum(1 for x in tail if x) / float(len(tail)))
    boost = 1.0 + loss_ratio * 1.35
    focus = min(sample_count, max(4, len(tail) * 3))
    for idx in range(sample_count - focus, sample_count):
        weights[idx] = boost
    wins = sum(1 for x in tail if x)
    if wins >= len(tail) - 1 and len(tail) >= 4:
        dampen = max(0.75, 1.0 - wins / float(len(tail) + 1))
        for idx in range(sample_count - focus, sample_count):
            weights[idx] *= dampen
    last_dir = getattr(orch, "_last_loss_direction", None)
    if last_dir and targets and len(targets) == sample_count and tail and not tail[-1]:
        want_label = 1.0 if str(last_dir).upper() == "CALL" else 0.0
        for idx in range(sample_count):
            if float(targets[idx]) == want_label:
                weights[idx] *= 1.25
            else:
                weights[idx] *= 1.55
    return weights
