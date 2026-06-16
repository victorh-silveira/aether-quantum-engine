"""Gates de treino inicial e historico minimo para Deep Learning."""

from src.application.services.deep_learning.dl_symbol_runtime import get_symbol_runtime


def min_dl_inference_len(params: dict) -> int:
    """Minimo de velas OHLC para inferencia DL sem janela de treino completa."""
    lookback = int(params["lookback"])
    infer_bars = int(params.get("inference_history_bars", lookback + 20))
    return max(lookback + 5, infer_bars)


def min_dl_history_len(params: dict) -> int:
    """Calcula o minimo de velas OHLC exigidas para treino e inferencia DL."""
    if not params.get("online_training", True):
        return min_dl_inference_len(params)
    split_floor = int(params["lookback"]) + int(params["validation_bars"]) + 20
    train_window = int(params.get("training_history_bars", 0))
    return max(split_floor, train_window) if train_window > 0 else split_floor


def runtime_in_training(runtime: dict, params: dict) -> bool:
    """Indica se o modelo do simbolo ainda nao concluiu o primeiro treino valido da sessao."""
    if not runtime.get("session_trained", False):
        return True
    brier = float(runtime.get("val_brier", 0.0))
    return brier + 1e-9 >= float(params.get("brier_untrained_floor", 0.99))


def training_priority_symbols(orch, dl_config: dict, params: dict) -> frozenset[str]:
    """Lista simbolos sem primeiro treino valido que tem prioridade no slot de treino."""
    pending = []
    for symbol in orch.symbols:
        runtime = get_symbol_runtime(orch, symbol, dl_config, params)
        if runtime_in_training(runtime, params):
            pending.append(str(symbol))
    return frozenset(pending)
