"""Ponte de decisão do Deep Learning para o Orquestrador."""

import logging
from pathlib import Path

import torch

from src.application.services.deep_learning.model import (
    MarketDirectionClassifier,
    predict_next_direction,
    train_model_online,
)


logger = logging.getLogger("AETH")


async def collect_deep_learning_decisions(orch) -> dict[str, dict]:
    """Retorna um mapa de decisões de direção para cada símbolo utilizando modelos PyTorch."""
    decisions = {}
    dl_config = orch.config.get("deep_learning", {})
    if not dl_config.get("enabled", True):
        logger.warning("DL: Deep learning está desativado na configuração.")
        return decisions

    lookback = dl_config.get("lookback", 20)
    epochs = dl_config.get("training_epochs", 10)
    lr = dl_config.get("learning_rate", 0.01)
    min_conviction = dl_config.get("min_conviction_execute", 0.53)

    if not hasattr(orch, "_dl_models"):
        orch._dl_models = {}

    for symbol in orch.symbols:
        # Pega a série temporal do histórico de velas para o par correspondente
        prices = orch.stream.get_numpy_series(symbol, "close")

        if len(prices) < lookback + 10:
            logger.debug(f"DL: Histórico insuficiente para {symbol} ({len(prices)}/{lookback + 10} velas). Pulando.")
            decisions[symbol] = {
                "direction": None,
                "metrics": {
                    "conviction": 0.0,
                    "execute": False,
                    "duration": 1,
                    "llm_note": "Insufficient historical data",
                },
            }
            continue

        # Instancia modelo se não existir
        if symbol not in orch._dl_models:
            input_dim = 4  # returns, RSI, volatility, ema_spread
            model = MarketDirectionClassifier(input_dim)
            # Tenta carregar pesos salvos se existirem
            model_path = dl_config.get("model_path", "data/deep_learning_model.pth")
            if Path(model_path).exists():
                try:
                    # nosec B614 (Segurança: weights_only=True adicionado para evitar deserialização arbitrária)
                    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))
                    logger.info(f"DL: Pesos carregados com sucesso para {symbol} a partir de {model_path}.")
                except Exception as e:
                    logger.warning(f"DL: Falha ao carregar pesos salvos: {e}. Inicializando novo modelo.")
            orch._dl_models[symbol] = model

        model = orch._dl_models[symbol]

        # Treinamento incremental/online
        try:
            train_model_online(model, prices, lookback=lookback, epochs=epochs, lr=lr)
        except Exception as e:
            logger.error(f"DL: Erro no treinamento incremental para {symbol}: {e}")

        # Predição de direção
        try:
            direction, prob = predict_next_direction(model, prices, lookback=lookback)
            execute = prob >= min_conviction

            note = f"DL Predict: {direction.name if direction else 'NONE'} (Prob={prob:.2f})"
            decisions[symbol] = {
                "direction": direction,
                "metrics": {"conviction": prob, "execute": execute, "duration": 1, "llm_note": note},
            }
            logger.info(
                f"DL: Símbolo {symbol} | Direção: {direction.name if direction else 'NONE'} | Confiança: {prob:.2f} | Executar: {execute}"
            )
        except Exception as e:
            logger.error(f"DL: Falha na predição para {symbol}: {e}")
            decisions[symbol] = {
                "direction": None,
                "metrics": {"conviction": 0.0, "execute": False, "duration": 1, "llm_note": f"Inference failure: {e}"},
            }

    return decisions
