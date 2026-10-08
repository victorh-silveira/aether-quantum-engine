"""Fabrica de arquiteturas TCN/LSTM/GRU para classificacao de direcao."""

from torch import nn

from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_lstm import RecurrentDirectionClassifier
from src.application.services.deep_learning.dl_model_types import DEFAULT_ARCH
from src.application.services.deep_learning.dl_tcn import TemporalDirectionClassifier


def create_direction_model(
    *,
    arch: str = DEFAULT_ARCH,
    input_dim: int = FEATURE_DIM,
    tcn_channels: tuple[int, ...] | None = None,
    tcn_dropout: float = 0.2,
    rnn_hidden_size: int = 64,
    rnn_num_layers: int = 2,
    rnn_dropout: float = 0.2,
) -> nn.Module:
    """Fabrica modelo de direcao conforme arquitetura configurada."""
    name = str(arch).strip().lower()
    if name == "tcn":
        channels = tcn_channels if tcn_channels else (64, 64, 32)
        return TemporalDirectionClassifier(input_dim=input_dim, channels=channels, dropout=tcn_dropout)
    if name in ("lstm", "gru"):
        return RecurrentDirectionClassifier(
            input_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout,
            rnn_type=name,
        )
    return TemporalDirectionClassifier(input_dim=input_dim, channels=(64, 64, 32), dropout=tcn_dropout)
