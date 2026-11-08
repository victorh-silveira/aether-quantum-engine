"""Classificador LSTM/GRU para direcao de mercado em janelas temporais."""

import torch
from torch import nn


class RecurrentDirectionClassifier(nn.Module):
    """LSTM ou GRU bidirecional sobre sequencia (batch, lookback, features)."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "lstm",
    ):
        super().__init__()
        rnn_cls = nn.GRU if str(rnn_type).lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_dim,
            int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Linear(int(hidden_size) * 2, 1)

    def forward(self, x: torch.Tensor, *, logits: bool = False) -> torch.Tensor:
        """Projeta sequencia em probabilidade ou logits de alta."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        pooled = out[:, -1, :]
        raw = self.head(pooled)
        if logits:
            return raw.squeeze(-1)
        return torch.sigmoid(raw)
