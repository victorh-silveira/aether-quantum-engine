"""Classificador TCN para direcao de mercado em janelas temporais."""

import torch
from torch import nn


class _Chomp1d(nn.Module):
    """Remove padding causal ao final da convolucao temporal."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica corte de cauda na dimensao temporal."""
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    """Bloco convolucional dilatado com conexao residual."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward do bloco temporal com duas convolucoes dilatadas."""
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalDirectionClassifier(nn.Module):
    """TCN dilatado sobre sequencia (batch, lookback, features)."""

    def __init__(self, input_dim: int, channels: tuple[int, ...] = (32, 32, 16), dropout: float = 0.25):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_dim
        for idx, out_ch in enumerate(channels):
            layers.append(_TemporalBlock(in_ch, out_ch, kernel_size=3, dilation=2**idx, dropout=dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(in_ch)
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor, *, logits: bool = False) -> torch.Tensor:
        """Projeta sequencia (batch, lookback, features) em probabilidade ou logits de alta."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        out = self.network(x)
        pooled = out.mean(dim=2)
        pooled = self.norm(pooled)
        raw = self.head(pooled)
        if logits:
            return raw.squeeze(-1)
        return torch.sigmoid(raw)
