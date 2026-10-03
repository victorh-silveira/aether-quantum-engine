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


class _TemporalAttention(nn.Module):
    """Atencao temporal 1D para ponderar relevância causal dos passos passados."""

    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, max(4, channels // 2), kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(max(4, channels // 2), 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna vetor ponderado por atencao ao longo da dimensao temporal."""
        weights = torch.softmax(self.attn(x), dim=2)
        return torch.sum(x * weights, dim=2)


class TemporalDirectionClassifier(nn.Module):
    """TCN dilatado causal sobre sequencia (batch, lookback, features)."""

    def __init__(self, input_dim: int, channels: tuple[int, ...] = (32, 32, 16), dropout: float = 0.25):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_dim
        for idx, out_ch in enumerate(channels):
            layers.append(_TemporalBlock(in_ch, out_ch, kernel_size=3, dilation=2**idx, dropout=dropout))
            in_ch = out_ch
        self.channels = tuple(channels)
        self.network = nn.Sequential(*layers)
        self.attn_pool = _TemporalAttention(in_ch)
        self.input_proj = nn.Linear(input_dim, in_ch)
        self.fusion = nn.Sequential(nn.Linear(in_ch * 3, in_ch), nn.GELU())
        self.head = nn.Linear(in_ch, 1)
        self.regression_head = nn.Linear(in_ch, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        """Inicializa pesos convolucionais com Kaiming e lineares com ganho para conviccao."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        logits: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Projeta sequencia em probabilidade direcional e delta de preco auxiliar."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        out = self.network(x)
        h_last = out[:, :, -1]
        h_att = self.attn_pool(out)
        raw_last = self.input_proj(x[:, :, -1])
        fused = self.fusion(torch.cat([h_last, h_att, raw_last], dim=-1))
        raw = self.head(fused)
        aux = self.regression_head(fused)
        if logits:
            logits_out = raw.squeeze(-1)
            aux_out = aux.squeeze(-1)
            if return_aux:
                return logits_out, aux_out
            return logits_out
        prob = torch.sigmoid(raw)
        if return_aux:
            return prob, aux.squeeze(-1)
        return prob
