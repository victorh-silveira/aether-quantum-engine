"""Modelos e utilitários de Deep Learning baseados em PyTorch."""

import logging

import numpy as np
import polars as pl
import torch
from torch import nn, optim

from src.domain.models.trade import TradeDirection


logger = logging.getLogger("AETH")


class MarketDirectionClassifier(nn.Module):
    """Modelo PyTorch para classificar a direção do mercado (CALL/PUT)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa a passagem de dados (forward pass) pela rede neural."""
        return self.net(x)


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calcula o RSI (Relative Strength Index) em NumPy."""
    if len(prices) < period + 1:
        return np.full_like(prices, 50.0)
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / (down + 1e-10)
    rsi = np.zeros_like(prices)
    rsi[:period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            up_val = delta
            down_val = 0.0
        else:
            up_val = 0.0
            down_val = -delta
        up = (up * (period - 1) + up_val) / period
        down = (down * (period - 1) + down_val) / period
        rs = up / (down + 1e-10)
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def extract_features(prices: np.ndarray, lookback: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Extrai features quantitativas e targets para o modelo a partir dos preços de fechamento."""
    n = len(prices)
    if n < lookback + 5:
        return np.empty((0, 4)), np.empty((0,))

    # Feature 1: Diffs percentuais (retornos)
    returns = np.diff(prices) / (prices[:-1] + 1e-10)
    # Feature 2: Volatilidade rolling
    vol = np.zeros(n)
    for i in range(10, n):
        vol[i] = np.std(returns[max(0, i - 10) : i])
    # Feature 3: RSI
    rsi = calculate_rsi(prices) / 100.0  # normalizado [0, 1]
    # Feature 4: Spread de médias móveis exponenciais (EMA)
    df = pl.DataFrame({"close": prices})
    ema_fast = df.select(pl.col("close").ewm_mean(span=5)).to_numpy().flatten()
    ema_slow = df.select(pl.col("close").ewm_mean(span=15)).to_numpy().flatten()
    ema_spread = (ema_fast - ema_slow) / (ema_slow + 1e-10)

    features = []
    targets = []
    for i in range(lookback, n - 1):
        # Feature vector: [último retorno, RSI atual, volatilidade atual, spread de EMA]
        feat = [returns[i - 1], rsi[i], vol[i], ema_spread[i]]
        # Target: 1.0 se a próxima vela subir, 0.0 se descer ou ficar igual
        target = 1.0 if prices[i + 1] > prices[i] else 0.0
        features.append(feat)
        targets.append(target)

    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)


def train_model_online(
    model: MarketDirectionClassifier, prices: np.ndarray, lookback: int, epochs: int, lr: float
) -> float:
    """Treina o modelo local incrementalmente usando os dados mais recentes."""
    x_data, y = extract_features(prices, lookback=lookback)
    if len(x_data) < 10:
        return 0.0  # Dados insuficientes para treinar

    x_tensor = torch.tensor(x_data)
    y_tensor = torch.tensor(y).unsqueeze(1)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.train()
    total_loss = 0.0
    for _epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(x_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / epochs
    logger.debug(f"DL_TRAIN: Treinado por {epochs} épocas. Perda Média: {avg_loss:.4f}")
    return avg_loss


def predict_next_direction(
    model: MarketDirectionClassifier, prices: np.ndarray, lookback: int
) -> tuple[TradeDirection | None, float]:
    """Realiza a inferência com base na última janela de preços."""
    n = len(prices)
    if n < lookback:
        return None, 0.5

    # Calcula features para a última barra
    returns = np.diff(prices) / (prices[:-1] + 1e-10)
    vol = np.std(returns[-10:])
    rsi = calculate_rsi(prices)[-1] / 100.0
    df = pl.DataFrame({"close": prices})
    ema_fast = df.select(pl.col("close").ewm_mean(span=5)).to_numpy().flatten()[-1]
    ema_slow = df.select(pl.col("close").ewm_mean(span=15)).to_numpy().flatten()[-1]
    ema_spread = (ema_fast - ema_slow) / (ema_slow + 1e-10)

    last_feat = np.array([[returns[-1], rsi, vol, ema_spread]], dtype=np.float32)

    model.eval()
    with torch.no_grad():
        prob = model(torch.tensor(last_feat)).item()

    if prob > 0.5:
        # Probabilidade de subir
        return TradeDirection.CALL, prob
    # Probabilidade de descer
    return TradeDirection.PUT, 1.0 - prob
