"""Motor matemático do Medallion: Filtro de Kalman, Cointegração PCA e HMM."""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Filtro de Kalman unidimensional para suavização de séries ruidosas sem lag."""

    def __init__(self, q: float = 1e-5, r: float = 1e-3):
        self.q = q
        self.r = r
        self.x = None  # Estado estimado (preço suavizado)
        self.p = 1.0  # Covariância do erro

    def update(self, measurement: float) -> float:
        """Atualiza recursivamente o filtro com uma nova medição de preço."""
        if self.x is None:
            self.x = measurement
            return measurement

        # Predição
        self.p = self.p + self.q

        # Atualização/Correção
        k_gain = self.p / (self.p + self.r)
        self.x = self.x + k_gain * (measurement - self.x)
        self.p = (1.0 - k_gain) * self.p

        return self.x

    def filter_series(self, series: list[float] | np.ndarray) -> list[float]:
        """Aplica o filtro de Kalman recursivo em toda a série temporal."""
        out = []
        for val in series:
            out.append(self.update(val))
        return out


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    """Calcula a densidade de probabilidade normal (Gaussiana)."""
    sigma = max(1e-8, sigma)
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


class MarketHMMClassifier:
    """Classificador Bayesiano de Regimes de Mercado (HMM de 2 Estados).

    Estado 0: Reversão à Média (Volatilidade Baixa)
    Estado 1: Tendência/Rompimento (Volatilidade Alta)
    """

    def __init__(
        self,
        transition_matrix: np.ndarray | None = None,
        sigma_low: float = 0.0004,
        sigma_high: float = 0.0016,
    ):
        # Matriz de transição padrão (regimes tendem a persistir)
        if transition_matrix is None:
            self.A = np.array([[0.92, 0.08], [0.08, 0.92]])
        else:
            self.A = np.asarray(transition_matrix)

        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.prior = np.array([0.7, 0.3])  # Inicia com preferência por Reversão à Média

    def update_regime(self, log_return: float, recent_returns: list[float] | None = None) -> tuple[int, float]:
        """Atualiza a probabilidade posterior do estado de Markov com o último retorno logarítmico.

        Permite calibração dinâmica da volatilidade se uma lista de retornos recentes for fornecida.
        """
        # Calibração adaptativa de volatilidade baseada em lookback dinâmico
        if recent_returns and len(recent_returns) >= 5:
            std = float(np.std(recent_returns))
            if std > 1e-6:
                self.sigma_low = std * 0.5
                self.sigma_high = std * 2.0

        # Predição de estado
        pred = self.A.T @ self.prior

        # Likelihoods de emissão sob hipótese de volatilidades distintas
        l_low = normal_pdf(log_return, 0.0, self.sigma_low)
        l_high = normal_pdf(log_return, 0.0, self.sigma_high)

        # Atualização Bayesiana posterior
        posterior = pred * np.array([l_low, l_high])
        p_sum = posterior.sum()

        posterior = posterior / p_sum if p_sum > 1e-12 else np.array([0.5, 0.5])

        self.prior = posterior
        active_state = int(np.argmax(posterior))
        return active_state, float(posterior[active_state])


def compute_pca_cointegration_zscores(
    closes_map: dict[str, list[float]],
    symbols: list[str],
    *,
    lookback: int = 15,
) -> dict[str, float]:
    """Calcula resíduos de cointegração multivariada via PCA e Z-Scores de reversão.

    Usa decomposição espectral em NumPy puro sobre preços logarítmicos suavizados por Kalman.
    """
    zscores: dict[str, float] = {}
    if not symbols:
        return zscores

    # Alinhamento e limpeza das séries
    aligned_data = []
    min_len = 999999
    for sym in symbols:
        closes = closes_map.get(sym, [])
        if len(closes) < 3:
            return dict.fromkeys(symbols, 0.0)
        min_len = min(min_len, len(closes))

    lookback_window = min(lookback, min_len)

    # Filtragem com Kalman e conversão para log-preço
    for sym in symbols:
        closes = closes_map[sym][-lookback_window:]
        kf = KalmanFilter(q=1e-5, r=1e-3)
        denoised = kf.filter_series(closes)
        aligned_data.append(np.log(denoised))

    # Matriz x_matrix de dados suavizados: shape (num_assets, lookback_window)
    x_matrix = np.array(aligned_data)

    # Centralização dos dados
    means = x_matrix.mean(axis=1, keepdims=True)
    x_centered = x_matrix - means

    # Cálculo da matriz de covariância
    cov = np.cov(x_centered) if lookback_window > 1 else np.zeros((len(symbols), len(symbols)))

    # Se a covariância for nula ou inválida, retorna Z-Scores neutros
    if np.allclose(cov, 0.0) or np.any(np.isnan(cov)):
        return dict.fromkeys(symbols, 0.0)

    # Decomposição Espectral (eigh retorna autovalores ordenados em ordem ascendente)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except Exception:
        return dict.fromkeys(symbols, 0.0)

    # O primeiro componente principal representa a tendência de mercado comum (maior autovalor)
    pc1 = eigenvectors[:, -1]

    # Projeção dos preços centralizados no componente principal principal
    # common_trend shape: (lookback_window,)
    common_trend = pc1.T @ x_centered

    # Cálculo do resíduo (idiosincrasia de spread em relação ao fator comum)
    # residual shape: (num_assets, lookback_window)
    residual = x_centered - np.outer(pc1, common_trend)

    # Z-Score dos resíduos no último período (t = -1)
    for i, sym in enumerate(symbols):
        res_series = residual[i]
        res_last = res_series[-1]
        res_std = np.std(res_series)

        if res_std > 1e-8:
            zscores[sym] = float(res_last / res_std)
        else:
            zscores[sym] = 0.0

    return zscores
