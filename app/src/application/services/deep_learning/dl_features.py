"""Extracao de features sequenciais e rotulos binarios para treino."""

from src.application.services.deep_learning import (
    dl_feature_build as _dl_feature_build,
    dl_feature_matrix as _dl_feature_matrix,
    dl_sequence_extract as _dl_sequence_extract,
)
from src.application.services.deep_learning.dl_feature_indicators import calculate_rsi as _calculate_rsi
from src.application.services.deep_learning.dl_feature_orthogonal import FEATURE_DIM as _ORTH_DIM


FEATURE_DIM = _ORTH_DIM
MICRO_FEATURE_DIM = 0
TRADITIONAL_FEATURE_DIM = FEATURE_DIM
VOLATILITY_FEATURE_DIM = 0
PERSISTENCE_FEATURE_DIM = 0
build_feature_row = _dl_feature_matrix.build_feature_row
build_feature_matrix = _dl_feature_matrix.build_feature_matrix
build_sequence_tensor = _dl_feature_matrix.build_sequence_tensor
calculate_rsi = _calculate_rsi
precompute_price_series = _dl_feature_build.precompute_price_series
symbol_vol_target = _dl_feature_build.symbol_vol_target
extract_features = _dl_sequence_extract.extract_features
extract_sequences = _dl_sequence_extract.extract_sequences
