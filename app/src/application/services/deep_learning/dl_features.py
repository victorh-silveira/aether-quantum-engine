"""Extracao de features sequenciais e rotulos binarios para treino."""

from src.application.services.deep_learning import (
    dl_feature_build as _dl_feature_build,
    dl_sequence_extract as _dl_sequence_extract,
)


FEATURE_DIM = _dl_feature_build.FEATURE_DIM
MICRO_FEATURE_DIM = _dl_feature_build.MICRO_FEATURE_DIM
TRADITIONAL_FEATURE_DIM = _dl_feature_build.TRADITIONAL_FEATURE_DIM
VOLATILITY_FEATURE_DIM = _dl_feature_build.VOLATILITY_FEATURE_DIM
PERSISTENCE_FEATURE_DIM = _dl_feature_build.PERSISTENCE_FEATURE_DIM
build_feature_row = _dl_feature_build.build_feature_row
build_sequence_tensor = _dl_feature_build.build_sequence_tensor
calculate_rsi = _dl_feature_build.calculate_rsi
precompute_price_series = _dl_feature_build.precompute_price_series
symbol_vol_target = _dl_feature_build.symbol_vol_target
extract_features = _dl_sequence_extract.extract_features
extract_sequences = _dl_sequence_extract.extract_sequences
