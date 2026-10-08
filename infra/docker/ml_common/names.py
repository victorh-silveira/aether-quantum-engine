from __future__ import annotations

LOSS_FEATURE_DIM = 24
LOSS_FEATURE_NAMES: tuple[str, ...] = (
    "direction_margin",
    "calibrated_prob",
    "cal_raw_discord",
    "scale_discordance",
    "regime_explosion",
    "regime_retraction",
    "regime_chop",
    "tape_vs_tcn",
    "linear_norm",
    "pending_norm",
    "predicted_payoff_edge",
    "conviction",
    "live_n_norm",
    "hurst",
    "scale_tape_strong",
    "scale_mili_oppose_tcn",
    "adx",
    "variance_ratio",
    "raw_prob",
    "tick_accel",
    "val_accuracy",
    "keltner_deviation",
    "bb_width_z",
    "mini_oppose",
)

META_FEATURE_DIM = 23
META_FEATURE_NAMES: tuple[str, ...] = (
    *(f"feature_{index}" for index in range(14)),
    "micro_bid_ask_spread_momentum",
    "micro_bid_ask_spread_momentum_zscore",
    "volatility_shadow_ratio",
    "volatility_shadow_ratio_zscore",
    "micro_price_velocity",
    "micro_tick_count_norm",
    "implied_vol_centered",
    "micro_tick_acceleration",
    "keltner_deviation_ratio",
)
