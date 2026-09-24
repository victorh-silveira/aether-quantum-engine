"""Splits temporais purged com embargo para treino walk-forward."""


def settlement_train_sample_count(sample_count: int, horizon_bars: int, gate_cfg: dict | None) -> int:
    """Reserva janela final intocada para avaliar o vencimento do contrato."""
    if not isinstance(gate_cfg, dict) or not gate_cfg.get("enabled", True):
        return sample_count
    eval_bars = max(0, int(gate_cfg.get("mini_bars", 0)))
    return sample_count - eval_bars - max(1, int(horizon_bars)) - 2 if eval_bars else sample_count


def splits_valid(val_end: int, val_start: int, calib_end: int, calib_start: int) -> bool:
    """Indica se fatias de validacao e calibracao tem comprimento positivo."""
    return calib_end > calib_start and val_end > val_start


def purged_temporal_splits(
    sample_count: int,
    validation_bars: int,
    *,
    calib_ratio: float = 0.15,
    embargo: int = 1,
    stride: int = 1,
) -> tuple[slice, slice, slice] | None:
    """Divide amostras em treino, validacao e calibracao com embargo e stride."""
    if sample_count < 20:
        return None
    val_size = max(5, int(validation_bars))
    calib_size = max(3, int(sample_count * calib_ratio))
    holdout = val_size + calib_size + embargo * 2
    if sample_count <= holdout + 10:
        calib_size = max(3, sample_count // 10)
        val_size = max(5, validation_bars)
        holdout = val_size + calib_size + embargo * 2
    if sample_count <= holdout + 10:
        return None
    train_end = sample_count - holdout
    val_start = train_end + embargo
    val_end = val_start + val_size
    calib_start = val_end + embargo
    calib_end = min(sample_count, calib_start + calib_size)
    if not splits_valid(val_end, val_start, calib_end, calib_start):
        return None
    step = max(1, int(stride))
    return slice(0, train_end), slice(val_start, val_end, step), slice(calib_start, calib_end, step)
