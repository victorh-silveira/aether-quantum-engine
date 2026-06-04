"""Splits temporais purged com embargo para treino walk-forward."""


def splits_valid(val_end: int, val_start: int, calib_end: int, calib_start: int) -> bool:
    """Indica se fatias de validacao e calibracao tem comprimento positivo."""
    return calib_end > calib_start and val_end > val_start


def purged_temporal_splits(
    sample_count: int,
    validation_bars: int,
    *,
    calib_ratio: float = 0.15,
    embargo: int = 1,
) -> tuple[slice, slice, slice] | None:
    """Divide amostras em treino, validacao e calibracao com embargo."""
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
    return slice(0, train_end), slice(val_start, val_end), slice(calib_start, calib_end)
