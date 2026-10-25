"""Vinculo da ordem com a versao e a vela do indicador."""


def attach_indicator_identity(params: dict, metrics: dict | None) -> None:
    """Preserva periodo e versao ate a proposta e auditoria."""
    if not isinstance(metrics, dict) or not metrics.get("_register_before_tracking"):
        return
    signal_epoch = metrics.get("indicator_candle_epoch") or metrics.get("model_input_candle_epoch")
    if signal_epoch:
        params["_signal_candle_epoch"] = int(signal_epoch)
    if metrics.get("indicator_model_version"):
        params["_indicator_model_version"] = metrics["indicator_model_version"]
        params["_indicator_period_seconds"] = int(metrics.get("indicator_period_seconds") or 300)


def set_indicator_duration(params: dict, metrics: dict | None) -> None:
    """Alinha a duracao do contrato ao periodo aprovado."""
    if not isinstance(metrics, dict) or not metrics.get("indicator_controls_execution"):
        return
    period = int(metrics.get("indicator_period_seconds") or 300)
    allowed = metrics.get("allowed_period_seconds")
    if str(metrics.get("indicator_model_version", "")).startswith("pullback-1-"):
        if allowed is not None and period != allowed:
            raise RuntimeError("Periodo do pullback divergente do permitido")
        if allowed is None and period != 300:
            raise RuntimeError("Setup pullback opera somente em M5")
    if period not in (300, 900, 3600):
        raise RuntimeError("Periodo do indicador invalido para contrato")
    params["duration"] = period // 60
    params["duration_unit"] = "m"
