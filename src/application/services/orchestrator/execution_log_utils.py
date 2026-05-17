"""Montagem de linhas de log para execucao de ordens."""


def build_exec_log_lines(
    symbol: str,
    direction_name: str,
    stake: float,
    duration: int | str,
    duration_unit: str,
    metrics: dict,
    mode: str,
    *,
    is_recovery: bool,
    cycle_id: int = 0,
    order_n: int = 0,
) -> list[str]:
    """Monta linhas compactas de EXEC para um envio de ordem."""
    conv = float(metrics.get("conviction", 0.0)) * 100.0
    suffix = f" | {mode}" if is_recovery and mode else ""
    src = str(metrics.get("decision_source") or "")
    src_tag = " | fonte=LLM" if src == "llm" else ""
    cycle_tag = f" | ciclo={cycle_id:04d}" if cycle_id > 0 else ""
    order_tag = f" | ordem={order_n:02d}" if order_n > 0 else ""
    body = (
        f"EXEC | {symbol} | {direction_name} | ${stake:.2f} | "
        f"{duration}{duration_unit} | conv {conv:.0f}%{cycle_tag}{order_tag}{src_tag}{suffix}"
    )
    return [body]
