"""Resumo de resultado do cluster liquidado."""

from src.application.services.orchestrator.metrics_utils import neutral_metrics


def log_cluster_summary(orch):
    """Emite resumo de performance do cluster encerrado."""
    orch.logger.debug(
        "[C%04d] BANCA FINAL: $%.2f | ACUMULADO: %dW / %dL",
        orch._last_result_cycle_id,
        orch.state.balance,
        orch._session_wins,
        orch._session_losses,
    )
    orch.logger.debug("")
    orch._cluster_results = []
    orch._last_anchor_metrics = neutral_metrics()
