"""Instrumentacao automatica de metricas de negocio seguindo o padrao OpenTelemetry."""

from __future__ import annotations

import importlib
import logging


logger = logging.getLogger("AETH")

try:
    otel_metrics = importlib.import_module("opentelemetry.metrics")
    _OTEL_SDK_AVAILABLE = True
except ImportError:
    otel_metrics = None
    _OTEL_SDK_AVAILABLE = False


class BusinessMetricsInstrumentor:
    """Instrumentador semantico de metricas de negocio para trading quantitativo."""

    def __init__(self, service_name: str = "aether_quantum_engine") -> None:
        """Inicializa medidores compativeis com OpenTelemetry e buffers locais."""
        self._service_name = service_name
        self._current_balance: float = 0.0
        self._session_start_balance: float = 0.0
        self._session_target_win: float = 0.0
        self._active_contracts_count: int = 0
        self._total_pnl: float = 0.0
        self._high_water_mark: float = 0.0
        self._max_drawdown_pct: float = 0.0
        self._recent_probs: list[float] = []
        self._recent_outcomes: list[int] = []
        self._contract_counts: dict[tuple[str, str, str], int] = {}
        self._gate_verdicts: dict[tuple[str, str, str], int] = {}
        self._radar: dict[str, dict[str, float | None]] = {}

        self._meter = None
        if _OTEL_SDK_AVAILABLE and otel_metrics is not None:
            try:
                self._meter = otel_metrics.get_meter(self._service_name)
            except Exception as exc:
                logger.debug("OTEL: Falha ao obter meter: %s", exc)
                self._meter = None

    @staticmethod
    def is_otel_sdk_available() -> bool:
        """Indica se a biblioteca oficial OpenTelemetry Metrics esta instalada."""
        return _OTEL_SDK_AVAILABLE

    def set_session_targets(self, start_balance: float, target_win: float) -> None:
        """Define o saldo de abertura e a meta de stop win financeiro da sessao."""
        self._session_start_balance = float(start_balance)
        self._session_target_win = float(target_win)
        if self._current_balance <= 0.0 and self._session_start_balance > 0.0:
            self.update_balance(self._session_start_balance)

    def update_active_contracts_count(self, count: int) -> None:
        """Atualiza contagem instantanea de contratos em aberto no broker."""
        self._active_contracts_count = max(0, int(count))

    def update_balance(self, balance_usd: float) -> None:
        """Atualiza a banca operacional e recalcula o drawdown instantaneo."""
        val = float(balance_usd)
        self._current_balance = val
        self._high_water_mark = max(self._high_water_mark, val)
        if self._high_water_mark > 0.0:
            dd = (self._high_water_mark - val) / self._high_water_mark * 100.0
            self._max_drawdown_pct = max(self._max_drawdown_pct, dd)

    def record_cycle(
        self,
        symbol: str,
        *,
        is_execution: bool,
        reason: str = "ok",
        duration_ms: float = 0.0,
    ) -> None:
        """Registra automaticamente conclusao do ciclo com veredito de gates."""
        _ = duration_ms
        verdict = "EXEC" if is_execution else "SKIP"
        clean_reason = str(reason).strip() or "ok"
        key = (str(symbol), verdict, clean_reason)
        self._gate_verdicts[key] = self._gate_verdicts.get(key, 0) + 1

    def record_inference_radar(
        self,
        symbol: str,
        *,
        prob: float = 0.5,
        cal: float = 0.5,
        margin: float = 0.0,
        edge: float = 0.0,
        conviction: float = 0.0,
        p_loss: float = 0.5,
        p_eff: float = 0.5,
        is_flip: bool = False,
        anti_trend_lock: bool = False,
        direction: str = "FLAT",
        stake_usd: float = 0.0,
        stake: float | None = None,
        metrics: dict | None = None,
    ) -> None:
        """Armazena snapshot inferencial e de risco do ultimo ciclo para o radar Grafana."""
        if metrics is not None:
            prob = float(metrics.get("prob", prob))
            cal_raw = metrics.get("calibrated_prob", metrics.get("cal"))
            cal = float(cal_raw) if cal_raw is not None else None
            margin = float(metrics.get("directional_margin", metrics.get("margin", margin)))
            edge_raw = metrics.get("predicted_payoff_edge", metrics.get("payoff_edge", metrics.get("edge")))
            edge = float(edge_raw) if edge_raw is not None else None
            conviction = float(metrics.get("conviction", conviction))
            p_loss = float(metrics.get("p_loss", p_loss))
            p_eff_raw = metrics.get("loss_clf_p_eff", metrics.get("p_eff"))
            p_eff = float(p_eff_raw) if p_eff_raw is not None else None
            is_flip = bool(metrics.get("loss_clf_flip", metrics.get("is_flip", is_flip)))
            anti_trend_lock = bool(
                metrics.get("anti_trend_lock_active", metrics.get("anti_trend_lock", anti_trend_lock))
            )
        if stake is not None:
            stake_usd = float(stake)

        dir_upper = str(direction).upper()
        dir_val = 1.0 if dir_upper == "CALL" else (-1.0 if dir_upper == "PUT" else 0.0)
        self._radar[str(symbol)] = {
            "prob": float(prob),
            "cal": float(cal) if cal is not None else None,
            "margin": float(margin),
            "edge": float(edge) if edge is not None else None,
            "conviction": float(conviction),
            "p_loss": float(p_loss),
            "p_eff": float(p_eff) if p_eff is not None else None,
            "is_flip": 1.0 if is_flip else 0.0,
            "anti_trend_lock": 1.0 if anti_trend_lock else 0.0,
            "direction_num": dir_val,
            "stake_usd": float(stake_usd),
        }

    def record_trade(
        self,
        symbol: str = "1HZ75V",
        direction: str = "CALL",
        *,
        is_win: bool | None = None,
        won: bool | None = None,
        profit_usd: float | None = None,
        profit: float | None = None,
        predicted_prob: float = 0.5,
        stake: float = 0.0,
    ) -> None:
        """Registra liquidacao contratual atualizando PnL e Brier Score de negocio."""
        _ = stake
        win = is_win if is_win is not None else (won if won is not None else True)
        pnl = float(profit_usd if profit_usd is not None else (profit if profit is not None else 0.0))
        outcome = "WIN" if win else "LOSS"
        dir_name = str(direction).upper()
        sym_name = str(symbol)
        key = (sym_name, dir_name, outcome)
        self._contract_counts[key] = self._contract_counts.get(key, 0) + 1

        self._total_pnl += pnl
        self.update_balance(self._current_balance + pnl)

        self._recent_probs.append(max(0.0, min(1.0, float(predicted_prob))))
        self._recent_outcomes.append(1 if win else 0)
        if len(self._recent_outcomes) > 20:
            self._recent_probs.pop(0)
            self._recent_outcomes.pop(0)

    def compute_rolling_brier_score(self) -> float:
        """Calcula o Brier Score de calibracao sobre os ultimos 20 contratos."""
        if not self._recent_outcomes:
            return 0.0
        sq_errors = [(p - y) ** 2 for p, y in zip(self._recent_probs, self._recent_outcomes, strict=False)]
        return float(sum(sq_errors) / len(sq_errors))

    def format_prometheus_metrics(self) -> str:
        """Serializa todas as metricas de negocio no formato OpenMetrics com labels semanticos."""
        target_bal = self._session_start_balance + self._session_target_win
        profit = (
            (self._current_balance - self._session_start_balance)
            if self._session_start_balance > 0.0
            else self._total_pnl
        )
        rem = max(0.0, self._session_target_win - profit)
        prog = (
            min(100.0, max(0.0, (profit / self._session_target_win * 100.0))) if self._session_target_win > 0.0 else 0.0
        )
        roi = (profit / self._session_start_balance * 100.0) if self._session_start_balance > 0.0 else 0.0
        lines: list[str] = [
            f"aether_trading_balance_usd {round(self._current_balance, 2)}",
            f"aether_trading_pnl_usd {round(self._total_pnl, 4)}",
            f"aether_trading_max_drawdown_pct {round(self._max_drawdown_pct, 2)}",
            f"aether_trading_brier_score {round(self.compute_rolling_brier_score(), 4)}",
        ]
        if self._session_start_balance > 0.0 and self._session_target_win > 0.0:
            lines.extend(
                (
                    f"aether_session_balance_usd {round(self._current_balance, 2)}",
                    f"aether_session_start_balance_usd {round(self._session_start_balance, 2)}",
                    f"aether_session_profit_usd {round(profit, 4)}",
                    f"aether_session_target_win_usd {round(self._session_target_win, 2)}",
                    f"aether_session_target_balance_usd {round(target_bal, 2)}",
                    f"aether_session_remaining_usd {round(rem, 2)}",
                    f"aether_session_progress_pct {round(prog, 2)}",
                    f"aether_session_roi_pct {round(roi, 2)}",
                    f"aether_session_active_trades {self._active_contracts_count}",
                )
            )

        for sym, data in self._radar.items():
            lines.append(f'aether_inference_prob{{symbol="{sym}"}} {round(data["prob"], 4)}')
            if data["cal"] is not None:
                lines.append(f'aether_inference_calibrated{{symbol="{sym}"}} {round(data["cal"], 4)}')
            lines.append(f'aether_inference_directional_margin{{symbol="{sym}"}} {round(data["margin"], 4)}')
            if data["edge"] is not None:
                lines.append(f'aether_inference_payoff_edge{{symbol="{sym}"}} {round(data["edge"], 4)}')
            lines.append(f'aether_inference_conviction{{symbol="{sym}"}} {round(data["conviction"], 4)}')
            lines.append(f'aether_loss_classifier_p_loss{{symbol="{sym}"}} {round(data["p_loss"], 4)}')
            if data["p_eff"] is not None:
                lines.append(f'aether_loss_classifier_p_eff{{symbol="{sym}"}} {round(data["p_eff"], 4)}')
            lines.append(f'aether_loss_classifier_flip_active{{symbol="{sym}"}} {int(data["is_flip"])}')
            lines.append(f'aether_anti_trend_lock_active{{symbol="{sym}"}} {int(data["anti_trend_lock"])}')
            lines.append(f'aether_trading_direction{{symbol="{sym}"}} {round(data["direction_num"], 1)}')
            lines.append(f'aether_trading_stake_usd{{symbol="{sym}"}} {round(data["stake_usd"], 2)}')

        for (sym, direction, outcome), count in self._contract_counts.items():
            line = (
                f'aether_trading_contracts_total{{symbol="{sym}",direction="{direction}",outcome="{outcome}"}} {count}'
            )
            lines.append(line)

        for (sym, verdict, reason), count in self._gate_verdicts.items():
            line = f'aether_trading_gate_verdicts_total{{symbol="{sym}",verdict="{verdict}",reason="{reason}"}} {count}'
            lines.append(line)

        return "\n".join(lines) + "\n"


OtelBusinessInstrumentor = BusinessMetricsInstrumentor
