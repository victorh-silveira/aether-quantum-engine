"""Dashboard Rich em tempo real para telemetria e estado do motor Deep Learning."""

import json
import logging
import re
import sys
import time
from pathlib import Path
from threading import Thread


_APP = Path(__file__).resolve().parents[2]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from rich.live import Live

from aether_paths import repo_path
from scripts.monitor.monitor_redis import refresh_session_targets_from_redis
from scripts.monitor.monitor_state import DashboardState
from scripts.monitor.monitor_ui import (
    generate_header,
    generate_radar,
    generate_stats,
    make_layout,
)


LOG_PATH = repo_path("logs", "engine.log")
STATE_PATH = repo_path("data", "state.json")
SESSION_STATE_PATH = repo_path("data", "session_state.json")
CONFIG_PATH = repo_path("config", "settings.json")

repo_path("logs").mkdir(parents=True, exist_ok=True)
repo_path("data").mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.ERROR, filename=str(repo_path("logs", "monitor.log")))
logger = logging.getLogger("MONITOR")

_CLUSTER_RE = re.compile(
    r"\[CLUSTER\]\s+(?P<tf>\S+)\s+\|\|\s+(?P<body>.+)$",
    re.IGNORECASE,
)
_CLUSTER_TOKEN_RE = re.compile(
    r"(?P<symbol>RDBEAR|RDBULL):\s+(?P<ord>CALL|PUT|FLAT)"
    r"(?:\s+\((?:Prob:\s+(?P<prob>-?[\d.]+)\s+Cal:\s+(?P<cal>-?[\d.]+)\s+Edge:\s+(?P<edge>[+-]?[\d.]+)|(?P<veto>[A-Z0-9_]+))\))?",
    re.IGNORECASE,
)
_EXEC_RE = re.compile(
    r"\]\s*EXEC\s*\|\|\s*(?P<ord>CALL|PUT)\s+\[(?P<symbol>RDBEAR|RDBULL)\]\s*\|\|\s*"
    r"STAKE:\s*(?P<stake>-?[\d.]+)\s+\((?P<mode>[A-Z0-9_]+)\)\s*\|\s*"
    r"PEND:\s*(?P<pend>-?[\d.]+)"
    r"(?:\s*\|\s*LIN:\s*(?P<lin>-?\d+))?"
    r"(?:\s*\|\s*CAP:\s*(?P<cap>-?[\d.]+))?"
    r"\s*\|\s*BANCA:\s*(?P<banca>-?[\d.]+)"
    r"(?:\s*\|\s*RECOVERY_INFEASIBLE)?"
    r"\s*\|\|\s*"
    r"CID:\s*(?P<cid>\d+)\s*\|\s*PAY:\s*(?P<pay>-?[\d.]+)",
    re.IGNORECASE,
)
_IND_RE = re.compile(
    r"\]\s*IND\s*\|\|\s*"
    r"RSI:\s*(?P<rsi>[+-]?[\d.]+)\s*\|\s*ADX:\s*(?P<adx>[+-]?[\d.]+)\s*\|\s*HURST:\s*(?P<hurst>[+-]?[\d.]+)\s*\|\|\s*"
    r"ATR:\s*(?P<atr>[+-]?[\d.]+)\s*\|\s*BBW:\s*(?P<bbw>[+-]?[\d.]+)\s*\|\s*VOL_R:\s*(?P<vol_r>[+-]?[\d.]+)\s*\|\|\s*"
    r"Z:\s*(?P<z_edge>[+-]?[\d.]+)\s*\|\s*ACC:\s*(?P<acc>[+-]?[\d.]+)"
    r"(?:\s*\|\|\s*MARGIN:\s*(?P<margin>[+-]?[\d.]+)\s*\|\s*NEUTRAL:\s*(?P<neutral>\S+)\s*\|\s*META_VETO:\s*(?P<meta_veto>\S+))?",
    re.IGNORECASE,
)
_SESSION_START_RE = re.compile(r"Alvo de [\d.]+%:\s*\$([\d,]+\.?\d*)", re.IGNORECASE)
_SESSION_START_FIXED_RE = re.compile(r"Alvo fixo micro-banca:\s*\$([\d,]+\.?\d*)", re.IGNORECASE)


def _safe_load_json(path: Path, retries: int = 3, delay: float = 0.05):
    for i in range(max(1, retries)):
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            if i == retries - 1:
                return None
            time.sleep(delay)
    return None


class LogParser:
    def __init__(self, state: DashboardState):
        self.state = state

    def process_line(self, line: str):
        line = line.strip()
        if not line:
            return
        self._parse_cluster(line)
        self._parse_ind(line)
        self._parse_exec(line)
        self._parse_session_bootstrap(line)
        self._parse_balance(line)

    def _parse_cluster(self, line: str) -> None:
        if "[CLUSTER]" not in line:
            return
        match = _CLUSTER_RE.search(line)
        if not match:
            return
        try:
            tokens = list(_CLUSTER_TOKEN_RE.finditer(match.group("body")))
            if not tokens:
                return
            chosen = next((token for token in tokens if not token.group("veto")), tokens[0])
            self.state.last_telemetry["symbol"] = chosen.group("symbol").upper()
            self.state.last_telemetry["dir"] = chosen.group("ord").upper()
            self.state.last_telemetry["dl_dir"] = chosen.group("ord").upper()
            edge = chosen.group("edge")
            if edge is not None:
                self.state.last_telemetry["conv"] = f"{float(edge):.2f}"
            elif chosen.group("prob") is not None:
                self.state.last_telemetry["conv"] = f"{float(chosen.group('prob')):.2f}"
        except Exception as exc:
            logger.error("Parser Error CLUSTER: %s", exc)

    def _parse_ind(self, line: str) -> None:
        if "IND ||" not in line:
            return
        match = _IND_RE.search(line)
        if not match:
            return
        try:
            z_edge = float(match.group("z_edge"))
            acc = float(match.group("acc"))
            extras = f"Z={z_edge:+.2f} ACC={acc:.4f}"
            margin = match.group("margin")
            if margin is not None:
                extras += f" MARGIN={float(margin):.3f}"
            neutral = match.group("neutral")
            if neutral is not None:
                extras += f" NEUTRAL={neutral}"
            meta_veto = match.group("meta_veto")
            if meta_veto is not None:
                extras += f" META_VETO={meta_veto}"
            self.state.last_telemetry["metrics"] = extras
        except Exception as exc:
            logger.error("Parser Error IND: %s", exc)

    def _parse_exec(self, line: str) -> None:
        if "EXEC ||" not in line:
            return
        match = _EXEC_RE.search(line)
        if not match:
            return
        try:
            self.state.last_telemetry["symbol"] = match.group("symbol").upper()
            self.state.last_telemetry["dir"] = match.group("ord").upper()
            self.state.last_telemetry["dl_dir"] = match.group("ord").upper()
            stake = float(match.group("stake"))
            self.state.last_telemetry["conv"] = f"{stake:.2f}"
            mode = match.group("mode")
            pend = float(match.group("pend"))
            pay = float(match.group("pay"))
            lin = match.group("lin")
            cap = match.group("cap")
            extras = f"mode={mode} pend={pend:.2f} pay={pay:.2f}"
            if lin is not None:
                extras += f" lin={lin}"
            if cap is not None:
                extras += f" cap={float(cap):.2f}"
            self.state.last_telemetry["metrics"] = extras
            banca = float(match.group("banca"))
            if banca > 0.0:
                self.state.balance = banca
        except Exception as exc:
            logger.error("Parser Error EXEC: %s", exc)

    def _parse_session_bootstrap(self, line: str) -> None:
        if "SESSAO INICIADA" not in line.upper():
            return
        try:
            match = _SESSION_START_FIXED_RE.search(line) or _SESSION_START_RE.search(line)
            if match:
                target = float(match.group(1).replace(",", ""))
                if target > 0.0:
                    self.state.session_target_win = target
        except Exception as exc:
            logger.error("Parser Error SESSAO INICIADA: %s", exc)

    def _parse_balance(self, line: str) -> None:
        upper = line.upper()
        if "SALDO ATUAL:" not in upper:
            return
        try:
            val = upper.split("SALDO ATUAL:")[1].split("|")[0].strip().replace("$", "").replace(",", "")
            balance = float(val)
            if balance > 0.0:
                self.state.balance = balance
        except Exception as exc:
            logger.debug("Balance parsing error: %s", exc)


def main():
    state = DashboardState()
    parser = LogParser(state)

    def tail_logs():
        if not LOG_PATH.exists():
            return
        with LOG_PATH.open("r", encoding="utf-8") as f:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if line:
                    parser.process_line(line)
                else:
                    time.sleep(0.1)

    def refresh_state():
        while True:
            _refresh_config(state)
            _refresh_data(state)
            time.sleep(0.5)

    Thread(target=tail_logs, daemon=True).start()
    Thread(target=refresh_state, daemon=True).start()
    _run_dashboard(state)


def _refresh_config(state: DashboardState) -> None:
    if not CONFIG_PATH.exists():
        return
    try:
        cfg = _safe_load_json(CONFIG_PATH)
        if not isinstance(cfg, dict):
            return
        symbols = cfg.get("symbols")
        if isinstance(symbols, list) and symbols:
            state.active_symbols = tuple(str(s) for s in symbols)

        dl = cfg.get("deep_learning")
        if isinstance(dl, dict):
            state.dl_arch = str(dl.get("arch", state.dl_arch))

        orch = cfg.get("orchestrator")
        if isinstance(orch, dict):
            execution = orch.get("execution")
            if isinstance(execution, dict):
                mandatory = bool(execution.get("mandatory_trade_each_cycle", False))
                state.decision_mode = "CONTÍNUO" if mandatory else "SELETIVO"

        rm = cfg.get("risk_management")
        if isinstance(rm, dict):
            params = rm.get("params")
            if isinstance(params, dict):
                state.compounding_enabled = bool(params.get("compounding_enabled", True))
                state.compounding_rate = float(params.get("compounding_rate_daily", state.compounding_rate))

        tm = cfg.get("trading")
        if isinstance(tm, dict) and "mode" in tm:
            state.trading_mode = str(tm["mode"])

        infra = cfg.get("infra")
        if isinstance(infra, dict):
            redis_cfg = infra.get("redis")
            if isinstance(redis_cfg, dict):
                state.redis_url = str(redis_cfg.get("url", state.redis_url))
                state.redis_key_prefix = str(redis_cfg.get("key_prefix", state.redis_key_prefix))
    except Exception as exc:
        logger.error("Config refresh error: %s", exc)


def _refresh_data(state: DashboardState) -> None:
    refresh_session_targets_from_redis(state)
    _refresh_session_state_file(state)
    _refresh_runtime_state(state)


def _refresh_session_state_file(state: DashboardState) -> None:
    if not SESSION_STATE_PATH.exists():
        return
    try:
        data = _safe_load_json(SESSION_STATE_PATH)
        if not isinstance(data, dict):
            return
        initial = float(data.get("initial_balance", 0.0))
        current = float(data.get("current_balance", 0.0))
        target = float(data.get("daily_stop_win_target", 0.0))
        if initial > 0.0 and state.session_start_balance <= 0.0:
            state.session_start_balance = initial
        if target > 0.0 and state.session_target_win <= 0.0:
            state.session_target_win = target
        if initial > 0.0 and current > 0.0:
            state.session_profit = current - initial
    except Exception as exc:
        logger.error("Session state load error: %s", exc)


def _refresh_runtime_state(state: DashboardState) -> None:
    if not STATE_PATH.exists():
        return
    try:
        data = _safe_load_json(STATE_PATH)
        if not isinstance(data, dict):
            return
        balance = data.get("balance")
        if balance is not None:
            b = float(balance)
            if b > 0.0:
                state.balance = b
        profit = data.get("total_session_profit")
        if profit is not None:
            state.session_profit = float(profit)
        contracts = data.get("active_contracts")
        if isinstance(contracts, dict):
            state.active_contracts = contracts
        risk = data.get("risk")
        if isinstance(risk, dict):
            ini = risk.get("initial_bankroll")
            if ini is not None:
                bankroll = float(ini)
                if bankroll > 0.0 and state.session_start_balance <= 0.0:
                    state.session_start_balance = bankroll
    except Exception as exc:
        logger.error("State load error: %s", exc)


def _run_dashboard(state: DashboardState):
    layout = make_layout()
    with Live(layout, refresh_per_second=4, screen=True):
        while True:
            layout["header"].update(generate_header())
            layout["stats"].update(generate_stats(state))
            layout["radar"].update(generate_radar(state))
            time.sleep(0.2)


if __name__ == "__main__":
    main()
