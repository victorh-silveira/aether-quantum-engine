"""Dashboard Rich em tempo real para telemetria e estado do motor Medallion."""

import json
import logging
import time
from pathlib import Path
from threading import Thread

from rich.live import Live

from scripts.monitor.monitor_ui import (
    generate_header,
    generate_radar,
    generate_stats,
    make_layout,
)


LOG_PATH = Path("logs/engine.log")
STATE_PATH = Path("data/state.json")
CONFIG_PATH = Path("config/settings.json")

Path("logs").mkdir(parents=True, exist_ok=True)
Path("data").mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.ERROR, filename="logs/monitor.log")
logger = logging.getLogger("MONITOR")


class DashboardState:
    def __init__(self):
        self.balance = 0.0
        self.initial_bankroll = 0.0
        self.stop_win_pct = 15.0
        self.small_threshold = 100.0
        self.small_stop_win = 10.0
        self.active_contracts = {}
        self.last_telemetry = {}
        self.total_profit = 0.0
        self.llm_enabled = True
        self.trading_mode = "N/A"


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

        self._parse_telemetry(line)
        self._parse_llm_response(line)
        self._parse_llm_dados(line)
        self._parse_balance(line)

    def _parse_telemetry(self, line: str):
        if "LLM_AUDIT" in line:
            try:
                content = line.split("LLM_AUDIT", 1)[1]
                parts = content.split("||") if "||" in content else content.split("|")
                for p in parts:
                    if "=" in p:
                        k, v = p.strip().split("=")[0], p.strip().split("=")[1]
                        self.state.last_telemetry[k.strip().lower()] = v.strip().split(" ")[0]
            except Exception as e:
                logger.error(f"Parser Error LLM_AUDIT: {e}")

    def _parse_llm_response(self, line: str):
        if "LLM_RESPOSTA" in line:
            try:
                if "[CALL]" in line:
                    self.state.last_telemetry["dir"] = "CALL"
                elif "[PUT]" in line:
                    self.state.last_telemetry["dir"] = "PUT"
                if "prob=" in line:
                    val = line.split("prob=")[1].split("%", maxsplit=1)[0]
                    self.state.last_telemetry["conv"] = f"{float(val) / 100.0:.2f}"
            except Exception as e:
                logger.error(f"Parser Error LLM_RESPOSTA: {e}")

    def _parse_llm_dados(self, line: str):
        if "LLM_DADOS" in line and "[MTF]" in line:
            try:
                tags_str = line.split("[MTF]")[1].strip()
                tags = [t.strip() for t in tags_str.split("|") if ":" in t]
                self.state.last_telemetry["patterns"] = ",".join(tags)
            except Exception as e:
                logger.error(f"Parser Error LLM_DADOS: {e}")

    def _parse_balance(self, line: str):
        if "SALDO ATUAL:" in line.upper():
            try:
                val = line.upper().split("SALDO ATUAL:")[1].split("|")[0].strip().replace("$", "").replace(",", "")
                self.state.balance = float(val)
            except Exception as e:
                logger.debug(f"Balance parsing error: {e}")


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


def _refresh_config(state: DashboardState):
    if CONFIG_PATH.exists():
        try:
            cfg = _safe_load_json(CONFIG_PATH)
            if not isinstance(cfg, dict):
                return
            rm = cfg.get("risk_management")
            if isinstance(rm, dict):
                if "large_account_stop_win_pct" in rm:
                    state.stop_win_pct = float(rm["large_account_stop_win_pct"])
                elif "stop_win_percentage" in rm:
                    state.stop_win_pct = float(rm["stop_win_percentage"])

                if "small_account_threshold" in rm:
                    state.small_threshold = float(rm["small_account_threshold"])
                if "small_account_stop_win" in rm:
                    state.small_stop_win = float(rm["small_account_stop_win"])

            tm = cfg.get("trading")
            if isinstance(tm, dict) and "mode" in tm:
                state.trading_mode = str(tm["mode"])

            llm = cfg.get("llm")
            if isinstance(llm, dict):
                state.llm_enabled = bool(llm.get("enabled", True))
        except Exception as e:
            logger.error(f"Config refresh error: {e}")


def _refresh_data(state: DashboardState):
    if STATE_PATH.exists():
        try:
            data = _safe_load_json(STATE_PATH)
            if not isinstance(data, dict):
                return
            if "balance" in data:
                b = float(data["balance"])
                if b > 0:
                    state.balance = b
            if "total_session_profit" in data:
                state.total_profit = float(data["total_session_profit"])
            if "active_contracts" in data and isinstance(data["active_contracts"], dict):
                state.active_contracts = data["active_contracts"]
            risk = data.get("risk")
            if isinstance(risk, dict) and "initial_bankroll" in risk:
                ini = float(risk["initial_bankroll"])
                if ini > 0:
                    state.initial_bankroll = ini
        except Exception as e:
            logger.error(f"State load error: {e}")


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
