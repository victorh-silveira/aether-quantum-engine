"""Componentes de UI para o Live Monitor do Aether Engine."""

from datetime import datetime

from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

from scripts.monitor.monitor_state import (
    active_symbols_label,
    decision_engine_label,
    resolve_session_financials,
)


def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(Layout(name="header", size=3), Layout(name="body"))
    layout["body"].split_row(Layout(name="stats", ratio=1), Layout(name="radar", ratio=2))
    return layout


def generate_header() -> Panel:
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="right", ratio=1)
    grid.add_row(
        "[bold magenta]Aether Quantum Engine[/]",
        "[bold white]TELEMETRY[/]",
        f"[bold cyan]{datetime.now().strftime('%H:%M:%S')}[/]",
    )
    return Panel(grid, style="on grey11", border_style="bright_black")


def generate_stats(state) -> Panel:
    fin = resolve_session_financials(state)
    profit_style = "green" if state.session_profit >= 0 else "red"

    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="cyan")
    table.add_column(style="bold white", justify="right")

    table.add_row("ACCOUNT BALANCE", f"[bold white]${state.balance:,.2f}[/]")
    table.add_row("SESSION PROFIT", f"[bold {profit_style}]${state.session_profit:,.2f}[/]")
    table.add_row("SESSION ROI", f"[bold {profit_style}]{fin.roi_pct:+.2f}%[/]")
    table.add_row("[dim]" + "-" * 20, "[dim]" + "-" * 10)
    table.add_row("SESSION START", f"${fin.start_balance:,.2f}")
    table.add_row(
        "STOP WIN TARGET",
        f"[bold green]${fin.target_win:,.2f}[/] [dim]{fin.goal_label}[/]",
    )
    table.add_row("TARGET BALANCE", f"${fin.target_balance:,.2f}")
    table.add_row("REMAINING TO GOAL", f"[bold yellow]${fin.remaining:,.2f}[/]")

    prog = Progress(
        BarColumn(bar_width=30, complete_style="green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    prog.add_task("Goal", total=100, completed=fin.progress_pct)

    summary_table = Table.grid(expand=True)
    summary_table.add_row(table)
    summary_table.add_row("")
    summary_table.add_row("[bold white]PROGRESS TO STOP WIN[/]")
    summary_table.add_row(prog)
    summary_table.add_row("")
    summary_table.add_row(f"[bold]ACTIVE TRADES: [yellow]{len(state.active_contracts)}[/]")
    summary_table.add_row(f"[bold]TRADING MODE:  [magenta]{state.trading_mode.upper()}[/]")

    return Panel(summary_table, title="[bold green]FINANCIAL STATE[/]", border_style="green", style="on grey11")


def generate_radar(state) -> Panel:
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="left", ratio=2)

    telemetry = state.last_telemetry

    def get_val(key: str, default: str = "N/A") -> str:
        return str(telemetry.get(key.lower(), default))

    grid.add_row("[bold white]DECISION_ENGINE[/]", f"[bold cyan]{decision_engine_label(state)}[/]")
    grid.add_row("[dim]" + active_symbols_label(state) + "[/]", "")
    grid.add_row("", "")

    direction = get_val("dir")
    dir_color = "green" if direction == "CALL" else "red" if direction == "PUT" else "yellow"
    grid.add_row("[bold white]LAST_EXEC_DIR[/]", f"[bold {dir_color}]{direction}[/]")
    grid.add_row("[bold white]LAST_SYMBOL[/]", f"[bold white]{get_val('symbol')}[/]")

    conv_val = get_val("conv")
    conv_color = "green" if conv_val != "N/A" and float(conv_val) >= 0.68 else "white"
    grid.add_row("[bold white]TRADE_SCORE[/]", f"[bold {conv_color}]{conv_val}[/]")
    grid.add_row("[bold white]DL_DIRECTION[/]", f"[bold cyan]{get_val('dl_dir')}[/]")
    grid.add_row("", "")

    metrics_raw = get_val("metrics", "-")
    metrics_table = Table(expand=True, box=None, show_header=False, padding=(0, 1))
    metrics_table.add_column(ratio=1)
    if metrics_raw and metrics_raw != "-":
        for item in metrics_raw.split("|")[:6]:
            metric = item.strip()
            if metric:
                metrics_table.add_row(f"[bold cyan]{metric}[/]")
    else:
        metrics_table.add_row("[dim]aguardando EXEC do motor[/]")

    grid.add_row("[bold white]SIGNAL[/]", metrics_table)
    grid.add_row("", "")

    status_label = (
        "[bold green]AETHER_QUANTUM_ENGINE[/]" if state.trading_mode != "N/A" else "[bold yellow]INITIALIZING[/]"
    )
    grid.add_row("[bold white]STATUS[/]", status_label)

    return Panel(grid, title="[bold magenta]ENGINE OVERVIEW[/]", border_style="magenta", style="on grey11")
