"""Componentes de UI para o Live Monitor do Aether Engine."""

from datetime import datetime

from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table


def make_layout() -> Layout:
    """Cria a estrutura de layout básica do dashboard."""
    layout = Layout()
    layout.split_column(Layout(name="header", size=3), Layout(name="body"))
    layout["body"].split_row(Layout(name="stats", ratio=1), Layout(name="radar", ratio=2))
    return layout


def generate_header() -> Panel:
    """Gera o painel de cabeçalho com o título e a hora atual."""
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
    """Gera o painel de estado financeiro e progresso de Stop Win."""
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="cyan")
    table.add_column(style="bold white", justify="right")

    profit_style = "green" if state.total_profit >= 0 else "red"

    is_small = 0 < state.initial_bankroll < state.small_threshold
    if is_small:
        stop_win_val = state.small_stop_win
        regime_label = "[bold yellow]SMALL BANKROLL REGIME[/]"
        logic_label = f"[dim](FIXED ${state.small_stop_win:.2f})[/]"
    else:
        stop_win_val = state.initial_bankroll * state.stop_win_pct / 100
        regime_label = "[bold blue]LARGE BANKROLL REGIME[/]"
        logic_label = f"[dim]({state.stop_win_pct}% RELATIVE)[/]"

    target_bankroll = state.initial_bankroll + stop_win_val
    remaining = max(0, stop_win_val - state.total_profit) if stop_win_val > 0 else 0

    progress_pct = (state.total_profit / stop_win_val * 100) if stop_win_val > 0 else 0
    progress_pct = min(100.0, max(0.0, progress_pct))

    roi = (state.total_profit / state.initial_bankroll * 100) if state.initial_bankroll > 0 else 0

    table.add_row("ACCOUNT BALANCE", f"[bold white]${state.balance:,.2f}[/]")
    table.add_row("SESSION PROFIT", f"[bold {profit_style}]${state.total_profit:,.2f}[/]")
    table.add_row("SESSION ROI", f"[bold {profit_style}]{roi:+.2f}%[/]")
    table.add_row("[dim]" + "-" * 20, "[dim]" + "-" * 10)
    table.add_row("INITIAL BANKROLL", f"${state.initial_bankroll:,.2f}")
    table.add_row("REGIME", regime_label)
    table.add_row("STOP WIN TARGET", f"[bold green]${stop_win_val:,.2f}[/] {logic_label}")
    table.add_row("TARGET BALANCE", f"${target_bankroll:,.2f}")
    table.add_row("REMAINING TO GOAL", f"[bold yellow]${remaining:,.2f}[/]")

    prog = Progress(
        BarColumn(bar_width=30, complete_style="green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    prog.add_task("Goal", total=100, completed=progress_pct)

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
    """Gera o painel de visão geral do motor e padrões detectados."""
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="left", ratio=2)

    phys = state.last_physics

    def get_val(key, default="N/A"):
        return phys.get(key.lower(), default)

    mode = str(state.direction_mode).upper()
    grid.add_row("[bold white]DIRECTION_MODE[/]", f"[bold cyan]{mode}[/]")
    grid.add_row("[dim](servidor)[/]", "[dim]call | put | alternate[/]")
    grid.add_row("", "")

    direction = get_val("dir")
    dir_color = "green" if direction == "CALL" else "red" if direction == "PUT" else "yellow"
    grid.add_row("[bold white]LAST_TELEMETRY_DIR[/]", f"[bold {dir_color}]{direction}[/]")
    conv_val = get_val("conv")
    conv_color = "green" if conv_val != "N/A" and float(conv_val) > 0.8 else "white"
    grid.add_row("[bold white]CONVICTION[/]", f"[bold {conv_color}]{conv_val}[/]")
    grid.add_row("", "")

    patterns_raw = get_val("patterns", "-")
    patterns = patterns_raw.split(",") if patterns_raw and patterns_raw != "-" else []
    pat_table = Table(expand=True, box=None, show_header=False, padding=(0, 1))
    pat_table.add_column(ratio=1)
    if patterns:
        for tag in patterns[:6]:
            pat_table.add_row(f"[bold cyan]{tag.strip()}[/]")
    else:
        pat_table.add_row("[dim]motor simples (sem padroes RADAR)[/]")

    grid.add_row("[bold white]PATTERNS[/]", pat_table)
    grid.add_row("", "")

    status_label = (
        "[bold green]AETHER_QUANTUM_ENGINE[/]" if state.trading_mode != "N/A" else "[bold yellow]INITIALIZING[/]"
    )
    grid.add_row("[bold white]STATUS[/]", status_label)

    return Panel(grid, title="[bold magenta]ENGINE OVERVIEW[/]", border_style="magenta", style="on grey11")
