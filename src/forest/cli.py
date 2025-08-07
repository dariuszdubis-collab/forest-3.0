"""Forest CLI – back‑test, grid‑search, dashboard.

Uruchomienie:
    poetry run forest --help
    poetry run forest bt       --fast 8 --slow 30   # alias back‑test
    poetry run forest gr       -j 4                 # alias grid
    poetry run forest db       --port 8501          # alias dashboard
    poetry run forest ver
    poetry run forest bt --debug …                  # pełne logi + traceback
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

import forest  # wersja pakietu
from forest.utils.log import setup_logger

console = Console()


# ────────────────────────────────[ grupa główna ]──────────────────────────────── #

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--debug",
    is_flag=True,
    help="Włącza logowanie DEBUG i pokazuje pełne tracebacki przy błędach.",
)
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """Główne polecenie CLI (back‑test, grid, dashboard…)."""
    setup_logger("DEBUG" if debug else "INFO")
    ctx.obj = {"debug": debug}


# ────────────────────────────────[ back‑test ]─────────────────────────────────── #

@cli.command(name="backtest", short_help="Uruchom pojedynczy back‑test.")
@click.option("--fast", type=int, default=10, show_default=True, help="Okres EMA‑fast")
@click.option("--slow", type=int, default=30, show_default=True, help="Okres EMA‑slow")
@click.pass_context
def backtest(ctx: click.Context, fast: int, slow: int) -> None:  # noqa: D401
    """Back‑test prostej strategii EMA‑cross na danych demo."""
    from forest.backtest.engine import _example_dataframe, run_backtest
    from forest.backtest.risk import RiskManager

    df_demo = _example_dataframe()
    out = run_backtest(df_demo, RiskManager(capital=10_000), fast=fast, slow=slow)
    console.rule("[bold]Back‑test wyniki[/bold]")
    console.print(out.tail())

# alias „bt”
cli.add_command(backtest, name="bt")


# ────────────────────────────────[ grid‑search ]──────────────────────────────── #

@cli.command(name="grid", short_help="Przeskanuj siatkę parametrów.")
@click.option("--fast-min", type=int, default=5, show_default=True)
@click.option("--fast-max", type=int, default=20, show_default=True)
@click.option("--slow-min", type=int, default=30, show_default=True)
@click.option("--slow-max", type=int, default=60, show_default=True)
@click.option("-j", "--jobs", type=int, default=1, show_default=True, help="Równoległe wątki")
@click.pass_context
def grid(
    ctx: click.Context,
    fast_min: int,
    fast_max: int,
    slow_min: int,
    slow_max: int,
    jobs: int,
) -> None:
    """Grid‑search demo (EMA‑cross) i tabelka RAR/DD."""
    from forest.backtest.engine import _example_dataframe
    from forest.backtest.grid import param_grid, run_grid
    from forest.backtest.risk import RiskManager

    df_demo = _example_dataframe()
    grid_df = run_grid(
        df_demo,
        param_grid(
            fast=list(range(fast_min, fast_max + 1, 5)),
            slow=list(range(slow_min, slow_max + 1, 10)),
        ),
        make_risk=lambda: RiskManager(capital=10_000),
        n_jobs=jobs,
    )
    best = grid_df.sort_values("rar", ascending=False).head()
    console.rule("[bold]TOP wyniki RAR[/bold]")
    console.print(best)

# alias „gr”
cli.add_command(grid, name="gr")


# ────────────────────────────────[ dashboard ]────────────────────────────────── #

@cli.command(name="dashboard", short_help="Startuje dashboard Streamlit.")
@click.option("--port", type=int, default=8501, show_default=True)
@click.pass_context
def dashboard(ctx: click.Context, port: int) -> None:
    """Otwórz lokalne UI (Streamlit)."""
    from forest.dashboard.app import run_dashboard

    run_dashboard(port=port)

# alias „db”
cli.add_command(dashboard, name="db")


# ────────────────────────────────[ wersja ]───────────────────────────────────── #

@cli.command(name="ver", short_help="Pokaż wersję pakietu.")
def version_cmd() -> None:
    """Szybki podgląd wersji Forest i ścieżki pyproject."""
    ver = forest.__version__
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    console.print(
        f"[bold green]Forest[/bold green] {ver}  🗎  [dim]{pyproject}[/dim]"
        if pyproject.exists()
        else f"Forest {ver}"
    )

# alias „ver” niepotrzebny (nazwa = komenda)


# ────────────────────────────────[ entrypoint helper ]────────────────────────── #

def _main() -> None:  # pragma: no cover
    try:
        cli()              # pozwala clickowi obsłużyć błędy validation
    except Exception:
        if "--debug" in sys.argv:
            raise          # w trybie DEBUG wyrzucamy pełny traceback
        console.print_exception(show_locals=False)
        sys.exit(1)


if __name__ == "__main__":  # python src/forest/cli.py
    _main()

