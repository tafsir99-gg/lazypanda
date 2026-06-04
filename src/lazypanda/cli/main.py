"""
LazyPanda v0.1.0 — CLI Entry Point

This file is the ONLY entry point for the `lazypanda` command.

WHY structure it this way?
    Typer allows us to compose sub-commands from separate files.
    This keeps main.py clean — it only registers commands, it never
    contains business logic. Each command lives in its own file.

REGISTERED COMMANDS (added progressively per milestone):
    lazypanda analyze <file>   — Run all analyzers, show report in terminal
    lazypanda clean <file>     — Run full cleaning pipeline
    lazypanda report <file>    — Generate reports from existing analysis
    lazypanda version          — Show tool version
"""

import os
from pathlib import Path

import typer
from dotenv import load_dotenv

from lazypanda.cli.commands.analyze import analyze
from lazypanda.cli.commands.clean import clean
from lazypanda.cli.commands.report import report
from lazypanda.cli.commands.wizard import wizard
from lazypanda.cli.console import console, error_console
from lazypanda.utils.logging_setup import setup_logging

# Load .env file BEFORE anything else so all modules can access env vars
load_dotenv()

# ─── Application Definition ───────────────────────────────────────────────────
app = typer.Typer(
    name="lazypanda",
    help=(
        "[bold cyan]LazyPanda v0.1.0[/bold cyan] — Professional ML dataset cleaning tool.\n\n"
        "Analyzes and cleans CSV datasets with deterministic Python + optional AI insights.\n"
        "Run [bold]lazypanda <command> --help[/bold] for detailed usage."
    ),
    add_completion=False,   # Disable shell completion for simplicity in v1
    rich_markup_mode="rich",
    pretty_exceptions_enable=True,
    pretty_exceptions_show_locals=False,  # Don't show variable values in errors (security)
)

# ─── Register commands ────────────────────────────────────────────────────────
app.command(name="analyze")(analyze)
app.command(name="clean")(clean)
app.command(name="report")(report)
app.command(name="wizard")(wizard)


# ─── Global Callback (runs before every command) ──────────────────────────────
@app.callback()
def global_options(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging to console."),
    log_dir: Path = typer.Option(Path("./logs"), "--log-dir", help="Directory for log files."),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colored output."),
):
    """
    Global options that apply to ALL commands.

    This callback runs before any sub-command, allowing us to configure
    logging and other global state once, cleanly.
    """
    # Configure logging based on verbosity flag
    log_level = "DEBUG" if verbose else os.getenv("LOG_LEVEL", "INFO")
    setup_logging(log_level=log_level, log_dir=log_dir)


# ─── Version Command ──────────────────────────────────────────────────────────
@app.command()
def version():
    """Show the LazyPanda version and exit."""
    console.print(
        "\n[bold cyan]LazyPanda v0.1.0[/bold cyan] "
        "[bold white]v0.1.0[/bold white]\n"
        "[muted]Production-grade ML dataset cleaning tool[/muted]\n"
    )


# ─── Application Entry Point ──────────────────────────────────────────────────
def main():
    """CLI entry point for the lazypanda command."""
    app()

if __name__ == "__main__":
    main()
