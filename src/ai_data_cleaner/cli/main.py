"""
AI Data Cleaner — CLI Entry Point

This file is the ONLY entry point for the `adc` command.

WHY structure it this way?
    Typer allows us to compose sub-commands from separate files.
    This keeps main.py clean — it only registers commands, it never
    contains business logic. Each command lives in its own file.

REGISTERED COMMANDS (added progressively per milestone):
    adc analyze <file>   — Run all analyzers, show report in terminal
    adc clean <file>     — Run full cleaning pipeline
    adc report <file>    — Generate reports from existing analysis
    adc version          — Show tool version
"""

import os
from pathlib import Path

import typer
from dotenv import load_dotenv

from ai_data_cleaner.cli.commands.analyze import analyze
from ai_data_cleaner.cli.console import console, error_console
from ai_data_cleaner.utils.logging_setup import setup_logging

# Load .env file BEFORE anything else so all modules can access env vars
load_dotenv()

# ─── Application Definition ───────────────────────────────────────────────────
app = typer.Typer(
    name="adc",
    help=(
        "[bold cyan]AI Data Cleaner[/bold cyan] — Professional ML dataset cleaning tool.\n\n"
        "Analyzes and cleans CSV datasets with deterministic Python + optional AI insights.\n"
        "Run [bold]adc <command> --help[/bold] for detailed usage."
    ),
    add_completion=False,   # Disable shell completion for simplicity in v1
    rich_markup_mode="rich",
    pretty_exceptions_enable=True,
    pretty_exceptions_show_locals=False,  # Don't show variable values in errors (security)
)

# ─── Register commands ────────────────────────────────────────────────────────
app.command(name="analyze")(analyze)

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
    """Show the AI Data Cleaner version and exit."""
    console.print(
        "\n[bold cyan]AI Data Cleaner[/bold cyan] "
        "[bold white]v0.1.0[/bold white]\n"
        "[muted]Production-grade ML dataset cleaning tool[/muted]\n"
    )


# ─── Application Entry Point ──────────────────────────────────────────────────
if __name__ == "__main__":
    app()
