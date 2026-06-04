"""
`lazypanda clean` command — run the full pipeline and save a cleaned CSV.

WHAT IT DOES:
    1. Loads a dataset
    2. Runs all 7 analyzers (same as `lazypanda analyze`)
    3. Runs all 6 cleaners automatically based on analyzer results
    4. Enriches results with Gemini AI (if enabled and key present)
    5. Saves the cleaned DataFrame to disk
    6. Optionally saves a lightweight JSON report

USAGE:
    lazypanda clean data/train.csv
    lazypanda clean data/train.csv --output cleaned/train_cleaned.csv
    lazypanda clean data/train.csv --dry-run          # Preview only — no file written
    lazypanda clean data/train.csv --no-ai            # Skip AI calls entirely
    lazypanda clean data/train.csv --action cap       # Override outlier action
    lazypanda clean data/train.csv --config my.yaml   # Use custom config
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from lazypanda.cli.console import console, error_console
from lazypanda.cli.display import print_before_after, print_pipeline_table
from lazypanda.core.config_manager import ConfigManager
from lazypanda.core.dataset import Dataset
from lazypanda.core.pipeline import CleaningPipeline, PipelineResult
from lazypanda.utils.exceptions import AIDataCleanerError

# Severity icons (reused from analyze.py)
_APPLIED_STYLE = {"True": "bold green", "False": "dim"}
_APPLIED_ICON  = {"True": "✅", "False": "⏭"}


def clean(
    file: Path = typer.Argument(..., help="Path to the CSV file to clean."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output path for the cleaned CSV. Defaults to <file>_cleaned.csv in the same directory.",
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to a custom config YAML file."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n",
        help="Simulate the pipeline but do NOT write any files. Shows what would change.",
    ),
    no_ai: bool = typer.Option(
        False, "--no-ai", help="Disable all AI calls (fully deterministic run)."
    ),
    action: Optional[str] = typer.Option(
        None, "--action",
        help="Override outlier action: 'flag', 'cap', or 'drop'. Overrides config.",
    ),
    encoding: str = typer.Option("utf-8", "--encoding", help="CSV file encoding."),
    save_report: bool = typer.Option(
        False, "--save-report", help="Also save a JSON analysis report alongside the cleaned CSV."
    ),
):
    """
    Clean a CSV dataset using all deterministic analyzers and cleaners.

    Runs the full pipeline: analyze → clean → save.
    Use --dry-run to preview changes without writing any files.
    """
    try:
        _run_clean(file, output, config, dry_run, no_ai, action, encoding, save_report)
    except AIDataCleanerError as e:
        error_console.print(f"\n[bold red]✗ Error:[/bold red] {e}\n")
        raise typer.Exit(code=1)


def _run_clean(
    file: Path,
    output: Optional[Path],
    config_path: Optional[Path],
    dry_run: bool,
    no_ai: bool,
    action_override: Optional[str],
    encoding: str,
    save_report: bool,
) -> PipelineResult:
    """Core pipeline logic — separated from CLI boilerplate for testability."""

    # ── 1. Header ─────────────────────────────────────────────────────────────
    console.print()
    mode_label = "[bold yellow]DRY RUN[/bold yellow] — no files will be written" if dry_run else "Writing cleaned output"
    console.print(Panel(
        f"[bold cyan]LazyPanda v0.1.0[/bold cyan] — Clean Run\n"
        f"[muted]File:[/muted] [white]{file}[/white]\n"
        f"[muted]Mode:[/muted] {mode_label}"
        + (f"\n[warning]⚠  AI disabled (--no-ai)[/warning]" if no_ai else ""),
        border_style="cyan",
        expand=False,
    ))

    # ── 2. Load config ─────────────────────────────────────────────────────────
    overrides: dict = {}
    if action_override:
        overrides = {"outliers": {"action": action_override}}

    app_config = ConfigManager().load(user_config_path=config_path, overrides=overrides)

    # ── 3. Load dataset ────────────────────────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as p:
        p.add_task("Loading dataset...", total=None)
        dataset = Dataset.from_csv(file, encoding=encoding, delimiter=app_config.dataset.delimiter)

    console.print(
        f"\n[success]✓[/success] Loaded [bold]{dataset.meta.n_rows:,}[/bold] rows × "
        f"[bold]{dataset.meta.n_cols}[/bold] columns\n"
    )

    # ── 4. Run pipeline ────────────────────────────────────────────────────────
    pipeline = CleaningPipeline(app_config)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as p:
        p.add_task("Running cleaning pipeline...", total=None)
        result = pipeline.run(
            dataset.copy(),
            dry_run=dry_run,
            use_ai=not no_ai,
            source_file=file.name,
        )

    # ── 5. Print results ───────────────────────────────────────────────────────
    print_pipeline_table(result)
    print_before_after(result)

    # ── 6. Save output ─────────────────────────────────────────────────────────
    if not dry_run:
        out_path = output or file.parent / f"{file.stem}_cleaned.csv"
        pipeline.save(result, out_path)
        console.print(f"[success]✓[/success] Cleaned data saved to [bold white]{out_path}[/bold white]")

        if save_report:
            report_path = out_path.with_suffix(".analysis.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            console.print(f"[success]✓[/success] Report saved to [bold white]{report_path}[/bold white]")
    else:
        console.print("\n[bold yellow]DRY RUN complete — no files written.[/bold yellow]")

    console.print()
    return result

