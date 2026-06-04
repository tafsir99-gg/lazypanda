"""
`lazypanda report` command — run the full pipeline and generate all output files.

WHAT IT DOES:
    1. Loads a dataset
    2. Runs ALL 7 analyzers + ALL 6 cleaners (same as `lazypanda clean`)
    3. Enriches results with Gemini AI (if enabled and key present)
    4. Writes a comprehensive suite of artifacts to the output directory:
       - The cleaned CSV
       - A detailed Markdown Audit Report (with AI insights)
       - A standalone Python script containing deterministic pandas code
       - The full Pipeline Configuration JSON

USAGE:
    lazypanda report data/train.csv
    lazypanda report data/train.csv --output-dir reports/titanic/
    lazypanda report data/train.csv --no-script    # Skip Python script export
    lazypanda report data/train.csv --dry-run      # Preview without writing files

WHY IT MATTERS:
    Transparency is critical in ML. This command generates an artifact suite
    that can be checked into git, attached to a PR, or shared with stakeholders
    to prove exactly what transformations were applied to the raw data and why.

    `lazypanda clean` is for fast, interactive use. `lazypanda report` is for producing
    the permanent artifacts needed for enterprise compliance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn


from lazypanda.cli.console import console, error_console
from lazypanda.cli.display import (
    print_ai_insights_summary,
    print_outputs_table,
    print_pipeline_table,
)
from lazypanda.core.config_manager import ConfigManager
from lazypanda.core.dataset import Dataset
from lazypanda.core.pipeline import CleaningPipeline, PipelineResult
from lazypanda.exporters.config_exporter import ConfigExporter
from lazypanda.exporters.pipeline_exporter import PipelineExporter
from lazypanda.reporters.json_reporter import JSONReporter
from lazypanda.reporters.markdown_reporter import MarkdownReporter
from lazypanda.utils.exceptions import AIDataCleanerError


def report(
    file: Path = typer.Argument(..., help="Path to the CSV file to analyze and clean."),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-d",
        help="Directory for all output files. Defaults to <file_dir>/<stem>_adc_report/",
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to a custom config YAML file."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n",
        help="Simulate the pipeline but do NOT write any files.",
    ),
    no_ai: bool = typer.Option(
        False, "--no-ai", help="Disable all AI calls (fully deterministic run)."
    ),
    encoding: str = typer.Option("utf-8", "--encoding", help="CSV encoding."),
    no_script: bool = typer.Option(
        False, "--no-script", help="Skip generating the standalone Python cleaning script."
    ),
    no_config: bool = typer.Option(
        False, "--no-config", help="Skip saving the run config YAML."
    ),
):
    """
    Run the full pipeline and generate all output artifacts.

    Produces: cleaned CSV + Markdown report + JSON summary + Python script + run config.
    Use --dry-run to preview what would be generated without writing any files.
    """
    try:
        _run_report(file, output_dir, config, dry_run, no_ai, encoding, no_script, no_config)
    except AIDataCleanerError as e:
        error_console.print(f"\n[bold red]✗ Error:[/bold red] {e}\n")
        raise typer.Exit(code=1)


def _run_report(
    file: Path,
    output_dir: Optional[Path],
    config_path: Optional[Path],
    dry_run: bool,
    no_ai: bool,
    encoding: str,
    no_script: bool,
    no_config: bool,
) -> None:
    """Core reporting logic — separated from CLI boilerplate for testability."""

    # ── 1. Header ──────────────────────────────────────────────────────────────
    console.print()
    mode_label = (
        "[bold yellow]DRY RUN[/bold yellow] — no files will be written"
        if dry_run else "Generating full artifact set"
    )
    console.print(Panel(
        f"[bold cyan]LazyPanda v0.1.0[/bold cyan] — Report Run\n"
        f"[muted]File:[/muted] [white]{file}[/white]\n"
        f"[muted]Mode:[/muted] {mode_label}"
        + (f"\n[warning]⚠  AI disabled (--no-ai)[/warning]" if no_ai else ""),
        border_style="cyan",
        expand=False,
    ))

    # ── 2. Resolve output directory ─────────────────────────────────────────────
    file = file.resolve()
    stem = file.stem
    if output_dir is None:
        output_dir = file.parent / f"{stem}_adc_report"
    output_dir = Path(output_dir)

    # ── 3. Load config ──────────────────────────────────────────────────────────
    app_config = ConfigManager().load(user_config_path=config_path)

    # ── 4. Load dataset ─────────────────────────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as p:
        p.add_task("Loading dataset...", total=None)
        dataset = Dataset.from_csv(file, encoding=encoding, delimiter=app_config.dataset.delimiter)

    console.print(
        f"\n[success]✓[/success] Loaded [bold]{dataset.meta.n_rows:,}[/bold] rows × "
        f"[bold]{dataset.meta.n_cols}[/bold] columns\n"
    )

    # ── 5. Run pipeline ─────────────────────────────────────────────────────────
    pipeline = CleaningPipeline(app_config)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as p:
        p.add_task("Running cleaning pipeline...", total=None)
        result = pipeline.run(
            dataset.copy(),
            dry_run=dry_run,
            use_ai=not no_ai,
            source_file=file.name,
        )

    # ── 6. Plan output paths ────────────────────────────────────────────────────
    paths = {
        "Cleaned CSV":       output_dir / f"{stem}_cleaned.csv",
        "Markdown Report":   output_dir / f"{stem}_report.md",
        "JSON Summary":      output_dir / f"{stem}_summary.json",
        "Cleaning Script":   output_dir / f"{stem}_cleaning_script.py",
        "Run Config":        output_dir / f"{stem}_run_config.yaml",
    }
    if no_script:
        del paths["Cleaning Script"]
    if no_config:
        del paths["Run Config"]

    output_files_str = {label: str(path) for label, path in paths.items()}

    # ── 7. Write all artifacts ──────────────────────────────────────────────────
    written: dict[str, Path] = {}

    if not dry_run:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as p:
            task = p.add_task("Writing output files...", total=None)

            # 7a. Cleaned CSV
            pipeline.save(result, paths["Cleaned CSV"])
            written["Cleaned CSV"] = paths["Cleaned CSV"]

            # 7b. Markdown report
            MarkdownReporter().write(
                result,
                paths["Markdown Report"],
                source_file=str(file),
                output_files=output_files_str,
            )
            written["Markdown Report"] = paths["Markdown Report"]

            # 7c. JSON summary
            JSONReporter().write(
                result,
                paths["JSON Summary"],
                source_file=str(file),
                output_files=output_files_str,
            )
            written["JSON Summary"] = paths["JSON Summary"]

            # 7d. Python cleaning script
            if "Cleaning Script" in paths:
                PipelineExporter().write(
                    result,
                    paths["Cleaning Script"],
                    source_file=str(file),
                    output_file=str(paths["Cleaned CSV"]),
                )
                written["Cleaning Script"] = paths["Cleaning Script"]

            # 7e. Run config YAML
            if "Run Config" in paths:
                ConfigExporter().write(app_config, paths["Run Config"])
                written["Run Config"] = paths["Run Config"]

    # ── 8. Print results table ──────────────────────────────────────────────────
    print_pipeline_table(result)
    print_outputs_table(written if not dry_run else paths, dry_run)

    if dry_run:
        console.print("\n[bold yellow]DRY RUN complete — no files written.[/bold yellow]")
    else:
        print_ai_insights_summary(result)
        console.print(
            f"\n[success]✓[/success] All {len(written)} artifact(s) written to "
            f"[bold white]{output_dir}[/bold white]"
        )
    console.print()

