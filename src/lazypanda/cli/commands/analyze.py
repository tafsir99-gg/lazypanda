"""
`lazypanda analyze` command — run all analyzers and display results in the terminal.

WHAT IT DOES:
    1. Loads the CSV file
    2. Loads configuration (default or user-provided)
    3. Runs all 7 analyzers
    4. Prints a beautiful Rich summary table
    5. Optionally saves the full analysis to a JSON file

USAGE:
    lazypanda analyze data/train.csv
    lazypanda analyze data/train.csv --config my_config.yaml
    lazypanda analyze data/train.csv --no-ai
    lazypanda analyze data/train.csv --output results/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn


from lazypanda.analyzers.base import AnalysisResult
from lazypanda.analyzers.cardinality import CardinalityAnalyzer
from lazypanda.analyzers.category_consistency import CategoryConsistencyAnalyzer
from lazypanda.analyzers.duplicates import DuplicateAnalyzer
from lazypanda.analyzers.missing_values import MissingValueAnalyzer
from lazypanda.analyzers.outliers import OutlierAnalyzer
from lazypanda.analyzers.suspicious_values import SuspiciousValueAnalyzer
from lazypanda.analyzers.type_inference import TypeInferenceAnalyzer
from lazypanda.cli.console import console, error_console
from lazypanda.cli.display import print_analysis_table, print_issue_summary
from lazypanda.core.config_manager import ConfigManager
from lazypanda.core.dataset import Dataset
from lazypanda.utils.exceptions import AIDataCleanerError

app = typer.Typer()


@app.command()
def analyze(
    file: Path = typer.Argument(..., help="Path to the CSV file to analyze."),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to a custom config YAML file."
    ),
    no_ai: bool = typer.Option(
        False, "--no-ai", help="Disable all AI calls (fully deterministic run)."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Directory to save analysis results (JSON)."
    ),
    encoding: str = typer.Option("utf-8", "--encoding", help="CSV file encoding."),
):
    """
    Analyze a CSV dataset and report all data quality issues.

    Runs all 7 deterministic analyzers (zero AI calls) and prints
    a structured summary. Use --output to also save a JSON report.
    """
    try:
        _run_analysis(file, config, no_ai, output, encoding)
    except AIDataCleanerError as e:
        error_console.print(f"\n[bold red]✗ Error:[/bold red] {e}\n")
        raise typer.Exit(code=1)


def _run_analysis(
    file: Path,
    config_path: Optional[Path],
    no_ai: bool,
    output_dir: Optional[Path],
    encoding: str,
) -> list[AnalysisResult]:
    """Core analysis logic — separated from CLI boilerplate for testability."""

    # ── 1. Print header ───────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold cyan]LazyPanda v0.1.0[/bold cyan] — Analysis Run\n"
        f"[muted]File:[/muted] [white]{file}[/white]"
        + (f"\n[muted]Config:[/muted] [white]{config_path}[/white]" if config_path else "")
        + (f"\n[warning]⚠  AI disabled (--no-ai)[/warning]" if no_ai else ""),
        border_style="cyan",
        expand=False,
    ))

    # ── 2. Load configuration ─────────────────────────────────────────────────
    app_config = ConfigManager().load(user_config_path=config_path)
    if no_ai:
        # Patch AI config without modifying the Pydantic model
        ai_cfg = app_config.ai.model_copy(update={"enabled": False})
    else:
        ai_cfg = app_config.ai

    # ── 3. Load dataset ───────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Loading dataset...", total=None)
        dataset = Dataset.from_csv(
            file,
            encoding=encoding,
            delimiter=app_config.dataset.delimiter,
        )

    console.print(
        f"\n[success]✓[/success] Loaded [bold]{dataset.meta.n_rows:,}[/bold] rows × "
        f"[bold]{dataset.meta.n_cols}[/bold] columns "
        f"([muted]{dataset.meta.memory_usage_mb:.2f} MB[/muted])\n"
    )

    # ── 4. Build analyzers ────────────────────────────────────────────────────
    cfg = app_config.model_dump()
    analyzers = [
        MissingValueAnalyzer(cfg["missing_values"]),
        DuplicateAnalyzer(cfg["duplicates"]),
        TypeInferenceAnalyzer(cfg.get("type_inference", {})),
        OutlierAnalyzer(cfg["outliers"]),
        CardinalityAnalyzer(cfg["cardinality"]),
        CategoryConsistencyAnalyzer(cfg["categories"]),
        SuspiciousValueAnalyzer(cfg.get("suspicious_values", {})),
    ]

    # ── 5. Run analyzers with progress spinner ────────────────────────────────
    results: list[AnalysisResult] = []
    working_df = dataset.copy()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        for analyzer in analyzers:
            task = progress.add_task(f"Analyzing: {analyzer.display_name}...", total=None)
            result = analyzer._safe_analyze(working_df)
            results.append(result)
            progress.remove_task(task)

    # ── 6. Print results table ────────────────────────────────────────────────────────
    print_analysis_table(results)
    print_issue_summary(results, dataset)

    console.print(
        "[muted]Run [bold]lazypanda clean <file>[/bold] to apply automatic fixes.[/muted]\n"
    )

    # ── 7. Optionally save JSON ─────────────────────────────────────────────────────
    if output_dir:
        _save_json_results(results, dataset, output_dir, file)

    return results




def _save_json_results(
    results: list[AnalysisResult],
    dataset: Dataset,
    output_dir: Path,
    source_file: Path,
) -> None:
    """Save analysis results to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source_file.stem
    out_file = output_dir / f"{stem}_analysis.json"

    payload = {
        "source_file": str(source_file),
        "dataset": {
            "rows": dataset.meta.n_rows,
            "columns": dataset.meta.n_cols,
            "memory_mb": dataset.meta.memory_usage_mb,
        },
        "results": [r.to_dict() for r in results],
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    console.print(f"[success]✓[/success] Analysis saved to [white]{out_file}[/white]\n")
