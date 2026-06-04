"""
`lazypanda wizard` command — interactive step-by-step guided cleaning mode.

WHAT IT DOES:
    Walks the user through every decision with interactive prompts:

    1. Model Selection  — pick a Gemini model tier or skip AI entirely
    2. Max Output Tokens — configure the token budget for AI responses
    3. API Key Check     — detect or securely prompt for GEMINI_API_KEY
    4. Input File Path   — select the CSV to clean
    5. Output Directory  — choose where artifacts are written
    6. "Before" Report   — run analysis, print defects table, save dirty report
    7. Confirmation      — ask the user before running the cleaning pipeline
    8. Cleaning + AI     — run the full pipeline with the user's model choice
    9. Output Artifacts  — write cleaned CSV + audit report
   10. Summary Tables    — print the same beautiful tables as `lazypanda report`

WHY A WIZARD?
    The non-interactive `lazypanda clean` and `lazypanda report` commands are designed for
    power users and CI/CD. The wizard is designed for first-time users who
    want to understand each decision and customize the run interactively.

USAGE:
    lazypanda wizard
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import questionary
import typer
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from lazypanda.cli.console import console, error_console
from lazypanda.cli.display import (
    print_ai_insights_summary,
    print_analysis_table,
    print_before_after,
    print_issue_summary,
    print_outputs_table,
    print_pipeline_table,
)
from lazypanda.core.config_manager import ConfigManager
from lazypanda.core.dataset import Dataset
from lazypanda.core.pipeline import CleaningPipeline, PipelineResult
from lazypanda.reporters.markdown_reporter import MarkdownReporter
from lazypanda.utils.exceptions import AIDataCleanerError

logger = logging.getLogger("lazypanda")


# ── Model choice menu ─────────────────────────────────────────────────────────

# Maps display label → Gemini model ID (or None for offline mode)
_MODEL_CHOICES = [
    questionary.Choice(
        title="gemini-3.5-flash — Recommended · Blazing fast, high-efficiency",
        value="gemini-3.5-flash",
    ),
    questionary.Choice(
        title="gemini-3.1-pro   — Premium · Deep reasoning, best for complex anomalies",
        value="gemini-3.1-pro",
    ),
    questionary.Choice(
        title="gemini-2.5-flash — Stable production baseline",
        value="gemini-2.5-flash",
    ),
    questionary.Choice(
        title="Skip AI / 100% Offline",
        value=None,
    ),
]


def wizard():
    """
    Interactive guided cleaning — choose your model, file, and output step by step.

    Walks you through every decision with beautiful prompts.
    No flags to memorize — just answer the questions.
    """
    try:
        _run_wizard()
    except KeyboardInterrupt:
        console.print("\n[warning]⚠ Wizard cancelled by user.[/warning]\n")
        raise typer.Exit(code=130)
    except AIDataCleanerError as e:
        error_console.print(f"\n[bold red]✗ Error:[/bold red] {e}\n")
        raise typer.Exit(code=1)


def _run_wizard() -> None:
    """Core wizard logic — separated from CLI boilerplate for testability."""

    # ── 1. Welcome banner ─────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold cyan]LazyPanda 🐼[/bold cyan] — Interactive Wizard\n"
        "[muted]Step-by-step guided cleaning mode[/muted]\n"
        "[muted]Answer each prompt to configure your run.[/muted]",
        border_style="cyan",
        expand=False,
    ))
    console.print()

    # ── 2. Model selection ────────────────────────────────────────────────────
    model_choice: str | None = questionary.select(
        "Select AI model tier:",
        choices=_MODEL_CHOICES,
        default="gemini-3.5-flash",
    ).ask()

    # questionary.ask() returns None on Ctrl+C
    if model_choice is False:
        raise KeyboardInterrupt

    use_ai = model_choice is not None

    if use_ai:
        console.print(f"  [success]✓[/success] Model: [bold]{model_choice}[/bold]")
    else:
        console.print("  [success]✓[/success] Mode: [bold]100% Offline[/bold] — no API calls")

    # ── 3. Max output tokens ──────────────────────────────────────────────────
    max_tokens = 8192
    if use_ai:
        token_input = questionary.text(
            "Max output tokens (default 8192):",
            default="8192",
            validate=lambda val: (
                True if val.strip().isdigit() and int(val.strip()) >= 256
                else "Must be an integer ≥ 256"
            ),
        ).ask()

        if token_input is None:
            raise KeyboardInterrupt

        max_tokens = int(token_input.strip())
        console.print(f"  [success]✓[/success] Token budget: [bold]{max_tokens:,}[/bold]")

    # ── 4. API key validation ─────────────────────────────────────────────────
    if use_ai:
        existing_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if existing_key:
            masked = existing_key[:4] + "…" + existing_key[-4:]
            console.print(f"  [success]✓[/success] GEMINI_API_KEY detected ({masked})")
        else:
            console.print(
                "\n  [warning]⚠  GEMINI_API_KEY not found in environment.[/warning]"
            )
            api_key = questionary.password(
                "Enter your Gemini API key (input hidden):"
            ).ask()

            if not api_key or not api_key.strip():
                console.print("  [dim]No key provided — switching to offline mode.[/dim]")
                use_ai = False
                model_choice = None
            else:
                os.environ["GEMINI_API_KEY"] = api_key.strip()
                console.print("  [success]✓[/success] API key set for this session")

    # ── 5. Input file path ────────────────────────────────────────────────────
    file_path_str = questionary.path(
        "Path to your CSV file:",
        validate=lambda p: (
            True if Path(p).is_file() and Path(p).suffix.lower() == ".csv"
            else "File must exist and have a .csv extension"
        ),
    ).ask()

    if file_path_str is None:
        raise KeyboardInterrupt

    file_path = Path(file_path_str).resolve()
    stem = file_path.stem
    console.print(f"  [success]✓[/success] Input: [bold]{file_path}[/bold]")

    # ── 6. Output directory ───────────────────────────────────────────────────
    output_dir_str = questionary.text(
        "Output directory (default ./outputs):",
        default="./outputs",
    ).ask()

    if output_dir_str is None:
        raise KeyboardInterrupt

    output_dir = Path(output_dir_str.strip()).resolve()
    os.makedirs(output_dir, exist_ok=True)
    console.print(f"  [success]✓[/success] Output: [bold]{output_dir}[/bold]")
    console.print()

    # ── 7. "Before" report — dirty analysis ───────────────────────────────────
    console.print(Panel(
        "[bold cyan]Phase 1:[/bold cyan] Pre-Cleaning Analysis\n"
        "[muted]Running all 7 analyzers on the raw dataset...[/muted]",
        border_style="yellow",
        expand=False,
    ))

    # Load config with token override
    overrides = {}
    if use_ai:
        overrides["ai"] = {"max_tokens_per_run": max_tokens}
    app_config = ConfigManager().load(overrides=overrides if overrides else None)

    # Load dataset
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as p:
        p.add_task("Loading dataset...", total=None)
        dataset = Dataset.from_csv(file_path, encoding=app_config.dataset.encoding)

    console.print(
        f"\n[success]✓[/success] Loaded [bold]{dataset.meta.n_rows:,}[/bold] rows × "
        f"[bold]{dataset.meta.n_cols}[/bold] columns "
        f"([muted]{dataset.meta.memory_usage_mb:.2f} MB[/muted])\n"
    )

    # Run analysis-only to produce the "before" report
    pipeline = CleaningPipeline(app_config)
    analysis_results = pipeline._run_analyzers(dataset.copy())

    # Print the analysis table in the terminal
    print_analysis_table(analysis_results)
    print_issue_summary(analysis_results, dataset)

    # Build a minimal PipelineResult for the "dirty" Markdown report
    dirty_result = PipelineResult(
        original_shape=dataset.df.shape,
        final_shape=dataset.df.shape,
        analysis_results=analysis_results,
        cleaning_results=[],
        cleaned_df=dataset.df,
        dry_run=True,
        stats={
            "rows_original": dataset.df.shape[0],
            "rows_final": dataset.df.shape[0],
            "rows_removed": 0,
            "cols_original": dataset.df.shape[1],
            "cols_final": dataset.df.shape[1],
            "cols_removed": 0,
            "issues_found": sum(1 for r in analysis_results if r.issues_found),
            "cleaners_applied": 0,
            "cleaners_skipped": 0,
        },
    )

    dirty_report_path = output_dir / f"{stem}_dirty_report.md"
    MarkdownReporter().write(
        dirty_result,
        dirty_report_path,
        source_file=str(file_path),
    )
    console.print(
        f"[success]✓[/success] Pre-cleaning report saved to "
        f"[bold white]{dirty_report_path}[/bold white]\n"
    )

    # ── 8. Confirmation prompt ────────────────────────────────────────────────
    proceed = questionary.confirm(
        "Proceed to execute automated cleaning filters?",
        default=True,
    ).ask()

    if not proceed:
        console.print("\n[warning]⚠ Wizard stopped by user. No cleaning was performed.[/warning]")
        console.print(
            f"[muted]Your pre-cleaning report is still available at: {dirty_report_path}[/muted]\n"
        )
        raise typer.Exit(code=0)

    # ── 9. Cleaning + AI enrichment ───────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold cyan]Phase 2:[/bold cyan] Cleaning Pipeline"
        + (f" + AI ({model_choice})" if use_ai else " (Offline)")
        + "\n[muted]Running all cleaners"
        + (" and Gemini AI enrichment..." if use_ai else "...[/muted]"),
        border_style="green",
        expand=False,
    ))

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as p:
        p.add_task("Running cleaning pipeline...", total=None)
        result = pipeline.run(
            dataset.copy(),
            dry_run=False,
            use_ai=use_ai,
            source_file=file_path.name,
            model_override=model_choice,
        )

    # ── 10. Write output artifacts ────────────────────────────────────────────
    cleaned_csv_path = output_dir / f"{stem}_cleaned.csv"
    audit_report_path = output_dir / f"{stem}_audit_report.md"

    pipeline.save(result, cleaned_csv_path)

    output_files_str = {
        "Cleaned CSV": str(cleaned_csv_path),
        "Audit Report": str(audit_report_path),
        "Dirty Report": str(dirty_report_path),
    }

    MarkdownReporter().write(
        result,
        audit_report_path,
        source_file=str(file_path),
        output_files=output_files_str,
    )

    written = {
        "Cleaned CSV":  cleaned_csv_path,
        "Audit Report": audit_report_path,
        "Dirty Report": dirty_report_path,
    }

    # ── 11. Terminal summary tables ───────────────────────────────────────────
    print_pipeline_table(result)
    print_before_after(result)
    print_ai_insights_summary(result)
    print_outputs_table(written, dry_run=False)

    console.print(
        f"\n[success]✓[/success] Wizard complete — "
        f"[bold]{len(written)}[/bold] artifact(s) written to "
        f"[bold white]{output_dir}[/bold white]"
    )
    console.print()
