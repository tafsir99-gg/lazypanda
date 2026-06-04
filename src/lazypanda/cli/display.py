"""
Shared CLI Display Utilities.

WHAT IT DOES:
    Provides reusable Rich table-printing functions used by multiple CLI
    commands (analyze, clean, report, wizard). Centralizing these avoids
    copy-paste duplication and ensures consistent terminal output.

WHY A SEPARATE MODULE?
    Before this module, each CLI command had its own private _print_*
    functions. The wizard command needs the same tables, so we extracted
    them here. Now any command can render analysis tables, cleaning tables,
    output artifact tables, and AI insight summaries with a single import.

USAGE:
    from lazypanda.cli.display import (
        print_analysis_table,
        print_issue_summary,
        print_pipeline_table,
        print_before_after,
        print_outputs_table,
        print_ai_insights_summary,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich import box
from rich.panel import Panel
from rich.table import Table

from lazypanda.cli.console import console

if TYPE_CHECKING:
    from lazypanda.analyzers.base import AnalysisResult
    from lazypanda.core.dataset import Dataset
    from lazypanda.core.pipeline import PipelineResult


# ── Severity styling (for analysis results) ──────────────────────────────────

_SEVERITY_STYLE = {
    "critical": "bold red",
    "warning":  "bold yellow",
    "info":     "cyan",
    "ok":       "bold green",
}
_SEVERITY_ICON = {
    "critical": "🔴",
    "warning":  "🟡",
    "info":     "🔵",
    "ok":       "✅",
}


# ── Analysis Tables ───────────────────────────────────────────────────────────

def print_analysis_table(results: list[AnalysisResult]) -> None:
    """Print a Rich table summarizing all analyzer findings."""
    table = Table(
        title="Analysis Results",
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
        header_style="bold white",
        expand=False,
    )
    table.add_column("Analyzer",       style="white",  min_width=22)
    table.add_column("Status",         style="bold",   min_width=10, justify="center")
    table.add_column("Severity",       min_width=10,   justify="center")
    table.add_column("Affected Cols",  min_width=6,    justify="center")
    table.add_column("Finding",        min_width=40)

    for result in results:
        severity_style = _SEVERITY_STYLE.get(result.severity, "white")
        severity_icon  = _SEVERITY_ICON.get(result.severity, "")
        status = "[bold green]PASS[/bold green]" if not result.issues_found else "[bold red]FAIL[/bold red]"
        affected_count = str(len(result.affected_columns)) if result.affected_columns else "—"
        summary = result.summary if len(result.summary) <= 55 else result.summary[:52] + "..."

        table.add_row(
            result.display_name,
            status,
            f"[{severity_style}]{severity_icon} {result.severity.upper()}[/{severity_style}]",
            affected_count,
            f"[{severity_style}]{summary}[/{severity_style}]",
        )

    console.print(table)


def print_issue_summary(results: list[AnalysisResult], dataset: Dataset) -> None:
    """Print a brief issue summary with recommendations."""
    issues = [r for r in results if r.issues_found]
    passes = [r for r in results if not r.issues_found]

    console.print()
    console.print(
        f"[bold]Summary:[/bold] "
        f"[bold red]{len(issues)} issue(s)[/bold red] detected, "
        f"[bold green]{len(passes)} check(s)[/bold green] passed"
    )

    if issues:
        console.print()
        console.print("[bold white]Recommendations:[/bold white]")
        for r in issues:
            icon = _SEVERITY_ICON.get(r.severity, "•")
            style = _SEVERITY_STYLE.get(r.severity, "white")
            console.print(f"  [{style}]{icon} {r.display_name}:[/{style}] {r.recommendation}")

    console.print()


# ── Cleaning Pipeline Tables ──────────────────────────────────────────────────

def print_pipeline_table(result: PipelineResult) -> None:
    """Print a compact cleaning results table."""
    table = Table(
        title="Cleaning Pipeline",
        box=box.ROUNDED, border_style="cyan",
        show_header=True, header_style="bold white",
    )
    table.add_column("Cleaner",   style="white",  min_width=22)
    table.add_column("Status",    justify="center", min_width=10)
    table.add_column("Rows",      justify="center", min_width=14)
    table.add_column("Summary",   min_width=40)

    for r in result.cleaning_results:
        if r.applied:
            status = "[bold green]APPLIED[/bold green]"
            row_str = f"{r.rows_before:,} → [bold]{r.rows_after:,}[/bold]"
        else:
            status = "[dim]SKIPPED[/dim]"
            row_str = f"[dim]{r.rows_before:,}[/dim]"

        table.add_row(r.display_name, status, row_str, r.summary)

    console.print()
    console.print(table)


def print_before_after(result: PipelineResult) -> None:
    """Print a before/after summary panel."""
    orig_rows, orig_cols = result.original_shape
    final_rows, final_cols = result.final_shape
    rows_removed = orig_rows - final_rows
    cols_removed = orig_cols - final_cols
    cleaners_applied = sum(1 for r in result.cleaning_results if r.applied)

    row_delta = f"[bold red]-{rows_removed}[/bold red]" if rows_removed > 0 else "[dim]±0[/dim]"
    col_delta = f"[bold red]-{cols_removed}[/bold red]" if cols_removed > 0 else "[dim]±0[/dim]"

    console.print()
    console.print(Panel(
        f"[bold white]Before:[/bold white] {orig_rows:,} rows × {orig_cols} cols\n"
        f"[bold white]After: [/bold white] {final_rows:,} rows × {final_cols} cols  "
        f"({row_delta} rows, {col_delta} cols)\n"
        f"[bold white]Applied:[/bold white] {cleaners_applied}/{len(result.cleaning_results)} cleaner(s)",
        title="[bold cyan]Before → After[/bold cyan]",
        border_style="green" if not result.dry_run else "yellow",
        expand=False,
    ))


# ── Output Artifacts Table ────────────────────────────────────────────────────

def print_outputs_table(
    paths: dict[str, Path],
    dry_run: bool,
) -> None:
    """Print a table of all output files that would be / were written."""
    table = Table(
        title="Output Artifacts" + (" [DRY RUN]" if dry_run else ""),
        box=box.ROUNDED, border_style="green" if not dry_run else "yellow",
        show_header=True, header_style="bold white",
    )
    table.add_column("Artifact",  style="white",  min_width=18)
    table.add_column("Path",      style="dim",    min_width=50)
    table.add_column("Status",    justify="center", min_width=10)

    status_str = "[bold yellow]PREVIEW[/bold yellow]" if dry_run else "[bold green]WRITTEN[/bold green]"

    for label, path in paths.items():
        table.add_row(label, str(path), status_str)

    console.print()
    console.print(table)


# ── AI Insights Summary ──────────────────────────────────────────────────────

def print_ai_insights_summary(result: PipelineResult) -> None:
    """Print a one-line AI insights status in the terminal."""
    ai = result.ai_insights
    if ai is None:
        return

    if not ai.skipped:
        cache_tag = " [dim](from cache)[/dim]" if ai.from_cache else ""
        console.print(
            f"\n[bold cyan]✨ AI Insights:[/bold cyan]{cache_tag} "
            f"Quality score [bold]{ai.data_quality_score}/100[/bold] · "
            f"{len(ai.recommendations)} recommendation(s) · "
            f"{ai.tokens_used} tokens used"
        )
    else:
        console.print(
            f"\n[dim]AI insights skipped: {ai.skip_reason}[/dim]"
        )
