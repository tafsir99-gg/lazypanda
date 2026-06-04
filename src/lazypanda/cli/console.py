"""
Centralized Rich console configuration.

WHY centralize this?
    Rich's Console object controls ALL terminal output — colors, tables,
    progress bars. Having one shared instance ensures consistent styling
    and makes it easy to globally disable colors (e.g., for CI pipelines).

USAGE in any command file:
    from lazypanda.cli.console import console, error_console
    console.print("[green]✓[/green] Analysis complete")
    error_console.print("[red]✗[/red] File not found")
"""

from rich.console import Console
from rich.theme import Theme

# ─── Custom Color Theme ───────────────────────────────────────────────────────
# Defining semantic color names means we can change the look in one place.
_theme = Theme({
    "success":  "bold green",
    "warning":  "bold yellow",
    "error":    "bold red",
    "info":     "bold cyan",
    "muted":    "dim white",
    "header":   "bold white",
    "critical": "bold red on dark_red",
    "ai":       "bold magenta",  # Used for AI-generated content
})

# ─── Main Console (stdout) ────────────────────────────────────────────────────
# Used for all normal output: analysis results, progress, success messages.
console = Console(theme=_theme)

# ─── Error Console (stderr) ───────────────────────────────────────────────────
# Used ONLY for error messages. Separating stdout/stderr lets users pipe
# the real output while still seeing errors in their terminal.
error_console = Console(stderr=True, theme=_theme)
