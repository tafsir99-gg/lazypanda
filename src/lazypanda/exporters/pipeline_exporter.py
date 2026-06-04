"""
Pipeline Exporter.

WHAT IT DOES:
    Generates a standalone Python script that exactly reproduces the
    cleaning operations that were applied to the dataset.

WHY IS THIS VALUABLE?
    Reproducibility is one of the most important properties in ML engineering.

    Scenario: You run `lazypanda clean train.csv` and get a cleaned file. Six months
    later, you receive new raw data that needs the same cleaning. Instead of
    re-running the full tool and hoping the config is the same, you run the
    exported script — guaranteed identical operations.

    The exported script:
    - Has ZERO external dependencies except pandas and numpy
    - Has ZERO dependency on ai-data-cleaner (it's standalone)
    - Is readable, commented, and shows exactly what was done
    - Can be committed to your ML project's repo

WHAT THE SCRIPT CONTAINS:
    1. Header comment — file, date, config summary
    2. Imports section
    3. One function per applied cleaner
    4. A main() function that calls them in order
    5. A __main__ guard so it's importable AND runnable

IMPORTANT: Only APPLIED cleaners are exported. Skipped cleaners are omitted.

CODE GENERATION DESIGN:
    Instead of dedent(f-string) templates (which are tricky with dynamic
    multi-line substitutions), we use a simple CodeBuilder helper that
    tracks indent level and appends lines. This is the same approach used
    by Python's compile module and many serious code generators.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from lazypanda.cleaners.base import CleaningResult
from lazypanda.core.pipeline import PipelineResult

logger = logging.getLogger("lazypanda")

_TOOL_VERSION = "0.1.0"
_I = "    "  # 4-space indent unit


# ── Minimal code builder ──────────────────────────────────────────────────────

class _Code:
    """
    Lightweight line-by-line code builder.

    Usage:
        c = _Code()
        c.line("def foo():")
        with c.indent():
            c.line('return 42')
        print(c.text())
    """
    def __init__(self):
        self._lines: list[str] = []
        self._indent_level: int = 0

    def line(self, text: str = "") -> None:
        if text.strip() == "":
            self._lines.append("")
        else:
            self._lines.append(_I * self._indent_level + text)

    def blank(self) -> None:
        self._lines.append("")

    def text(self) -> str:
        return "\n".join(self._lines)

    class _IndentCtx:
        def __init__(self, code: "_Code"):
            self._code = code
        def __enter__(self):
            self._code._indent_level += 1
            return self
        def __exit__(self, *_):
            self._code._indent_level -= 1

    def indent(self) -> "_IndentCtx":
        return self._IndentCtx(self)


# ── Exporter ──────────────────────────────────────────────────────────────────

class PipelineExporter:
    """
    Generates a standalone Python script that reproduces the cleaning pipeline.

    USAGE:
        exporter = PipelineExporter()
        path = exporter.write(result, output_path=Path("outputs/cleaning_script.py"),
                              source_file="train.csv")
    """

    def generate(
        self,
        result: PipelineResult,
        source_file: str = "input.csv",
        output_file: str = "cleaned_output.csv",
    ) -> str:
        """
        Generate the full Python script as a string.

        Args:
            result:       The PipelineResult from which to read applied operations.
            source_file:  The original input file path (used in the script header).
            output_file:  The default output path embedded in the script.

        Returns:
            Complete Python script as a string.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        applied = [cr for cr in result.cleaning_results if cr.applied]

        parts: list[str] = []
        parts.append(self._header(source_file, now, len(applied)))
        parts.append(self._imports())

        for cr in applied:
            fn = self._cleaner_function(cr)
            if fn:
                parts.append(fn)

        parts.append(self._main_function(source_file, output_file, applied))
        parts.append(self._entry_point())

        return "\n\n".join(parts) + "\n"

    def write(
        self,
        result: PipelineResult,
        output_path: Path,
        source_file: str = "input.csv",
        output_file: str = "cleaned_output.csv",
    ) -> Path:
        """
        Generate and write the Python script to disk.

        Returns the path that was written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        script = self.generate(result, source_file=source_file, output_file=output_file)
        output_path.write_text(script, encoding="utf-8")
        logger.info("PipelineExporter: wrote cleaning script to %s", output_path)
        return output_path

    # ── Private code generation helpers ──────────────────────────────────────

    def _header(self, source_file: str, generated_at: str, n_applied: int) -> str:
        lines = [
            "# ============================================================",
            "# LazyPanda 🐼 — Reproducible Cleaning Script",
            "# ============================================================",
            f"# Generated:     {generated_at}",
            f"# Source file:   {source_file}",
            f"# Tool version:  {_TOOL_VERSION}",
            f"# Cleaners applied: {n_applied}",
            "#",
            "# HOW TO USE:",
            "#   python cleaning_script.py",
            "#   python cleaning_script.py --input new_data.csv --output cleaned_new_data.csv",
            "#",
            "# DEPENDENCIES: pandas, numpy (no ai-data-cleaner required)",
            "# ============================================================",
        ]
        return "\n".join(lines)

    def _imports(self) -> str:
        return (
            "import argparse\n"
            "from pathlib import Path\n"
            "\n"
            "import numpy as np\n"
            "import pandas as pd"
        )

    def _cleaner_function(self, cr: CleaningResult) -> str | None:
        """Dispatch to the correct code generator for each cleaner type."""
        generators = {
            "drop_duplicates":     self._gen_drop_duplicates,
            "constant_columns":    self._gen_constant_columns,
            "type_caster":         self._gen_type_caster,
            "category_normalizer": self._gen_category_normalizer,
            "missing_values":      self._gen_missing_values,
            "outlier_handler":     self._gen_outlier_handler,
        }
        gen = generators.get(cr.cleaner_name)
        if gen is None:
            logger.warning("PipelineExporter: no generator for cleaner '%s'", cr.cleaner_name)
            return None
        return gen(cr)

    def _gen_drop_duplicates(self, cr: CleaningResult) -> str:
        drop_ops = [op for op in cr.operations if op.type == "drop_rows"]
        keep = "first"
        for op in cr.operations:
            if "keep" in op.details:
                keep = op.details["keep"]
        removed = drop_ops[0].details.get("rows_dropped", 0) if drop_ops else cr.rows_removed

        c = _Code()
        c.line("def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:")
        with c.indent():
            c.line('"""')
            c.line("Remove exact duplicate rows.")
            c.line(f"Applied: Removed {removed} duplicate row(s) using keep='{keep}'.")
            c.line('"""')
            c.line(f'return df.drop_duplicates(keep="{keep}").reset_index(drop=True)')
        return c.text()

    def _gen_constant_columns(self, cr: CleaningResult) -> str:
        drop_ops = [op for op in cr.operations if op.type == "drop_column"]
        cols = [op.details.get("column", "?") for op in drop_ops]
        cols_repr = repr(cols)

        c = _Code()
        c.line("def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:")
        with c.indent():
            c.line('"""')
            c.line("Drop columns that contain only one unique value.")
            c.line(f"Applied: Dropped {len(cols)} column(s): {cols}")
            c.line('"""')
            c.line(f"cols_to_drop = {cols_repr}")
            c.line("return df.drop(columns=[col for col in cols_to_drop if col in df.columns])")
        return c.text()

    def _gen_type_caster(self, cr: CleaningResult) -> str:
        cast_ops = [op for op in cr.operations if op.type == "cast_column"]
        if not cast_ops:
            return ""

        summary = "; ".join(
            f"{op.details.get('column')} → {op.details.get('to_dtype')}" for op in cast_ops
        )

        c = _Code()
        c.line("def cast_column_types(df: pd.DataFrame) -> pd.DataFrame:")
        with c.indent():
            c.line('"""')
            c.line("Cast columns to their inferred correct dtypes.")
            c.line(f"Applied: {summary}")
            c.line('"""')
            c.line("df = df.copy()")
            for op in cast_ops:
                col = op.details.get("column", "?")
                to_dtype = op.details.get("to_dtype", "float64")
                if "datetime" in to_dtype:
                    c.line(f'df["{col}"] = pd.to_datetime(df["{col}"], errors="coerce")')
                else:
                    c.line(f'df["{col}"] = pd.to_numeric(df["{col}"], errors="coerce")')
            c.line("return df")
        return c.text()

    def _gen_category_normalizer(self, cr: CleaningResult) -> str:
        norm_ops = [op for op in cr.operations if op.type == "normalize_category"]
        cols = [op.details.get("column", "?") for op in norm_ops]
        cols_repr = repr(cols)

        c = _Code()
        c.line("def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:")
        with c.indent():
            c.line('"""')
            c.line("Lowercase and strip whitespace from categorical columns.")
            c.line(f"Applied to: {cols}")
            c.line('"""')
            c.line("df = df.copy()")
            c.line(f"for col in {cols_repr}:")
            with c.indent():
                c.line("if col in df.columns:")
                with c.indent():
                    c.line("df[col] = df[col].astype(str).str.lower().str.strip()")
                    c.line('df[col] = df[col].replace("nan", pd.NA)')
            c.line("return df")
        return c.text()

    def _gen_missing_values(self, cr: CleaningResult) -> str:
        drop_col_ops = [op for op in cr.operations if op.type == "drop_column"]
        drop_row_ops = [op for op in cr.operations if op.type == "drop_rows"]
        impute_ops   = [op for op in cr.operations if op.type == "impute_column"]

        dropped_cols = repr([op.details.get("column", "?") for op in drop_col_ops])
        rows_dropped = drop_row_ops[0].details.get("rows_dropped", 0) if drop_row_ops else 0
        threshold    = drop_row_ops[0].details.get("threshold", 0.5) if drop_row_ops else 0.5

        c = _Code()
        c.line("def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:")
        with c.indent():
            c.line('"""')
            c.line("Drop high-missing columns and rows, then impute remaining missing values.")
            c.line(
                f"Applied: dropped {len(drop_col_ops)} col(s), "
                f"{rows_dropped} row(s), imputed {len(impute_ops)} col(s)."
            )
            c.line('"""')
            c.line("df = df.copy()")
            c.line(f"# Step 1: Drop high-missing columns")
            c.line(f"cols_to_drop = {dropped_cols}")
            c.line("df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])")
            c.line(f"# Step 2: Drop rows with too many missing values (threshold={threshold})")
            c.line("n_cols = len(df.columns)")
            c.line("import math")
            c.line(f"min_valid = math.floor(n_cols * (1.0 - {threshold})) + 1")
            c.line("df = df.dropna(thresh=min_valid).reset_index(drop=True)")
            c.line("# Step 3: Impute remaining columns")
            if not impute_ops:
                c.line("pass  # No imputation applied")
            else:
                for op in impute_ops:
                    col      = op.details.get("column", "?")
                    strategy = op.details.get("strategy", "median")
                    fill_val = op.details.get("fill_value")
                    if strategy == "median":
                        c.line(f'df["{col}"] = df["{col}"].fillna(df["{col}"].median())')
                    elif strategy == "mean":
                        c.line(f'df["{col}"] = df["{col}"].fillna(df["{col}"].mean())')
                    elif strategy == "mode":
                        c.line(f'_mode = df["{col}"].mode()')
                        c.line(f'df["{col}"] = df["{col}"].fillna(_mode[0] if not _mode.empty else pd.NA)')
                    elif strategy in ("zero", "constant"):
                        fill_repr = repr(fill_val) if fill_val is not None else "0"
                        c.line(f'df["{col}"] = df["{col}"].fillna({fill_repr})')
            c.line("return df")
        return c.text()

    def _gen_outlier_handler(self, cr: CleaningResult) -> str:
        flag_ops = [op for op in cr.operations if op.type == "flag_outlier"]
        cap_ops  = [op for op in cr.operations if op.type == "cap_outlier"]
        drop_ops = [op for op in cr.operations if op.type == "drop_rows"]

        c = _Code()
        c.line("def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:")
        with c.indent():
            if flag_ops:
                c.line('"""')
                c.line(f"Flag {len(flag_ops)} column(s) with outlier indicator boolean columns.")
                c.line('"""')
                c.line("df = df.copy()")
                for op in flag_ops:
                    col   = op.details.get("column", "?")
                    lower = op.details.get("lower_fence", 0.0)
                    upper = op.details.get("upper_fence", 0.0)
                    flag  = op.details.get("flag_column", f"{col}_is_outlier")
                    c.line(
                        f'df["{flag}"] = (df["{col}"] < {lower!r}) | (df["{col}"] > {upper!r})'
                    )
            elif cap_ops:
                c.line('"""')
                c.line(f"Cap {len(cap_ops)} column(s) to IQR fences (Winsorization).")
                c.line('"""')
                c.line("df = df.copy()")
                for op in cap_ops:
                    col   = op.details.get("column", "?")
                    lower = op.details.get("lower_fence", 0.0)
                    upper = op.details.get("upper_fence", 0.0)
                    c.line(f'df["{col}"] = df["{col}"].clip(lower={lower!r}, upper={upper!r})')
            elif drop_ops:
                rows = drop_ops[0].details.get("rows_dropped", 0)
                c.line('"""')
                c.line(f"Drop {rows} row(s) with outlier values (IQR method).")
                c.line('"""')
                c.line("df = df.copy()")
                c.line("def _iqr_outlier(s, mult=1.5):")
                with c.indent():
                    c.line("q1, q3 = s.quantile(0.25), s.quantile(0.75)")
                    c.line("iqr = q3 - q1")
                    c.line("return (s < q1 - mult * iqr) | (s > q3 + mult * iqr)")
                c.line("mask = pd.Series(False, index=df.index)")
                for op in cr.operations:
                    if op.type in ("flag_outlier", "mark_for_drop_outlier"):
                        col = op.details.get("column", "?")
                        c.line(f'mask |= _iqr_outlier(df["{col}"])')
                c.line("df = df[~mask].reset_index(drop=True)")
            else:
                c.line('"""No outlier operations recorded."""')
                c.line("pass")
            c.line("return df")
        return c.text()

    def _main_function(
        self,
        source_file: str,
        output_file: str,
        applied: list[CleaningResult],
    ) -> str:
        fn_map = {
            "drop_duplicates":     "drop_duplicates",
            "constant_columns":    "drop_constant_columns",
            "type_caster":         "cast_column_types",
            "category_normalizer": "normalize_categories",
            "missing_values":      "handle_missing_values",
            "outlier_handler":     "handle_outliers",
        }

        c = _Code()
        c.line(f'def main(input_path: str = "{source_file}", output_path: str = "{output_file}"):')
        with c.indent():
            c.line('"""')
            c.line("Load, clean, and save the dataset.")
            c.line("Reproduces the exact cleaning pipeline run by LazyPanda.")
            c.line('"""')
            c.line('print(f"Loading: {input_path}")')
            c.line("df = pd.read_csv(input_path, low_memory=False)")
            c.line('print(f"  Original shape: {df.shape[0]} rows × {df.shape[1]} cols")')
            c.line('print("\\nApplying cleaning steps:")')
            for cr in applied:
                fn = fn_map.get(cr.cleaner_name)
                if fn:
                    c.line(f"df = {fn}(df)")
                    c.line(
                        f"print(f'  ✓ {cr.display_name}: "
                        "{df.shape[0]} rows × {df.shape[1]} cols')"
                    )
            c.line('print(f"\\nSaving cleaned data to: {output_path}")')
            c.line("Path(output_path).parent.mkdir(parents=True, exist_ok=True)")
            c.line("df.to_csv(output_path, index=False)")
            c.line('print(f"  Final shape: {df.shape[0]} rows × {df.shape[1]} cols")')
            c.line('print("Done.")')
        return c.text()

    def _entry_point(self) -> str:
        c = _Code()
        c.line('if __name__ == "__main__":')
        with c.indent():
            c.line('parser = argparse.ArgumentParser(description="Reproducible LazyPanda pipeline")')
            c.line('parser.add_argument("--input",  default=None, help="Input CSV path")')
            c.line('parser.add_argument("--output", default=None, help="Output CSV path")')
            c.line("args = parser.parse_args()")
            c.line("kwargs = {}")
            c.line("if args.input:")
            with c.indent():
                c.line('kwargs["input_path"] = args.input')
            c.line("if args.output:")
            with c.indent():
                c.line('kwargs["output_path"] = args.output')
            c.line("main(**kwargs)")
        return c.text()
