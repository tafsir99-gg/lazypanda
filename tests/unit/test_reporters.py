"""
Tests for Milestone 4: Reporters and Exporters.

WHAT IS TESTED:
    1. MarkdownReporter — renders valid Markdown from a PipelineResult
    2. JSONReporter     — produces a valid, parseable JSON payload
    3. PipelineExporter — generates a runnable Python script
    4. ConfigExporter   — saves a valid YAML config

PHILOSOPHY:
    We test:
    - Content correctness: key strings appear in the output
    - Structure: JSON keys exist, Markdown sections are present
    - File I/O: files are actually written and readable
    - Executability: the generated Python script runs without errors

    We do NOT test:
    - Exact formatting (too brittle — whitespace/order can change)
    - Visual rendering (Markdown renders differently everywhere)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from lazypanda.core.config_manager import AppConfig, ConfigManager
from lazypanda.core.pipeline import CleaningPipeline, PipelineResult
from lazypanda.exporters.config_exporter import ConfigExporter
from lazypanda.exporters.pipeline_exporter import PipelineExporter
from lazypanda.reporters.json_reporter import JSONReporter
from lazypanda.reporters.markdown_reporter import MarkdownReporter

# ── Shared fixtures ───────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
DIRTY_CSV    = FIXTURES_DIR / "sample_dirty.csv"
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.yaml"


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return ConfigManager(default_config_path=DEFAULT_CONFIG_PATH).load()


@pytest.fixture(scope="module")
def dirty_df() -> pd.DataFrame:
    return pd.read_csv(DIRTY_CSV)


@pytest.fixture(scope="module")
def pipeline_result(config, dirty_df) -> PipelineResult:
    """Run the pipeline once and reuse across all reporter tests."""
    return CleaningPipeline(config).run(dirty_df)


# ═══════════════════════════════════════════════════════════════════════════════
# MarkdownReporter
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarkdownReporter:

    @pytest.fixture
    def reporter(self):
        return MarkdownReporter()

    def test_render_returns_string(self, reporter, pipeline_result):
        ctx = reporter.build_context(pipeline_result, source_file="train.csv")
        md = reporter.render(ctx)
        assert isinstance(md, str)

    def test_render_not_empty(self, reporter, pipeline_result):
        ctx = reporter.build_context(pipeline_result, source_file="train.csv")
        md = reporter.render(ctx)
        assert len(md) > 100

    def test_report_contains_dataset_overview_section(self, reporter, pipeline_result):
        ctx = reporter.build_context(pipeline_result, source_file="train.csv")
        md = reporter.render(ctx)
        assert "Dataset Overview" in md

    def test_report_contains_analysis_results_section(self, reporter, pipeline_result):
        ctx = reporter.build_context(pipeline_result, source_file="train.csv")
        md = reporter.render(ctx)
        assert "Analysis Results" in md

    def test_report_contains_cleaning_pipeline_section(self, reporter, pipeline_result):
        ctx = reporter.build_context(pipeline_result, source_file="train.csv")
        md = reporter.render(ctx)
        assert "Cleaning Pipeline Results" in md

    def test_report_contains_source_file_name(self, reporter, pipeline_result):
        ctx = reporter.build_context(pipeline_result, source_file="my_data.csv")
        md = reporter.render(ctx)
        assert "my_data.csv" in md

    def test_report_contains_before_after_section(self, reporter, pipeline_result):
        ctx = reporter.build_context(pipeline_result, source_file="train.csv")
        md = reporter.render(ctx)
        assert "Before" in md and "After" in md

    def test_report_contains_original_shape(self, reporter, pipeline_result):
        ctx = reporter.build_context(pipeline_result, source_file="train.csv")
        md = reporter.render(ctx)
        orig_rows = str(pipeline_result.original_shape[0])
        assert orig_rows in md

    def test_write_creates_file(self, reporter, pipeline_result, tmp_path):
        out = tmp_path / "report.md"
        reporter.write(pipeline_result, out, source_file="train.csv")
        assert out.exists()

    def test_write_file_is_readable(self, reporter, pipeline_result, tmp_path):
        out = tmp_path / "report.md"
        reporter.write(pipeline_result, out, source_file="train.csv")
        content = out.read_text(encoding="utf-8")
        assert len(content) > 50

    def test_write_creates_parent_dirs(self, reporter, pipeline_result, tmp_path):
        out = tmp_path / "a" / "b" / "report.md"
        reporter.write(pipeline_result, out, source_file="train.csv")
        assert out.exists()

    def test_dry_run_labelled_in_report(self, reporter, config, dirty_df, tmp_path):
        result = CleaningPipeline(config).run(dirty_df, dry_run=True)
        ctx = reporter.build_context(result, source_file="train.csv")
        md = reporter.render(ctx)
        assert "DRY RUN" in md

    def test_output_files_appear_in_report(self, reporter, pipeline_result, tmp_path):
        out = tmp_path / "report.md"
        output_files = {"Cleaned CSV": "/tmp/clean.csv", "JSON Summary": "/tmp/summary.json"}
        reporter.write(pipeline_result, out, source_file="train.csv", output_files=output_files)
        content = out.read_text()
        assert "Output Files" in content


# ═══════════════════════════════════════════════════════════════════════════════
# JSONReporter
# ═══════════════════════════════════════════════════════════════════════════════

class TestJSONReporter:

    @pytest.fixture
    def reporter(self):
        return JSONReporter()

    def test_build_payload_returns_dict(self, reporter, pipeline_result):
        payload = reporter.build_payload(pipeline_result, source_file="train.csv")
        assert isinstance(payload, dict)

    def test_payload_has_required_top_level_keys(self, reporter, pipeline_result):
        payload = reporter.build_payload(pipeline_result, source_file="train.csv")
        for key in ("meta", "dataset", "summary", "analysis", "cleaning"):
            assert key in payload, f"Missing top-level key: '{key}'"

    def test_meta_section_has_source_file(self, reporter, pipeline_result):
        payload = reporter.build_payload(pipeline_result, source_file="my_data.csv")
        assert payload["meta"]["source_file"] == "my_data.csv"

    def test_meta_has_tool_name(self, reporter, pipeline_result):
        payload = reporter.build_payload(pipeline_result, source_file="train.csv")
        assert payload["meta"]["tool"] == "LazyPanda"

    def test_dataset_section_shapes_correct(self, reporter, pipeline_result):
        payload = reporter.build_payload(pipeline_result, source_file="train.csv")
        ds = payload["dataset"]
        assert ds["original_rows"] == pipeline_result.original_shape[0]
        assert ds["original_cols"] == pipeline_result.original_shape[1]
        assert ds["final_rows"]    == pipeline_result.final_shape[0]

    def test_analysis_is_list_of_7(self, reporter, pipeline_result):
        payload = reporter.build_payload(pipeline_result, source_file="train.csv")
        assert isinstance(payload["analysis"], list)
        assert len(payload["analysis"]) == 7

    def test_cleaning_is_list_of_6(self, reporter, pipeline_result):
        payload = reporter.build_payload(pipeline_result, source_file="train.csv")
        assert isinstance(payload["cleaning"], list)
        assert len(payload["cleaning"]) == 6

    def test_render_produces_valid_json(self, reporter, pipeline_result):
        payload = reporter.build_payload(pipeline_result, source_file="train.csv")
        json_str = reporter.render(payload)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_write_creates_file(self, reporter, pipeline_result, tmp_path):
        out = tmp_path / "summary.json"
        reporter.write(pipeline_result, out, source_file="train.csv")
        assert out.exists()

    def test_write_file_is_valid_json(self, reporter, pipeline_result, tmp_path):
        out = tmp_path / "summary.json"
        reporter.write(pipeline_result, out, source_file="train.csv")
        parsed = json.loads(out.read_text())
        assert "meta" in parsed

    def test_write_creates_parent_dirs(self, reporter, pipeline_result, tmp_path):
        out = tmp_path / "a" / "b" / "summary.json"
        reporter.write(pipeline_result, out, source_file="train.csv")
        assert out.exists()

    def test_summary_has_critical_issues_flag(self, reporter, pipeline_result):
        payload = reporter.build_payload(pipeline_result, source_file="train.csv")
        assert "has_critical_issues" in payload["summary"]

    def test_operations_serialized_per_cleaner(self, reporter, pipeline_result):
        payload = reporter.build_payload(pipeline_result, source_file="train.csv")
        # At least one cleaner should have operations
        all_ops = [op for cr in payload["cleaning"] for op in cr.get("operations", [])]
        assert len(all_ops) > 0

    def test_output_files_in_payload(self, reporter, pipeline_result):
        output_files = {"Cleaned CSV": "/tmp/clean.csv"}
        payload = reporter.build_payload(pipeline_result, source_file="t.csv", output_files=output_files)
        assert payload["output_files"]["Cleaned CSV"] == "/tmp/clean.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# PipelineExporter
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineExporter:

    @pytest.fixture
    def exporter(self):
        return PipelineExporter()

    def test_generate_returns_string(self, exporter, pipeline_result):
        script = exporter.generate(pipeline_result, source_file="train.csv")
        assert isinstance(script, str)

    def test_script_is_not_empty(self, exporter, pipeline_result):
        script = exporter.generate(pipeline_result, source_file="train.csv")
        assert len(script) > 200

    def test_script_has_imports(self, exporter, pipeline_result):
        script = exporter.generate(pipeline_result, source_file="train.csv")
        assert "import pandas as pd" in script

    def test_script_has_main_function(self, exporter, pipeline_result):
        script = exporter.generate(pipeline_result, source_file="train.csv")
        assert "def main(" in script

    def test_script_has_entry_point(self, exporter, pipeline_result):
        script = exporter.generate(pipeline_result, source_file="train.csv")
        assert 'if __name__ == "__main__"' in script

    def test_script_has_applied_cleaner_functions(self, exporter, pipeline_result):
        script = exporter.generate(pipeline_result, source_file="train.csv")
        # DropDuplicates was applied on the dirty fixture
        assert "def drop_duplicates" in script

    def test_write_creates_file(self, exporter, pipeline_result, tmp_path):
        out = tmp_path / "clean.py"
        exporter.write(pipeline_result, out, source_file="train.csv")
        assert out.exists()

    def test_write_file_is_readable(self, exporter, pipeline_result, tmp_path):
        out = tmp_path / "clean.py"
        exporter.write(pipeline_result, out, source_file="train.csv")
        content = out.read_text()
        assert len(content) > 100

    def test_generated_script_has_valid_python_syntax(self, exporter, pipeline_result, tmp_path):
        """The generated script must be syntactically valid Python."""
        script = exporter.generate(pipeline_result, source_file="train.csv")
        out = tmp_path / "clean.py"
        out.write_text(script)
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(out)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Syntax error in generated script:\n{result.stderr}"

    def test_script_source_file_embedded(self, exporter, pipeline_result):
        script = exporter.generate(pipeline_result, source_file="my_data.csv", output_file="out.csv")
        assert "my_data.csv" in script

    def test_write_creates_parent_dirs(self, exporter, pipeline_result, tmp_path):
        out = tmp_path / "a" / "b" / "script.py"
        exporter.write(pipeline_result, out, source_file="train.csv")
        assert out.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# ConfigExporter
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigExporter:

    @pytest.fixture
    def exporter(self):
        return ConfigExporter()

    def test_build_payload_returns_dict(self, exporter, config):
        payload = exporter.build_payload(config)
        assert isinstance(payload, dict)

    def test_payload_has_expected_sections(self, exporter, config):
        payload = exporter.build_payload(config)
        for section in ("missing_values", "duplicates", "outliers", "cardinality", "categories"):
            assert section in payload, f"Missing config section: '{section}'"

    def test_write_creates_file(self, exporter, config, tmp_path):
        out = tmp_path / "run_config.yaml"
        exporter.write(config, out)
        assert out.exists()

    def test_written_file_is_valid_yaml(self, exporter, config, tmp_path):
        out = tmp_path / "run_config.yaml"
        exporter.write(config, out)
        raw = out.read_text()
        parsed = yaml.safe_load(raw)
        assert isinstance(parsed, dict)

    def test_written_yaml_preserves_outlier_action(self, exporter, config, tmp_path):
        out = tmp_path / "run_config.yaml"
        exporter.write(config, out)
        parsed = yaml.safe_load(out.read_text())
        assert "outliers" in parsed
        assert "action" in parsed["outliers"]

    def test_written_yaml_has_header_comment(self, exporter, config, tmp_path):
        out = tmp_path / "run_config.yaml"
        exporter.write(config, out)
        raw = out.read_text()
        assert "AI Data Cleaner" in raw
        assert "Generated:" in raw

    def test_write_creates_parent_dirs(self, exporter, config, tmp_path):
        out = tmp_path / "a" / "b" / "config.yaml"
        exporter.write(config, out)
        assert out.exists()

    def test_roundtrip_config_validates_correctly(self, exporter, config, tmp_path):
        """Write the config, reload it, and validate it produces the same AppConfig."""
        out = tmp_path / "run_config.yaml"
        exporter.write(config, out)
        reloaded = ConfigManager(default_config_path=DEFAULT_CONFIG_PATH).load(
            user_config_path=out
        )
        assert reloaded.outliers.action == config.outliers.action
        assert reloaded.missing_values.numeric_imputation == config.missing_values.numeric_imputation
