"""
Unit tests for the interactive wizard command and display utilities.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from lazypanda.analyzers.base import AnalysisResult
from lazypanda.cleaners.base import CleaningResult, CleaningOperation
from lazypanda.cli.commands.wizard import _run_wizard
from lazypanda.cli.display import (
    print_ai_insights_summary,
    print_analysis_table,
    print_before_after,
    print_issue_summary,
    print_outputs_table,
    print_pipeline_table,
)
from lazypanda.cli.main import app
from lazypanda.core.dataset import Dataset
from lazypanda.core.pipeline import PipelineResult


@pytest.fixture
def dummy_pipeline_result():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    
    ar = AnalysisResult(
        analyzer_name="test_analyzer",
        display_name="Test Analyzer",
        issues_found=True,
        severity="warning",
        affected_columns=["a"],
        summary="Found issues",
        recommendation="Fix it",
        details={},
    )
    
    cr = CleaningResult(
        cleaner_name="test_cleaner",
        display_name="Test Cleaner",
        applied=True,
        rows_before=2,
        rows_after=2,
        cols_before=2,
        cols_after=2,
        summary="Cleaned",
        operations=[CleaningOperation("modify", "fixed it")],
    )
    
    ai_insights = MagicMock()
    ai_insights.skipped = False
    ai_insights.data_quality_score = 90
    ai_insights.recommendations = ["Rec 1"]
    ai_insights.tokens_used = 1200
    ai_insights.from_cache = False
    
    return PipelineResult(
        original_shape=(2, 2),
        final_shape=(2, 2),
        analysis_results=[ar],
        cleaning_results=[cr],
        cleaned_df=df,
        dry_run=False,
        stats={},
        ai_insights=ai_insights,
    )


# ─── Test Display Utilities ───────────────────────────────────────────────────

def test_print_pipeline_table(dummy_pipeline_result):
    """Test pipeline table renders without crashing."""
    print_pipeline_table(dummy_pipeline_result)


def test_print_outputs_table():
    """Test output table renders without crashing."""
    paths = {"Cleaned CSV": Path("/tmp/test.csv")}
    print_outputs_table(paths, dry_run=False)
    print_outputs_table(paths, dry_run=True)


def test_print_analysis_table(dummy_pipeline_result):
    """Test analysis table renders without crashing."""
    print_analysis_table(dummy_pipeline_result.analysis_results)


def test_print_issue_summary(dummy_pipeline_result):
    """Test issue summary renders without crashing."""
    ds = Dataset(dummy_pipeline_result.cleaned_df, MagicMock())
    print_issue_summary(dummy_pipeline_result.analysis_results, ds)


def test_print_before_after(dummy_pipeline_result):
    """Test before/after panel renders without crashing."""
    print_before_after(dummy_pipeline_result)


def test_print_ai_insights_summary(dummy_pipeline_result):
    """Test AI insights summary renders."""
    # Test valid insights
    print_ai_insights_summary(dummy_pipeline_result)
    
    # Test skipped insights
    dummy_pipeline_result.ai_insights.skipped = True
    dummy_pipeline_result.ai_insights.skip_reason = "Offline mode"
    print_ai_insights_summary(dummy_pipeline_result)
    
    # Test None
    dummy_pipeline_result.ai_insights = None
    print_ai_insights_summary(dummy_pipeline_result)


# ─── Test Wizard Flow ─────────────────────────────────────────────────────────

@patch("lazypanda.cli.commands.wizard.questionary")
@patch("lazypanda.cli.commands.wizard.CleaningPipeline")
@patch("lazypanda.cli.commands.wizard.Dataset")
@patch("lazypanda.cli.commands.wizard.MarkdownReporter")
def test_wizard_full_ai_flow(mock_reporter, mock_dataset, mock_pipeline_cls, mock_questionary, tmp_path, monkeypatch, dummy_pipeline_result):
    """Test the full wizard flow with AI enabled."""
    
    # Setup mock answers for questionary
    mock_select = MagicMock()
    mock_select.ask.return_value = "gemini-3.5-flash"
    
    mock_text = MagicMock()
    mock_text.ask.side_effect = ["1024", str(tmp_path / "outputs")]
    
    mock_path = MagicMock()
    mock_path.ask.return_value = "data.csv"
    
    mock_confirm = MagicMock()
    mock_confirm.ask.return_value = True

    mock_questionary.select.return_value = mock_select
    mock_questionary.text.return_value = mock_text
    mock_questionary.path.return_value = mock_path
    mock_questionary.confirm.return_value = mock_confirm
    
    # Mock the dataset so formatting works
    mock_dataset_inst = mock_dataset.from_csv.return_value
    mock_dataset_inst.meta.n_rows = 100
    mock_dataset_inst.meta.n_cols = 5
    mock_dataset_inst.meta.memory_usage_mb = 1.0
    mock_dataset_inst.df.shape = (100, 5)

    # Mock the pipeline return value to prevent unpacking errors
    mock_pipeline_inst = mock_pipeline_cls.return_value
    mock_pipeline_inst.run.return_value = dummy_pipeline_result

    # Inject an API key to bypass the password prompt
    monkeypatch.setenv("GEMINI_API_KEY", "test_key")
    
    # Run wizard directly (bypass Typer exit codes)
    _run_wizard()
    
    # Verify pipeline was called with model_override="gemini-3.5-flash"
    mock_pipeline_inst = mock_pipeline_cls.return_value
    mock_pipeline_inst.run.assert_called_once()
    kwargs = mock_pipeline_inst.run.call_args[1]
    assert kwargs["use_ai"] is True
    assert kwargs["model_override"] == "gemini-3.5-flash"
    

@patch("lazypanda.cli.commands.wizard.questionary")
@patch("lazypanda.cli.commands.wizard.CleaningPipeline")
@patch("lazypanda.cli.commands.wizard.Dataset")
@patch("lazypanda.cli.commands.wizard.MarkdownReporter")
def test_wizard_offline_flow(mock_reporter, mock_dataset, mock_pipeline_cls, mock_questionary, tmp_path, dummy_pipeline_result):
    """Test the wizard flow when 'Skip AI' is selected."""
    
    # Setup mock answers for questionary
    mock_select = MagicMock()
    mock_select.ask.return_value = None  # None is the value for "Skip AI"
    
    mock_text = MagicMock()
    mock_text.ask.return_value = str(tmp_path / "outputs")
    
    mock_path = MagicMock()
    mock_path.ask.return_value = "data.csv"
    
    mock_confirm = MagicMock()
    mock_confirm.ask.return_value = True
    
    mock_questionary.select.return_value = mock_select
    mock_questionary.text.return_value = mock_text
    mock_questionary.path.return_value = mock_path
    mock_questionary.confirm.return_value = mock_confirm

    # Mock the dataset so formatting works
    mock_dataset_inst = mock_dataset.from_csv.return_value
    mock_dataset_inst.meta.n_rows = 100
    mock_dataset_inst.meta.n_cols = 5
    mock_dataset_inst.meta.memory_usage_mb = 1.0
    mock_dataset_inst.df.shape = (100, 5)

    # Mock the pipeline return value to prevent unpacking errors
    mock_pipeline_inst = mock_pipeline_cls.return_value
    mock_pipeline_inst.run.return_value = dummy_pipeline_result
    
    _run_wizard()
    
    mock_pipeline_inst = mock_pipeline_cls.return_value
    mock_pipeline_inst.run.assert_called_once()
    kwargs = mock_pipeline_inst.run.call_args[1]
    assert kwargs["use_ai"] is False
    assert kwargs["model_override"] is None
