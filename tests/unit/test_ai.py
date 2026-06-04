"""
Tests for Milestone 5: AI Integration.

WHAT IS TESTED:
    1. AIResponseCache      — get/set/TTL/miss/clear
    2. TokenBudgetManager   — budget math, can_afford, record_usage, estimation
    3. PromptBuilder        — privacy (no raw rows), token estimate, required keys
    4. ResponseParser       — valid JSON, code-fenced JSON, malformed, field validation
    5. AIEnricher           — full gate flow (no-ai, no-key, dry-run, budget, cache, call)
    6. Pipeline + AI        — pipeline.run(use_ai=True/False) integration
    7. Reporters + AI       — MarkdownReporter and JSONReporter with AIInsights

PHILOSOPHY:
    - ZERO real Gemini API calls — all mocked
    - Tests are deterministic and run offline
    - Each test covers exactly one concern (not multiple)
    - Failure messages are self-explanatory
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lazypanda.ai.cache import AIResponseCache
from lazypanda.ai.enricher import AIEnricher, AIInsights
from lazypanda.ai.prompt_builder import PromptBuilder
from lazypanda.ai.response_parser import ResponseParser
from lazypanda.ai.token_budget import TokenBudgetManager
from lazypanda.core.config_manager import AppConfig, ConfigManager
from lazypanda.core.pipeline import CleaningPipeline, PipelineResult
from lazypanda.reporters.json_reporter import JSONReporter
from lazypanda.reporters.markdown_reporter import MarkdownReporter

# ── Shared fixtures ───────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
DIRTY_CSV    = FIXTURES_DIR / "sample_dirty.csv"
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.yaml"

# Canonical mock Gemini response dict (valid, parseable)
_MOCK_RESPONSE_DICT = {
    "overall_summary": "The Titanic dataset is a classic ML benchmark with moderate quality issues.",
    "data_quality_score": 62,
    "score_rationale": "Several missing values and a few outliers detected. Cleaning is recommended before ML use.",
    "column_insights": [
        {"column": "age", "insight": "Age has ~10% missing values; median imputation is appropriate."},
        {"column": "fare", "insight": "Fare shows strong right skew with high-value outliers."},
    ],
    "recommendations": [
        "Encode categorical columns (sex, cabin) before training.",
        "Consider feature engineering on fare (log transform).",
        "Review outlier flags before dropping records.",
    ],
    "cleaning_assessment": "The applied cleaning decisions are reasonable for a supervised learning task.",
}


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return ConfigManager(default_config_path=DEFAULT_CONFIG_PATH).load()


@pytest.fixture(scope="module")
def dirty_df() -> pd.DataFrame:
    return pd.read_csv(DIRTY_CSV)


@pytest.fixture(scope="module")
def pipeline_result(config, dirty_df) -> PipelineResult:
    """Run deterministic pipeline once, reuse across all AI tests."""
    return CleaningPipeline(config).run(dirty_df.copy())


@pytest.fixture
def mock_insights() -> AIInsights:
    return AIInsights.from_parsed(
        _MOCK_RESPONSE_DICT,
        model_used="gemini-1.5-flash",
        tokens_used=850,
        from_cache=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AIResponseCache
# ═══════════════════════════════════════════════════════════════════════════════

class TestAIResponseCache:

    @pytest.fixture
    def cache(self, tmp_path):
        return AIResponseCache(cache_dir=tmp_path / "cache")

    def test_make_key_is_deterministic(self, cache):
        k1 = cache.make_key("hello world")
        k2 = cache.make_key("hello world")
        assert k1 == k2

    def test_make_key_is_64_hex_chars(self, cache):
        key = cache.make_key("test prompt")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_make_key_differs_for_different_inputs(self, cache):
        k1 = cache.make_key("prompt A")
        k2 = cache.make_key("prompt B")
        assert k1 != k2

    def test_get_returns_none_on_miss(self, cache):
        assert cache.get("nonexistent_key") is None

    def test_set_then_get_returns_value(self, cache):
        payload = {"answer": 42}
        cache.set("mykey", payload)
        result = cache.get("mykey", ttl_hours=24)
        assert result == payload

    def test_get_respects_ttl_expiry(self, cache, tmp_path):
        """Simulate an expired entry by writing a past timestamp."""
        key = "expiredkey"
        # Write entry manually with a past timestamp
        past_time = "2000-01-01T00:00:00+00:00"
        path = tmp_path / "cache" / key[:2] / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "cached_at": past_time,
            "prompt_sha256": key,
            "response": {"answer": "old"},
        }))
        result = cache.get(key, ttl_hours=1)
        assert result is None  # should be expired

    def test_get_with_ttl_zero_never_expires(self, cache):
        payload = {"answer": "forever"}
        cache.set("ttlzerokey", payload)
        result = cache.get("ttlzerokey", ttl_hours=0)
        assert result == payload

    def test_clear_removes_all_entries(self, cache):
        cache.set("key1", {"a": 1})
        cache.set("key2", {"b": 2})
        removed = cache.clear()
        assert removed >= 2
        assert cache.get("key1") is None

    def test_size_returns_entry_count(self, cache):
        cache.set("sz1", {"x": 1})
        cache.set("sz2", {"x": 2})
        assert cache.size() >= 2

    def test_cache_uses_subdirectory_sharding(self, cache, tmp_path):
        """Verify files are placed in <key[:2]>/ subdirectory."""
        key = cache.make_key("shard test")
        cache.set(key, {"data": 1})
        expected_dir = tmp_path / "cache" / key[:2]
        assert expected_dir.exists()
        assert any(f.suffix == ".json" for f in expected_dir.iterdir())

    def test_corrupted_cache_entry_returns_none(self, cache, tmp_path):
        key = "corrupt_key_abc123"
        path = tmp_path / "cache" / key[:2] / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("NOT_VALID_JSON{{{")
        assert cache.get(key) is None


# ═══════════════════════════════════════════════════════════════════════════════
# TokenBudgetManager
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenBudgetManager:

    def test_default_max_is_2000(self):
        b = TokenBudgetManager()
        assert b.max == 2000

    def test_initial_used_is_zero(self):
        b = TokenBudgetManager(max_tokens=1000)
        assert b.used == 0

    def test_remaining_equals_max_initially(self):
        b = TokenBudgetManager(max_tokens=500)
        assert b.remaining == 500

    def test_can_afford_within_budget(self):
        b = TokenBudgetManager(max_tokens=1000)
        assert b.can_afford(estimated_input=400, estimated_output=400)

    def test_can_afford_exactly_at_budget(self):
        b = TokenBudgetManager(max_tokens=1000)
        assert b.can_afford(estimated_input=500, estimated_output=500)

    def test_cannot_afford_over_budget(self):
        b = TokenBudgetManager(max_tokens=1000)
        assert not b.can_afford(estimated_input=600, estimated_output=500)

    def test_record_usage_decreases_remaining(self):
        b = TokenBudgetManager(max_tokens=1000)
        b.record_usage(300)
        assert b.remaining == 700
        assert b.used == 300

    def test_is_exhausted_after_full_usage(self):
        b = TokenBudgetManager(max_tokens=500)
        b.record_usage(500)
        assert b.is_exhausted

    def test_is_not_exhausted_before_full(self):
        b = TokenBudgetManager(max_tokens=500)
        b.record_usage(499)
        assert not b.is_exhausted

    def test_reset_clears_usage(self):
        b = TokenBudgetManager(max_tokens=500)
        b.record_usage(200)
        b.reset()
        assert b.used == 0
        assert b.remaining == 500

    def test_estimate_tokens_empty_string(self):
        assert TokenBudgetManager.estimate_tokens("") == 0

    def test_estimate_tokens_short_string(self):
        # "hello" = 5 chars → ceil(5/3.5) = 2
        est = TokenBudgetManager.estimate_tokens("hello")
        assert est >= 1

    def test_estimate_tokens_scales_with_length(self):
        short = TokenBudgetManager.estimate_tokens("short")
        long  = TokenBudgetManager.estimate_tokens("x" * 1000)
        assert long > short

    def test_invalid_max_tokens_raises(self):
        with pytest.raises(ValueError):
            TokenBudgetManager(max_tokens=0)

    def test_record_negative_tokens_raises(self):
        b = TokenBudgetManager(max_tokens=1000)
        with pytest.raises(ValueError):
            b.record_usage(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# PromptBuilder
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptBuilder:

    @pytest.fixture
    def builder(self):
        return PromptBuilder()

    def test_build_returns_string_and_int(self, builder, pipeline_result):
        prompt, tokens = builder.build(pipeline_result, source_file="train.csv")
        assert isinstance(prompt, str)
        assert isinstance(tokens, int)

    def test_prompt_is_valid_json(self, builder, pipeline_result):
        prompt, _ = builder.build(pipeline_result, source_file="train.csv")
        parsed = json.loads(prompt)
        assert isinstance(parsed, dict)

    def test_prompt_has_task_field(self, builder, pipeline_result):
        prompt, _ = builder.build(pipeline_result, source_file="train.csv")
        parsed = json.loads(prompt)
        # New compact format uses 'role' as the task description key
        assert "role" in parsed

    def test_prompt_has_dataset_section(self, builder, pipeline_result):
        prompt, _ = builder.build(pipeline_result, source_file="train.csv")
        parsed = json.loads(prompt)
        assert "dataset" in parsed

    def test_prompt_has_analysis_findings(self, builder, pipeline_result):
        prompt, _ = builder.build(pipeline_result, source_file="train.csv")
        parsed = json.loads(prompt)
        # Compact format uses 'analysis' (not 'analysis_findings')
        assert "analysis" in parsed

    def test_prompt_has_cleaning_decisions(self, builder, pipeline_result):
        prompt, _ = builder.build(pipeline_result, source_file="train.csv")
        parsed = json.loads(prompt)
        # Compact format uses 'cleaning' (not 'cleaning_decisions')
        assert "cleaning" in parsed

    def test_prompt_has_response_format(self, builder, pipeline_result):
        prompt, _ = builder.build(pipeline_result, source_file="train.csv")
        parsed = json.loads(prompt)
        # Compact format uses 'output_schema' for field-name guidance
        assert "output_schema" in parsed


    def test_prompt_does_not_contain_raw_rows(self, builder, pipeline_result):
        """THE CARDINAL RULE: no raw data rows in the prompt."""
        prompt, _ = builder.build(pipeline_result, source_file="train.csv")
        parsed = json.loads(prompt)
        dataset = parsed["dataset"]
        # Dataset section must not contain a list of raw rows
        # It should contain 'cols' (column metadata) but NOT actual data values
        assert isinstance(dataset.get("cols"), list)
        # Each col entry is a 'name:dtype' string, not a data row
        for col_entry in dataset["cols"]:
            assert isinstance(col_entry, str)
            assert ":" in col_entry  # format is 'colname:dtype'

    def test_estimated_tokens_is_positive(self, builder, pipeline_result):
        _, tokens = builder.build(pipeline_result, source_file="train.csv")
        assert tokens > 0

    def test_estimated_tokens_under_1000(self, builder, pipeline_result):
        """Compact prompt must fit in 1000 input tokens (well under 8000 budget)."""
        _, tokens = builder.build(pipeline_result, source_file="train.csv")
        assert tokens < 1000, f"Prompt too large: {tokens} tokens (target < 1000)"

    def test_source_file_basename_in_dataset(self, builder, pipeline_result):
        prompt, _ = builder.build(pipeline_result, source_file="/some/path/train.csv")
        parsed = json.loads(prompt)
        # Compact format uses 'file' key in the dataset section
        assert parsed["dataset"]["file"] == "train.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# ResponseParser
# ═══════════════════════════════════════════════════════════════════════════════

class TestResponseParser:

    @pytest.fixture
    def parser(self):
        return ResponseParser()

    def _valid_json(self):
        return json.dumps(_MOCK_RESPONSE_DICT)

    def test_parse_valid_json_returns_dict(self, parser):
        result = parser.parse(self._valid_json())
        assert isinstance(result, dict)

    def test_parse_valid_json_has_all_keys(self, parser):
        result = parser.parse(self._valid_json())
        for key in ("overall_summary", "data_quality_score", "score_rationale"):
            assert key in result

    def test_parse_score_is_clamped_to_0_100(self, parser):
        bad = dict(_MOCK_RESPONSE_DICT)
        bad["data_quality_score"] = 999
        result = parser.parse(json.dumps(bad))
        assert result["data_quality_score"] == 100

    def test_parse_score_clamped_below_zero(self, parser):
        bad = dict(_MOCK_RESPONSE_DICT)
        bad["data_quality_score"] = -50
        result = parser.parse(json.dumps(bad))
        assert result["data_quality_score"] == 0

    def test_parse_json_in_markdown_fence(self, parser):
        fenced = f"```json\n{self._valid_json()}\n```"
        result = parser.parse(fenced)
        assert result is not None
        assert "overall_summary" in result

    def test_parse_json_in_plain_code_fence(self, parser):
        fenced = f"```\n{self._valid_json()}\n```"
        result = parser.parse(fenced)
        assert result is not None

    def test_parse_json_embedded_in_prose(self, parser):
        """JSON buried inside prose text — should still extract."""
        text = f"Here is my analysis:\n{self._valid_json()}\nEnd."
        result = parser.parse(text)
        assert result is not None

    def test_parse_empty_string_returns_none(self, parser):
        assert parser.parse("") is None

    def test_parse_pure_prose_returns_none(self, parser):
        assert parser.parse("This is just plain text with no JSON.") is None

    def test_parse_malformed_json_returns_none(self, parser):
        assert parser.parse("{broken: json, missing quotes}") is None

    def test_parse_missing_required_field_returns_none(self, parser):
        incomplete = {"data_quality_score": 70, "score_rationale": "ok"}  # missing overall_summary
        assert parser.parse(json.dumps(incomplete)) is None

    def test_parse_optional_fields_have_defaults(self, parser):
        minimal = {
            "overall_summary": "Good dataset.",
            "data_quality_score": 80,
            "score_rationale": "Few issues.",
        }
        result = parser.parse(json.dumps(minimal))
        assert result is not None
        assert result["column_insights"] == []
        assert result["recommendations"] == []
        assert result["cleaning_assessment"] == ""

    def test_parse_filters_invalid_column_insights(self, parser):
        mixed = dict(_MOCK_RESPONSE_DICT)
        mixed["column_insights"] = [
            {"column": "age", "insight": "valid"},
            "not a dict",
            {"column": "fare"},       # missing 'insight'
            {"insight": "no column"}, # missing 'column'
        ]
        result = parser.parse(json.dumps(mixed))
        assert len(result["column_insights"]) == 1
        assert result["column_insights"][0]["column"] == "age"

    def test_parse_filters_non_string_recommendations(self, parser):
        mixed = dict(_MOCK_RESPONSE_DICT)
        mixed["recommendations"] = ["valid rec", 42, None, "another valid"]
        result = parser.parse(json.dumps(mixed))
        # Only strings should survive
        assert all(isinstance(r, str) for r in result["recommendations"])


# ═══════════════════════════════════════════════════════════════════════════════
# AIInsights dataclass
# ═══════════════════════════════════════════════════════════════════════════════

class TestAIInsights:

    def test_skipped_result_factory(self):
        ins = AIInsights.skipped_result("no key")
        assert ins.skipped is True
        assert ins.skip_reason == "no key"
        assert ins.overall_summary == ""

    def test_from_parsed_factory(self):
        ins = AIInsights.from_parsed(
            _MOCK_RESPONSE_DICT,
            model_used="gemini-1.5-flash",
            tokens_used=500,
            from_cache=False,
        )
        assert not ins.skipped
        assert ins.data_quality_score == 62
        assert ins.model_used == "gemini-1.5-flash"
        assert ins.tokens_used == 500
        assert len(ins.recommendations) == 3

    def test_to_dict_is_serializable(self):
        ins = AIInsights.from_parsed(
            _MOCK_RESPONSE_DICT, model_used="test", tokens_used=100, from_cache=True
        )
        d = ins.to_dict()
        json_str = json.dumps(d)  # must not raise
        assert isinstance(json_str, str)

    def test_from_cache_flag(self):
        ins = AIInsights.from_parsed(
            _MOCK_RESPONSE_DICT, model_used="(from cache)", tokens_used=0, from_cache=True
        )
        assert ins.from_cache is True
        assert ins.tokens_used == 0


# ═══════════════════════════════════════════════════════════════════════════════
# AIEnricher — gating logic (no real API calls)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAIEnricher:

    @pytest.fixture
    def enricher(self, tmp_path):
        """Enricher with isolated cache directory."""
        cache = AIResponseCache(cache_dir=tmp_path / "test_cache")
        return AIEnricher(cache=cache)

    @pytest.fixture
    def high_budget_config(self) -> AppConfig:
        """Config with a generous token budget so enricher tests reach the API stage."""
        cfg = ConfigManager(default_config_path=DEFAULT_CONFIG_PATH).load()
        object.__setattr__(cfg.ai, "max_tokens_per_run", 100_000)
        return cfg

    def test_skip_when_use_ai_false(self, enricher, pipeline_result, config):
        result = enricher.enrich(pipeline_result, config=config, use_ai=False)
        assert result.skipped
        assert "--no-ai" in result.skip_reason

    def test_skip_when_config_ai_disabled(self, enricher, pipeline_result):
        from lazypanda.core.config_manager import AIConfig
        cfg = ConfigManager(default_config_path=DEFAULT_CONFIG_PATH).load()
        # Monkeypatch ai.enabled=False
        object.__setattr__(cfg.ai, "enabled", False)
        result = enricher.enrich(pipeline_result, config=cfg, use_ai=True)
        assert result.skipped
        assert "disabled" in result.skip_reason.lower()

    def test_skip_on_dry_run(self, enricher, config, dirty_df):
        dry_result = CleaningPipeline(config).run(dirty_df.copy(), dry_run=True)
        result = enricher.enrich(dry_result, config=config, use_ai=True)
        assert result.skipped
        assert "dry run" in result.skip_reason.lower()

    def test_skip_when_api_key_missing(self, enricher, pipeline_result, config, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = enricher.enrich(pipeline_result, config=config, use_ai=True)
        assert result.skipped
        assert "GEMINI_API_KEY" in result.skip_reason

    def test_skip_when_budget_too_low(self, enricher, pipeline_result, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key_for_test")
        # Create config with very low token budget
        cfg = ConfigManager(default_config_path=DEFAULT_CONFIG_PATH).load()
        object.__setattr__(cfg.ai, "max_tokens_per_run", 10)  # impossibly small
        result = enricher.enrich(pipeline_result, config=cfg, use_ai=True)
        assert result.skipped
        assert "budget" in result.skip_reason.lower() or "token" in result.skip_reason.lower()

    def test_cache_hit_returns_cached_insights(self, enricher, pipeline_result, high_budget_config, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key_for_test")
        # Pre-populate cache
        builder = PromptBuilder()
        prompt, _ = builder.build(pipeline_result, source_file="train.csv")
        key = AIResponseCache.make_key(prompt)
        enricher._cache.set(key, _MOCK_RESPONSE_DICT)

        result = enricher.enrich(pipeline_result, config=high_budget_config, use_ai=True, source_file="train.csv")
        assert not result.skipped
        assert result.from_cache is True
        assert result.data_quality_score == 62

    def test_successful_api_call_with_mock(self, enricher, pipeline_result, high_budget_config, monkeypatch):
        """Full enricher flow with mocked Gemini client."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key_for_test")

        mock_client = MagicMock()
        mock_client.generate.return_value = (json.dumps(_MOCK_RESPONSE_DICT), 850)
        mock_client.model_name = "gemini-2.0-flash"

        with patch("lazypanda.ai.enricher.GeminiClient", return_value=mock_client):
            result = enricher.enrich(
                pipeline_result, config=high_budget_config, use_ai=True, source_file="train.csv"
            )

        assert not result.skipped
        assert result.data_quality_score == 62
        assert result.tokens_used == 850
        assert result.from_cache is False
        assert len(result.recommendations) == 3

    def test_api_failure_returns_skipped(self, enricher, pipeline_result, high_budget_config, monkeypatch):
        """GeminiAPIError → skipped, not a crash."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key_for_test")
        from lazypanda.ai.client import GeminiAPIError

        mock_client = MagicMock()
        mock_client.generate.side_effect = GeminiAPIError("Connection refused")

        with patch("lazypanda.ai.enricher.GeminiClient", return_value=mock_client):
            result = enricher.enrich(pipeline_result, config=high_budget_config, use_ai=True)

        assert result.skipped
        assert "failed" in result.skip_reason.lower()

    def test_unparseable_response_returns_skipped(self, enricher, pipeline_result, high_budget_config, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key_for_test")

        mock_client = MagicMock()
        mock_client.generate.return_value = ("This is not JSON at all!!!", 100)
        mock_client.model_name = "gemini-2.0-flash"

        with patch("lazypanda.ai.enricher.GeminiClient", return_value=mock_client):
            result = enricher.enrich(pipeline_result, config=high_budget_config, use_ai=True)

        assert result.skipped
        assert "unparse" in result.skip_reason.lower() or "response" in result.skip_reason.lower()

    def test_successful_call_is_cached(self, enricher, pipeline_result, high_budget_config, monkeypatch):
        """After a successful call, the response should be in the cache."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key_for_test")

        mock_client = MagicMock()
        mock_client.generate.return_value = (json.dumps(_MOCK_RESPONSE_DICT), 850)
        mock_client.model_name = "gemini-2.0-flash"

        with patch("lazypanda.ai.enricher.GeminiClient", return_value=mock_client):
            enricher.enrich(
                pipeline_result, config=high_budget_config, use_ai=True, source_file="train.csv"
            )

        # Verify it's now in the cache
        assert enricher._cache.size() > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline integration with AI
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineWithAI:

    def test_pipeline_run_without_ai_has_no_insights(self, config, dirty_df):
        result = CleaningPipeline(config).run(dirty_df.copy(), use_ai=False)
        assert result.ai_insights is None

    def test_pipeline_run_with_no_key_sets_skipped_insights(self, config, dirty_df, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = CleaningPipeline(config).run(dirty_df.copy(), use_ai=True)
        assert result.ai_insights is not None
        assert result.ai_insights.skipped

    def test_pipeline_run_with_mock_enricher(self, config, dirty_df):
        """Inject a mock enricher directly into pipeline.run()."""
        mock_enricher = MagicMock()
        mock_enricher.enrich.return_value = AIInsights.from_parsed(
            _MOCK_RESPONSE_DICT, model_used="mock-model", tokens_used=100, from_cache=False
        )

        result = CleaningPipeline(config).run(
            dirty_df.copy(),
            use_ai=True,
            enricher=mock_enricher,
        )
        assert result.ai_insights is not None
        assert not result.ai_insights.skipped
        assert result.ai_insights.data_quality_score == 62

    def test_pipeline_ai_failure_does_not_crash(self, config, dirty_df):
        """Even if enricher raises, the pipeline result is still returned."""
        crashing_enricher = MagicMock()
        crashing_enricher.enrich.side_effect = RuntimeError("Unexpected crash")

        result = CleaningPipeline(config).run(
            dirty_df.copy(),
            use_ai=True,
            enricher=crashing_enricher,
        )
        # Pipeline must not raise — it should return a skipped AIInsights
        assert result is not None
        assert result.ai_insights is not None
        assert result.ai_insights.skipped

    def test_dry_run_never_calls_enricher(self, config, dirty_df):
        mock_enricher = MagicMock()
        CleaningPipeline(config).run(
            dirty_df.copy(),
            dry_run=True,
            use_ai=True,
            enricher=mock_enricher,
        )
        mock_enricher.enrich.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Reporters with AI insights
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarkdownReporterWithAI:

    @pytest.fixture
    def result_with_insights(self, pipeline_result, mock_insights):
        """Clone the pipeline result and inject mock AI insights."""
        from copy import copy
        r = copy(pipeline_result)
        r.ai_insights = mock_insights
        return r

    @pytest.fixture
    def result_with_skipped_insights(self, pipeline_result):
        from copy import copy
        r = copy(pipeline_result)
        r.ai_insights = AIInsights.skipped_result("No API key set")
        return r

    def test_markdown_with_insights_contains_quality_score(self, result_with_insights):
        reporter = MarkdownReporter()
        ctx = reporter.build_context(result_with_insights, source_file="train.csv")
        md = reporter.render(ctx)
        assert "62" in md  # quality score

    def test_markdown_with_insights_contains_section_7(self, result_with_insights):
        reporter = MarkdownReporter()
        ctx = reporter.build_context(result_with_insights, source_file="train.csv")
        md = reporter.render(ctx)
        assert "AI Insights" in md

    def test_markdown_with_insights_contains_summary(self, result_with_insights):
        reporter = MarkdownReporter()
        ctx = reporter.build_context(result_with_insights, source_file="train.csv")
        md = reporter.render(ctx)
        assert "Titanic" in md

    def test_markdown_with_insights_contains_recommendations(self, result_with_insights):
        reporter = MarkdownReporter()
        ctx = reporter.build_context(result_with_insights, source_file="train.csv")
        md = reporter.render(ctx)
        assert "Encode categorical" in md or "categorical" in md

    def test_markdown_skipped_shows_skip_reason(self, result_with_skipped_insights):
        reporter = MarkdownReporter()
        ctx = reporter.build_context(result_with_skipped_insights, source_file="train.csv")
        md = reporter.render(ctx)
        assert "No API key set" in md

    def test_markdown_no_insights_shows_not_requested(self, pipeline_result):
        from copy import copy
        r = copy(pipeline_result)
        r.ai_insights = None
        reporter = MarkdownReporter()
        ctx = reporter.build_context(r, source_file="train.csv")
        md = reporter.render(ctx)
        assert "not requested" in md


class TestJSONReporterWithAI:

    @pytest.fixture
    def result_with_insights(self, pipeline_result, mock_insights):
        from copy import copy
        r = copy(pipeline_result)
        r.ai_insights = mock_insights
        return r

    def test_json_payload_has_ai_insights_key(self, result_with_insights):
        payload = JSONReporter().build_payload(result_with_insights, source_file="train.csv")
        assert "ai_insights" in payload

    def test_json_ai_insights_has_quality_score(self, result_with_insights):
        payload = JSONReporter().build_payload(result_with_insights, source_file="train.csv")
        assert payload["ai_insights"]["data_quality_score"] == 62

    def test_json_ai_insights_not_skipped(self, result_with_insights):
        payload = JSONReporter().build_payload(result_with_insights, source_file="train.csv")
        assert payload["ai_insights"]["skipped"] is False

    def test_json_without_ai_insights_is_null(self, pipeline_result):
        from copy import copy
        r = copy(pipeline_result)
        r.ai_insights = None
        payload = JSONReporter().build_payload(r, source_file="train.csv")
        assert payload["ai_insights"] is None

    def test_json_skipped_insights_serialized_correctly(self, pipeline_result):
        from copy import copy
        r = copy(pipeline_result)
        r.ai_insights = AIInsights.skipped_result("No key")
        payload = JSONReporter().build_payload(r, source_file="train.csv")
        assert payload["ai_insights"]["skipped"] is True
        assert "No key" in payload["ai_insights"]["skip_reason"]

    def test_json_with_insights_is_valid_json_string(self, result_with_insights):
        reporter = JSONReporter()
        payload = reporter.build_payload(result_with_insights, source_file="train.csv")
        json_str = reporter.render(payload)
        parsed = json.loads(json_str)
        assert parsed["ai_insights"]["data_quality_score"] == 62
