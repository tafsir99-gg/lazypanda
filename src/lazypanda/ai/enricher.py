"""
AI Enricher — orchestrates the full AI enrichment flow.

WHAT IT DOES:
    Orchestrates the complete AI enrichment pipeline:

    1. Gate:          Check if AI is enabled and not explicitly disabled
    2. Budget check:  Estimate prompt tokens → can we afford this call?
    3. Cache lookup:  Is there a fresh cached response for this prompt?
    4. API call:      If no cache hit → call Gemini
    5. Parse:         Extract + validate the JSON response
    6. Cache store:   Save successful response for next time
    7. Return:        AIInsights dataclass with all results

    At EVERY step, if something goes wrong → return AIInsights(skipped=True)
    instead of raising an exception. The pipeline continues deterministically.

WHY AN `AIInsights` DATACLASS INSTEAD OF `dict | None`?
    - Type safety: reporters know exactly what fields to expect
    - The `skipped` flag lets reporters render a meaningful "AI Disabled"
      section instead of silently omitting insights
    - `skip_reason` tells the user WHY AI was skipped (no key? budget? error?)
    - `from_cache` lets the CLI show "⚡ From cache" to reward good behavior

THE `skipped` FLAG PATTERN:
    Instead of returning None (which callers must check everywhere),
    we return an AIInsights with `skipped=True`. This is the Null Object
    pattern — reporters can always call insights.overall_summary safely.

    skipped=True means one of:
        - --no-ai flag was passed
        - GEMINI_API_KEY is not set
        - config.ai.enabled = false
        - Token budget would be exceeded
        - API call failed after all retries
        - Response could not be parsed

API KEY LOOKUP ORDER:
    1. Environment variable: GEMINI_API_KEY
    2. .env file (loaded by python-dotenv)
    If neither is set → skip with clear message.

USAGE:
    enricher = AIEnricher()
    insights = enricher.enrich(result, config=app_config, use_ai=True, source_file="train.csv")

    if not insights.skipped:
        print(insights.overall_summary)
        print(f"Quality score: {insights.data_quality_score}/100")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from lazypanda.ai.cache import AIResponseCache
from lazypanda.ai.client import GeminiAPIError, GeminiClient
from lazypanda.ai.prompt_builder import PromptBuilder
from lazypanda.ai.response_parser import ResponseParser
from lazypanda.ai.token_budget import TokenBudgetManager
from lazypanda.core.config_manager import AIConfig, AppConfig
from lazypanda.core.pipeline import PipelineResult

logger = logging.getLogger("lazypanda")

# Load .env file for GEMINI_API_KEY (non-fatal if file doesn't exist)
load_dotenv()

_ENV_KEY_NAME = "GEMINI_API_KEY"

# Estimated output tokens for the structured JSON response
_ESTIMATED_OUTPUT_TOKENS = 600


@dataclass
class AIInsights:
    """
    Structured output from one AI enrichment run.

    Always returned by AIEnricher.enrich() — check `skipped` before using.

    Fields:
        skipped:            True if AI was not called for any reason.
        skip_reason:        Human-readable explanation of why it was skipped.
        from_cache:         True if the response came from the disk cache.
        model_used:         The Gemini model name that was called.
        tokens_used:        Actual tokens reported by the API (0 if from cache or skipped).
        overall_summary:    2-3 sentence dataset narrative.
        data_quality_score: 0-100 quality score BEFORE cleaning (100 = perfect).
        score_rationale:    1-2 sentence explanation of the score.
        column_insights:    List of {column, insight} dicts for key columns.
        recommendations:    List of actionable next-step strings.
        cleaning_assessment: 1-2 sentence assessment of the cleaning decisions.
    """
    # Status
    skipped:             bool = False
    skip_reason:         str  = ""
    from_cache:          bool = False
    model_used:          str  = ""
    tokens_used:         int  = 0

    # Content (empty strings/lists when skipped)
    overall_summary:      str = ""
    data_quality_score:   int = 0
    score_rationale:      str = ""
    column_insights:      list[dict] = field(default_factory=list)
    recommendations:      list[str]  = field(default_factory=list)
    cleaning_assessment:  str = ""

    @classmethod
    def skipped_result(cls, reason: str) -> "AIInsights":
        """Factory for a skipped result with a clear reason."""
        logger.info("AI enrichment skipped: %s", reason)
        return cls(skipped=True, skip_reason=reason)

    @classmethod
    def from_parsed(
        cls,
        parsed: dict,
        model_used: str,
        tokens_used: int,
        from_cache: bool,
    ) -> "AIInsights":
        """Factory for a successful enrichment result."""
        return cls(
            skipped=False,
            skip_reason="",
            from_cache=from_cache,
            model_used=model_used,
            tokens_used=tokens_used,
            overall_summary=parsed.get("overall_summary", ""),
            data_quality_score=parsed.get("data_quality_score", 0),
            score_rationale=parsed.get("score_rationale", ""),
            column_insights=parsed.get("column_insights", []),
            recommendations=parsed.get("recommendations", []),
            cleaning_assessment=parsed.get("cleaning_assessment", ""),
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for JSON reporter)."""
        return {
            "skipped":            self.skipped,
            "skip_reason":        self.skip_reason,
            "from_cache":         self.from_cache,
            "model_used":         self.model_used,
            "tokens_used":        self.tokens_used,
            "overall_summary":    self.overall_summary,
            "data_quality_score": self.data_quality_score,
            "score_rationale":    self.score_rationale,
            "column_insights":    self.column_insights,
            "recommendations":    self.recommendations,
            "cleaning_assessment": self.cleaning_assessment,
        }


class AIEnricher:
    """
    Orchestrates the AI enrichment pipeline.

    Injects:
        - AIResponseCache  (injectable for testing)
        - GeminiClient     (lazy-initialized only if needed)
        - PromptBuilder    (injectable for testing)
        - ResponseParser   (injectable for testing)

    All components are injectable so tests can mock any layer independently.
    """

    def __init__(
        self,
        cache: AIResponseCache | None = None,
        prompt_builder: PromptBuilder | None = None,
        response_parser: ResponseParser | None = None,
    ):
        self._cache          = cache or AIResponseCache()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._response_parser = response_parser or ResponseParser()

    def enrich(
        self,
        result: PipelineResult,
        config: AppConfig,
        use_ai: bool = True,
        source_file: str = "dataset.csv",
        model_override: str | None = None,
    ) -> AIInsights:
        """
        Enrich a PipelineResult with AI-generated insights.

        Args:
            result:         The completed pipeline result.
            config:         The AppConfig for this run (reads config.ai.*).
            use_ai:         If False, skip immediately (--no-ai flag).
            source_file:    Source file basename (for prompt context only).
            model_override: If set, use this Gemini model instead of the default.
                            Used by the wizard command to inject the user's chosen model.

        Returns:
            AIInsights — always returned, check .skipped before using content.
        """
        ai_cfg = config.ai

        # ── Gate 1: --no-ai flag ───────────────────────────────────────────────
        if not use_ai:
            return AIInsights.skipped_result("--no-ai flag was set")

        # ── Gate 2: config.ai.enabled = false ─────────────────────────────────
        if not ai_cfg.enabled:
            return AIInsights.skipped_result("AI disabled in config (ai.enabled: false)")

        # ── Gate 3: dry run ────────────────────────────────────────────────────
        if result.dry_run:
            return AIInsights.skipped_result("Dry run mode — AI insights not generated")

        # ── Gate 4: GEMINI_API_KEY ─────────────────────────────────────────────
        api_key = os.environ.get(_ENV_KEY_NAME, "").strip()
        if not api_key:
            return AIInsights.skipped_result(
                f"{_ENV_KEY_NAME} environment variable is not set. "
                f"Set it or use --no-ai for a fully free run."
            )

        # ── Build prompt ───────────────────────────────────────────────────────
        try:
            prompt, estimated_input_tokens = self._prompt_builder.build(
                result, source_file=source_file
            )
        except Exception as e:
            logger.error("PromptBuilder failed: %s", e, exc_info=True)
            return AIInsights.skipped_result(f"Prompt building failed: {e}")

        # ── Gate 5: Token budget ───────────────────────────────────────────────
        budget = TokenBudgetManager(max_tokens=ai_cfg.max_tokens_per_run)
        if not budget.can_afford(estimated_input_tokens, _ESTIMATED_OUTPUT_TOKENS):
            return AIInsights.skipped_result(
                f"Token budget ({ai_cfg.max_tokens_per_run} max) would be exceeded. "
                f"Estimated: {estimated_input_tokens + _ESTIMATED_OUTPUT_TOKENS} tokens. "
                f"Increase ai.max_tokens_per_run in config to enable AI for this dataset."
            )

        # ── Cache lookup ───────────────────────────────────────────────────────
        cache_key = AIResponseCache.make_key(prompt)

        if ai_cfg.use_cache:
            cached = self._cache.get(cache_key, ttl_hours=ai_cfg.cache_ttl_hours)
            if cached is not None:
                logger.info("AI enrichment: using cached response (%s)", cache_key[:12])
                return AIInsights.from_parsed(
                    cached,
                    model_used="(from cache)",
                    tokens_used=0,
                    from_cache=True,
                )

        # ── API call ───────────────────────────────────────────────────────────
        client_kwargs: dict = {
            "api_key": api_key,
            "max_output_tokens": ai_cfg.max_tokens_per_run,
        }
        if model_override:
            client_kwargs["model"] = model_override
        client = GeminiClient(**client_kwargs)
        try:
            raw_response, actual_tokens = client.generate(prompt)
        except GeminiAPIError as e:
            return AIInsights.skipped_result(f"Gemini API call failed: {e}")

        budget.record_usage(actual_tokens)

        # ── Parse response ─────────────────────────────────────────────────────
        parsed = self._response_parser.parse(raw_response)
        if parsed is None:
            return AIInsights.skipped_result(
                "Gemini returned an unparseable response. "
                "The cleaning pipeline result is still complete and valid."
            )

        # ── Cache store ────────────────────────────────────────────────────────
        if ai_cfg.use_cache:
            self._cache.set(cache_key, parsed)

        logger.info(
            "AI enrichment: success — quality score=%d, %d recommendation(s), %d tokens",
            parsed.get("data_quality_score", 0),
            len(parsed.get("recommendations", [])),
            actual_tokens,
        )

        return AIInsights.from_parsed(
            parsed,
            model_used=client.model_name,
            tokens_used=actual_tokens,
            from_cache=False,
        )
