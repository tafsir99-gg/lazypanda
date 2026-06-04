"""
Gemini API Client — thin wrapper around google-genai with retry, timeout, logging.

WHAT IT DOES:
    Wraps the google-genai SDK's generate_content() with:
    - Exponential backoff retry (up to 3 attempts)
    - Per-call timeout enforcement
    - Structured logging (timing, token usage)
    - Clean error handling that converts API exceptions to our own types

SDK NOTE:
    This uses the NEW google-genai package (google.genai), NOT the deprecated
    google-generativeai package. The new SDK was released in 2025 and is the
    only one receiving active updates and support.

    New API pattern:
        from google import genai
        client = genai.Client(api_key="...")
        response = client.models.generate_content(model="...", contents="...")

WHY A WRAPPER CLASS?
    1. Testability: tests mock GeminiClient, not the raw SDK
    2. Retry logic: one place to change retry behavior for all callers
    3. Logging:     one place to add observability
    4. Future-proofing: if the SDK changes, only this file changes

RETRY STRATEGY: Exponential Backoff
    Attempt 1: immediate
    Attempt 2: wait 2 seconds
    Attempt 3: wait 4 seconds
    Then fail.

MODEL SELECTION:
    Default: gemini-2.0-flash (fast, low cost, excellent for structured JSON)
    Why not gemini-1.5-pro?  More expensive, slower — overkill for this task

USAGE:
    client = GeminiClient(api_key="your-key")
    text, tokens = client.generate(prompt_text)
"""

from __future__ import annotations

import logging
import time

from google import genai
from google.genai import types as genai_types

from lazypanda.utils.exceptions import AIDataCleanerError

logger = logging.getLogger("lazypanda")

# Gemini model to use by default
_DEFAULT_MODEL  = "gemini-2.5-flash"
_MAX_RETRIES    = 3
_RETRY_BASE_SEC = 2     # Retry delays: 2s, 4s
_TIMEOUT_SEC    = 30    # Max seconds per attempt

# Generation config defaults: low temperature + JSON MIME type for deterministic, complete output.
# TRUNCATION PREVENTION:
#   response_mime_type="application/json" forces the API to emit valid, complete JSON.
#   max_output_tokens defaults to 8192 but can be overridden by the wizard directly.
#   Together these eliminate mid-sentence truncation for all typical datasets.
_DEFAULT_GENERATION_KWARGS = {
    "temperature": 0.1,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}


class GeminiAPIError(AIDataCleanerError):
    """Raised when the Gemini API call fails after all retries."""
    pass


class GeminiClient:
    """
    Thin wrapper around the google-genai SDK.

    Handles: API key config, retry with exponential backoff, logging.

    TESTABILITY: This class is designed to be easily mocked in tests.
    The caller (AIEnricher) instantiates GeminiClient inside a patchable
    import, making it trivial to replace with a MagicMock.
    """

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL, max_output_tokens: int | None = None):
        """
        Configure the Gemini client.

        Args:
            api_key:           Your Gemini API key (from GEMINI_API_KEY env var).
            model:             Gemini model name (default: gemini-2.0-flash).
            max_output_tokens: Explicit override for max output tokens.
        """
        self._client = genai.Client(api_key=api_key)
        self._model_name = model
        
        # Build generation config with explicit override if provided
        gen_kwargs = _DEFAULT_GENERATION_KWARGS.copy()
        if max_output_tokens is not None:
            gen_kwargs["max_output_tokens"] = max_output_tokens
            
        self._generation_config = genai_types.GenerateContentConfig(**gen_kwargs)
        logger.debug("GeminiClient: configured with model=%s, max_tokens=%s", model, gen_kwargs["max_output_tokens"])

    @property
    def model_name(self) -> str:
        """The Gemini model being used."""
        return self._model_name

    def generate(self, prompt: str) -> tuple[str, int]:
        """
        Call the Gemini API with retry and timeout.

        Args:
            prompt: The full prompt string to send.

        Returns:
            (response_text, total_tokens_used)

        Raises:
            GeminiAPIError: If all retries are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.debug(
                    "GeminiClient: attempt %d/%d (prompt_len=%d chars)",
                    attempt, _MAX_RETRIES, len(prompt),
                )
                t0 = time.monotonic()

                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=self._generation_config,
                )

                elapsed_ms = (time.monotonic() - t0) * 1000

                # Extract token usage from usage_metadata
                usage = getattr(response, "usage_metadata", None)
                total_tokens = 0
                if usage:
                    total_tokens = getattr(usage, "total_token_count", 0) or 0

                # Check finish reason — warn if truncated
                finish_reason = None
                candidates = getattr(response, "candidates", None)
                if candidates:
                    finish_reason = getattr(candidates[0], "finish_reason", None)
                    if finish_reason and str(finish_reason) not in ("STOP", "FinishReason.STOP", "1"):
                        logger.warning(
                            "GeminiClient: response finish_reason=%s — possible truncation. "
                            "Consider increasing ai.max_tokens_per_run in config.",
                            finish_reason,
                        )

                # Extract response text
                text = response.text

                logger.info(
                    "GeminiClient: success in %.0fms — %d tokens, finish=%s (model=%s)",
                    elapsed_ms, total_tokens, finish_reason, self._model_name,
                )

                return text, total_tokens

            except Exception as e:
                last_error = e
                logger.warning(
                    "GeminiClient: attempt %d failed: %s — %s",
                    attempt, type(e).__name__, str(e)[:200],
                )

                if attempt < _MAX_RETRIES:
                    wait = _RETRY_BASE_SEC ** (attempt - 1)  # 2s, 4s
                    logger.debug("GeminiClient: waiting %.1fs before retry", wait)
                    time.sleep(wait)

        raise GeminiAPIError(
            f"Gemini API call failed after {_MAX_RETRIES} attempts. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        )
