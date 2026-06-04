"""
Token Budget Manager.

WHAT IT DOES:
    Enforces a hard token cap on all AI usage in a single pipeline run.
    If a call would exceed the budget, it is skipped — the tool runs
    deterministically instead.

WHY A HARD CAP?
    Without a hard cap, a large dataset (1000 columns, lots of issues)
    could generate a very long prompt and blow past your intended spend.
    The cap is a circuit breaker: exceed it → skip AI → run stays free.

    This is non-negotiable for the "minimize Gemini API usage" constraint.

TOKEN ESTIMATION:
    We estimate token counts conservatively (characters / 3.5, rounded up).
    This is a rough approximation — the Gemini API reports exact counts
    after a call. We err on the side of over-estimating to stay under budget.

    WHY 3.5 chars/token?
    - English text: ~4 chars/token (OpenAI rough rule)
    - Code/JSON: ~3-4 chars/token
    - 3.5 is a conservative middle ground

USAGE:
    budget = TokenBudgetManager(max_tokens=2000)

    # Before making an API call:
    if budget.can_afford(estimated_input=800, estimated_output=500):
        response = call_gemini(prompt)
        budget.record_usage(actual_tokens=1150)
    else:
        # Skip AI — budget would be exceeded
        ...

    print(f"Used {budget.used} / {budget.max} tokens")
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger("lazypanda")

# Characters per token — conservative estimate for JSON/English mix
_CHARS_PER_TOKEN = 3.5


class TokenBudgetManager:
    """
    Tracks and enforces a per-run token budget for Gemini API calls.

    This is not a sophisticated cost model — it's a simple circuit breaker
    to prevent runaway spending on large datasets.
    """

    def __init__(self, max_tokens: int = 2000):
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        self._max = max_tokens
        self._used = 0

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def max(self) -> int:
        """The configured maximum token budget for this run."""
        return self._max

    @property
    def used(self) -> int:
        """Tokens consumed so far in this run."""
        return self._used

    @property
    def remaining(self) -> int:
        """Tokens remaining in the budget."""
        return max(0, self._max - self._used)

    @property
    def is_exhausted(self) -> bool:
        """True if the budget is fully consumed."""
        return self._used >= self._max

    # ── Core API ───────────────────────────────────────────────────────────────

    def can_afford(self, estimated_input: int = 0, estimated_output: int = 0) -> bool:
        """
        Check if a call fits within the remaining budget.

        Args:
            estimated_input:  Estimated input tokens for the prompt.
            estimated_output: Estimated output tokens for the response.

        Returns:
            True if (estimated_input + estimated_output) <= remaining budget.
        """
        total_estimated = estimated_input + estimated_output
        affordable = total_estimated <= self.remaining

        if not affordable:
            logger.warning(
                "TokenBudget: call would cost ~%d tokens but only %d remain (max=%d)",
                total_estimated,
                self.remaining,
                self._max,
            )
        return affordable

    def record_usage(self, tokens_used: int) -> None:
        """
        Record actual tokens consumed by a completed API call.

        Args:
            tokens_used: The exact token count reported by the API.
        """
        if tokens_used < 0:
            raise ValueError(f"tokens_used must be >= 0, got {tokens_used}")
        self._used += tokens_used
        logger.debug(
            "TokenBudget: recorded %d tokens (used=%d / max=%d, remaining=%d)",
            tokens_used,
            self._used,
            self._max,
            self.remaining,
        )
        if self._used > self._max * 0.9:
            logger.warning(
                "TokenBudget: %.0f%% consumed (%d / %d). Next call may be skipped.",
                (self._used / self._max) * 100,
                self._used,
                self._max,
            )

    def reset(self) -> None:
        """Reset the usage counter to zero (for testing)."""
        self._used = 0

    # ── Static helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Rough token estimate from text length.

        Uses the 3.5 chars/token heuristic — conservative enough to avoid
        underestimating and accidentally exceeding the budget.

        Args:
            text: The text to estimate tokens for.

        Returns:
            Estimated token count (always >= 1 for non-empty strings).
        """
        if not text:
            return 0
        return max(1, math.ceil(len(text) / _CHARS_PER_TOKEN))
