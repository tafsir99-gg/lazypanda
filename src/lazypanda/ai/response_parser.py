"""
Response Parser — validates and parses Gemini JSON responses.

WHAT IT DOES:
    Takes the raw text returned by the Gemini API and converts it into a
    structured, validated Python dict (or returns None if parsing fails).

WHY A DEDICATED PARSER?
    LLMs sometimes return JSON wrapped in markdown code fences:
        ```json
        { "key": "value" }
        ```
    Or with trailing comments, or with slightly wrong key names.
    A dedicated parser handles all these cases gracefully.

DESIGN PRINCIPLE: Fail softly, never crash.
    If parsing fails for ANY reason, return None.
    The enricher will then set `skipped=True` with a clear reason.
    The pipeline continues deterministically — the user just doesn't get
    AI insights this run.

    WHY NOT raise an exception?
    A parsing failure means the LLM returned something unexpected.
    This is not a programming error — it's an expected failure mode.
    Crashing a data cleaning run because AI returned bad JSON would be
    extremely annoying for the user.

VALIDATED FIELDS:
    - overall_summary: str
    - data_quality_score: int, 0-100
    - score_rationale: str
    - column_insights: list of {column, insight} dicts
    - recommendations: list of strings
    - cleaning_assessment: str (optional — fall back to empty string)

USAGE:
    parser = ResponseParser()
    insights_dict = parser.parse(raw_text)
    if insights_dict is None:
        # Parsing failed — use deterministic-only mode
        ...
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger("lazypanda")

# Regex to extract JSON from a markdown code fence
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class ResponseParser:
    """
    Parse and validate a raw Gemini API response string.

    Handles:
    - Plain JSON strings
    - JSON wrapped in ```json ... ``` markdown code fences
    - Partial/malformed JSON (returns None — never raises)
    """

    def parse(self, raw_text: str) -> dict | None:
        """
        Parse raw Gemini response text into a validated dict.

        Args:
            raw_text: The raw string returned by the Gemini API.

        Returns:
            A validated dict with the AI insights, or None if parsing fails.
        """
        if not raw_text or not raw_text.strip():
            logger.warning("ResponseParser: empty response from Gemini")
            return None

        # Early truncation check: a complete JSON object must end with }
        stripped = raw_text.strip()
        if stripped and not stripped.endswith("}"):
            # Could still be a code-fenced response — don't short-circuit, but warn
            last_brace = stripped.rfind("}")
            if last_brace == -1:
                logger.warning(
                    "ResponseParser: response appears truncated — no closing '}' found. "
                    "This usually means max_output_tokens was hit. "
                    "Response ends with: %r",
                    stripped[-80:],
                )

        # Step 1: Try to extract JSON (plain or from code fence)
        parsed = self._extract_json(raw_text)
        if parsed is None:
            logger.warning(
                "ResponseParser: could not extract JSON from response "
                "(first 200 chars): %s",
                raw_text[:200],
            )
            return None

        # Step 2: Validate required fields
        return self._validate(parsed)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_json(self, text: str) -> dict | None:
        """Try multiple strategies to find and parse a JSON object."""
        text = text.strip()

        # Strategy 1: Direct JSON parse
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Strategy 2: JSON inside a markdown code fence
        match = _JSON_FENCE_RE.search(text)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find the first { ... } block in the text
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(text[start:end + 1])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return None

    def _validate(self, data: dict) -> dict | None:
        """
        Validate the parsed dict has the required structure.

        Missing optional fields are filled with safe defaults.
        Missing required fields → return None.
        """
        required = {
            "overall_summary":    str,
            "data_quality_score": (int, float),
            "score_rationale":    str,
        }

        for field, expected_type in required.items():
            if field not in data:
                logger.warning("ResponseParser: missing required field '%s'", field)
                return None
            if not isinstance(data[field], expected_type):
                # Try coercion for numeric fields
                if expected_type == (int, float) and isinstance(data[field], str):
                    try:
                        data[field] = float(data[field])
                    except ValueError:
                        logger.warning(
                            "ResponseParser: field '%s' is not numeric: %r",
                            field,
                            data[field],
                        )
                        return None
                else:
                    logger.warning(
                        "ResponseParser: field '%s' expected %s, got %s",
                        field,
                        expected_type,
                        type(data[field]).__name__,
                    )
                    return None

        # Clamp score to [0, 100]
        score = data["data_quality_score"]
        data["data_quality_score"] = max(0, min(100, int(score)))

        # Optional fields — fill defaults if missing or wrong type
        data.setdefault("column_insights", [])
        data.setdefault("recommendations", [])
        data.setdefault("cleaning_assessment", "")

        # Validate list types
        if not isinstance(data["column_insights"], list):
            data["column_insights"] = []
        if not isinstance(data["recommendations"], list):
            data["recommendations"] = []

        # Clean up column_insights entries
        clean_insights = []
        for item in data["column_insights"]:
            if isinstance(item, dict) and "column" in item and "insight" in item:
                clean_insights.append({
                    "column":  str(item["column"]),
                    "insight": str(item["insight"]),
                })
        data["column_insights"] = clean_insights

        # Clean up recommendations — keep only strings
        data["recommendations"] = [
            str(r) for r in data["recommendations"] if r and str(r).strip()
        ]

        return data
