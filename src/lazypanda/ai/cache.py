"""
AI response cache — disk-based, TTL-aware.

WHAT IT DOES:
    Saves Gemini API responses to disk so identical requests never hit the API
    twice. This is the primary cost-saving mechanism for the AI layer.

WHY CACHE AI RESPONSES?
    Scenario: You run `lazypanda clean train.csv` twice with the same data and config.
    Without cache: 2 API calls, 2x cost, 2x latency.
    With cache: 1 API call. The second run reads from disk — instant and free.

    In practice, during iterative Kaggle work (adjust config → re-clean →
    check → adjust), you'll hit the cache almost every time after the first run.

HOW THE CACHE KEY WORKS:
    key = SHA256(prompt_text)
    file = ~/.cache/lazypanda/<key[:2]>/<key>.json

    WHY SHA256?
    - Collision-resistant (two different prompts → two different keys)
    - Deterministic (same prompt → same key, always)
    - Fast to compute

    WHY SUBDIRECTORY (<key[:2]>)?
    - Avoids having thousands of files in one flat directory (OS performance)
    - Same trick used by git's object store

CACHE FILE FORMAT:
    {
        "cached_at": "2025-01-01T12:00:00Z",
        "ttl_hours": 24,
        "prompt_sha256": "abc123...",
        "response": { ... }  ← the parsed Gemini response dict
    }

TTL BEHAVIOR:
    - On read: if (now - cached_at) > ttl_hours → miss (treat as cache miss)
    - On write: always write (even if a cached entry exists — it gets refreshed)
    - TTL of 0 → entries never expire

USAGE:
    cache = AIResponseCache(cache_dir=Path.home() / ".cache" / "lazypanda")
    key = cache.make_key(prompt_text)
    hit = cache.get(key, ttl_hours=24)   # returns dict or None
    if hit is None:
        response = call_gemini(prompt_text)
        cache.set(key, response)
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("lazypanda")

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "lazypanda"


class AIResponseCache:
    """
    Disk-based SHA256 cache for Gemini API responses.

    Thread-safety: NOT thread-safe. Designed for single-process CLI use.
    """

    def __init__(self, cache_dir: Path | None = None):
        self._cache_dir = Path(cache_dir or _DEFAULT_CACHE_DIR)

    # ── Public interface ───────────────────────────────────────────────────────

    @staticmethod
    def make_key(prompt_text: str) -> str:
        """
        Compute the SHA256 cache key for a prompt.

        Args:
            prompt_text: The full prompt string sent to Gemini.

        Returns:
            64-character hex SHA256 digest.
        """
        return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

    def get(self, key: str, ttl_hours: int = 24) -> dict | None:
        """
        Retrieve a cached response.

        Args:
            key:       SHA256 key from make_key().
            ttl_hours: Maximum age in hours. 0 = never expire.

        Returns:
            The cached response dict if valid, or None on miss/expiry.
        """
        path = self._key_path(key)
        if not path.exists():
            logger.debug("Cache MISS: %s (file not found)", key[:12])
            return None

        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Cache: could not read entry %s: %s", key[:12], e)
            return None

        # TTL check
        if ttl_hours > 0:
            cached_at = datetime.fromisoformat(record["cached_at"])
            age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
            if age_hours > ttl_hours:
                logger.debug("Cache EXPIRED: %s (age=%.1fh, ttl=%dh)", key[:12], age_hours, ttl_hours)
                return None

        logger.info("Cache HIT: %s (key prefix=%s)", path.name, key[:12])
        return record["response"]

    def set(self, key: str, response: dict) -> None:
        """
        Store a response in the cache.

        Args:
            key:      SHA256 key from make_key().
            response: The parsed response dict to cache.
        """
        path = self._key_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "prompt_sha256": key,
            "response": response,
        }

        try:
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.debug("Cache SET: %s", key[:12])
        except OSError as e:
            # Cache write failure is non-fatal — just log and continue
            logger.warning("Cache: could not write entry %s: %s", key[:12], e)

    def clear(self) -> int:
        """
        Delete all cached entries.

        Returns:
            Number of entries deleted.
        """
        count = 0
        if not self._cache_dir.exists():
            return 0
        for f in self._cache_dir.rglob("*.json"):
            try:
                f.unlink()
                count += 1
            except OSError:
                pass
        logger.info("Cache cleared: %d entries deleted from %s", count, self._cache_dir)
        return count

    def size(self) -> int:
        """Return the number of cached entries."""
        if not self._cache_dir.exists():
            return 0
        return sum(1 for _ in self._cache_dir.rglob("*.json"))

    # ── Private ───────────────────────────────────────────────────────────────

    def _key_path(self, key: str) -> Path:
        """Return the filesystem path for a given cache key."""
        # Use first 2 chars as a subdirectory (same trick as git objects)
        return self._cache_dir / key[:2] / f"{key}.json"
