"""
AMM Utilities - Helper functions for Adaptive Memory Module

Provides utility functions for memory processing, deduplication, etc.
"""

import hashlib
import logging
from typing import Any, List

logger = logging.getLogger(__name__)


def dedup_memories_by_content(memories: List[Any]) -> List[Any]:
    """
    Deduplicate a list of episodic memory objects based on their textual content.

    - Uses a fast hash (SHA1) over the `content` field.
    - Preserves original order of the first occurrence of each unique content.
    - Works with both dict-like and object-like memories.

    Args:
        memories: List of memory objects (dict or Letta ArchivalMemorySearchResult)

    Returns:
        A new list with duplicates (same content) removed, preserving ordering.
    """
    seen_hashes = set()
    unique_memories: List[Any] = []

    for m in memories:
        # Try to extract `content` robustly:
        content = None
        if hasattr(m, "content"):
            content = m.content
        elif isinstance(m, dict):
            content = m.get("content")
        else:
            # Fallback: treat full object as content (stringified). Rare, but keeps us robust.
            content = str(m)

        if content is None:
            # If for some reason there is no content, just keep it
            unique_memories.append(m)
            continue

        # Hash **only the content** so timestamps, ids, etc. don't break dedup
        h = hashlib.sha1(content.encode("utf-8")).hexdigest()

        if h in seen_hashes:
            continue

        seen_hashes.add(h)
        unique_memories.append(m)

    # Optional log (we'll log at call sites instead of here to avoid spam)
    return unique_memories


