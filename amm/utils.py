"""
AMM Utilities - Helper functions for Adaptive Memory Module

Provides utility functions for memory processing, deduplication, etc.
"""

import hashlib
import logging
from typing import Any, List

logger = logging.getLogger(__name__)


def get_em_content(em: Any) -> str:
    """
    Extract content from an episodic memory object.
    
    Works with both dict-like and object-like memories.
    
    Args:
        em: Episodic memory object (dict or object with content attribute)
        
    Returns:
        Content string, or empty string if content not found
    """
    content = getattr(em, "content", None)
    if content is None and isinstance(em, dict):
        content = em.get("content", "")
    return str(content or "")


def is_structured_episodic_memory(em: Any) -> bool:
    """
    Heuristically determine whether an episodic memory is in the
    newer structured format:
      - Must contain 'TASK:' and 'STATE:' in content.
      - Ideally also ACTION / OBSERVATION / REWARD / WHY_REWARDED / TAGS.
    
    Returns True if content looks structured; False for old narrative-style EMs.
    """
    content = getattr(em, "content", None)
    if content is None and isinstance(em, dict):
        content = em.get("content")
    
    if not content:
        return False
    
    text = str(content)
    has_task = "TASK:" in text
    has_state = "STATE:" in text
    
    if not (has_task and has_state):
        return False
    
    # Optional: soft check for at least one detail field
    has_any_detail = any(
        key in text
        for key in ("ACTION:", "OBSERVATION:", "REWARD:", "WHY_REWARDED:", "TAGS:")
    )
    
    # For now, require TASK+STATE; log missing detail fields elsewhere if useful.
    return has_task and has_state


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


