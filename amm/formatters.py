"""
AMM Formatters - Structured Memory Formatting

Provides formatting functions for episodic memories in a canonical, fielded structure.
"""

import logging
import re
from typing import Iterable, Optional, List, Sequence, Any

from amm.utils import is_structured_episodic_memory, get_em_content

logger = logging.getLogger(__name__)


def _fmt_inv(items: Iterable[str]) -> str:
    """Format inventory items as comma-separated string"""
    items_list = list(items or [])
    if not items_list:
        return ""
    return ", ".join(items_list)


def _last_n(xs: Iterable[str], n: int = 5) -> List[str]:
    """Get last n items from iterable"""
    xs = list(xs or [])
    return xs[-n:]


def _fmt_signed_reward(r: float) -> str:
    """Format reward with sign and one decimal (e.g., +7.0, +20.0, -5.0)"""
    return f"{r:+.1f}"


def _fmt_score(prev: Optional[float], curr: Optional[float]) -> str:
    """Format score transition with one decimal (e.g., 77.0→77.0, 80.0→100.0)"""
    if prev is None:
        prev = 0.0
    if curr is None:
        curr = 0.0
    return f"{prev:.1f}→{curr:.1f}"


def _parse_inventory_text(inventory_text: str) -> List[str]:
    """
    Parse inventory text string into list of items.
    
    Handles formats like:
    - "In your inventory, you see:\n\tan orange"
    - "an orange, a cup"
    - "orange"
    - Empty string
    """
    if not inventory_text:
        return []
    
    # Remove common prefixes
    text = inventory_text.strip()
    prefixes = [
        "In your inventory, you see:",
        "Inventory:",
        "inventory:",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    # Split by newlines and tabs, then by commas
    items = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Remove leading tabs/bullets
        line = line.lstrip('\t•- ')
        # Split by comma if present
        if ',' in line:
            items.extend([item.strip() for item in line.split(',')])
        else:
            items.append(line)
    
    # Clean up items (remove empty strings, "a", "an", "the")
    cleaned = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        # Remove leading articles
        words = item.split()
        if words and words[0].lower() in ['a', 'an', 'the']:
            item = ' '.join(words[1:])
        if item:
            cleaned.append(item)
    
    return cleaned if cleaned else []


def format_em_structured(
    *,
    goal_text: str,
    room: str,
    inventory_items: Iterable[str],
    action: str,
    observation: str,
    recent_actions: Iterable[str],
    recent_obs: Iterable[str],
    reward: float,
    score_prev: float,
    score_curr: float,
    primary_tag: str,
    subtag: Optional[str] = None,
    look: Optional[str] = None,
) -> str:
    """
    Format episodic memory in canonical structured format.
    
    Args:
        goal_text: Task or goal description
        room: Current room/location
        inventory_items: Iterable of inventory item strings
        action: Action that was executed
        observation: Observation received
        recent_actions: Recent actions (last 5 will be used)
        recent_obs: Recent observations (last 5 will be used)
        reward: Reward value (float)
        score_prev: Previous score (float)
        score_curr: Current score (float)
        primary_tag: Primary tag (e.g., "episodic_success")
        subtag: Optional subtag (e.g., "milestone")
        look: Optional room description/look string
        
    Returns:
        Formatted memory string with fielded structure
    """
    # Format inventory
    inv_items = list(inventory_items or [])
    inv_str = _fmt_inv(inv_items)
    
    # Get last 5 recent actions and observations
    ra = _last_n(recent_actions, 5)
    ro = _last_n(recent_obs, 5)
    
    # Format tags (primary, optional subtag)
    tags = primary_tag if not subtag else f"{primary_tag}, {subtag}"
    
    # Build structured lines
    lines = [
        f'TASK: "{goal_text}"',
        f"STATE: room={room}; inventory=[{inv_str}]",
    ]
    
    # Add LOOK line if provided
    if look:
        # Normalize look string (strip, but preserve structure)
        look_str = look.strip() if look else ""
        if look_str:
            lines.append(f"LOOK: {look_str}")
    
    lines.extend([
        f"ACTION: {action}",
        f"OBSERVATION: {observation}",
        f"RECENT_ACTIONS: [{', '.join(ra)}]",
        f"RECENT_OBS: [{', '.join(ro)}]",
        f"REWARD: {_fmt_signed_reward(reward)} ({_fmt_score(score_prev, score_curr)})",
        # WHY_REWARDED is intentionally omitted here; Letta will add it.
        f"TAGS: {tags}",
    ])
    
    return "\n".join(lines) + "\n"


# ============================================================================
# Swift Episodic Memory Integration (Backbone + Logging)
# ============================================================================

def get_em_tags(em: Any) -> List[str]:
    """
    Extract tags from an episodic memory object.
    
    Works with both dict-like and object-like memories.
    
    Args:
        em: Episodic memory object (dict or object with tags attribute)
        
    Returns:
        List of tag strings, or empty list if tags not found
    """
    tags = getattr(em, "tags", None)
    if tags is None and isinstance(em, dict):
        tags = em.get("tags")
    if tags is None:
        return []
    # Normalize to list
    if isinstance(tags, str):
        return [tags]
    return list(tags) if tags else []


def get_em_timestamp(em: Any) -> str:
    """
    Extract timestamp from an episodic memory object.
    
    Works with both dict-like and object-like memories.
    
    Args:
        em: Episodic memory object (dict or object with timestamp attribute)
        
    Returns:
        Timestamp string, or "N/A" if timestamp not found
    """
    ts = getattr(em, "timestamp", None)
    if ts is None and isinstance(em, dict):
        ts = em.get("timestamp")
    return str(ts or "N/A")


def format_episodic_memories_for_swift(
    episodic_memories: Sequence[Any],
    max_ems: int = 3,
) -> List[Any]:
    """
    Filter and cap episodic memories for Swift integration.
    
    Filters EMs by tags (success/near-miss vs avoidance) and caps the number
    of selected EMs at max_ems. Returns the filtered list of EM objects only
    (no string summarization yet).
    
    Args:
        episodic_memories: Sequence of episodic memory objects
        max_ems: Maximum number of EMs to select (default: 3)
        
    Returns:
        List of selected episodic memory objects (unchanged, no string conversion)
    """
    logger.info(
        f"[AMM SwiftMem] format_episodic_memories_for_swift: received "
        f"{len(episodic_memories)} EMs (max_ems={max_ems})"
    )
    
    # Filter out unstructured EMs before tag bucketing/capping
    structured = [em for em in episodic_memories if is_structured_episodic_memory(em)]
    if len(structured) < len(episodic_memories):
        dropped_count = len(episodic_memories) - len(structured)
        logger.info(
            f"[AMM SwiftMem] Filtered out {dropped_count} "
            "unstructured EMs before tag bucketing/capping."
        )
        # Log preview of one dropped EM for debugging
        for em in episodic_memories:
            if not is_structured_episodic_memory(em):
                preview = get_em_content(em)[:200].replace("\n", " ")
                logger.debug(
                    "[AMM SwiftMem] Example dropped unstructured EM preview: "
                    f"{preview}..."
                )
                break
    
    # If nothing structured remains, early-return
    if not structured:
        logger.info("[AMM SwiftMem] No structured EMs available for Swift.")
        return []
    
    # Bucket EMs by tags (operate on structured list)
    success_like = []
    avoidance_like = []
    
    for em in structured:
        try:
            tags = get_em_tags(em)
            tags_lower = [t.lower() for t in tags]
            
            # Check for success/near-miss tags
            if any(tag in tags_lower for tag in ["episodic_success", "episodic_nearmiss"]):
                success_like.append(em)
            # Check for avoidance tags
            elif any(tag in tags_lower for tag in ["avoidance", "episodic_failure"]):
                avoidance_like.append(em)
            else:
                # If no recognized tags, log warning but don't skip
                logger.debug(
                    f"[AMM SwiftMem] EM has unrecognized tags: {tags}, "
                    "treating as success-like for now"
                )
                success_like.append(em)
        except Exception as e:
            logger.warning(
                f"[AMM SwiftMem] Failed to process EM (skipping): {e}"
            )
            continue
    
    logger.info(
        f"[AMM SwiftMem] Tag bucket counts: "
        f"success_like={len(success_like)}, avoidance_like={len(avoidance_like)}"
    )
    
    # Build selected list: take from success_like first, then avoidance_like if needed
    selected = []
    
    # Take from success_like first (in order), up to max_ems
    remaining_slots = max_ems
    for em in success_like[:remaining_slots]:
        selected.append(em)
        remaining_slots -= 1
    
    # If we have room and avoidance EMs exist, optionally fill from avoidance_like
    # For now, we'll keep it simple and only use success_like
    # (This can be extended later if needed)
    if remaining_slots > 0 and avoidance_like:
        logger.debug(
            f"[AMM SwiftMem] {remaining_slots} slots remaining, "
            f"but not yet filling from avoidance_like (keeping simple for now)"
        )
    
    logger.info(
        f"[AMM SwiftMem] Selected {len(selected)} EMs for Swift "
        f"(before summarization / injection)"
    )
    
    # Log details for each selected EM
    for i, em in enumerate(selected):
        try:
            content = get_em_content(em)
            timestamp = get_em_timestamp(em)
            tags = get_em_tags(em)
            
            content_preview = content[:80] + "..." if len(content) > 80 else content
            tags_str = ", ".join(tags) if tags else "N/A"
            
            logger.info(
                f"[AMM SwiftMem] EM {i+1}: timestamp={timestamp}, "
                f"tags=[{tags_str}], content_preview=\"{content_preview}\""
            )
        except Exception as e:
            logger.warning(
                f"[AMM SwiftMem] Failed to log details for selected EM {i+1}: {e}"
            )
    
    return selected


def shorten_em_for_swift(em: Any, max_chars: int = 400) -> str:
    """
    Convert a structured episodic memory into a compact textual snippet for Swift.
    
    - Keep key fields (TASK, STATE, ACTION, OBSERVATION, REWARD, WHY_REWARDED, TAGS).
    - Drop or heavily truncate LOOK to avoid huge room dumps.
    - Hard truncate at max_chars with an ellipsis if needed.
    
    Args:
        em: Episodic memory object
        max_chars: Maximum character length for the snippet (default: 400)
        
    Returns:
        Compact textual snippet of the episodic memory
    """
    text = get_em_content(em)
    lines = text.splitlines()
    
    kept_lines: List[str] = []
    inside_look_block = False
    
    for line in lines:
        stripped = line.strip()
        
        # Start of LOOK section
        if stripped.startswith("LOOK:") or stripped.startswith("LOOK::"):
            # Drop LOOK entirely
            inside_look_block = True
            continue
        
        # Skip lines inside the LOOK block (room dump)
        if inside_look_block:
            # End the LOOK block when we see another high-level field
            if any(stripped.startswith(prefix) for prefix in (
                "ACTION:", "OBSERVATION:", "REWARD:", "WHY_REWARDED:", "TAGS:", "TASK:", "STATE:"
            )):
                inside_look_block = False
            else:
                continue  # still inside LOOK details, skip
        
        # At this point, either not in LOOK, or we just detected a new top-level field
        
        # Only keep key fields
        if any(stripped.startswith(prefix) for prefix in (
            "TASK:", "STATE:", "ACTION:", "OBSERVATION:", "REWARD:", "WHY_REWARDED:", "TAGS:"
        )):
            kept_lines.append(stripped)
    
    compact = "\n".join(kept_lines).strip()
    
    if len(compact) > max_chars:
        compact = compact[: max_chars - 3].rstrip() + "..."
    
    return compact


def build_swift_memories_block(
    input_str: str,
    episodic_memories: Optional[Sequence[Any]],
    trigger_context: Optional[str] = None,
) -> Optional[str]:
    """
    Build a Swift memories block from episodic memories (backbone stub).
    
    For now, this function only logs what would be done and does NOT modify
    the Swift prompt. It returns None to indicate that the original input_str
    should be used unchanged.
    
    Args:
        input_str: Current Swift input string (for logging context)
        episodic_memories: Optional sequence of episodic memory objects
        trigger_context: Optional context string (e.g., "T1", "T2", "T3") for logging
        
    Returns:
        None (for now, indicating no modification to input_str)
    """
    # Entry log - always log when this function is called
    logger.info(
        f"[AMM SwiftMem] build_swift_memories_block: input_str_len={len(input_str)}, "
        f"episodic_memories={len(episodic_memories) if episodic_memories else 0}, "
        f"trigger_context={trigger_context!r}"
    )
    
    if not episodic_memories:
        logger.info("[AMM SwiftMem] No episodic memories provided for Swift integration.")
        return None
    
    logger.info(
        f"[AMM SwiftMem] Received {len(episodic_memories)} episodic memories for "
        "potential Swift integration."
    )
    
    selected = format_episodic_memories_for_swift(episodic_memories, max_ems=3)
    if not selected:
        logger.info("[AMM SwiftMem] No EMs selected for Swift (after filtering/capping).")
        return None
    
    # For now, do NOT change the Swift prompt and do NOT build any textual block
    # Just log what would be used later
    logger.info(
        f"[AMM SwiftMem] Returning None (no prompt modification yet); "
        f"would inject {len(selected)} EMs into Swift prompt in future phase"
    )
    
    return None

