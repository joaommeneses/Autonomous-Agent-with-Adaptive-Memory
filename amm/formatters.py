"""
AMM Formatters - Structured Memory Formatting

Provides formatting functions for episodic memories in a canonical, fielded structure.
"""

import logging
from typing import Iterable, Optional, List, Sequence, Any

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
        f"[AMM SwiftMem] format_episodic_memories_for_swift: "
        f"received {len(episodic_memories)} EMs (max_ems={max_ems})"
    )
    
    if not episodic_memories:
        logger.info("[AMM SwiftMem] No episodic memories provided, returning empty list")
        return []
    
    # Bucket EMs by tags
    success_like = []
    avoidance_like = []
    
    for em in episodic_memories:
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


def build_swift_memories_block(
    input_str: str,
    episodic_memories: Optional[Sequence[Any]],
) -> Optional[str]:
    """
    Build a Swift memories block from episodic memories (backbone stub).
    
    For now, this function only logs what would be done and does NOT modify
    the Swift prompt. It returns None to indicate that the original input_str
    should be used unchanged.
    
    Args:
        input_str: Current Swift input string (for logging context)
        episodic_memories: Optional sequence of episodic memory objects
        
    Returns:
        None (for now, indicating no modification to input_str)
    """
    if episodic_memories is None or len(episodic_memories) == 0:
        logger.info(
            "[AMM SwiftMem] No episodic memories provided for Swift integration "
            "(episodic_memories is empty or None)"
        )
        return None
    
    n = len(episodic_memories)
    logger.info(
        f"[AMM SwiftMem] Received {n} episodic memories for potential Swift integration"
    )
    
    # Call helper to filter/cap EMs
    filtered_ems = format_episodic_memories_for_swift(episodic_memories, max_ems=3)
    
    # Log tag distribution
    tag_counts = {}
    for em in filtered_ems:
        try:
            tags = get_em_tags(em)
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        except Exception:
            pass
    
    tag_dist_str = ", ".join([f"{tag}={count}" for tag, count in tag_counts.items()])
    if tag_dist_str:
        logger.info(
            f"[AMM SwiftMem] After tag filtering/capping: {len(filtered_ems)} EMs selected "
            f"(max_ems=3), tag distribution: {tag_dist_str}"
        )
    else:
        logger.info(
            f"[AMM SwiftMem] After tag filtering/capping: {len(filtered_ems)} EMs selected "
            f"(max_ems=3)"
        )
    
    # Log preview per EM (timestamp + first 80 chars of content, tags)
    for i, em in enumerate(filtered_ems):
        try:
            content = get_em_content(em)
            timestamp = get_em_timestamp(em)
            tags = get_em_tags(em)
            
            content_preview = content[:80] + "..." if len(content) > 80 else content
            tags_str = ", ".join(tags) if tags else "N/A"
            
            logger.info(
                f"[AMM SwiftMem] Preview EM {i+1}: timestamp={timestamp}, "
                f"tags=[{tags_str}], content=\"{content_preview}\""
            )
        except Exception as e:
            logger.warning(
                f"[AMM SwiftMem] Failed to log preview for EM {i+1}: {e}"
            )
    
    # For now, do NOT change the Swift prompt and do NOT build any textual block
    # Just log what would be used later
    logger.info(
        "[AMM SwiftMem] Returning None (no prompt modification yet); "
        f"would inject {len(filtered_ems)} EMs into Swift prompt in future phase"
    )
    
    return None

