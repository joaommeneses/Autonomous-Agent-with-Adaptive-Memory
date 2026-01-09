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


def build_sage_planning_memories_block(
    episodic_memories: Sequence[Any],
    max_ems: int = 5,
    logger: Optional[Any] = None
) -> Optional[str]:
    """
    Build the Sage Planning memory augmentation block from episodic memories.
    
    Args:
        episodic_memories: List of episodic memory dicts with 'content' and optionally 'timestamp'
        max_ems: Maximum number of EMs to include (default: 5)
        logger: Optional logger for logging
        
    Returns:
        Formatted memory block string, or None if no memories
    """
    if not episodic_memories:
        if logger:
            logger.info("[AMM SageMem] No episodic memories provided; skipping injection")
        return None
    
    # Cap to max_ems
    ems_to_use = list(episodic_memories[:max_ems])
    num_received = len(episodic_memories)
    num_injected = len(ems_to_use)
    
    if logger:
        logger.info(f"[AMM SageMem] Building memory block: {num_received} EMs received, {num_injected} EMs will be injected")
    
    # Build the block
    lines = [
        "====================",
        "",
        "RELEVANT PAST EPISODES (FROM MEMORY)",
        "",
        "You are given a small set of past episodes retrieved from memory. These may be from similar tasks or similar states.",
        "",
        "Use them as *hints* to answer Questions 1–5 more efficiently.",
        "",
        "How to use them:",
        "",
        "- Extract actionable patterns: useful objects/containers, common successful subgoals, typical mistake-fixes, and critical \"FOCUS\" timing.",
        "",
        "- Prefer the current observations when there is any conflict.",
        "",
        "- Do not assume every item mentioned in memory exists in the current run; if missing, propose the closest substitute in this environment.",
        "",
        "- Do not invent new focus targets beyond what the task allows.",
        "",
        "- You do NOT need to quote memory verbatim; only use the relevant details.",
        "",
        "====================",
        ""
    ]
    
    # Add each EM
    for i, em in enumerate(ems_to_use, 1):
        timestamp = get_em_timestamp(em)
        content = get_em_content(em)
        
        # Defensive: ensure we have content
        if not content:
            content = str(em) if em else "N/A"
        
        lines.append(f"[Memory Episode {i}] timestamp={timestamp}")
        lines.append("")
        lines.append(content)
        lines.append("")
    
    lines.append("====================")
    
    block = "\n".join(lines)
    
    if logger:
        logger.info(f"[AMM SageMem] Memory block built: {len(block)} chars")
    
    return block


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
    # Filter out unstructured EMs before tag bucketing/capping
    structured = [em for em in episodic_memories if is_structured_episodic_memory(em)]
    if len(structured) < len(episodic_memories):
        dropped_count = len(episodic_memories) - len(structured)
        logger.debug(
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
    
    # Log details for each selected EM
    for i, em in enumerate(selected):
        try:
            content = get_em_content(em)
            timestamp = get_em_timestamp(em)
            tags = get_em_tags(em)
            
            content_preview = content[:80] + "..." if len(content) > 80 else content
            tags_str = ", ".join(tags) if tags else "N/A"
            
            logger.debug(
                f"[AMM SwiftMem] EM {i+1}: timestamp={timestamp}, "
                f"tags=[{tags_str}], content_preview=\"{content_preview}\""
            )
        except Exception as e:
            logger.warning(
                f"[AMM SwiftMem] Failed to log details for selected EM {i+1}: {e}"
            )
    
    return selected


def _truncate(s: str, max_len: int) -> str:
    """Helper to safely truncate a string and append ... if needed."""
    if not s:
        return ""
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[:max_len - 3].rstrip() + "..."


def _parse_state_line(state_line: str) -> tuple[str, str]:
    """
    Parse STATE line to extract room and inventory.
    
    Format: STATE: room=kitchen; inventory=[orange, cup]
    
    Returns:
        (room, inventory_str) where inventory_str is the full [item1, item2] portion
    """
    room = "unknown"
    inventory_str = "[]"
    
    if not state_line:
        return room, inventory_str
    
    # Extract room= value
    room_match = re.search(r'room=([^;]+)', state_line)
    if room_match:
        room = room_match.group(1).strip()
    
    # Extract inventory=[...] portion
    inv_match = re.search(r'inventory=\[(.*?)\]', state_line)
    if inv_match:
        inventory_str = f"[{inv_match.group(1).strip()}]"
    
    return room, inventory_str


def _parse_field_from_content(content: str, field_name: str) -> str:
    """
    Parse a field value from structured EM content.
    
    Looks for lines like "FIELD_NAME: value" and extracts the value.
    Handles multi-line values by taking everything after the colon until the next field.
    """
    if not content:
        return ""
    
    lines = content.splitlines()
    field_prefix = f"{field_name}:"
    value_parts = []
    collecting = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(field_prefix):
            # Found the field, extract value after colon
            value = stripped[len(field_prefix):].strip()
            if value:
                value_parts.append(value)
            collecting = True
        elif collecting:
            # Check if this is the start of another field
            if any(stripped.startswith(prefix + ":") for prefix in (
                "TASK", "STATE", "LOOK", "ACTION", "OBSERVATION", "REWARD", 
                "WHY_REWARDED", "TAGS", "RECENT_ACTIONS", "RECENT_OBS"
            )):
                break
            elif stripped:
                value_parts.append(stripped)
    
    result = " ".join(value_parts).strip()
    # Remove quotes if present
    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1]
    return result


def _classify_episode_type(tags_str: str) -> str:
    """
    Classify episode type based on tags string.
    
    Returns: "success", "nearmiss", "avoidance", or "other"
    """
    tags_lower = tags_str.lower()
    
    if "episodic_success" in tags_lower:
        return "success"
    elif "episodic_nearmiss" in tags_lower or "nearmiss" in tags_lower:
        return "nearmiss"
    elif "avoidance" in tags_lower or "invalid_action" in tags_lower or "episodic_failure" in tags_lower:
        return "avoidance"
    else:
        return "other"


def compact_em_for_swift(em: Any, idx: int) -> str:
    """
    Convert a structured episodic memory into a compact, single-line snippet for Swift.
    
    Format: [Past Episode {idx} — {episode_type}] room={room}; inv={inventory}; 
            did: "{action}" → obs: "{observation}" (reward {reward}, tags: {tags})
    
    Args:
        em: Episodic memory object (dict or object with content attribute)
        idx: 1-based index for this EM
        
    Returns:
        Single-line compact snippet string
    """
    content = get_em_content(em)
    
    # Parse STATE line
    state_line = ""
    for line in content.splitlines():
        if line.strip().startswith("STATE:"):
            state_line = line.strip()
            break
    
    room, inventory_str = _parse_state_line(state_line)
    room = _truncate(room, max_len=40)
    inventory_str = _truncate(inventory_str, max_len=60)
    
    # Parse ACTION
    action = _parse_field_from_content(content, "ACTION")
    action = _truncate(action, max_len=80)
    if not action:
        action = "N/A"
    
    # Parse OBSERVATION
    observation = _parse_field_from_content(content, "OBSERVATION")
    observation = _truncate(observation, max_len=80)
    if not observation:
        observation = "N/A"
    
    # Parse REWARD
    reward_str = _parse_field_from_content(content, "REWARD")
    if not reward_str:
        reward_str = "N/A"
    else:
        # Extract just the reward value (e.g., "+50.0" from "+50.0 (20.0→70.0)")
        reward_match = re.search(r'([+-]?\d+\.?\d*)', reward_str)
        if reward_match:
            reward_str = reward_match.group(1)
            # Ensure sign is present
            if not reward_str.startswith(('+', '-')):
                reward_str = '+' + reward_str
    
    # Parse TAGS
    tags_str = _parse_field_from_content(content, "TAGS")
    tags_str = _truncate(tags_str, max_len=80)
    if not tags_str:
        tags_str = "N/A"
    
    # Classify episode type
    episode_type = _classify_episode_type(tags_str)
    
    # Build snippet
    snippet = (
        f"[Past Episode {idx} — {episode_type}] "
        f"room={room}; inv={inventory_str}; "
        f'did: "{action}" → obs: "{observation}" '
        f"(reward {reward_str}, tags: {tags_str})"
    )
    
    # Ensure single line (replace any newlines with spaces)
    snippet = " ".join(snippet.split())
    
    # Log the snippet
    logger.debug(f"[AMM SwiftMem] Swift snippet {idx}: {snippet}")
    
    return snippet


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
    Build and inject a Swift memories block from episodic memories into the input string.
    
    If no usable EMs are available, returns None so caller uses original input_str.
    Otherwise, returns augmented input_str with EM block injected before "What action should you do next? </s>".
    
    Args:
        input_str: Current Swift input string (must be full Swift prompt from compose_instance)
        episodic_memories: Optional sequence of episodic memory objects
        trigger_context: Optional context string (e.g., "T1", "T2", "T3") for logging
        
    Returns:
        Augmented input string with memories block, or None if no EMs or injection failed
    """
    # INVARIANT CHECK: Ensure input_str is a real Swift prompt, not a placeholder or task_description
    marker = "What action should you do next? </s>"
    if marker not in input_str:
        logger.warning(
            f"[AMM SwiftMem] ERROR: Swift EM injection called with non-Swift prompt "
            f"(missing final question marker). input_str_len={len(input_str)}, "
            f"trigger_context={trigger_context!r}. Preview: {input_str[:100]}..."
        )
        return None
    
    if not episodic_memories:
        logger.debug("[AMM SwiftMem] No episodic memories provided for Swift integration.")
        return None
    
    selected = format_episodic_memories_for_swift(episodic_memories, max_ems=3)
    if not selected:
        logger.debug("[AMM SwiftMem] No EMs selected for Swift (after filtering/capping).")
        return None
    
    # Generate compact snippets for each selected EM
    snippets: List[str] = []
    for idx, em in enumerate(selected, start=1):
        try:
            snippet = compact_em_for_swift(em, idx)
            snippets.append(snippet)
        except Exception as e:
            logger.warning(
                f"[AMM SwiftMem] Failed to create compact snippet for EM {idx}: {e}. Skipping."
            )
            continue
    
    if not snippets:
        logger.warning("[AMM SwiftMem] No valid snippets generated from selected EMs.")
        return None
    
    # Build the memories block
    # Format: </s> Relevant past episodes (from memory): </s> [snippet1] | [snippet2] | [snippet3] </s>
    mem_block = (
        "</s> Relevant past episodes (from memory): </s> " +
        " | ".join(snippets) +
        " </s> "
    )
    
    # Inject into input_str right before "What action should you do next? </s>"
    marker = "What action should you do next? </s>"
    
    if marker not in input_str:
        logger.warning(
            "[AMM SwiftMem] WARNING: Could not find 'What action should you do next? </s>' "
            "in input_str; skipping EM injection."
        )
        return None
    
    # Replace marker with mem_block + marker (only first occurrence)
    augmented_input_str = input_str.replace(marker, mem_block + marker, 1)
    
    logger.info(
        f"[AMM SwiftMem] Injected {len(snippets)} EMs into Swift prompt "
        f"(len: {len(input_str)} → {len(augmented_input_str)}, trigger: {trigger_context})"
    )
    
    return augmented_input_str

