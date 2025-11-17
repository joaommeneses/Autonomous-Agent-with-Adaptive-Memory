"""
AMM Formatters - Structured Memory Formatting

Provides formatting functions for episodic memories in a canonical, fielded structure.
"""

from typing import Iterable, Optional, List


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

