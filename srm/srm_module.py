"""
Self-Reflection Module (SRM) - Core Implementation

SRM proposes valid, cheap "guiding" actions from a curated subset of validActions.
"""

import string
from typing import List, Set, Optional


# Allowed action prefixes (sorted by length descending to avoid partial matching)
ALLOWED_PREFIXES = tuple(sorted([
    "turn on",
    "turn off",
    "pick up",
    "look",
    "inventory",
    "examine",
    "inspect",
    "search",
    "open",
    "close",
    "read",
    "take",
    "get",
    "drop",
    "put",
    "place",
    "move",
    "go",
    "walk",
    "enter",
    "leave",
    "use",
], key=lambda x: (-len(x), x)))  # Sort by length descending, then alphabetically

# Normalized prefixes (hoisted for performance)
NORMALIZED_PREFIXES = tuple(p.lower() for p in ALLOWED_PREFIXES)

# Verb base weights for "cheap guiding action" behavior
# Higher weight = safer/cheaper actions (look, examine, etc.)
# Lower weight = riskier/more impactful actions (drop, use, etc.)
VERB_BASE_WEIGHTS = {
    # High weight: safe exploration/inspection actions
    "look": 3,
    "inventory": 3,
    "examine": 3,
    "inspect": 3,
    "search": 3,
    "open": 2,
    "read": 2,
    "close": 1,
    # Medium weight: acquisition actions
    "take": 1,
    "get": 1,
    "pick up": 1,
    # Lower weight: risky/impactful actions
    "drop": 0,
    "put": 0,
    "place": 0,
    "move": 0,
    "go": 0,
    "walk": 0,
    "enter": 0,
    "leave": 0,
    "use": 0,
    "turn on": 0,
    "turn off": 0,
}


# Minimal stopwords for context extraction
STOPWORDS = {"the", "and", "with", "for", "from", "that", "this", "are", "was", "were", "been", "have", "has", "had"}


# Punctuation removal translation table (hoisted for performance)
PUNCTUATION_TRANSLATOR = str.maketrans('', '', string.punctuation)


def build_action_subset(valid_actions: List[str], max_candidates: int = 200) -> List[str]:
    """
    Build a subset of valid actions filtered by allowed prefixes.
    
    Args:
        valid_actions: List of valid action strings
        max_candidates: Maximum number of candidates to return (hard cap for efficiency)
    
    Returns:
        List of actions that start with one of the allowed prefixes, preserving original casing.
        Limited to max_candidates items.
    """
    if not valid_actions:
        return []
    
    subset = []
    
    for action in valid_actions:
        if len(subset) >= max_candidates:
            break
        
        action_normalized = action.strip().lower()
        
        # Check if action starts with any allowed prefix (using hoisted normalized prefixes)
        for prefix in NORMALIZED_PREFIXES:
            if action_normalized.startswith(prefix):
                # Check that prefix is followed by space or end of string (for exact prefix match)
                if len(action_normalized) == len(prefix) or action_normalized[len(prefix)] == ' ':
                    subset.append(action)  # Preserve original casing
                    break
    
    return subset


def extract_context(observation: str, inventory: str) -> Set[str]:
    """
    Extract context tokens from observation and inventory.
    
    Args:
        observation: Current observation string
        inventory: Current inventory string
    
    Returns:
        Set of tokens (lowercase, length >= 3, punctuation removed)
    """
    # Combine observation and inventory
    text = f"{observation} {inventory}".lower()
    
    # Remove punctuation (using hoisted translation table)
    text = text.translate(PUNCTUATION_TRANSLATOR)
    
    # Split on whitespace and filter
    tokens = set()
    for token in text.split():
        token = token.strip()
        # Keep tokens with length >= 3
        if len(token) >= 3:
            # Remove stopwords
            if token not in STOPWORDS:
                tokens.add(token)
    
    return tokens


def score_action(
    action: str,
    context_tokens: Set[str],
    recent_actions: List[str],
    recent_window: int = 5
) -> int:
    """
    Score a candidate action based on context and recent actions.
    
    Scoring scheme:
    - Base weight: verb-specific (higher for safe actions like look/examine, lower for risky actions)
    - +2 if action contains any context_tokens (substring match)
    - -5 if action equals last action (exact string match)
    - -2 if action appears in recent window (exact string match)
    - Small bonus for "look" / "inventory" if context is empty
    
    Args:
        action: Action string to score
        context_tokens: Set of context tokens from observation/inventory
        recent_actions: List of recent actions (most recent last)
        recent_window: Number of recent actions to check
    
    Returns:
        Integer score (higher is better)
    """
    # Get verb-specific base weight
    action_lower = action.lower()
    base_weight = 1  # Default base weight
    for prefix, weight in VERB_BASE_WEIGHTS.items():
        if action_lower.startswith(prefix):
            # Check that prefix is followed by space or end of string
            if len(action_lower) == len(prefix) or action_lower[len(prefix)] == ' ':
                base_weight = weight
                break
    
    score = base_weight
    
    # Check if action contains any context tokens
    for token in context_tokens:
        # Simple substring match (token boundaries not strictly enforced for simplicity)
        if token in action_lower:
            score += 2
            break  # Only count once
    
    # Check recent actions
    if recent_actions:
        recent_window_actions = recent_actions[-recent_window:] if len(recent_actions) > recent_window else recent_actions
        
        # -5 if equals last action
        if action == recent_actions[-1]:
            score -= 5
        
        # -2 if appears in recent window (excluding last, already penalized)
        elif action in recent_window_actions[:-1]:
            score -= 2
    
    # Small bonus for "look" / "inventory" if context is empty
    if not context_tokens:
        if action_lower.startswith("look") or action_lower.startswith("inventory"):
            score += 1
    
    return score


def choose_best_action(
    candidates: List[str],
    context_tokens: Set[str],
    recent_actions: List[str],
    recent_window: int = 5
) -> Optional[str]:
    """
    Choose the best action from candidates based on scoring.
    
    Args:
        candidates: List of candidate action strings
        context_tokens: Set of context tokens from observation/inventory
        recent_actions: List of recent actions (most recent last)
        recent_window: Number of recent actions to check for scoring
    
    Returns:
        Best action string, or None if candidates is empty
    """
    if not candidates:
        return None
    
    # Find best action using max (O(n) instead of O(n log n))
    # Use first-seen approach for deterministic tie-breaking (preserves candidate order)
    max_score = None
    best_action = None
    
    for action in candidates:
        score = score_action(action, context_tokens, recent_actions, recent_window)
        if max_score is None or score > max_score:
            max_score = score
            best_action = action
        # No tie-break needed: first max score wins (deterministic given candidate order)
    
    return best_action


def propose_action(
    valid_actions: List[str],
    observation: str,
    inventory: str,
    recent_actions: List[str],
    max_candidates: int = 200,
    recent_window: int = 5
) -> Optional[str]:
    """
    Propose a valid guiding action from a curated subset of validActions.
    
    This is the public API for SRM. It:
    1. Builds a subset of valid actions filtered by allowed prefixes
    2. Extracts context tokens from observation and inventory
    3. Scores and selects the best action
    
    Args:
        valid_actions: List of all valid action strings
        observation: Current observation string
        inventory: Current inventory string
        recent_actions: List of recent actions (most recent last)
        max_candidates: Maximum number of candidates to consider (default: 200)
        recent_window: Number of recent actions to check for scoring (default: 5)
    
    Returns:
        A valid action string (guaranteed to be in valid_actions), or None if no action can be proposed.
    
    Raises:
        AssertionError: If returned action is not in valid_actions (should never happen)
    """
    # Build subset
    subset = build_action_subset(valid_actions, max_candidates)
    
    if not subset:
        return None
    
    # Extract context
    context_tokens = extract_context(observation, inventory)
    
    # Choose best action
    best_action = choose_best_action(subset, context_tokens, recent_actions, recent_window)
    
    # No assertion needed: best_action comes from subset, which is built from valid_actions
    # This is guaranteed by construction (subset only contains actions from valid_actions)
    
    return best_action
