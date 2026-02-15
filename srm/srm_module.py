"""
Self-Reflection Module (SRM) v2 - Core Implementation

SRM v2 proposes useful "probe" actions when Swift fails, focusing on task-relevant
and context-relevant actions with deterministic scoring.
"""

import string
from typing import List, Set, Optional, Tuple


# SRM v2 verb set (sorted by length descending for longest-prefix matching)
SRM_V2_VERBS = tuple(sorted([
    "pick up",
    "turn on",
    "turn off",
    "disconnect",
    "activate",
    "deactivate",
    "examine",
    "inspect",
    "search",
    "open",
    "use",
    "pour",
    "mix",
    "dunk",
    "put",
    "move",
    "wait",
], key=lambda x: (-len(x), x)))  # Sort by length descending, then alphabetically

# Normalized verbs (hoisted for performance)
NORMALIZED_VERBS = tuple(v.lower() for v in SRM_V2_VERBS)

# Verb base utility weights (SRM v2)
VERB_BASE_WEIGHTS = {
    "open": 6,
    "search": 6,
    "examine": 5,
    "inspect": 5,
    "activate": 4,
    "deactivate": 4,
    "connect": 4,
    "disconnect": 4,
    "use": 3,
    "put": 3,
    "pour": 2,
    "mix": 2,
    "dunk": 2,
    "pick up": 2,
    "move": 2,
    "wait": 0,
}

# Progress keywords for wait gating
PROGRESS_KEYWORDS = {
    "boil", "boiling", "heat", "heating", "warm", "warming",
    "cook", "cooking", "melt", "melting", "brew", "brewing",
    "steep", "steeping", "fill", "filling", "drain", "draining",
    "dissolve", "dissolving"
}

# Minimal stopwords for token extraction
STOPWORDS = {"the", "and", "with", "for", "from", "that", "this", "are", "was", "were", "been", "have", "has", "had"}

# Disfavored target tokens (apply strong penalty if these appear in action arguments)
DISFAVORED_TOKENS = {"air", "outside", "agent"}

# Failure keywords for cooldown (if last_obs contains these, apply strong penalty to that action)
FAILURE_KEYWORDS = {"doesn't", "not sure how"}

# Punctuation removal translation table (hoisted for performance)
PUNCTUATION_TRANSLATOR = str.maketrans('', '', string.punctuation)


def extract_tokens(text: str) -> Set[str]:
    """
    Extract tokens from text (lowercase, punctuation removed, stopwords removed, len>=3).
    
    Args:
        text: Input text string
    
    Returns:
        Set of tokens
    """
    if not text:
        return set()
    
    text_lower = text.lower()
    text_clean = text_lower.translate(PUNCTUATION_TRANSLATOR)
    
    tokens = set()
    for token in text_clean.split():
        token = token.strip()
        if len(token) >= 3 and token not in STOPWORDS:
            tokens.add(token)
    
    return tokens


def parse_action(action: str) -> Tuple[Optional[str], str, Set[str], Optional[str]]:
    """
    Parse action into verb, arg_text, arg_tokens, and main_object.
    Uses longest-prefix matching from SRM v2 verb set.
    
    Args:
        action: Action string (e.g., "examine apple", "pick up red paint")
    
    Returns:
        Tuple of (verb, arg_text, arg_tokens, main_object)
        - verb: Matched verb from SRM v2 verb set, or None
        - arg_text: String after verb
        - arg_tokens: Tokens extracted from arg_text
        - main_object: First token in arg_tokens, or None
    """
    action_normalized = action.strip().lower()
    
    # Find longest matching verb prefix
    matched_verb = None
    matched_len = 0
    
    for verb in NORMALIZED_VERBS:
        if action_normalized.startswith(verb):
            # Check that verb is followed by space or end of string
            if len(action_normalized) == len(verb) or action_normalized[len(verb)] == ' ':
                if len(verb) > matched_len:
                    matched_verb = verb
                    matched_len = len(verb)
    
    if matched_verb is None:
        return None, action, set(), None
    
    # Extract arg_text (everything after verb)
    arg_text = action[len(matched_verb):].strip()
    
    # Extract arg_tokens
    arg_tokens = extract_tokens(arg_text)
    
    # Get main_object (first token in arg_tokens)
    main_object = next(iter(sorted(arg_tokens))) if arg_tokens else None
    
    return matched_verb, arg_text, arg_tokens, main_object


def filter_candidates(
    valid_actions: List[str],
    task_tokens: Set[str],
    context_tokens: Set[str],
    max_candidates: int = 200
) -> List[str]:
    """
    Filter candidates by verb + overlap gate.
    
    Keep action iff:
    - verb in SRM verb set AND
    - (arg_tokens overlaps task_tokens OR overlaps context_tokens) OR verb == "wait" (always allowed in first pass)
    
    NOTE: max_candidates is kept for API compatibility but we scan ALL valid_actions
    to avoid early-capping bias. The actual filtering happens here, but scoring scans all.
    
    Args:
        valid_actions: List of valid action strings
        task_tokens: Tokens from task description
        context_tokens: Tokens from look + inventory
        max_candidates: Maximum number of candidates to return (for compatibility, but full scan preferred)
    
    Returns:
        List of filtered candidate actions (no early capping - scans all valid_actions)
    """
    candidates = []
    
    # Scan ALL valid_actions (no early capping based on iteration order)
    for action in valid_actions:
        verb, arg_text, arg_tokens, main_object = parse_action(action)
        
        if verb is None:
            continue
        
        # Wait is always allowed in first pass
        if verb == "wait":
            candidates.append(action)
            continue
        
        # Check overlap with task or context
        task_overlap = bool(arg_tokens & task_tokens)
        context_overlap = bool(arg_tokens & context_tokens)
        
        if task_overlap or context_overlap:
            candidates.append(action)
    
    return candidates


def score_action_v2(
    action: str,
    verb: str,
    arg_tokens: Set[str],
    main_object: Optional[str],
    task_description: str,
    task_tokens: Set[str],
    look_tokens: Set[str],
    inv_tokens: Set[str],
    recent_actions: List[str],
    look: str = "",
    recent_obs: Optional[List[str]] = None,
    recent_window: int = 5,
    impossible_actions: Optional[Set[str]] = None
) -> int:
    """
    Score action using SRM v2 scoring cascade.
    
    Args:
        action: Full action string
        verb: Parsed verb
        arg_tokens: Tokens from action arguments
        main_object: Main object token
        task_description: Task description string
        task_tokens: Tokens from task description
        look_tokens: Tokens from look
        inv_tokens: Tokens from inventory
        recent_actions: List of recent actions (most recent last)
        look: Current look string
        recent_obs: Optional list of recent observations
        recent_window: Number of recent actions to check
        impossible_actions: Set of actions that failed recently (for cooldown)
    
    Returns:
        Integer score (higher is better)
    """
    # Base utility weight (verb utility dominates)
    base_weight = VERB_BASE_WEIGHTS.get(verb, 0)
    score = base_weight
    
    # Task relevance
    task_overlap_count = len(arg_tokens & task_tokens)
    task_relevance_bonus = 4 * min(task_overlap_count, 3)  # Cap to +12
    score += task_relevance_bonus
    
    # Optional: verb string appears in task description
    task_lower = task_description.lower()
    if verb in task_lower:
        score += 2
    
    # Context relevance
    context_relevance_bonus = 0
    if arg_tokens & look_tokens:
        context_relevance_bonus += 3
    if arg_tokens & inv_tokens:
        context_relevance_bonus += 2
    score += context_relevance_bonus
    
    # B.4) Disfavored target penalties
    if arg_tokens & DISFAVORED_TOKENS:
        score -= 10
    
    # B.5) Strongly gate/penalize "use" actions
    # "use" should lose against open/search/examine/inspect/activate/deactivate/move unless clearly relevant
    if verb == "use":
        # Only allow "use" to compete if it has strong relevance signals
        # If task/context relevance is weak, apply penalty
        # if task_relevance_bonus == 0 and context_relevance_bonus == 0:
        score -= 10  # Penalty for "use" without clear relevance
        # Even with relevance, "use" base weight (3) is lower than open/search (6), examine/inspect (5)
        # So it will naturally lose unless relevance is very strong
    
    # B.6) Failure-conditioned cooldown based on recent_obs[-1]
    if recent_obs and len(recent_obs) > 0:
        last_obs_lower = recent_obs[-1].lower()
        # Check if last observation contains failure keywords
        has_failure_keyword = any(keyword in last_obs_lower for keyword in FAILURE_KEYWORDS)
        if has_failure_keyword:
            # Apply very strong penalty to the action that caused this failure
            if recent_actions and len(recent_actions) > 0 and action == recent_actions[-1]:
                score -= 50  # Very strong penalty for action that caused failure
                # Store in impossible_actions set (handled by caller)
    
    # Check impossible_actions set (if provided)
    if impossible_actions is not None and action in impossible_actions:
        score -= 50  # Very strong penalty for actions that previously failed
    
    # Anti-loop penalties
    if recent_actions:
        recent_window_actions = recent_actions[-recent_window:] if len(recent_actions) > recent_window else recent_actions
        
        # -10 if exact action == last action
        if action == recent_actions[-1]:
            score -= 10
        
        # -4 if exact action appears in last 5 actions excluding last
        elif action in recent_window_actions[:-1]:
            score -= 4
        
        # -2 if same (verb, main_object) appears in last 5 actions
        for past_action in recent_window_actions:
            past_verb, _, _, past_main = parse_action(past_action)
            if past_verb == verb and past_main == main_object:
                score -= 2
                break
    
    # Wait gating
    if verb == "wait":
        allow_wait = False
        
        # Check if last action was wait AND (look changed OR last obs changed)
        if recent_actions and recent_actions[-1].lower().startswith("wait"):
            if recent_obs and len(recent_obs) >= 2:
                # Simple check: if last two obs are different, something changed
                if recent_obs[-1] != recent_obs[-2]:
                    allow_wait = True
        
        # Check for progress keywords in recent observations or look
        if not allow_wait:
            obs_text = ""
            if recent_obs:
                obs_text = " ".join(recent_obs[-3:]).lower()  # Last 3 observations
            if look:
                obs_text += " " + look.lower()
            
            # Check for progress keywords
            for keyword in PROGRESS_KEYWORDS:
                if keyword in obs_text:
                    allow_wait = True
                    break
        
        if not allow_wait:
            score -= 20  # Strong penalty
    
    return score


def propose_action(
    valid_actions: List[str],
    task_description: str,
    look: str,
    inventory: str,
    recent_actions: List[str],
    recent_obs: Optional[List[str]] = None,
    max_candidates: int = 200,
    recent_window: int = 5
) -> Optional[str]:
    """
    Propose a valid guiding action using SRM v2 policy.
    
    This is the public API for SRM v2. It:
    1. Extracts tokens from task, look, and inventory
    2. Filters candidates by verb + overlap gate (scans ALL valid_actions, no early capping)
    3. Scores candidates using SRM v2 scoring cascade
    4. Selects best action (first max score wins)
    
    Args:
        valid_actions: List of all valid action strings
        task_description: Task description string
        look: Current look/observation string
        inventory: Current inventory string
        recent_actions: List of recent actions (most recent last)
        recent_obs: Optional list of recent observations (for wait gating and failure cooldown)
        max_candidates: Maximum number of candidates to consider (for compatibility, but full scan is used)
        recent_window: Number of recent actions to check for scoring (default: 5)
    
    Returns:
        A valid action string (guaranteed to be in valid_actions), or None if no action can be proposed.
    """
    if not valid_actions:
        return None
    
    # Extract tokens
    task_tokens = extract_tokens(task_description)
    look_tokens = extract_tokens(look)
    inv_tokens = extract_tokens(inventory)
    context_tokens = look_tokens | inv_tokens
    
    # B.6) Build impossible_actions set from recent_obs failure keywords
    impossible_actions = set()
    if recent_obs and len(recent_obs) > 0 and len(recent_actions) > 0:
        last_obs_lower = recent_obs[-1].lower()
        has_failure_keyword = any(keyword in last_obs_lower for keyword in FAILURE_KEYWORDS)
        if has_failure_keyword and recent_actions:
            # Add the last action that caused the failure to impossible set
            impossible_actions.add(recent_actions[-1])
    
    # Filter candidates (scans ALL valid_actions, no early capping)
    candidates = filter_candidates(valid_actions, task_tokens, context_tokens, max_candidates)
    
    if not candidates:
        return None
    
    # Prevent SRM from returning "wait" unless justified
    # If there are non-wait candidates, remove "wait" from candidates
    # This avoids SRM becoming a "wait generator" when it's unsure
    non_wait_candidates = [c for c in candidates if not c.strip().lower().startswith("wait")]
    if non_wait_candidates:
        # Remove wait from candidates if there are alternatives
        candidates = non_wait_candidates
    
    # B.3) Score ALL candidates (no early capping - full scan)
    # This ensures we don't miss good actions due to iteration order
    max_score = None
    best_action = None
    best_verb = None
    best_main_object = None
    
    for action in candidates:
        verb, arg_text, arg_tokens, main_object = parse_action(action)
        if verb is None:
            continue
        
        score = score_action_v2(
            action=action,
            verb=verb,
            arg_tokens=arg_tokens,
            main_object=main_object,
            task_description=task_description,
            task_tokens=task_tokens,
            look_tokens=look_tokens,
            inv_tokens=inv_tokens,
            recent_actions=recent_actions,
            look=look,
            recent_obs=recent_obs,
            recent_window=recent_window,
            impossible_actions=impossible_actions
        )
        
        if max_score is None or score > max_score:
            max_score = score
            best_action = action
            best_verb = verb
            best_main_object = main_object
    
    # B.8) Minimal debug info (verb + main_object for logging)
    # This is returned implicitly via best_action, but caller can parse if needed
    # For now, we just return the action (caller already logs it)
    
    return best_action
