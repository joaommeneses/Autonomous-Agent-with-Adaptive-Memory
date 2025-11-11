"""
AMM Tagging System

Implements structured episodic memory tagging with primary tags and optional subtags.
All tags are embedded directly into the content field as natural language.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ==================== TAXONOMY ====================

PRIMARY_TAGS = {
    "episodic_success",
    "episodic_nearmiss",
    "avoidance"
}

SUBTAGS_BY_PRIMARY = {
    "episodic_success": {
        "milestone",
        "terminal_task_completion",
        "subgoal_focus"
    },
    "episodic_nearmiss": {
        "partial",
        "delayed_reward"
    },
    "avoidance": {
        "invalid_action",
        "terminal_incorrect"
    }
}


class TaggingError(ValueError):
    """Raised when tagging validation fails"""
    pass


def validate_tags(primary: str, subtag: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Validate that primary tag exists and subtag is valid for the primary.
    
    Args:
        primary: Primary tag (must be in PRIMARY_TAGS)
        subtag: Optional subtag (must be in SUBTAGS_BY_PRIMARY[primary] if provided)
        
    Returns:
        Tuple of (primary, subtag) if valid
        
    Raises:
        TaggingError: If validation fails
    """
    if primary not in PRIMARY_TAGS:
        raise TaggingError(f"Invalid primary tag: {primary}. Must be one of {PRIMARY_TAGS}")
    
    if subtag is not None:
        if primary not in SUBTAGS_BY_PRIMARY:
            raise TaggingError(f"Primary tag {primary} has no subtags defined")
        
        if subtag not in SUBTAGS_BY_PRIMARY[primary]:
            raise TaggingError(
                f"Invalid subtag '{subtag}' for primary '{primary}'. "
                f"Valid subtags: {SUBTAGS_BY_PRIMARY[primary]}"
            )
    
    return (primary, subtag)


def is_eventful(reward: float, done: Optional[bool], observation: str) -> bool:
    """
    Determine if a step is eventful (should trigger EM writing).
    
    Eventful when ANY of:
    - done == True, OR
    - reward > 0, OR
    - avoidance signal detected (invalid/blocked/etc.)
    
    Args:
        reward: Reward value
        done: Whether episode/task is completed
        observation: Observation string
        
    Returns:
        True if step is eventful and should trigger EM writing
    """
    # Check done flag
    if done is True:
        return True
    
    # Check reward
    if reward is not None and reward > 0:
        return True
    
    # Check for avoidance signals
    obs_norm = (observation or "").strip().lower()
    avoidance_phrases = [
        "no known action matches that input",
        "you can't do that",
        "that doesn't exist",
        "invalid action",
        "not possible",
        "unknown action"
    ]
    
    for phrase in avoidance_phrases:
        if phrase in obs_norm:
            return True
    
    return False


def append_tags_to_content(content: str, primary: str, subtag: Optional[str] = None) -> str:
    """
    Append tags to content string in the format: TAGS: <primary>[, <subtag>]
    
    Ensures exactly one trailing newline before TAGS: and a trailing newline after.
    Idempotent: removes any existing TAGS line before appending new one.
    
    Args:
        content: Original content string
        primary: Primary tag (required)
        subtag: Optional subtag
        
    Returns:
        Content string with tags appended
    """
    # Validate tags first
    validate_tags(primary, subtag)
    
    # Remove any existing TAGS line (idempotent)
    lines = content.split('\n')
    filtered_lines = []
    for line in lines:
        if not line.strip().startswith('TAGS:'):
            filtered_lines.append(line)
    content = '\n'.join(filtered_lines)
    
    # Normalize content: strip trailing whitespace, ensure single trailing newline
    content = content.rstrip()
    if not content.endswith('\n'):
        content += '\n'
    
    # Build tag string
    tag_str = primary
    if subtag:
        tag_str += f", {subtag}"
    
    # Append TAGS line with trailing newline
    content += f"TAGS: {tag_str}\n"
    
    return content


def classify_episode(
    action: str,
    observation: str,
    reward: float,
    score_prev: Optional[float],
    score_curr: Optional[float],
    done: Optional[bool],
    goal_text: Optional[str] = None,
    milestone_threshold: float = 20.0,
    small_reward_threshold: float = 5.0,
    shaping_actions: Optional[set] = None
) -> Optional[Tuple[str, Optional[str]]]:
    """
    Classify an episode based on action, observation, reward, and context.
    
    Classification rules (in precedence order):
    1. Gating: Check if eventful (done=True OR reward>0 OR avoidance signal); if not → return None (SKIP)
    2. Terminal: If done=True:
       - If score_curr == 100 → episodic_success + terminal_task_completion
       - Else → avoidance + terminal_incorrect
    3. Avoidance: If avoidance phrase matches → avoidance + invalid_action
    4. Success: If reward > 0:
       - If focus action → episodic_success + subgoal_focus
       - Else if reward >= milestone_threshold → episodic_success + milestone
    5. Nearmiss: If reward > 0 (and not success):
       - If wait action → episodic_nearmiss + delayed_reward
       - Else if reward < milestone_threshold → episodic_nearmiss + partial
    6. Else → return None (SKIP - no fallback writing)
    
    Args:
        action: Action string that was executed
        observation: Observation string received
        reward: Reward value (float) - TRUE reward from environment
        score_prev: Previous score (optional) - score before this step
        score_curr: Current score (optional) - TRUE score from environment
        done: Whether episode/task is completed (optional) - TRUE done flag from environment
        goal_text: Goal/subgoal text (optional, for context)
        milestone_threshold: Threshold for milestone rewards (default 20.0)
        small_reward_threshold: Threshold for small rewards (default 5.0, currently unused)
        shaping_actions: Set of shaping action strings (default: {"wait", "wait1", "wait2"})
        
    Returns:
        Tuple of (primary_tag, subtag_or_none) if eventful and classifiable, else None
    """
    if shaping_actions is None:
        shaping_actions = {"wait", "wait1", "wait2"}
    
    # GATING: Check if step is eventful
    if not is_eventful(reward, done, observation):
        return None  # SKIP - non-eventful step
    
    # Normalize for case-insensitive matching
    action_norm = (action or "").strip().lower()
    obs_norm = (observation or "").strip().lower()
    
    # ==================== TIE-BREAKING PRECEDENCE ====================
    
    # 1) If done=True: Check for terminal completion (correct vs incorrect)
    if done is True:
        # Check if score_curr == 100 for correct completion
        if score_curr is not None and score_curr == 100:
            return ("episodic_success", "terminal_task_completion")
        else:
            # Terminal but incorrect (score_curr != 100)
            return ("avoidance", "terminal_incorrect")
    
    # 2) Avoidance: Check for invalid action phrases
    invalid_action_phrases = [
        "no known action matches that input",
        "you can't do that",
        "that doesn't exist",
        "invalid action",
        "not possible",
        "unknown action"
    ]
    
    for phrase in invalid_action_phrases:
        if phrase in obs_norm:
            return ("avoidance", "invalid_action")
    
    # 3) Success classification
    # Primary: episodic_success if reward > 0 AND not clearly avoidance
    if reward is not None and reward > 0:
        # a) subgoal_focus
        if (action_norm.startswith("focus") or "you focus on" in obs_norm):
            return ("episodic_success", "subgoal_focus")
        
        # b) milestone (reward >= milestone_threshold)
        if reward >= milestone_threshold:
            return ("episodic_success", "milestone")
        
        # If reward > 0 but not milestone/subgoal_focus, continue to nearmiss rules below
    
    # 4) Nearmiss classification
    # (STRICT: both delayed_reward and partial require reward > 0)
    if reward is not None and reward > 0:
        # a) delayed_reward (wait actions with reward > 0)
        is_shaping = action_norm in shaping_actions or any(
            action_norm.startswith(shape_action) for shape_action in shaping_actions
        )
        wait_obs_patterns = ["decide to wait", "you wait", "waiting for", "wait for"]
        is_wait_obs = any(pattern in obs_norm for pattern in wait_obs_patterns)
        
        if is_shaping or is_wait_obs:
            return ("episodic_nearmiss", "delayed_reward")
        
        # b) partial (reward > 0 but < milestone_threshold, and not subgoal_focus)
        # This catches rewards that are positive but not large enough for milestone
        # and not from focus actions (which are already handled above)
        if reward < milestone_threshold:
            return ("episodic_nearmiss", "partial")
    
    # 5) No rule matched → SKIP (do not fallback-write)
    return None

