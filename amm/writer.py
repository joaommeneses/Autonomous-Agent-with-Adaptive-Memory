"""
AMM Writer - Memory Writing Functions

Implements the three core memory writing functions:
- write_success: For successful actions (reward > 0)
- write_nearmiss: For credible progress without reward
- write_avoidance: For invalid/blocked actions
"""

import logging
import time
from typing import Dict, Any, Optional
from .schema import MemoryRecord
from .client_letta import AMMLettaClient
from .tagging import classify_episode, TaggingError
from .config import DEFAULT_CONFIG
from .formatters import format_em_structured, _parse_inventory_text

logger = logging.getLogger(__name__)


def _now_ts() -> int:
    """Get current timestamp"""
    return int(time.time())


def write_success(client: AMMLettaClient, rec: MemoryRecord, tag: str = None, meta: dict = None) -> str:
    """
    Write a SUCCESS memory record.
    
    Args:
        client: AMM Letta client
        rec: Memory record (type will be set to episodic_success)
        tag: Optional tag (legacy, now ignored - classification happens automatically)
        meta: Optional additional metadata to merge into the record's meta
        
    Returns:
        Memory ID
    """
    # Check if EM writing is enabled
    if not DEFAULT_CONFIG.enable_em_write:
        logger.debug("[AMM Writer] EM writing is disabled (enable_em_write=False), skipping write_success")
        return ""
    
    rec.meta.setdefault("created_ts", _now_ts())
    rec.meta["last_seen_ts"] = rec.meta["created_ts"]
    
    # Merge additional meta if provided
    if meta:
        rec.meta.update(meta)
    
    # Classify episode using tagging system
    try:
        result = classify_episode(
            action=rec.action_text,
            observation=rec.obs_text,
            reward=rec.meta.get("reward", 0.0),
            score_prev=rec.meta.get("score_prev"),
            score_curr=rec.meta.get("score_curr"),
            done=rec.meta.get("done"),
            goal_text=rec.goal_signature,
            milestone_threshold=DEFAULT_CONFIG.MILESTONE_THRESHOLD,
            small_reward_threshold=DEFAULT_CONFIG.SMALL_REWARD_THRESHOLD,
            shaping_actions=DEFAULT_CONFIG.SHAPING_ACTIONS
        )
    except Exception as e:
        logger.warning(f"[AMM Writer] Classification failed: {e}")
        result = None
    
    # Skip writing if non-eventful or unclassifiable
    if result is None:
        logger.debug(
            f"[AMM Writer] EM skipped: non-eventful step. "
            f"action={rec.action_text[:50]}, reward={rec.meta.get('reward', 0.0)}, done={rec.meta.get('done')}"
        )
        return ""  # Return empty string to indicate no memory was written
    
    primary, subtag = result
    
    # Set type based on classification result
    rec.type = primary
    
    # Extract values for structured formatting
    reward_val = rec.meta.get('reward', 0.0) or 0.0
    score_prev = rec.meta.get('score_prev')
    score_curr = rec.meta.get('score_curr')
    room = rec.meta.get('room', '')
    inventory_text = rec.meta.get('inventory_text', '')
    look = rec.meta.get('look', '')
    recent_actions = rec.meta.get('recent_actions', [])
    recent_obs = rec.meta.get('recent_obs', [])
    
    # Parse inventory text into items
    inventory_items = _parse_inventory_text(inventory_text)
    
    # Log with detailed info: action, primary, subtag, computed_reward, s_prev→s_curr
    score_str = f"{score_prev} → {score_curr}" if (score_prev is not None and score_curr is not None) else "N/A"
    logger.info(
        f"[AMM Writer] Writing SUCCESS memory: "
        f"action='{rec.action_text[:50]}', primary={primary}, subtag={subtag}, "
        f"reward={reward_val}, score={score_str}"
    )
    
    # Format the memory content using structured formatter (tags are embedded)
    content = format_em_structured(
        goal_text=rec.goal_signature,
        room=room,
        inventory_items=inventory_items,
        action=rec.action_text,
        observation=rec.obs_text,
        recent_actions=recent_actions,
        recent_obs=recent_obs,
        reward=reward_val,
        score_prev=score_prev or 0.0,
        score_curr=score_curr or 0.0,
        primary_tag=primary,
        subtag=subtag,
        look=look,
    )
    
    # Build payload with only content (tags are already embedded in structured format)
    payload = {"content": content}
    
    # Log structured content preview (first 1-2 lines)
    content_lines = content.split('\n')
    preview = ' | '.join([line for line in content_lines[:2] if line.strip()])
    logger.info(f"[AMM Writer] Formatted content length: {len(content)} chars, preview: {preview}")
    
    # TODO: Add de-dup hook (amm/dedup.py) in future phases
    return client.add_tagged(payload)


def write_nearmiss(client: AMMLettaClient, rec: MemoryRecord, tag: str = None, meta: dict = None) -> str:
    """
    Write a NEARMISS memory record.
    
    Args:
        client: AMM Letta client
        rec: Memory record (type will be set to episodic_nearmiss)
        tag: Optional tag (legacy, now ignored - classification happens automatically)
        meta: Optional additional metadata to merge into the record's meta
        
    Returns:
        Memory ID
    """
    # Check if EM writing is enabled
    if not DEFAULT_CONFIG.enable_em_write:
        logger.debug("[AMM Writer] EM writing is disabled (enable_em_write=False), skipping write_nearmiss")
        return ""
    
    rec.meta.setdefault("created_ts", _now_ts())
    rec.meta["last_seen_ts"] = rec.meta["created_ts"]
    
    # Merge additional meta if provided
    if meta:
        rec.meta.update(meta)
    
    # Classify episode using tagging system
    try:
        result = classify_episode(
            action=rec.action_text,
            observation=rec.obs_text,
            reward=rec.meta.get("reward", 0.0),
            score_prev=rec.meta.get("score_prev"),
            score_curr=rec.meta.get("score_curr"),
            done=rec.meta.get("done"),
            goal_text=rec.goal_signature,
            milestone_threshold=DEFAULT_CONFIG.MILESTONE_THRESHOLD,
            small_reward_threshold=DEFAULT_CONFIG.SMALL_REWARD_THRESHOLD,
            shaping_actions=DEFAULT_CONFIG.SHAPING_ACTIONS
        )
    except Exception as e:
        logger.warning(f"[AMM Writer] Classification failed: {e}")
        result = None
    
    # Skip writing if non-eventful or unclassifiable
    if result is None:
        logger.debug(
            f"[AMM Writer] EM skipped: non-eventful step. "
            f"action={rec.action_text[:50]}, reward={rec.meta.get('reward', 0.0)}, done={rec.meta.get('done')}"
        )
        return ""  # Return empty string to indicate no memory was written
    
    primary, subtag = result
    
    # Set type based on classification result
    rec.type = primary
    
    # Extract values for structured formatting
    reward_val = rec.meta.get('reward', 0.0) or 0.0
    score_prev = rec.meta.get('score_prev')
    score_curr = rec.meta.get('score_curr')
    room = rec.meta.get('room', '')
    inventory_text = rec.meta.get('inventory_text', '')
    look = rec.meta.get('look', '')
    recent_actions = rec.meta.get('recent_actions', [])
    recent_obs = rec.meta.get('recent_obs', [])
    
    # Parse inventory text into items
    inventory_items = _parse_inventory_text(inventory_text)
    
    # Log with detailed info: action, primary, subtag, computed_reward, s_prev→s_curr
    score_str = f"{score_prev} → {score_curr}" if (score_prev is not None and score_curr is not None) else "N/A"
    logger.info(
        f"[AMM Writer] Writing NEARMISS memory: "
        f"action='{rec.action_text[:50]}', primary={primary}, subtag={subtag}, "
        f"reward={reward_val}, score={score_str}"
    )
    
    # Format the memory content using structured formatter (tags are embedded)
    content = format_em_structured(
        goal_text=rec.goal_signature,
        room=room,
        inventory_items=inventory_items,
        action=rec.action_text,
        observation=rec.obs_text,
        recent_actions=recent_actions,
        recent_obs=recent_obs,
        reward=reward_val,
        score_prev=score_prev or 0.0,
        score_curr=score_curr or 0.0,
        primary_tag=primary,
        subtag=subtag,
        look=look,
    )
    
    # Build payload with only content (tags are already embedded in structured format)
    payload = {"content": content}
    
    # Log structured content preview (first 1-2 lines)
    content_lines = content.split('\n')
    preview = ' | '.join([line for line in content_lines[:2] if line.strip()])
    logger.info(f"[AMM Writer] Formatted content length: {len(content)} chars, preview: {preview}")
    
    return client.add_tagged(payload)


def write_avoidance(client: AMMLettaClient, rec: MemoryRecord, tag: str = None, meta: dict = None) -> str:
    """
    Write an AVOIDANCE memory record.
    
    Args:
        client: AMM Letta client
        rec: Memory record (type will be set to avoidance)
        tag: Optional tag (legacy, now ignored - classification happens automatically)
        meta: Optional additional metadata to merge into the record's meta
        
    Returns:
        Memory ID
    """
    # Check if EM writing is enabled
    if not DEFAULT_CONFIG.enable_em_write:
        logger.debug("[AMM Writer] EM writing is disabled (enable_em_write=False), skipping write_avoidance")
        return ""
    
    rec.meta.setdefault("created_ts", _now_ts())
    rec.meta["last_seen_ts"] = rec.meta["created_ts"]
    
    # Add TTL for avoidance memories (50 episodes as per spec)
    rec.meta["ttl_steps"] = 50
    
    # Merge additional meta if provided
    if meta:
        rec.meta.update(meta)
    
    # Classify episode using tagging system
    try:
        result = classify_episode(
            action=rec.action_text,
            observation=rec.obs_text,
            reward=rec.meta.get("reward", 0.0),
            score_prev=rec.meta.get("score_prev"),
            score_curr=rec.meta.get("score_curr"),
            done=rec.meta.get("done"),
            goal_text=rec.goal_signature,
            milestone_threshold=DEFAULT_CONFIG.MILESTONE_THRESHOLD,
            small_reward_threshold=DEFAULT_CONFIG.SMALL_REWARD_THRESHOLD,
            shaping_actions=DEFAULT_CONFIG.SHAPING_ACTIONS
        )
    except Exception as e:
        logger.warning(f"[AMM Writer] Classification failed: {e}")
        result = None
    
    # Skip writing if non-eventful or unclassifiable
    if result is None:
        logger.debug(
            f"[AMM Writer] EM skipped: non-eventful step. "
            f"action={rec.action_text[:50]}, reward={rec.meta.get('reward', 0.0)}, done={rec.meta.get('done')}"
        )
        return ""  # Return empty string to indicate no memory was written
    
    primary, subtag = result
    
    # Set type based on classification result
    rec.type = primary
    
    # Extract values for structured formatting
    reward_val = rec.meta.get('reward', 0.0) or 0.0
    score_prev = rec.meta.get('score_prev')
    score_curr = rec.meta.get('score_curr')
    room = rec.meta.get('room', '')
    inventory_text = rec.meta.get('inventory_text', '')
    look = rec.meta.get('look', '')
    recent_actions = rec.meta.get('recent_actions', [])
    recent_obs = rec.meta.get('recent_obs', [])
    
    # Parse inventory text into items
    inventory_items = _parse_inventory_text(inventory_text)
    
    # Log with detailed info: action, primary, subtag, computed_reward, s_prev→s_curr
    score_str = f"{score_prev} → {score_curr}" if (score_prev is not None and score_curr is not None) else "N/A"
    logger.info(
        f"[AMM Writer] Writing AVOIDANCE memory: "
        f"action='{rec.action_text[:50]}', primary={primary}, subtag={subtag}, "
        f"reward={reward_val}, score={score_str}"
    )
    
    # Format the memory content using structured formatter (tags are embedded)
    content = format_em_structured(
        goal_text=rec.goal_signature,
        room=room,
        inventory_items=inventory_items,
        action=rec.action_text,
        observation=rec.obs_text,
        recent_actions=recent_actions,
        recent_obs=recent_obs,
        reward=reward_val,
        score_prev=score_prev or 0.0,
        score_curr=score_curr or 0.0,
        primary_tag=primary,
        subtag=subtag,
        look=look,
    )
    
    # Build payload with only content (tags are already embedded in structured format)
    payload = {"content": content}
    
    # Log structured content preview (first 1-2 lines)
    content_lines = content.split('\n')
    preview = ' | '.join([line for line in content_lines[:2] if line.strip()])
    logger.info(f"[AMM Writer] Formatted content length: {len(content)} chars, preview: {preview}")
    
    return client.add_tagged(payload)


def create_memory_record(
    goal_signature: str,
    action_text: str,
    obs_text: str,
    memory_type: str = "episodic_success",
    meta: dict = None
) -> MemoryRecord:
    """
    Create a basic memory record from action/observation data.
    
    This is a convenience function for creating minimal memory records
    in Phase 1. More sophisticated record creation will be added later.
    
    Args:
        goal_signature: Current goal/subgoal
        action_text: Action that was executed
        obs_text: Observation received
        memory_type: Type of memory (will be overridden by writer functions)
        meta: Optional additional metadata to merge (room, inventory, reward, etc.)
        
    Returns:
        MemoryRecord instance
    """
    # Start with base meta including timestamp
    base_meta = {"created_ts": _now_ts()}
    
    # Merge additional meta if provided
    if meta:
        try:
            base_meta.update(meta)
        except Exception as e:
            # Be tolerant of non-dict meta
            logger.warning(f"[AMM Writer] Could not merge meta: {e}")
            base_meta["meta_str"] = str(meta)
    
    return MemoryRecord(
        goal_signature=goal_signature,
        action_text=action_text,
        obs_text=obs_text,
        type=memory_type,
        state_fingerprint=[],  # TODO: Add canonical room/flags in future phases
        preconds_satisfied=[],  # TODO: Optional in Phase 1
        preconds_missing=[],    # TODO: Optional in Phase 1
        action_seq=[action_text],
        obs_seq=[obs_text],
        inventory_delta=[],    # TODO: Track inventory changes
        summary=f"Action '{action_text}' led to obs '{obs_text[:80]}'",
        success_weight=1,
        avoid_tags=[],
        meta=base_meta
    )
