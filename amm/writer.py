"""
AMM Writer - Memory Writing Functions

Implements the three core memory writing functions:
- write_success: For successful actions (reward > 0)
- write_nearmiss: For credible progress without reward
- write_avoidance: For invalid/blocked actions
"""

import logging
import time
from typing import Dict, Any
from .schema import MemoryRecord
from .client_letta import AMMLettaClient

logger = logging.getLogger(__name__)


def _now_ts() -> int:
    """Get current timestamp"""
    return int(time.time())


def format_episodic_memory_entry(goal: str, action: str, observation: str, meta: dict) -> str:
    """
    Format an episodic memory entry in a human-readable format.
    
    Args:
        goal: Task or goal description
        action: Action that was executed
        observation: Observation received
        meta: Metadata dict with context (inventory, room, rewards, etc.)
        
    Returns:
        Formatted memory string
    """
    return (
        f"While working on the task: \"{goal}\",\n"
        f"the action '{action}' caused: '{observation}'.\n"
        f"Inventory: {meta.get('inventory_text', '')}\n"
        f"Location: {meta.get('room', '')}\n"
        f"Recent actions: {meta.get('recent_actions', [])}\n"
        f"Recent obs: {meta.get('recent_obs', [])}\n"
        f"Resulted in reward: {meta.get('reward')} (score {meta.get('score_prev')} → {meta.get('score_curr')})\n"
    )


def write_success(client: AMMLettaClient, rec: MemoryRecord, tag: str = None, meta: dict = None) -> str:
    """
    Write a SUCCESS memory record.
    
    Args:
        client: AMM Letta client
        rec: Memory record (type will be set to episodic_success)
        tag: Optional tag for the success type (e.g., "terminal", "product-made", "reward-validated")
        meta: Optional additional metadata to merge into the record's meta
        
    Returns:
        Memory ID
    """
    rec.type = "episodic_success"
    rec.meta.setdefault("created_ts", _now_ts())
    rec.meta["last_seen_ts"] = rec.meta["created_ts"]
    
    # Add tag to meta if provided
    if tag:
        rec.meta["success_tag"] = tag
    
    # Merge additional meta if provided
    if meta:
        rec.meta.update(meta)
    
    logger.info(f"[AMM Writer] Writing SUCCESS memory ({tag or 'default'}): {rec.goal_signature}")
    
    # Format the memory content using the readable formatter
    content = format_episodic_memory_entry(
        goal=rec.goal_signature,
        action=rec.action_text,
        observation=rec.obs_text,
        meta=rec.meta
    )
    
    # Build payload with formatted content
    payload = {
        "content": content,
        "meta": rec.meta,
        "tags": []
    }
    
    # Build tags list
    tags = [rec.type]
    if tag:
        tags.append(tag)
    
    logger.info(f"[AMM Writer] Formatted content length: {len(content)} chars, tags: {tags}")
    
    # TODO: Add de-dup hook (amm/dedup.py) in future phases
    return client.add_tagged(payload, *tags)


def write_nearmiss(client: AMMLettaClient, rec: MemoryRecord, tag: str = None, meta: dict = None) -> str:
    """
    Write a NEARMISS memory record.
    
    Args:
        client: AMM Letta client
        rec: Memory record (type will be set to episodic_nearmiss)
        tag: Optional tag for the nearmiss type (e.g., "progress")
        meta: Optional additional metadata to merge into the record's meta
        
    Returns:
        Memory ID
    """
    rec.type = "episodic_nearmiss"
    rec.meta.setdefault("created_ts", _now_ts())
    rec.meta["last_seen_ts"] = rec.meta["created_ts"]
    
    # Add tag to meta if provided
    if tag:
        rec.meta["nearmiss_tag"] = tag
    
    # Merge additional meta if provided
    if meta:
        rec.meta.update(meta)
    
    logger.info(f"[AMM Writer] Writing NEARMISS memory ({tag or 'default'}): {rec.goal_signature}")
    
    # Format the memory content using the readable formatter
    content = format_episodic_memory_entry(
        goal=rec.goal_signature,
        action=rec.action_text,
        observation=rec.obs_text,
        meta=rec.meta
    )
    
    # Build payload with formatted content
    payload = {
        "content": content,
        "meta": rec.meta,
        "tags": []
    }
    
    # Build tags list
    tags = [rec.type]
    if tag:
        tags.append(tag)
    
    logger.info(f"[AMM Writer] Formatted content length: {len(content)} chars, tags: {tags}")
    
    return client.add_tagged(payload, *tags)


def write_avoidance(client: AMMLettaClient, rec: MemoryRecord, tag: str = None, meta: dict = None) -> str:
    """
    Write an AVOIDANCE memory record.
    
    Args:
        client: AMM Letta client
        rec: Memory record (type will be set to avoidance)
        tag: Optional tag for the avoidance type (e.g., "shaping-decoy", "exec-invalid")
        meta: Optional additional metadata to merge into the record's meta
        
    Returns:
        Memory ID
    """
    rec.type = "avoidance"
    rec.meta.setdefault("created_ts", _now_ts())
    rec.meta["last_seen_ts"] = rec.meta["created_ts"]
    
    # Add TTL for avoidance memories (50 episodes as per spec)
    rec.meta["ttl_steps"] = 50
    
    # Add tag to meta if provided
    if tag:
        rec.meta["avoidance_tag"] = tag
    
    # Merge additional meta if provided
    if meta:
        rec.meta.update(meta)
    
    logger.info(f"[AMM Writer] Writing AVOIDANCE memory ({tag or 'default'}): {rec.goal_signature}")
    
    # Format the memory content using the readable formatter
    content = format_episodic_memory_entry(
        goal=rec.goal_signature,
        action=rec.action_text,
        observation=rec.obs_text,
        meta=rec.meta
    )
    
    # Build payload with formatted content
    payload = {
        "content": content,
        "meta": rec.meta,
        "tags": []
    }
    
    # Build tags list
    tags = [rec.type]
    if tag:
        tags.append(tag)
    
    logger.info(f"[AMM Writer] Formatted content length: {len(content)} chars, tags: {tags}")
    
    return client.add_tagged(payload, *tags)


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
