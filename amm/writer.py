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


def write_success(client: AMMLettaClient, rec: MemoryRecord) -> str:
    """
    Write a SUCCESS memory record.
    
    Args:
        client: AMM Letta client
        rec: Memory record (type will be set to episodic_success)
        
    Returns:
        Memory ID
    """
    rec.type = "episodic_success"
    rec.meta.setdefault("created_ts", _now_ts())
    rec.meta["last_seen_ts"] = rec.meta["created_ts"]
    
    logger.info(f"[AMM Writer] Writing SUCCESS memory: {rec.goal_signature}")
    
    # TODO: Add de-dup hook (amm/dedup.py) in future phases
    return client.add_tagged(rec.to_dict(), "episodic_success")


def write_nearmiss(client: AMMLettaClient, rec: MemoryRecord) -> str:
    """
    Write a NEARMISS memory record.
    
    Args:
        client: AMM Letta client
        rec: Memory record (type will be set to episodic_nearmiss)
        
    Returns:
        Memory ID
    """
    rec.type = "episodic_nearmiss"
    rec.meta.setdefault("created_ts", _now_ts())
    rec.meta["last_seen_ts"] = rec.meta["created_ts"]
    
    logger.info(f"[AMM Writer] Writing NEARMISS memory: {rec.goal_signature}")
    
    return client.add_tagged(rec.to_dict(), "episodic_nearmiss")


def write_avoidance(client: AMMLettaClient, rec: MemoryRecord) -> str:
    """
    Write an AVOIDANCE memory record.
    
    Args:
        client: AMM Letta client
        rec: Memory record (type will be set to avoidance)
        
    Returns:
        Memory ID
    """
    rec.type = "avoidance"
    rec.meta.setdefault("created_ts", _now_ts())
    rec.meta["last_seen_ts"] = rec.meta["created_ts"]
    
    # Add TTL for avoidance memories (50 episodes as per spec)
    rec.meta["ttl_steps"] = 50
    
    logger.info(f"[AMM Writer] Writing AVOIDANCE memory: {rec.goal_signature}")
    
    return client.add_tagged(rec.to_dict(), "avoidance")


def create_memory_record(
    goal_signature: str,
    action_text: str,
    obs_text: str,
    memory_type: str = "episodic_success"
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
        
    Returns:
        MemoryRecord instance
    """
    return MemoryRecord(
        type=memory_type,
        goal_signature=goal_signature,
        state_fingerprint=[],  # TODO: Add canonical room/flags in future phases
        preconds_satisfied=[],  # TODO: Optional in Phase 1
        preconds_missing=[],    # TODO: Optional in Phase 1
        action_seq=[action_text],
        obs_seq=[obs_text],
        inventory_delta=[],    # TODO: Track inventory changes
        summary=f"Action '{action_text}' led to obs '{obs_text[:80]}'",
        success_weight=1,
        avoid_tags=[],
        meta={}
    )
