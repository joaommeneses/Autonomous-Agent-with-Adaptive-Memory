"""
AMM Instrumentation - Logging Helpers

Lightweight logging and instrumentation for the AMM.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def log_memory_write(memory_type: str, goal_signature: str, success: bool = True, error: Optional[str] = None):
    """
    Log memory write operations.
    
    Args:
        memory_type: Type of memory (episodic_success, episodic_nearmiss, avoidance)
        goal_signature: Goal/subgoal signature
        success: Whether the write was successful
        error: Error message if write failed
    """
    if success:
        logger.info(f"[AMM Instrumentation] Memory written - Type: {memory_type}, Goal: {goal_signature}")
    else:
        logger.error(f"[AMM Instrumentation] Memory write failed - Type: {memory_type}, Goal: {goal_signature}, Error: {error}")


def log_retrieval(query: str, num_results: int, success: bool = True, error: Optional[str] = None):
    """
    Log memory retrieval operations.
    
    Args:
        query: Retrieval query
        num_results: Number of results returned
        success: Whether retrieval was successful
        error: Error message if retrieval failed
    """
    if success:
        logger.info(f"[AMM Instrumentation] Memory retrieved - Query: {query[:50]}..., Results: {num_results}")
    else:
        logger.error(f"[AMM Instrumentation] Memory retrieval failed - Query: {query[:50]}..., Error: {error}")


def log_episode_stats(episode_id: str, stats: Dict[str, Any]):
    """
    Log episode-level statistics.
    
    Args:
        episode_id: Episode identifier
        stats: Dictionary of episode statistics
    """
    logger.info(f"[AMM Instrumentation] Episode {episode_id} stats: {stats}")


def create_episode_log_entry(episode_id: str, step: int, action: str, obs: str, reward: float, memory_writes: Dict[str, int]) -> Dict[str, Any]:
    """
    Create a structured log entry for an episode step.
    
    Args:
        episode_id: Episode identifier
        step: Step number
        action: Action taken
        obs: Observation received
        reward: Reward received
        memory_writes: Count of memory writes by type
        
    Returns:
        Structured log entry dictionary
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "episode_id": episode_id,
        "step": step,
        "action": action,
        "observation": obs[:100] + "..." if len(obs) > 100 else obs,  # Truncate long observations
        "reward": reward,
        "memory_writes": memory_writes
    }
