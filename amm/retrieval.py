"""
AMM Retrieval - Episodic Memory Retrieval

Provides retrieval functions for episodic memories using Letta's archival_memory_search tool.
Implements Template A (S1) retrieval for success-only episodic memories.
"""

import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def build_success_retrieval_query_s1(
    task_description: str,
    room_name: str,
    inventory_items: List[str],
    recent_rewards: List[float],
    current_score: float,
    look_description: Optional[str] = None,
    recent_actions: List[str] = None,
    recent_observations: List[str] = None,
) -> str:
    """
    Build a Template A (S1) retrieval query for success-only episodic memories.
    
    This query is used when Swift fails to find a valid action (T1 trigger).
    It retrieves memories that show successful strategies for similar tasks and states.
    
    Args:
        task_description: Natural language task description
        room_name: Current room/location name
        inventory_items: List of inventory item strings
        recent_rewards: Last up to 5 reward values
        current_score: Current score value
        recent_actions: Last 5 actions (defaults to empty list)
        recent_observations: Last 5 observations (aligned with recent_actions, defaults to empty list)
        
    Returns:
        Formatted query string for passages.search (plain semantic query, no tool invocation)
    """
    # Normalize inputs
    recent_actions = recent_actions or []
    recent_observations = recent_observations or []
    
    # Format inventory as comma-separated string
    inventory_str = ", ".join(inventory_items) if inventory_items else ""
    
    # Format recent rewards as Python list literal
    recent_reward_list = recent_rewards[-5:] if len(recent_rewards) > 5 else recent_rewards
    recent_reward_str = json.dumps(recent_reward_list)
    
    # Format recent actions and observations as Python list literals
    recent_actions_list = recent_actions[-5:] if len(recent_actions) > 5 else recent_actions
    recent_obs_list = recent_observations[-5:] if len(recent_observations) > 5 else recent_observations
    
    # Ensure lists are aligned (pad with "N/A" if needed)
    while len(recent_obs_list) < len(recent_actions_list):
        recent_obs_list.append("N/A")
    
    recent_actions_str = json.dumps(recent_actions_list)
    recent_obs_str = json.dumps(recent_obs_list)
    
    # Escape quotes in task_description
    task_desc_escaped = task_description.replace('"', '\\"')
    
    # Build the query following Template A, mode S1 format (without LOOK attribute)
    # This is now used as a plain semantic query string for passages.search
    query = f"""TASK: "{task_desc_escaped}"
STATE: room={room_name}; inventory=[{inventory_str}]; recent_reward={recent_reward_str}; current_score={current_score};
ACTION_CONTEXT: "The agent is planning its next steps and wants examples of clearly successful strategies for similar tasks and states so it can choose the next action correctly. Retrieve memories that show correct subgoal completion or full task completion, rather than failures."
RECENT_ACTIONS: {recent_actions_str}
RECENT_OBS: {recent_obs_str}
ISSUE: "swift_failure"
TAG_SCOPE: subgoal_focus, terminal_task_completion
TAGS_HINT: episodic_success
"""
    
    return query


def build_success_retrieval_query_s2(
    task_description: str,
    room_name: str,
    inventory_items: List[str],
    recent_rewards: List[float],
    current_score: float,
    look_description: Optional[str] = None,
    recent_actions: List[str] = None,
    recent_observations: List[str] = None,
) -> str:
    """
    Build a Template A (S2) retrieval query for success + partial/near-miss episodic memories.
    
    This query is used when Swift fails to find a valid action (T1 trigger) and swift_failure_count == 1.
    It retrieves memories that show successful strategies OR partial progress (near-miss) for similar tasks.
    
    Args:
        task_description: Natural language task description
        room_name: Current room/location name
        inventory_items: List of inventory item strings
        recent_rewards: Last up to 5 reward values
        current_score: Current score value
        recent_actions: Last 5 actions (defaults to empty list)
        recent_observations: Last 5 observations (aligned with recent_actions, defaults to empty list)
        
    Returns:
        Formatted query string for passages.search (plain semantic query, no tool invocation)
    """
    # Normalize inputs
    recent_actions = recent_actions or []
    recent_observations = recent_observations or []
    
    # Format inventory as comma-separated string
    inventory_str = ", ".join(inventory_items) if inventory_items else ""
    
    # Format recent rewards as Python list literal
    recent_reward_list = recent_rewards[-5:] if len(recent_rewards) > 5 else recent_rewards
    recent_reward_str = json.dumps(recent_reward_list)
    
    # Format recent actions and observations as Python list literals
    recent_actions_list = recent_actions[-5:] if len(recent_actions) > 5 else recent_actions
    recent_obs_list = recent_observations[-5:] if len(recent_observations) > 5 else recent_observations
    
    # Ensure lists are aligned (pad with "N/A" if needed)
    while len(recent_obs_list) < len(recent_actions_list):
        recent_obs_list.append("N/A")
    
    recent_actions_str = json.dumps(recent_actions_list)
    recent_obs_str = json.dumps(recent_obs_list)
    
    # Escape quotes in task_description
    task_desc_escaped = task_description.replace('"', '\\"')
    
    # Build the query following Template A, mode S2 format (success + partial/near-miss)
    # This is now used as a plain semantic query string for passages.search
    query = f"""TASK: "{task_desc_escaped}"
STATE: room={room_name}; inventory=[{inventory_str}]; recent_reward={recent_reward_str}; current_score={current_score};
ACTION_CONTEXT: "The agent is planning its next steps and wants examples of successful strategies OR partial progress (near-miss) for similar tasks and states. Retrieve memories that show correct subgoal completion, full task completion, or credible progress even without immediate reward."
RECENT_ACTIONS: {recent_actions_str}
RECENT_OBS: {recent_obs_str}
ISSUE: "swift_failure"
TAG_SCOPE: subgoal_focus, terminal_task_completion, partial
TAGS_HINT: episodic_success, episodic_nearmiss
"""
    
    return query


def retrieve_success_ems_s1(
    memory_agent_id: str,
    query_text: str,
    letta_client: Any,  # AMMLettaClient type
) -> List[Dict[str, Any]]:
    """
    Send S1 retrieval query to Letta memory agent using passages.search API,
    and return a list of episodic memory passages.
    
    Args:
        memory_agent_id: Letta agent ID for memory operations (unused, kept for API compatibility)
        query_text: Query string built by build_success_retrieval_query_s1()
        letta_client: AMMLettaClient instance
        
    Returns:
        List of episodic memory passage dictionaries (each has content, tags, timestamp, relevance)
        
    Raises:
        Exception: If retrieval fails
    """
    logger.info("[AMM Retrieval] Starting S1 retrieval via passages.search")
    logger.info(f"[AMM Retrieval] Query:\n{query_text}")
    
    try:
        # Call Letta using passages.search API via client method
        passages = letta_client.retrieve_memories(query_text, top_k=10)
        
        return passages
        
    except Exception as e:
        logger.error(f"[AMM Retrieval] Retrieval failed: {e}")
        raise


def retrieve_success_ems_s2(
    memory_agent_id: str,
    query_text: str,
    letta_client: Any,  # AMMLettaClient type
) -> List[Dict[str, Any]]:
    """
    Send S2 retrieval query to Letta memory agent using passages.search API,
    and return a list of episodic memory passages (success + partial/near-miss).
    
    Args:
        memory_agent_id: Letta agent ID for memory operations (unused, kept for API compatibility)
        query_text: Query string built by build_success_retrieval_query_s2()
        letta_client: AMMLettaClient instance
        
    Returns:
        List of episodic memory passage dictionaries (each has content, tags, timestamp, relevance)
        
    Raises:
        Exception: If retrieval fails
    """
    logger.info("[AMM Retrieval] Starting S2 retrieval via passages.search (success + near-miss)")
    logger.info(f"[AMM Retrieval] Query:\n{query_text}")
    
    try:
        # Call Letta using passages.search API via client method
        passages = letta_client.retrieve_memories(query_text, top_k=10)
        
        logger.info(f"[AMM Retrieval] Retrieved {len(passages)} episodic memory passages (S2: success + near-miss)")
        
        # Log retrieved passages in a clean format
        for i, passage in enumerate(passages):
            passage_preview = json.dumps(passage, indent=2)[:300] + "..." if len(json.dumps(passage)) > 300 else json.dumps(passage, indent=2)
            logger.info(f"[AMM Retrieval] Passage {i+1}:\n{passage_preview}")
        
        return passages
        
    except Exception as e:
        logger.error(f"[AMM Retrieval] S2 retrieval failed: {e}")
        raise


def _parse_archival_memory_search_output(response: Any) -> List[Dict[str, Any]]:
    """
    Parse the archival_memory_search tool output from Letta response.
    
    The response contains messages, and one of them should have a JSON array of episodic memories
    in its content field.
    
    Args:
        response: Response object from letta_client.agents.messages.create()
        
    Returns:
        List of episodic memory dictionaries
    """
    episodic_memories = []
    
    if not hasattr(response, 'messages'):
        logger.warning("[AMM Retrieval] Response has no 'messages' attribute")
        return episodic_memories
    
    # Search through all messages (start from latest)
    for i, msg in enumerate(reversed(response.messages)):
        # Try to get content - could be attribute or dict key
        content = None
        if hasattr(msg, 'content'):
            content = msg.content
        elif isinstance(msg, dict) and 'content' in msg:
            content = msg['content']
        else:
            logger.debug(f"[AMM Retrieval] Message {i} has no content field")
            continue
        
        logger.debug(f"[AMM Retrieval] Checking message {i}, content type: {type(content)}")
        
        # Case 1: Content is already a list (the JSON array)
        if isinstance(content, list):
            episodic_memories.extend(content)
            logger.debug(f"[AMM Retrieval] Found list in message content: {len(content)} items")
            break  # Found it, no need to continue
        
        # Case 2: Content is a string that can be parsed as JSON
        elif isinstance(content, str):
            try:
                # Try parsing as JSON
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    episodic_memories.extend(parsed)
                    logger.debug(f"[AMM Retrieval] Parsed JSON array from string: {len(parsed)} items")
                    break
                elif isinstance(parsed, dict):
                    # Check if dict has a 'content' field with the array
                    if 'content' in parsed and isinstance(parsed['content'], list):
                        episodic_memories.extend(parsed['content'])
                        logger.debug(f"[AMM Retrieval] Found array in dict.content: {len(parsed['content'])} items")
                        break
            except json.JSONDecodeError:
                # Not valid JSON, try to extract JSON array from the string
                try:
                    import re
                    # Look for JSON array pattern (more lenient)
                    json_match = re.search(r'\[.*\]', content, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group(0))
                        if isinstance(parsed, list):
                            episodic_memories.extend(parsed)
                            logger.debug(f"[AMM Retrieval] Extracted JSON array from string: {len(parsed)} items")
                            break
                except (json.JSONDecodeError, AttributeError):
                    continue
        
        # Case 3: Content is a dict - check for nested content/result fields
        elif isinstance(content, dict):
            # Check common field names that might contain the array
            for field in ['content', 'result', 'data', 'memories']:
                if field in content and isinstance(content[field], list):
                    episodic_memories.extend(content[field])
                    logger.debug(f"[AMM Retrieval] Found array in dict.{field}: {len(content[field])} items")
                    break
            if episodic_memories:
                break
    
    logger.info(f"[AMM Retrieval] Parsed {len(episodic_memories)} episodic memories from tool output")
    
    return episodic_memories

