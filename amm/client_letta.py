"""
AMM Letta Client

Refactored Letta client for the Adaptive Memory Module.
Centralizes agent creation and memory operations.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from letta_client import Letta, MessageCreate

logger = logging.getLogger(__name__)


@dataclass
class LettaConfig:
    """Configuration for Letta client"""
    api_token: str
    agent_id: str
    agent_name: str = "memory-agent"  # Kept for backwards compatibility, not used with cloud API


class AMMLettaClient:
    """
    Letta client wrapper for AMM operations.
    
    This refactors the existing Letta agent creation logic from eval_agent_fast_slow.py
    into a reusable component for the AMM package.
    """
    
    def __init__(self, cfg: LettaConfig):
        self.cfg = cfg
        self.client = Letta(token=cfg.api_token)
        self.agent_id = cfg.agent_id
        logger.info(f"AMM Letta client initialized with agent: {self.agent_id}")
    
    def _retry_with_backoff(self, func, max_retries: int = 5, initial_delay: float = 1.0):
        """
        Retry a function with exponential backoff.
        
        Args:
            func: Callable to execute (should raise exception on failure)
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds (will be doubled each retry)
            
        Returns:
            Result from func
            
        Raises:
            Last exception if all retries fail
        """
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Check if it's a retryable error (timeout, connection issues)
                is_retryable = any(keyword in error_msg for keyword in [
                    'timeout', 'timed out', 'connection', 'temporary', 'unavailable'
                ])
                
                if not is_retryable or attempt == max_retries - 1:
                    # Non-retryable error or last attempt - raise immediately
                    raise
                
                logger.warning(
                    f"[AMM Letta] Request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= 2  # Exponential backoff
        
        # Should never reach here, but just in case
        raise last_exception
    
    def add_memory(self, memory: Dict[str, Any]) -> str:
        """
        Insert a document/record into the Letta memory store.
        
        Args:
            memory: Dictionary containing memory data
            
        Returns:
            Memory ID (for now, returns a placeholder)
        """
        def _send_memory():
            # Format the memory as a message to the agent
            memory_text = self._format_memory_for_storage(memory)
            
            # Send to agent via streaming message
            stream = self.client.agents.messages.create_stream(
                agent_id=self.agent_id,
                messages=[
                    MessageCreate(
                        role="user",
                        content=memory_text
                    )
                ]
            )
            
            # Process the stream (for now, just log)
            for chunk in stream:
                logger.debug(f"[AMM Letta] Memory storage chunk: {chunk}")
            
            return "memory_placeholder_id"
        
        try:
            result = self._retry_with_backoff(_send_memory)
            logger.info(f"[AMM Letta] Memory stored successfully")
            return result
            
        except Exception as e:
            logger.error(f"[AMM Letta] Failed to store memory after retries: {e}")
            raise
    
    def add_tagged(self, record: Union[Dict[str, Any], str], *tags: str) -> str:
        """
        Insert memory using archival_memory_insert base tool.
        
        Args:
            record: Memory payload dict with "content" field (tags should already be embedded in content)
            *tags: Deprecated - tags should already be embedded in content. Ignored for backward compatibility.
            
        Returns:
            Memory ID if available, else ""
        """
        # Normalize payload to dict
        if isinstance(record, str):
            try:
                record = json.loads(record)
            except Exception:
                record = {"content": record}
        
        payload = dict(record)  # shallow copy is fine
        
        # Extract content (tags should already be embedded by tagging system)
        content = str(payload.get("content", "")).strip()
        
        if not content:
            raise ValueError("[AMM Letta] Content is required and cannot be empty")
        
        # Build the tool invocation message (content already includes tags)
        tool_cmd = (
            "Use the base tool `archival_memory_insert` with these args.\n"
            f"content:\n{content}\n"
        )
        
        logger.info(
            f"[AMM Letta] Invoking archival_memory_insert: "
            f"content_len={len(content)}"
        )
        # Log the final content that will be sent (showing tags if present)
        content_preview = content[:500] + "..." if len(content) > 500 else content
        logger.info(f"[AMM Letta] Content:\n{content_preview}")
        
        if not self.agent_id:
            raise RuntimeError("[AMM Letta] agent_id is not set; cannot send memory write.")
        
        def _send_tagged_memory():
            # Using async write (messages.create_async) to speed up EM insertion; we won't wait for the run to finish.
            # Send the tool invocation message asynchronously
            run = self.client.agents.messages.create_async(
                agent_id=self.agent_id,
                messages=[MessageCreate(role="user", content=tool_cmd)]
            )
            
            # Log the run ID for debugging (but don't wait for completion)
            run_id = getattr(run, 'id', None) or (run.get('id') if isinstance(run, dict) else None)
            if run_id:
                logger.debug(f"[AMM Letta] Async write initiated, run_id={run_id}")
            
            return ""
        
        try:
            result = self._retry_with_backoff(_send_tagged_memory)
            logger.info("[AMM Letta] Async write request submitted successfully")
            return result
        
        except Exception as e:
            logger.exception(f"[AMM Letta] archival_memory_insert failed after retries: {e}")
            raise
    
    def retrieve_memories(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on a query using non-streaming messages.create API.
        
        This uses the archival_memory_search tool via non-streaming API for synchronous retrieval.
        
        Args:
            query: Query string formatted for archival_memory_search tool
            top_k: Maximum number of memories to retrieve (default: 10)
            
        Returns:
            Response object from Letta API (contains messages with tool output)
        """
        logger.info(f"[AMM Letta] Retrieval requested (non-streaming): query_len={len(query)}")
        
        if not self.agent_id:
            raise RuntimeError("[AMM Letta] agent_id is not set; cannot retrieve memories.")
        
        def _retrieve():
            # Use non-streaming messages.create API with messages parameter
            response = self.client.agents.messages.create(
                agent_id=self.agent_id,
                messages=[
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
            )
            return response
        
        try:
            response = self._retry_with_backoff(_retrieve)
            logger.info("[AMM Letta] Retrieval completed successfully")
            return response
        except Exception as e:
            logger.exception(f"[AMM Letta] Retrieval failed after retries: {e}")
            raise
    
    def _format_memory_for_storage(self, memory: Dict[str, Any]) -> str:
        """
        Format memory data for storage in Letta.
        
        Args:
            memory: Memory record dictionary
            
        Returns:
            Formatted text for storage
        """
        memory_type = memory.get("type", "unknown")
        goal_signature = memory.get("goal_signature", "")
        summary = memory.get("summary", "")
        action_seq = memory.get("action_seq", [])
        obs_seq = memory.get("obs_seq", [])
        
        # Format based on memory type
        if memory_type == "episodic_success":
            return (
                f"SUCCESS MEMORY:\n"
                f"Goal: {goal_signature}\n"
                f"Actions: {'; '.join(action_seq)}\n"
                f"Observations: {'; '.join(obs_seq)}\n"
                f"Summary: {summary}\n"
                f"This resulted in a reward > 0, updating the score. Marked as SUCCESS."
            )
        elif memory_type == "episodic_nearmiss":
            return (
                f"NEARMISS MEMORY:\n"
                f"Goal: {goal_signature}\n"
                f"Actions: {'; '.join(action_seq)}\n"
                f"Observations: {'; '.join(obs_seq)}\n"
                f"Summary: {summary}\n"
                f"No reward was received but credible progress was made. Marked as NEARMISS."
            )
        elif memory_type == "avoidance":
            return (
                f"AVOIDANCE MEMORY:\n"
                f"Goal: {goal_signature}\n"
                f"Actions: {'; '.join(action_seq)}\n"
                f"Observations: {'; '.join(obs_seq)}\n"
                f"Summary: {summary}\n"
                f"Action was invalid or blocked. Marked as AVOIDANCE."
            )
        else:
            return (
                f"MEMORY:\n"
                f"Goal: {goal_signature}\n"
                f"Actions: {'; '.join(action_seq)}\n"
                f"Observations: {'; '.join(obs_seq)}\n"
                f"Summary: {summary}"
            )
