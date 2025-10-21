"""
AMM Letta Client

Refactored Letta client for the Adaptive Memory Module.
Centralizes agent creation and memory operations.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from letta_client import CreateBlock, Letta, MessageCreate

logger = logging.getLogger(__name__)


@dataclass
class LettaConfig:
    """Configuration for Letta client"""
    agent_name: str = "MemoryAgent"
    base_url: str = "https://0936-2001-8a0-57f3-d400-1951-5829-3cd4-ba4b.ngrok-free.app"
    model: str = "openai/letta-free"
    embedding: str = "hugging-face/letta-free"
    context_window_limit: int = 1000000
    include_base_tools: bool = True


class AMMLettaClient:
    """
    Letta client wrapper for AMM operations.
    
    This refactors the existing Letta agent creation logic from eval_agent_fast_slow.py
    into a reusable component for the AMM package.
    """
    
    def __init__(self, cfg: LettaConfig):
        self.cfg = cfg
        self.client = Letta(base_url=cfg.base_url)
        self.agent_id = self.get_or_create_agent(cfg.agent_name)
        logger.info(f"AMM Letta client initialized with agent: {self.agent_id}")
    
    def get_or_create_agent(self, agent_name: str) -> str:
        """
        Get existing agent or create a new one.
        
        This refactors the agent creation logic from eval_agent_fast_slow.py
        """
        try:
            # Try to get existing agent by hardcoded ID (matching existing code)
            mem_agent = self.client.agents.retrieve(agent_id="agent-0c985fd5-2b07-43c4-adba-0fb6ef6fe520")
            if mem_agent is not None:
                logger.info(f"Found existing Letta agent: {mem_agent}")
                return mem_agent.id
            else:
                raise Exception("Agent not found")
        except Exception as e:
            logger.info(f"Agent '{agent_name}' not found, creating it...")
            
            # System prompt for the memory agent
            system_prompt = (
                "You are a memory agent that stores and retrieves episodic memories "
                "for an AI agent working in ScienceWorld. You help the agent learn "
                "from past experiences by storing successful actions, near-misses, "
                "and avoidance patterns."
            )
            
            # Create new agent
            mem_agent = self.client.agents.create(
                name=agent_name,
                memory_blocks=[CreateBlock(
                    value="Memory: Episodic",
                    label="episodic_memory",
                )],
                system=system_prompt,
                agent_type="memgpt_agent",
                model=self.cfg.model,
                embedding=self.cfg.embedding,
                context_window_limit=self.cfg.context_window_limit,
                include_base_tools=self.cfg.include_base_tools
            )
            
            logger.info(f"Created new Letta agent: {mem_agent}")
            return mem_agent.id
    
    def add_memory(self, memory: Dict[str, Any]) -> str:
        """
        Insert a document/record into the Letta memory store.
        
        Args:
            memory: Dictionary containing memory data
            
        Returns:
            Memory ID (for now, returns a placeholder)
        """
        try:
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
            
            # Reset message context to avoid overflow
            self.client.agents.messages.reset(
                agent_id=self.agent_id, 
                add_default_initial_messages=True
            )
            
            logger.info(f"[AMM Letta] Memory stored successfully")
            return "memory_placeholder_id"  # TODO: Return actual memory ID when available
            
        except Exception as e:
            logger.error(f"[AMM Letta] Failed to store memory: {e}")
            raise
    
    def add_tagged(self, payload: Dict[str, Any], tag: str) -> str:
        """
        Convenience method to add memory with a specific tag.
        
        Args:
            payload: Memory data
            tag: Tag to add (episodic_success, episodic_nearmiss, avoidance)
            
        Returns:
            Memory ID
        """
        # Ensure tags field exists
        if "tags" not in payload:
            payload["tags"] = []
        
        # Add tag if not already present
        if tag not in payload["tags"]:
            payload["tags"].append(tag)
        
        return self.add_memory(payload)
    
    def retrieve_memories(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on a query.
        
        This will be implemented in Phase 2 for retrieval functionality.
        For now, returns empty list.
        """
        logger.info(f"[AMM Letta] Retrieval requested (not implemented in Phase 1): {query}")
        return []
    
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
