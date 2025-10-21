"""
Working Memory - Transient per-episode scratchpad

This maintains the current episode state for memory retrieval and writing.
It's not persisted and resets between episodes.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WorkingMemory:
    """
    Transient working memory for the current episode.
    
    This tracks the current state and context to inform memory retrieval
    and writing decisions. It's reset at the start of each episode.
    """
    # Current goal and context
    pending_subgoal: str = ""
    room: str = ""
    
    # Current state
    inventory: List[str] = field(default_factory=list)
    last_obs_keyphrases: List[str] = field(default_factory=list)
    
    # Preconditions tracking
    preconds_satisfied: List[str] = field(default_factory=list)
    preconds_missing: List[str] = field(default_factory=list)
    
    # Action history
    last_actions: List[str] = field(default_factory=list)
    
    # Progress tracking
    cycles_without_progress: int = 0
    difficulty_score: float = 0.0
    
    # Timestamps
    last_progress_ts: int = 0
    
    # Retrieval tracking (for future phases)
    last_retrieved_ids: List[str] = field(default_factory=list)
    
    def reset(self):
        """Reset all fields to initial state"""
        self.__init__()
    
    def record_action(self, action: str):
        """Record a new action, keeping only the last 5"""
        self.last_actions.append(action)
        self.last_actions = self.last_actions[-5:]
    
    def update_room(self, room: str):
        """Update current room"""
        self.room = room
    
    def update_inventory(self, inventory: List[str]):
        """Update current inventory"""
        self.inventory = inventory.copy()
    
    def update_obs_keyphrases(self, keyphrases: List[str]):
        """Update observation keyphrases"""
        self.last_obs_keyphrases = keyphrases.copy()
    
    def update_preconds(self, satisfied: List[str], missing: List[str]):
        """Update precondition tracking"""
        self.preconds_satisfied = satisfied.copy()
        self.preconds_missing = missing.copy()
    
    def increment_cycles_without_progress(self):
        """Increment the counter for cycles without progress"""
        self.cycles_without_progress += 1
    
    def reset_cycles_without_progress(self):
        """Reset the cycles without progress counter"""
        self.cycles_without_progress = 0
    
    def set_difficulty_score(self, score: float):
        """Set the difficulty score (0.0 to 1.0)"""
        self.difficulty_score = max(0.0, min(1.0, score))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for debugging/logging"""
        return {
            "pending_subgoal": self.pending_subgoal,
            "room": self.room,
            "inventory": self.inventory,
            "last_obs_keyphrases": self.last_obs_keyphrases,
            "preconds_satisfied": self.preconds_satisfied,
            "preconds_missing": self.preconds_missing,
            "last_actions": self.last_actions,
            "cycles_without_progress": self.cycles_without_progress,
            "difficulty_score": self.difficulty_score,
            "last_progress_ts": self.last_progress_ts,
            "last_retrieved_ids": self.last_retrieved_ids
        }
