"""
Working Memory - Transient per-episode scratchpad

This maintains the current episode state for memory retrieval and writing.
It's not persisted and resets between episodes.
"""

import copy as _copy
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional


def _safe_copy(x):
    """
    Safely copy any object, handling strings and other immutable types.
    
    Returns:
        Copy of the object if possible, otherwise the object itself
    """
    try:
        return x.copy()  # dict or list-like
    except Exception:
        # For immutables such as str/int/bool/None, return as-is.
        # For other objects, fallback to deepcopy.
        if isinstance(x, (str, int, float, bool, type(None))):
            return x
        try:
            return _copy.deepcopy(x)
        except Exception:
            return x


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
    
    # Current state - inventory handling
    inventory_text: str = ""  # Canonical text snapshot from env
    inventory_items: List[str] = field(default_factory=list)  # Normalized list
    inventory: Any = field(default_factory=list)  # Backward-compatible
    
    # Other state
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
    
    def update_inventory(self, inventory: Any):
        """
        Accepts the ScienceWorld inventory (often a STRING) and stores both:
          - inventory_text: str (canonical textual snapshot)
          - inventory_items: list[str] (best-effort normalization)
        Never calls .copy() on a string.
        """
        # Default reset
        self.inventory_text = ""
        self.inventory_items = []
        
        # Normalize based on type
        if isinstance(inventory, (list, tuple)):
            self.inventory_items = list(inventory)
            self.inventory_text = "; ".join(map(str, self.inventory_items))
        elif isinstance(inventory, dict):
            # Keep keys as items; preserve a readable text too
            self.inventory_items = list(inventory.keys())
            try:
                import json
                self.inventory_text = json.dumps(inventory)
            except Exception:
                self.inventory_text = str(inventory)
        else:
            # Most common case in ScienceWorld: a plain descriptive string
            self.inventory_text = str(inventory)
        
        # Keep backward-compatible attribute if used elsewhere
        self.inventory = _safe_copy(self.inventory_items) if self.inventory_items else self.inventory_text
        
        # Optional: type debug logging
        try:
            from amm.config import DEFAULT_CONFIG
            if getattr(DEFAULT_CONFIG, "debug_types", False):
                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    f"[WM] inventory_text(type={type(self.inventory_text).__name__}), "
                    f"inventory_items(type={type(self.inventory_items).__name__}, len={len(self.inventory_items)})"
                )
        except Exception:
            pass
    
    def update_obs_keyphrases(self, keyphrases):
        """Update observation keyphrases"""
        self.last_obs_keyphrases = _safe_copy(keyphrases)
    
    def update_preconds(self, satisfied, missing):
        """Update precondition tracking"""
        self.preconds_satisfied = _safe_copy(satisfied)
        self.preconds_missing = _safe_copy(missing)
    
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
            "inventory_text": self.inventory_text,
            "inventory_items": self.inventory_items,
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
