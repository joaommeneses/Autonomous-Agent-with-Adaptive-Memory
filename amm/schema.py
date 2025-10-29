"""
AMM Schema Definitions

Data structures for memory records and working memory.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal


# Memory type definitions
MemoryType = Literal["episodic_success", "episodic_nearmiss", "avoidance", "skill"]


def _json_safe(o: Any) -> Any:
    """
    Ensure a value is JSON-serializable.
    Converts numpy types, sets, etc. to plain Python types.
    """
    try:
        json.dumps(o)
        return o
    except (TypeError, ValueError):
        # Handle common non-serializable types
        if isinstance(o, set):
            return list(o)
        # Fall back to string representation
        return str(o)


@dataclass
class MemoryRecord:
    """
    Represents a single episodic memory record.
    
    This is the core data structure for storing memories in the AMM.
    All fields are optional to support incremental implementation.
    """
    goal_signature: str
    action_text: str = ""
    obs_text: str = ""
    type: str = "episodic"
    state_fingerprint: List[str] = field(default_factory=list)
    preconds_satisfied: List[str] = field(default_factory=list)
    preconds_missing: List[str] = field(default_factory=list)
    action_seq: List[str] = field(default_factory=list)
    obs_seq: List[str] = field(default_factory=list)
    inventory_delta: List[str] = field(default_factory=list)
    summary: str = ""
    success_weight: int = 1
    avoid_tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_payload(self) -> Dict[str, Any]:
        """
        Convert to payload dict for archival_memory_insert.
        
        Returns a dict with:
        - "content": formatted string for the episode step
        - "meta": metadata dict with all relevant fields
        - "tags": list of tags (can be empty, will be populated by client)
        """
        # Build content string in consistent format
        action_str = (self.action_text or "").strip()
        obs_str = (self.obs_text or "").strip()
        goal_str = (self.goal_signature or "").strip()
        type_str = (self.type or "episodic").strip()
        
        content = f"""[AMM {type_str}]
task: {goal_str}
action: {action_str}
obs: {obs_str}
"""
        
        # Build metadata dict with all relevant fields
        meta = {
            "goal_signature": goal_str,
            "action_text": action_str,
            "obs_text": obs_str,
        }
        
        # Add existing meta fields, ensuring JSON-serializable
        for key, value in self.meta.items():
            meta[key] = _json_safe(value)
        
        return {
            "content": content.strip(),
            "meta": meta,
            "tags": []  # Will be populated by client
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Alias for to_payload() for backward compatibility.
        Returns a dict, not a JSON string.
        """
        return self.to_payload()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        """Create from dictionary"""
        return cls(**data)


@dataclass 
class SkillRecord:
    """
    Represents a distilled skill promoted from repeated successes.
    
    This will be used in Phase 3 for skill matching and execution.
    """
    name: str
    goal_templates: List[str]
    preconds: List[str]
    steps: List[str]
    effects: List[str]
    failure_modes: List[str] = field(default_factory=list)
    success_count: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission"""
        return {
            "name": self.name,
            "goal_templates": self.goal_templates,
            "preconds": self.preconds,
            "steps": self.steps,
            "effects": self.effects,
            "failure_modes": self.failure_modes,
            "success_count": self.success_count,
            "meta": self.meta
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillRecord":
        """Create from dictionary"""
        return cls(**data)
