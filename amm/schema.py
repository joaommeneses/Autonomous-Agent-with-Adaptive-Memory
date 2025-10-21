"""
AMM Schema Definitions

Data structures for memory records and working memory.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal


# Memory type definitions
MemoryType = Literal["episodic_success", "episodic_nearmiss", "avoidance", "skill"]


@dataclass
class MemoryRecord:
    """
    Represents a single episodic memory record.
    
    This is the core data structure for storing memories in the AMM.
    All fields are optional to support incremental implementation.
    """
    type: MemoryType
    goal_signature: str
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission"""
        return {
            "type": self.type,
            "goal_signature": self.goal_signature,
            "state_fingerprint": self.state_fingerprint,
            "preconds_satisfied": self.preconds_satisfied,
            "preconds_missing": self.preconds_missing,
            "action_seq": self.action_seq,
            "obs_seq": self.obs_seq,
            "inventory_delta": self.inventory_delta,
            "summary": self.summary,
            "success_weight": self.success_weight,
            "avoid_tags": self.avoid_tags,
            "meta": self.meta
        }
    
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
