"""
AMM Configuration

Centralized configuration for the Adaptive Memory Module.
All constants and thresholds are defined here for easy tuning.
"""

from dataclasses import dataclass
import os

@dataclass
class AMMConfig:
    """Configuration for the Adaptive Memory Module"""
    
    # ==================== Agent Configuration ====================
    agent_name: str = "memory-agent"
    agent_id: str = os.getenv("LETTA_AGENT_ID", "")
    api_token: str = os.getenv("LETTA_API_TOKEN", "")
    
    # ==================== Tagging Thresholds ====================
    # Thresholds for episodic memory classification (raw score deltas)
    MILESTONE_THRESHOLD: float = 20.0  # Reward >= this → episodic_success + milestone
    SMALL_REWARD_THRESHOLD: float = 5.0  # Reward <= this → episodic_nearmiss + partial (if not milestone)
    
    # ==================== Action Classification ====================
    # Actions that are considered shaping/delayed reward actions
    SHAPING_ACTIONS: set = None  # Defaults to {"wait", "wait1", "wait2"} in __post_init__
    
    # ==================== Future Phase Configuration ====================
    # Retrieval configuration (for Phase 2+)
    retrieval_top_k: int = 20
    budgeted_k_easy: int = 3
    budgeted_k_medium: int = 5
    budgeted_k_hard: int = 7
    
    # MMR diversity configuration
    mmr_lambda: float = 0.4
    
    # Scoring weights
    cosine_weight: float = 1.0
    goal_overlap_weight: float = 0.5
    success_prior_weight: float = 0.3
    recency_weight: float = 0.2
    
    # Deduplication
    nearmiss_cap_per_room: int = 1
    avoidance_ttl_episodes: int = 50
    
    # Skill promotion
    skill_promotion_threshold: int = 3
    
    def __post_init__(self):
        """Initialize default values for mutable types"""
        if self.SHAPING_ACTIONS is None:
            self.SHAPING_ACTIONS = {"wait", "wait1", "wait2"}


# Default configuration instance
DEFAULT_CONFIG = AMMConfig()
