"""
AMM Configuration

Centralized configuration for the Adaptive Memory Module.
All constants and thresholds are defined here for easy tuning.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class AMMConfig:
    """Configuration for the Adaptive Memory Module"""
    
    # Agent configuration
    agent_name: str = "MemoryAgent"
    base_url: str = "https://0936-2001-8a0-57f3-d400-1951-5829-3cd4-ba4b.ngrok-free.app"
    
    # Memory writing thresholds
    success_reward_threshold: float = 0.0  # Write SUCCESS if reward > this
    nearmiss_keywords: List[str] = None  # Keywords that indicate credible progress
    avoidance_keywords: List[str] = None  # Keywords that indicate invalid/blocked actions
    
    # Retrieval configuration (for future phases)
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
        """Set default keyword lists if not provided"""
        if self.nearmiss_keywords is None:
            self.nearmiss_keywords = [
                "opened", "unlocked", "lit", "attached", "connected", 
                "placed", "mounted", "activated", "turned on"
            ]
        
        if self.avoidance_keywords is None:
            self.avoidance_keywords = [
                "can't", "cannot", "invalid", "doesn't", "won't work",
                "not possible", "blocked", "failed"
            ]


# Default configuration instance
DEFAULT_CONFIG = AMMConfig()
