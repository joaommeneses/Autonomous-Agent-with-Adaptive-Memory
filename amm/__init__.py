"""
Adaptive Memory Module (AMM) for SwiftSage

This package provides inference-time memory capabilities for the SwiftSage dual-process agent.
It stores and retrieves episodic memories to assist the SWIFT fast policy before escalating to SAGE.

Phase 1: Foundation - SUCCESS/NEARMISS/AVOIDANCE memory writing
Phase 2: Retrieval with hybrid reranking and MMR diversity
Phase 3: Skills promotion and matching
"""

from .schema import MemoryRecord, MemoryType
from .working_memory import WorkingMemory
from .client_letta import AMMLettaClient, LettaConfig
from .writer import write_success, write_nearmiss, write_avoidance
from .config import AMMConfig, DEFAULT_CONFIG
from .tagging import classify_episode, append_tags_to_content, validate_tags, TaggingError, is_eventful
from .retrieval import build_success_retrieval_query_s1, retrieve_success_ems_s1

__version__ = "0.1.0"
__all__ = [
    "MemoryRecord",
    "MemoryType", 
    "WorkingMemory",
    "AMMLettaClient",
    "LettaConfig",
    "write_success",
    "write_nearmiss", 
    "write_avoidance",
    "AMMConfig",
    "DEFAULT_CONFIG",
    "classify_episode",
    "append_tags_to_content",
    "validate_tags",
    "TaggingError",
    "is_eventful",
    "build_success_retrieval_query_s1",
    "retrieve_success_ems_s1",
]
