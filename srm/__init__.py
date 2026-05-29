"""Self-Reflection Module (SRM) for pre-execution validation and stagnation-triggered critic."""

from .srm_gate import SRMGate
from .stagnation import SRMStagnationDetector, StagnationReport
from .critic import build_critic_prompt, run_critic_once, parse_critic_actions

__all__ = [
    "SRMGate",
    "SRMStagnationDetector",
    "StagnationReport",
    "build_critic_prompt",
    "run_critic_once",
    "parse_critic_actions",
]
