"""
Self-Reflection Module (SRM)

SRM proposes valid, cheap "guiding" actions from a curated subset of validActions
to reduce Swift dead ends and reduce Sage invocations.
"""

from .srm_module import propose_action

__all__ = ["propose_action"]
