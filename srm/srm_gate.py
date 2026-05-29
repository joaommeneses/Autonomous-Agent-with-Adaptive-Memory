from dataclasses import dataclass
from typing import Dict, Optional

from .action_repairer import ActionRepairer
from .action_types import GateDecision, ValidationResult, normalize_action_text, parse_action, to_env_action
from .action_validator import ActionValidator


@dataclass
class FocusTracker:
    focus_category: Optional[str] = None
    focus_target: Optional[str] = None
    already_focused: bool = False


class SRMGate:
    def __init__(self):
        self.validator = ActionValidator()
        self.repairer = ActionRepairer()
        self.focus = FocusTracker()

    def set_focus_category(self, focus_category: Optional[str]):
        self.focus.focus_category = focus_category

    def maybe_set_focus_target_from_planning(self, planning_text: str):
        return

    def set_focus_target_from_planning(self, planning_text: str, focus_category: Optional[str] = None):
        return

    def maybe_set_focus_target_from_task(self, task_description: str):
        return

    def mark_focus_executed(self, action: str, obs: str):
        obs_l = (obs or "").lower()
        failed = (
            "no known action matches" in obs_l
            or "you can't" in obs_l
            or "you cannot" in obs_l
            or "doesn't seem possible" in obs_l
        )
        if (action or "").lower().startswith("focus on") and not failed:
            self.focus.already_focused = True

    def pre_execute(self, raw_action: str, source: str, state: Dict) -> GateDecision:
        normalized = normalize_action_text(raw_action)
        parsed = parse_action(normalized)
        state2 = dict(state or {})
        state2["source"] = source
        state2["focus_category"] = self.focus.focus_category
        state2["focus_target"] = self.focus.focus_target
        state2["already_focused"] = self.focus.already_focused

        if parsed is None:
            return GateDecision(
                kind="DROP_INVALID",
                action_env=None,
                normalized_action=normalized,
                reason_codes=["PARSE_FAILED"],
                source=source,
            )

        if parsed.verb == "focus":
            focus_limit = state2.get("focus_limit")
            focus_used = state2.get("focus_used", 0)
            if focus_limit is not None and focus_used is not None and int(focus_used) >= int(focus_limit):
                validation = ValidationResult(status="INVALID", reason_codes=["FOCUS_LIMIT_EXCEEDED"])
            else:
                validation = self.validator.validate(parsed, state2)
        else:
            validation = self.validator.validate(parsed, state2)

        if validation.status in ("VALID", "NOOP"):
            return GateDecision(
                kind="ACCEPT",
                action_env=to_env_action(parsed, valid_actions=state2.get("valid_actions")),
                normalized_action=normalized,
                reason_codes=validation.reason_codes or [],
                source=source,
            )

        repair = self.repairer.repair(raw_action, parsed, validation, state2)
        if repair.kind == "REPAIRED":
            return GateDecision(
                kind="REPAIR",
                action_env=repair.action_env,
                normalized_action=normalized,
                reason_codes=repair.reason_codes,
                source=source,
            )
        return GateDecision(
            kind="DROP_INVALID",
            action_env=None,
            normalized_action=normalized,
            reason_codes=repair.reason_codes,
            source=source,
        )

