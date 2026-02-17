import re
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


def extract_focus_target(planning_text: str, focus_category: Optional[str]) -> Optional[str]:
    text = planning_text or ""
    if not focus_category:
        return None

    cat = focus_category.lower().strip()
    if cat != "substance":
        return None

    # Prefer section-based extraction: "Substance:" / "Substances:"
    section = re.search(r"(?is)\bsubstances?\s*:\s*(.+?)(?:\n\s*\n|$)", text)
    if section:
        lines = [ln.strip() for ln in section.group(1).splitlines() if ln.strip()]
        for ln in lines:
            ln = re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip()
            ln = re.sub(r"\(.*?\)", "", ln).strip()
            if ln:
                return re.sub(r"\s+", " ", ln.lower())
    return None


def extract_focus_target_from_task(task_description: str, focus_category: Optional[str]) -> Optional[str]:
    if (focus_category or "").lower() != "substance":
        return None
    text = (task_description or "").lower()
    patterns = [
        r"\bboil\s+([a-z][a-z\s\-]+?)(?:\bwith\b|\bin\b|,|\.|$)",
        r"\bfocus on\s+([a-z][a-z\s\-]+?)(?:\bwith\b|\bin\b|,|\.|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    return None


class SRMGate:
    def __init__(self, debug: bool = False):
        self.validator = ActionValidator()
        self.repairer = ActionRepairer()
        self.focus = FocusTracker()
        self.debug = debug

    def set_focus_category(self, focus_category: Optional[str]):
        self.focus.focus_category = focus_category

    def maybe_set_focus_target_from_planning(self, planning_text: str):
        if self.focus.focus_target:
            return
        target = extract_focus_target(planning_text, self.focus.focus_category)
        if target:
            self.focus.focus_target = target

    def set_focus_target_from_planning(self, planning_text: str, focus_category: Optional[str] = None):
        if focus_category:
            self.set_focus_category(focus_category)
        self.maybe_set_focus_target_from_planning(planning_text)

    def maybe_set_focus_target_from_task(self, task_description: str):
        if self.focus.focus_target:
            return
        target = extract_focus_target_from_task(task_description, self.focus.focus_category)
        if target:
            self.focus.focus_target = target

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

        # Short-circuit repeated focus once task focus is already complete.
        if parsed.verb == "focus" and self.focus.already_focused:
            validation = ValidationResult(status="INVALID", reason_codes=["FOCUS_ALREADY_DONE"])
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

