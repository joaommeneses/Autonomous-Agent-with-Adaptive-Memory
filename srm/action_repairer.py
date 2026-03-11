from typing import Dict, List

from .action_types import RepairOutcome, normalize_action_text, parse_action, to_env_action


def _choose_non_focus_from_list(candidates: List[str], valid_actions: List[str]) -> str:
    va = set(valid_actions or [])
    for cand in candidates or []:
        if not cand:
            continue
        parsed = parse_action(normalize_action_text(cand))
        if parsed is None:
            continue
        if parsed.verb == "focus":
            continue
        env_action = to_env_action(parsed, valid_actions=valid_actions)
        if not va or env_action in va:
            return env_action
    return None


class ActionRepairer:
    def repair(self, raw_action: str, parsed_action, validation_result, state: Dict) -> RepairOutcome:
        if validation_result.status == "NOOP":
            # Do not skip NOOPs; execute to consume a step and avoid stalling.
            return RepairOutcome(
                kind="REPAIRED",
                action_env=to_env_action(parsed_action, valid_actions=state.get("valid_actions")),
                reason_codes=validation_result.reason_codes,
            )

        if validation_result.status == "VALID":
            return RepairOutcome(
                kind="UNCHANGED",
                action_env=to_env_action(parsed_action, valid_actions=state.get("valid_actions")),
                reason_codes=[],
            )

        reason_codes = validation_result.reason_codes or []
        if (
            parsed_action is not None
            and parsed_action.verb == "focus"
            and (
                "FOCUS_TARGET_NOT_OBSERVED_YET" in reason_codes
                or "NOT_IN_VALID_ACTIONS" in reason_codes
            )
        ):
            valid_actions = list(state.get("valid_actions") or [])
            inventory_text = str(state.get("inventory") or "").lower()
            target = (parsed_action.args[0] if parsed_action.args else "").strip().lower()
            alias_action = "focus on substance in inventory"
            if target and target in inventory_text and alias_action in valid_actions:
                return RepairOutcome(
                    kind="REPAIRED",
                    action_env=alias_action,
                    reason_codes=reason_codes + ["FOCUS_INVENTORY_ALIAS"],
                )

        if "FOCUS_ALREADY_DONE" in reason_codes or "FOCUS_LIMIT_EXCEEDED" in reason_codes or "FOCUS_LIMIT_REACHED" in reason_codes:
            source = (state.get("source") or "").lower()
            valid_actions = list(state.get("valid_actions") or [])
            if source.startswith("swift"):
                swift_predictions = list(state.get("swift_predictions") or [])
                next_action = _choose_non_focus_from_list(swift_predictions, valid_actions)
                if next_action:
                    return RepairOutcome(
                        kind="REPAIRED",
                        action_env=next_action,
                        reason_codes=reason_codes,
                    )
            if source.startswith("sage_buffer") or source.startswith("buffer"):
                buffer_next_actions = list(state.get("buffer_next_actions") or [])
                next_action = _choose_non_focus_from_list(buffer_next_actions, valid_actions)
                if next_action:
                    return RepairOutcome(
                        kind="REPAIRED",
                        action_env=next_action,
                        reason_codes=reason_codes,
                    )
            if "wait" in valid_actions:
                return RepairOutcome(kind="REPAIRED", action_env="wait", reason_codes=reason_codes)
            return RepairOutcome(kind="DROP_INVALID", action_env=None, reason_codes=reason_codes)

        if "FOCUS_TARGET_NOT_OBSERVED_YET" in reason_codes:
            # Milestone-1 key behavior: do not execute early focus; drop and move on.
            return RepairOutcome(kind="DROP_INVALID", action_env=None, reason_codes=reason_codes)

        if validation_result.suggested_repairs:
            return RepairOutcome(
                kind="REPAIRED",
                action_env=validation_result.suggested_repairs[0],
                reason_codes=reason_codes + ["REPAIRED_BY_MEMBERSHIP"],
            )

        # Deterministic alias fallback already handled via parser/to_env_action.
        # If no safe repair exists, drop and let caller try next candidate.
        return RepairOutcome(kind="DROP_INVALID", action_env=None, reason_codes=reason_codes or ["UNREPAIRED"])

