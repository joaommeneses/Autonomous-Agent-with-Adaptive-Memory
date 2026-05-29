import re
from typing import Dict, List, Set

from .action_types import ALLOWED_VERBS, ParsedAction, ValidationResult, to_env_action


ARITY = {
    "wait": 0,
    "look": (0, 1),
    "read": 1,
    "pick": 1,
    "open": 1,
    "close": 1,
    "activate": 1,
    "deactivate": 1,
    "examine": 1,
    "teleport": 1,
    "connect": 2,
    "move": 2,
    "use": 2,
    "pour": 2,
    "dunk": 2,
    "mix": 1,
    "focus": 1,
    "choose": 1,
}


def _norm_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _contains_obj(text: str, obj: str) -> bool:
    return obj and obj.lower() in (text or "").lower()


_OBS_STOPWORDS = {
    "a", "an", "the", "in", "on", "to", "from", "of", "and", "or",
    "called", "there", "is",
}


def _normalize_world_text(text: str) -> str:
    s = (text or "").lower()
    for phrase in ("a substance called", "substance called", "there is a", "there is an", "there is the"):
        s = s.replace(phrase, " ")
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_content_tokens(phrase: str) -> List[str]:
    norm = _normalize_world_text(phrase)
    if not norm:
        return []
    return [tok for tok in norm.split() if tok and tok not in _OBS_STOPWORDS]


def _is_phrase_observed(phrase: str, look: str, inventory: str) -> bool:
    tokens = _extract_content_tokens(phrase)
    if not tokens:
        return False

    ctx_norm = _normalize_world_text(f"{look}\n{inventory}")
    target_norm = " ".join(tokens)
    if target_norm and target_norm in ctx_norm:
        return True

    ctx_tokens = set(ctx_norm.split())
    return all(tok in ctx_tokens for tok in tokens)


def _is_target_observed(target: str, look: str, inventory: str) -> bool:
    return _is_phrase_observed(target, look, inventory)


def _open_noop(obj: str, valid_actions: Set[str]) -> bool:
    return (f"open {obj}" not in valid_actions) and (f"close {obj}" in valid_actions)


def _activate_noop(obj: str, valid_actions: Set[str]) -> bool:
    return (f"activate {obj}" not in valid_actions) and (f"deactivate {obj}" in valid_actions)


class ActionValidator:
    def validate(self, parsed_action: ParsedAction, state: Dict) -> ValidationResult:
        if parsed_action is None:
            return ValidationResult(status="INVALID", reason_codes=["PARSE_FAILED"])

        verb = parsed_action.verb
        args = parsed_action.args
        valid_actions = set(state.get("valid_actions") or [])
        look = state.get("look", "") or ""
        inventory = state.get("inventory", "") or ""
        focus_target = state.get("focus_target")

        if verb not in ALLOWED_VERBS:
            return ValidationResult(status="INVALID", reason_codes=["VERB_NOT_ALLOWED"])

        expected_arity = ARITY.get(verb)
        if isinstance(expected_arity, tuple):
            if len(args) not in expected_arity:
                return ValidationResult(status="INVALID", reason_codes=["ARITY_MISMATCH"])
        elif expected_arity is not None and len(args) != expected_arity:
            return ValidationResult(status="INVALID", reason_codes=["ARITY_MISMATCH"])

        if verb == "focus":
            target = _norm_text(args[0]) if args else ""
            # Exact equality is enforced only when an explicit focus_target is externally set.
            if focus_target and target != _norm_text(focus_target):
                return ValidationResult(status="INVALID", reason_codes=["FOCUS_WRONG_TARGET"])
            if not _is_target_observed(target, look, inventory):
                return ValidationResult(status="INVALID", reason_codes=["FOCUS_TARGET_NOT_OBSERVED_YET"])

        env_action = to_env_action(parsed_action, valid_actions=valid_actions if valid_actions else None)
        env_action_l = env_action.lower()

        # No-op handling from preconditions
        if valid_actions and verb == "open" and args:
            obj = args[0].lower().strip()
            if _open_noop(obj, set(a.lower() for a in valid_actions)):
                return ValidationResult(status="NOOP", reason_codes=["NOOP_ALREADY_OPEN"])
        if valid_actions and verb == "activate" and args:
            obj = args[0].lower().strip()
            if _activate_noop(obj, set(a.lower() for a in valid_actions)):
                return ValidationResult(status="NOOP", reason_codes=["NOOP_ALREADY_ACTIVE"])

        # Feasibility check
        if valid_actions:
            valid_actions_l = set(a.lower() for a in valid_actions)
            if env_action_l in valid_actions_l:
                return ValidationResult(status="VALID", reason_codes=[])
            return ValidationResult(
                status="INVALID",
                reason_codes=["NOT_IN_VALID_ACTIONS"],
                suggested_repairs=[a for a in valid_actions if a.lower() == env_action_l],
            )

        # Minimal fallback feasibility when valid_actions not available
        missing_obj = False
        for arg in args:
            if arg and not (_contains_obj(look, arg) or _contains_obj(inventory, arg)):
                missing_obj = True
                break
        if missing_obj:
            return ValidationResult(status="INVALID", reason_codes=["OBJECT_NOT_VISIBLE_OR_HELD"])
        return ValidationResult(status="VALID", reason_codes=[])

