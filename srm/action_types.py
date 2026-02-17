import re
from dataclasses import dataclass, field
from typing import List, Optional


ALLOWED_VERBS = {
    "wait",
    "look",
    "read",
    "pick",
    "open",
    "close",
    "activate",
    "deactivate",
    "examine",
    "teleport",
    "connect",
    "move",
    "use",
    "pour",
    "dunk",
    "mix",
    "focus",
    "choose",
}


@dataclass
class ParsedAction:
    verb: str
    args: List[str]
    source_format: str  # "FUNC" or "NL"
    canonical: str


@dataclass
class ValidationResult:
    status: str  # VALID | INVALID | NOOP
    reason_codes: List[str] = field(default_factory=list)
    suggested_repairs: List[str] = field(default_factory=list)


@dataclass
class RepairOutcome:
    kind: str  # REPAIRED | SKIP_NOOP | DROP_INVALID | UNCHANGED
    action_env: Optional[str] = None
    reason_codes: List[str] = field(default_factory=list)


@dataclass
class GateDecision:
    kind: str  # ACCEPT | REPAIR | SKIP_NOOP | DROP_INVALID
    action_env: Optional[str]
    normalized_action: str
    reason_codes: List[str] = field(default_factory=list)
    source: str = ""


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_action_text(raw: str) -> str:
    text = _clean_text(raw)
    if not text:
        return ""
    text = re.sub(r"^\s*Action\s*\d+\s*:\s*", "", text, flags=re.IGNORECASE)
    if "-->" in text:
        text = text.split("-->", 1)[0].strip()
    text = _clean_text(text)
    if not text:
        return ""

    # Normalize common NL aliases to function-like canonical forms.
    lowered = text.lower()
    if lowered.startswith("teleport to "):
        return f"TELEPORT({lowered.replace('teleport to ', '', 1).strip()})"
    if lowered.startswith("go to "):
        return f"TELEPORT({lowered.replace('go to ', '', 1).strip()})"
    if lowered.startswith("wait1"):
        return "WAIT()"
    return text


def _parse_function_like(canonical: str) -> Optional[ParsedAction]:
    m = re.match(r"^([A-Za-z_]+)\((.*)\)$", canonical.strip())
    if not m:
        return None
    verb_raw = m.group(1).lower().replace("_", " ")
    args_raw = m.group(2).strip()
    args = [a.strip().lower() for a in args_raw.split(",")] if args_raw else []
    args = [a for a in args if a]

    verb_aliases = {
        "pickup": "pick",
        "pick up": "pick",
        "go": "teleport",
        "turn on": "activate",
        "turn off": "deactivate",
        "see": "look",
        "observe": "examine",
        "open door": "open",
        "close door": "close",
    }
    verb = verb_aliases.get(verb_raw, verb_raw)
    return ParsedAction(verb=verb, args=args, source_format="FUNC", canonical=canonical)


def parse_action(canonical: str) -> Optional[ParsedAction]:
    text = _clean_text(canonical)
    if not text:
        return None

    func = _parse_function_like(text)
    if func is not None:
        return func

    lower = text.lower()
    patterns = [
        (r"^wait$", "wait", []),
        (r"^look around$", "look", []),
        (r"^look at (.+)$", "look", [1]),
        (r"^read (.+)$", "read", [1]),
        (r"^focus on (.+)$", "focus", [1]),
        (r"^pick up (.+)$", "pick", [1]),
        (r"^open (.+)$", "open", [1]),
        (r"^close (.+)$", "close", [1]),
        (r"^activate (.+)$", "activate", [1]),
        (r"^deactivate (.+)$", "deactivate", [1]),
        (r"^examine (.+)$", "examine", [1]),
        (r"^teleport to (.+)$", "teleport", [1]),
        (r"^go to (.+)$", "teleport", [1]),
        (r"^connect (.+) to (.+)$", "connect", [1, 2]),
        (r"^move (.+) to (.+)$", "move", [1, 2]),
        (r"^use (.+) on (.+)$", "use", [1, 2]),
        (r"^pour (.+) into (.+)$", "pour", [1, 2]),
        (r"^dunk (.+) into (.+)$", "dunk", [1, 2]),
        (r"^mix (.+)$", "mix", [1]),
        (r"^([01])$", "choose", [1]),
    ]
    for pattern, verb, groups in patterns:
        m = re.match(pattern, lower)
        if not m:
            continue
        args = [m.group(i).strip() for i in groups]
        return ParsedAction(verb=verb, args=args, source_format="NL", canonical=text)
    return None


def to_env_action(parsed: ParsedAction, valid_actions=None) -> str:
    verb = parsed.verb
    args = parsed.args
    candidates = []

    if verb == "wait":
        candidates = ["wait"]
    elif verb == "look":
        candidates = ["look around"] if not args else [f"look at {args[0]}"]
    elif verb == "read":
        candidates = [f"read {args[0]}"]
    elif verb == "focus":
        candidates = [f"focus on {args[0]}"]
    elif verb == "pick":
        candidates = [f"pick up {args[0]}"]
    elif verb == "open":
        candidates = [f"open {args[0]}", f"open door to {args[0]}"]
    elif verb == "close":
        candidates = [f"close {args[0]}", f"close door to {args[0]}"]
    elif verb == "activate":
        candidates = [f"activate {args[0]}"]
    elif verb == "deactivate":
        candidates = [f"deactivate {args[0]}"]
    elif verb == "examine":
        candidates = [f"examine {args[0]}"]
    elif verb == "teleport":
        candidates = [f"go to {args[0]}", f"teleport to {args[0]}"]
    elif verb == "connect":
        candidates = [f"connect {args[0]} to {args[1]}"]
    elif verb == "move":
        candidates = [f"move {args[0]} to {args[1]}"]
    elif verb == "use":
        candidates = [f"use {args[0]} on {args[1]}"]
    elif verb == "pour":
        candidates = [f"pour {args[0]} into {args[1]}", f"pour {args[0]} in {args[1]}"]
    elif verb == "dunk":
        candidates = [f"dunk {args[0]} into {args[1]}"]
    elif verb == "mix":
        candidates = [f"mix {args[0]}"]
    elif verb == "choose":
        candidates = [args[0]]
    else:
        return parsed.canonical

    if valid_actions:
        va = set(valid_actions)
        for cand in candidates:
            if cand in va:
                return cand
    return candidates[0] if candidates else parsed.canonical

