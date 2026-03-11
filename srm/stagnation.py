from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from data_utils.data_utils import sanitizeStr
from srm.action_types import normalize_action_text, parse_action


def _extract_verb(action: str) -> str:
    normalized = normalize_action_text(action or "")
    parsed = parse_action(normalized)
    if parsed is not None and parsed.verb:
        return parsed.verb
    tokens = normalized.lower().split()
    return tokens[0] if tokens else ""


def _inventory_signature(inventory_text: str) -> str:
    lines = (inventory_text or "").splitlines()
    cleaned: List[str] = []
    for line in lines:
        item = line.strip().lower()
        if not item:
            continue
        if item.startswith("in your inventory"):
            continue
        if item.startswith("your inventory"):
            continue
        cleaned.append(item)
    cleaned.sort()
    return "|".join(cleaned)


def _state_signature(room_name: str, inv_sig: str) -> Tuple[str, str]:
    return ((room_name or "").strip().lower(), inv_sig)


def _obs_signature(obs: str) -> str:
    return sanitizeStr(obs or "").strip().lower()


def _action_signature(action: str) -> str:
    return sanitizeStr(normalize_action_text(action or "")).strip().lower()


@dataclass
class StagnationReport:
    is_stagnated: bool
    reasons: List[str]
    metrics: Dict[str, Any]
    last_action: str
    last_obs_sig: str


class SRMStagnationDetector:
    def __init__(
        self,
        same_obs_k: int = 3,
        repeat_action_k: int = 2,
        repeat_verb_k: int = 3,
        window: int = 5,
        min_steps: int = 5,
    ):
        self.same_obs_k = same_obs_k
        self.repeat_action_k = repeat_action_k
        self.repeat_verb_k = repeat_verb_k
        self.window = window
        self.min_steps = min_steps
        self._history: Deque[Dict[str, Any]] = deque(maxlen=max(window, repeat_verb_k, same_obs_k))

    def _same_obs_streak(self) -> int:
        if not self._history:
            return 0
        last = self._history[-1]["obs_norm"]
        streak = 0
        for item in reversed(self._history):
            if item["obs_norm"] == last:
                streak += 1
            else:
                break
        return streak

    def _repeat_action_streak(self) -> int:
        if not self._history:
            return 0
        last = self._history[-1]["action_norm"]
        streak = 0
        for item in reversed(self._history):
            if item["action_norm"] == last:
                streak += 1
            else:
                break
        return streak

    def _verb_streak(self) -> int:
        if not self._history:
            return 0
        last = self._history[-1]["verb"]
        streak = 0
        for item in reversed(self._history):
            if item["verb"] == last:
                streak += 1
            else:
                break
        return streak

    @staticmethod
    def _is_nav(item: Dict[str, Any]) -> bool:
        action_norm = item["action_norm"]
        verb = item["verb"]
        return (
            verb in {"go", "teleport", "open_door"}
            or action_norm.startswith("go to ")
            or action_norm.startswith("teleport to ")
            or action_norm.startswith("open door to ")
        )

    def update(
        self,
        step: int,
        action: str,
        obs: str,
        room: str,
        inventory_text: str,
        score: float,
    ) -> Optional[StagnationReport]:
        entry = {
            "action_norm": _action_signature(action),
            "verb": _extract_verb(action),
            "obs_norm": _obs_signature(obs),
            "room": (room or "").strip().lower(),
            "inv_sig": _inventory_signature(inventory_text),
            "score": score,
        }
        entry["state_sig"] = _state_signature(entry["room"], entry["inv_sig"])
        self._history.append(entry)

        if step < self.min_steps or len(self._history) < 2:
            return None

        reasons: List[str] = []

        same_obs_streak = self._same_obs_streak()
        repeat_action_streak = self._repeat_action_streak()
        verb_streak = self._verb_streak()

        if same_obs_streak >= self.same_obs_k:
            reasons.append("SAME_OBS_3")

        if len(self._history) >= self.repeat_action_k:
            last_two = list(self._history)[-self.repeat_action_k :]
            if (
                len({x["action_norm"] for x in last_two}) == 1
                and len({x["obs_norm"] for x in last_two}) == 1
            ):
                reasons.append("REPEAT_ACTION_2_NO_EFFECT")

        if len(self._history) >= self.repeat_verb_k:
            last_three = list(self._history)[-self.repeat_verb_k :]
            all_same_verb = len({x["verb"] for x in last_three}) == 1
            obs_same_two = len({x["obs_norm"] for x in last_three[-2:]}) == 1
            obs_same_three = len({x["obs_norm"] for x in last_three}) == 1
            if all_same_verb and (obs_same_two or obs_same_three):
                reasons.append("REPEAT_VERB_3_NO_EFFECT")

        last_window = list(self._history)[-self.window :]
        nav_count = sum(1 for item in last_window if self._is_nav(item))
        unique_states = len({item["state_sig"] for item in last_window})
        if len(last_window) >= self.window and nav_count >= 4 and unique_states <= 2:
            reasons.append("MOVE_SPAM_4_OF_5")

        if not reasons:
            return None

        last_obs_sig = entry["obs_norm"][:120]
        return StagnationReport(
            is_stagnated=True,
            reasons=reasons,
            metrics={
                "same_obs_streak": same_obs_streak,
                "repeat_action_streak": repeat_action_streak,
                "verb_streak": verb_streak,
                "nav_count_5": nav_count,
                "unique_states_5": unique_states,
            },
            last_action=entry["action_norm"],
            last_obs_sig=last_obs_sig,
        )






