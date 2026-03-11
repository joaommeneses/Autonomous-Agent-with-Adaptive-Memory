import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple


def _norm_action_text(action: str) -> str:
    return re.sub(r"\s+", " ", (action or "").strip().lower())


def _filter_valid_actions_for_critic_internal(
    valid_actions: Sequence[str],
    task_description: str,
    look: str,
    inventory: str,
    rooms: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, int]]:
    del task_description  # reserved for future task-aware filtering
    del look              # reserved for future context-aware filtering
    del inventory         # reserved for future inventory-aware filtering

    room_set = {_norm_action_text(r) for r in (rooms or []) if r}
    air_agent_re = re.compile(r"\b(?:air|agent)\b", flags=re.IGNORECASE)

    kept: List[str] = []
    removed_counts: Counter = Counter()
    wait_candidates: List[str] = []

    for action in valid_actions or []:
        text = action or ""
        norm = _norm_action_text(text)
        if not norm:
            removed_counts["EMPTY_ACTION"] += 1
            continue

        if norm == "wait":
            wait_candidates.append(text)

        # Remove any action mentioning air/agent.
        if air_agent_re.search(norm):
            removed_counts["AIR_AGENT"] += 1
            continue

        # Remove look actions; LOOK context is already provided to Critic.
        if norm == "look around" or norm.startswith("look at ") or norm.startswith("look "):
            removed_counts["LOOK_ACTION"] += 1
            continue

        # Remove room inspection actions (keep navigation actions untouched).
        if room_set:
            room_match = re.match(r"^(?:look at|examine)\s+(.+)$", norm)
            if room_match:
                target = room_match.group(1).strip()
                if target in room_set:
                    removed_counts["ROOM_INSPECTION"] += 1
                    continue

        kept.append(text)

    # Ensure wait remains available if it exists in the original list.
    if wait_candidates and all(_norm_action_text(a) != "wait" for a in kept):
        kept.append(wait_candidates[0])

    return kept, dict(removed_counts)


def filter_valid_actions_for_critic(
    valid_actions: Sequence[str],
    task_description: str,
    look: str,
    inventory: str,
    rooms: Optional[Sequence[str]] = None,
) -> List[str]:
    kept, _ = _filter_valid_actions_for_critic_internal(
        valid_actions=valid_actions,
        task_description=task_description,
        look=look,
        inventory=inventory,
        rooms=rooms,
    )
    return kept


def filter_valid_actions_for_critic_with_stats(
    valid_actions: Sequence[str],
    task_description: str,
    look: str,
    inventory: str,
    rooms: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, int]]:
    return _filter_valid_actions_for_critic_internal(
        valid_actions=valid_actions,
        task_description=task_description,
        look=look,
        inventory=inventory,
        rooms=rooms,
    )


def build_critic_prompt(
    task_description: str,
    stagnation_reasons: Sequence[str],
    stagnation_metrics: dict,
    current_room: str,
    look: str,
    inventory: str,
    recent_history_lines: Sequence[str],
    valid_actions: Sequence[str],
    focus_used: int = 0,
    focus_limit: int = 1,
) -> str:
    history_block = "\n".join(recent_history_lines[-10:]) if recent_history_lines else "(none)"
    valid_block = "\n".join(f"- {a}" for a in valid_actions)
    return f"""You are SRM Critic for a ScienceWorld-style agent.
You are NOT the Planner. The Planner makes broad plans.
You are an immediate intervention layer used only when the agent is stagnated.
Your job is to break stagnation now with a short task-aligned action list (up to 5 actions).

TASK:
{task_description}

STAGNATION_REASONS:
{list(stagnation_reasons)}

STAGNATION_METRICS:
{stagnation_metrics}

CURRENT_ROOM:
{current_room}

LOOK:
{look}

INVENTORY:
{inventory}

RECENT_HISTORY (most recent last):
{history_block}

FOCUS_USAGE:
used={focus_used}, limit={focus_limit}

VALID_ACTIONS (EXHAUSTIVE; you MUST copy exact strings from here and MUST NOT invent new actions):
{valid_block}

Hard rules (do not break these):
- Your job is to break stagnation with task-aligned progress actions.
- Every action must directly advance TASK or set up an immediate prerequisite from LOOK/INVENTORY.
- Use STAGNATION_REASONS to avoid repeating the same failure mode.
- Prefer world-changing actions (open/activate/deactivate/pick up/move/pour/mix/use/read/examine). Use navigation only if needed to enable immediate progress.
- You MUST output only actions that appear EXACTLY (verbatim) in VALID_ACTIONS.
  Do NOT paraphrase. Do NOT rename objects. Copy exact strings.
- Before finalizing, verify each selected action is present verbatim in VALID_ACTIONS.
  If not, replace it with the closest alternative from VALID_ACTIONS.
- If you cannot find 5 good actions that are VERIFIABLY in VALID_ACTIONS, output fewer (1–4). NEVER invent actions.
- If focus_used >= focus_limit: NEVER output any "focus on ..." action.
- Avoid loops: avoid actions identical to any of the last 3 actions in RECENT_HISTORY unless clearly necessary.
- Avoid movement spam: output at most 1 navigation action (go/teleport/open door) in your ACTIONS list.

Output format (STRICT; do not add extra numbered lists outside ACTIONS):
DIAGNOSIS:
- (2–4 bullets) What is the agent doing that causes stagnation, and what is missing/blocked?

COURSE_CORRECTION:
- (1–2 bullets) The new immediate strategy to try.

ACTIONS:
1) <exact valid action string>
2) <exact valid action string>
3) <exact valid action string>
4) <exact valid action string>
5) <exact valid action string>
"""


def parse_critic_actions(response_text: str, valid_actions: Optional[Sequence[str]] = None) -> List[str]:
    text = response_text or ""
    valid_set = set(valid_actions) if valid_actions else set()

    actions: List[str] = []
    section_match = re.search(r"ACTIONS\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    section = section_match.group(1) if section_match else text
    for line in section.splitlines():
        m = re.match(r"^\s*(\d+)\)\s*(.+?)\s*$", line)
        if not m:
            continue
        candidate = m.group(2).strip()
        if candidate:
            actions.append(candidate)
        if len(actions) >= 5:
            break

    if actions and valid_set:
        actions = [a for a in actions if a in valid_set]

    if actions:
        return actions

    # Best-effort fallback: select lines that match any valid action exactly.
    if valid_set:
        fallback: List[str] = []
        for line in text.splitlines():
            candidate = line.strip()
            if candidate in valid_set:
                fallback.append(candidate)
            if len(fallback) >= 5:
                break
        return fallback
    return []


def run_critic_once(llm, prompt: str, logger=None) -> str:
    # Reuse the exact shared System-2 route (Qwen-vLLM or local wrapper)
    # so Critic and Sage use the same backend behavior.
    from eval_utils import call_system2

    return call_system2(
        prompt=prompt,
        llm_client=llm,
        logger=logger,
        max_tokens=1536,
        temperature=0.0,
        top_p=1.0,
    )

