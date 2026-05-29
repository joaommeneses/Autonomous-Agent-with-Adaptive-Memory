import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple


def _norm_action_text(action: str) -> str:
    return re.sub(r"\s+", " ", (action or "").strip().lower())


def _extract_recent_actions_to_avoid(recent_history_lines: Sequence[str], k: int = 3) -> List[str]:
    extracted: List[str] = []
    for line in recent_history_lines or []:
        text = str(line or "")
        action = ""
        if " | action=" in text:
            action = text.split(" | action=", 1)[1]
            if " | obs=" in action:
                action = action.split(" | obs=", 1)[0]
        elif "action=" in text:
            action = text.split("action=", 1)[1]
            if " | obs=" in action:
                action = action.split(" | obs=", 1)[0]
        action = action.strip()
        if action:
            extracted.append(action)
    return extracted[-k:]


def _derive_stagnation_mode(stagnation_reasons: Sequence[str]) -> str:
    reasons_upper = [str(r or "").upper() for r in (stagnation_reasons or [])]
    has_move_spam = any("MOVE_SPAM" in r for r in reasons_upper)
    has_repeat_no_effect = any(("REPEAT_ACTION" in r or "REPEAT_VERB" in r) for r in reasons_upper)
    has_same_obs = any("SAME_OBS" in r for r in reasons_upper)
    active = int(has_move_spam) + int(has_repeat_no_effect) + int(has_same_obs)
    if active >= 2:
        return "mixed_stagnation"
    if has_move_spam:
        return "move_spam"
    if has_repeat_no_effect:
        return "repeat_no_effect"
    if has_same_obs:
        return "same_obs_loop"
    return "other_stagnation"


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
    episodic_memories_block: Optional[str] = None,
) -> str:
    history_block = "\n".join(recent_history_lines[-10:]) if recent_history_lines else "(none)"
    valid_block = "\n".join(str(a) for a in valid_actions)
    focus_forbidden = "YES" if focus_used >= focus_limit else "NO"
    reasons_upper = [str(r or "").upper() for r in (stagnation_reasons or [])]
    nav_count_5 = int((stagnation_metrics or {}).get("nav_count_5", 0) or 0)
    nav_disfavored = "YES" if (any("MOVE_SPAM" in r for r in reasons_upper) or nav_count_5 >= 3) else "NO"
    recent_actions_to_avoid = _extract_recent_actions_to_avoid(recent_history_lines, k=3)
    stagnation_mode = _derive_stagnation_mode(stagnation_reasons)
    episodic_section = ""
    if episodic_memories_block and episodic_memories_block.strip():
        episodic_section = (
            "EPISODIC_MEMORIES (non-binding evidence):\n"
            f"{episodic_memories_block.strip()}\n\n"
        )
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

RUNTIME_CONSTRAINTS:
FOCUS_ACTIONS_FORBIDDEN={focus_forbidden}
NAVIGATION_STRONGLY_DISFAVORED={nav_disfavored}
RECENT_ACTIONS_TO_AVOID={recent_actions_to_avoid}
STAGNATION_MODE={stagnation_mode}

{episodic_section}VALID_ACTIONS (EXHAUSTIVE; RAW STRINGS, one per line; copy exact strings only):
{valid_block}

Hard rules (do not break these):
- Runtime constraints override episodic memories.
- If FOCUS_ACTIONS_FORBIDDEN=YES: never output any "focus on ..." action.
- If NAVIGATION_STRONGLY_DISFAVORED=YES: use navigation only when it is the only immediate enabler.
- Avoid proposing actions in RECENT_ACTIONS_TO_AVOID unless clearly justified as a strict enabler.
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
- Memory-use rules:
  - Episodic memories are non-binding evidence and may be stale.
  - Use episodic memories as hints, not commands.
  - Prefer current LOOK/INVENTORY/VALID_ACTIONS when memory conflicts with current state.
  - Runtime constraints override memory examples.
  - Do not copy memory actions unless they are valid now and appear verbatim in VALID_ACTIONS.
  - If a memory suggests a forbidden action now, infer the nearest admissible prerequisite or alternative.
  - Prioritize actions that change world state now; deprioritize sensing-only or repeated no-effect actions unless they unlock progress.

Selection procedure (strict order):
1) Read RUNTIME_CONSTRAINTS first.
2) Diagnose stagnation mode and immediate blocker.
3) Use episodic memories only as supporting evidence.
4) Select exact actions from VALID_ACTIONS.
5) Re-check final list against VALID_ACTIONS and RUNTIME_CONSTRAINTS.

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
    from eval_utils import call_system2

    return call_system2(
        prompt=prompt,
        llm_client=llm,
        logger=logger,
        max_tokens=1536,
        temperature=0.0,
        top_p=1.0,
    )

