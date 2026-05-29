from typing import Any, Dict, List, Optional, Sequence, Tuple

from amm.config import DEFAULT_CONFIG
from amm.formatters import _parse_inventory_text, build_critic_memories_block
from amm.retrieval import (
    build_avoidance_retrieval_query_b,
    build_success_retrieval_query_s2,
    retrieve_avoidance_ems_b,
    retrieve_success_ems_s2,
)


def _extract_recent_actions_obs(recent_history_lines: Sequence[str]) -> Tuple[List[str], List[str]]:
    actions: List[str] = []
    observations: List[str] = []
    for line in recent_history_lines or []:
        text = str(line or "")
        action = ""
        obs = ""
        if " | action=" in text:
            action = text.split(" | action=", 1)[1]
            if " | obs=" in action:
                action, obs = action.split(" | obs=", 1)
        elif "action=" in text:
            action = text.split("action=", 1)[1]
            if " | obs=" in action:
                action, obs = action.split(" | obs=", 1)
        if action:
            actions.append(action.strip())
        if obs:
            observations.append(obs.strip())
    return actions[-5:], observations[-5:]


def _has_failure_signal(
    stagnation_reasons: Sequence[str],
    recent_observations: Sequence[str],
) -> bool:
    reasons_upper = [str(r or "").upper() for r in (stagnation_reasons or [])]
    reason_match = any(
        any(key in r for key in ("INVALID", "NO_EFFECT", "REPEAT_ACTION", "REPEAT_VERB", "SAME_OBS"))
        for r in reasons_upper
    )
    recent_obs_text = " ".join(str(o or "").lower() for o in (recent_observations[-2:] if recent_observations else []))
    obs_keywords = (
        "no known action matches",
        "you can't",
        "you cannot",
        "not allowed",
        "doesn't seem possible",
        "not possible",
        "cannot",
    )
    obs_match = any(k in recent_obs_text for k in obs_keywords)
    return bool(reason_match or obs_match)


def _retarget_query_for_critic(query_text: str, issue: str, action_context: str) -> str:
    lines = []
    has_issue = False
    has_context = False
    for line in (query_text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("ISSUE:"):
            lines.append(f'ISSUE: "{issue}"')
            has_issue = True
            continue
        if stripped.startswith("ACTION_CONTEXT:"):
            lines.append(f'ACTION_CONTEXT: "{action_context}"')
            has_context = True
            continue
        lines.append(line)
    if not has_context:
        lines.append(f'ACTION_CONTEXT: "{action_context}"')
    if not has_issue:
        lines.append(f'ISSUE: "{issue}"')
    return "\n".join(lines) + "\n"


def retrieve_memories_for_critic(
    amm_client,
    task_description: str,
    current_room: str,
    look: str,
    inventory: str,
    recent_history_lines: List[str],
    stagnation_reasons,
    stagnation_metrics,
    logger,
    focus_used: int = 0,
    focus_limit: int = 1,
) -> Tuple[Optional[str], Dict[str, Any]]:
    del stagnation_metrics  # reserved for future weighting

    stats: Dict[str, Any] = {
        "worked_retrieved": 0,
        "avoid_retrieved": 0,
        "injected_chars": 0,
        "avoidance_used": False,
    }

    inventory_items = _parse_inventory_text(inventory or "")
    recent_rewards = [0.0]
    current_score = 0.0
    recent_actions, recent_observations = _extract_recent_actions_obs(recent_history_lines)

    query_s2 = build_success_retrieval_query_s2(
        task_description=task_description,
        room_name=current_room or "unknown",
        inventory_items=inventory_items,
        recent_rewards=recent_rewards,
        current_score=current_score,
        look_description=look,
        recent_actions=recent_actions,
        recent_observations=recent_observations,
    )
    query_s2 = _retarget_query_for_critic(
        query_text=query_s2,
        issue="critic_stagnation",
        action_context=(
            "The agent is stagnated and needs immediate course correction. Retrieve concise episodes that help break "
            "repeated no-effect behavior and suggest the next admissible progress-producing action."
        ),
    )
    worked_ems = retrieve_success_ems_s2(
        memory_agent_id=amm_client.agent_id,
        query_text=query_s2,
        letta_client=amm_client,
    )
    stats["worked_retrieved"] = len(worked_ems)

    avoidance_ems: List[Dict[str, Any]] = []
    reasons_upper = [str(r or "").upper() for r in (stagnation_reasons or [])]
    move_spam_only = bool(reasons_upper) and all("MOVE_SPAM" in r for r in reasons_upper)
    use_avoidance = DEFAULT_CONFIG.enable_t3_retrieval and (not move_spam_only) and _has_failure_signal(
        stagnation_reasons=stagnation_reasons,
        recent_observations=recent_observations,
    )
    if use_avoidance:
        query_b = build_avoidance_retrieval_query_b(
            task_description=task_description,
            room_name=current_room or "unknown",
            inventory_items=inventory_items,
            recent_rewards=recent_rewards,
            current_score=current_score,
            look_description=look,
            recent_actions=recent_actions,
            recent_observations=recent_observations,
        )
        query_b = _retarget_query_for_critic(
            query_text=query_b,
            issue="critic_stagnation_avoidance",
            action_context=(
                "The agent is stuck in repeated invalid/no-effect behavior. Retrieve concise avoidance patterns that "
                "warn what not to repeat and indicate a safer immediate alternative."
            ),
        )
        avoidance_ems = retrieve_avoidance_ems_b(
            memory_agent_id=amm_client.agent_id,
            query_text=query_b,
            letta_client=amm_client,
        )
        stats["avoidance_used"] = True
    stats["avoid_retrieved"] = len(avoidance_ems)

    block = build_critic_memories_block(
        worked_ems=worked_ems,
        avoidance_ems=avoidance_ems,
        max_worked=3,
        max_avoid=1,
        char_cap=12000,
        current_room=current_room or "",
        focus_actions_forbidden=bool(focus_used >= focus_limit),
        current_task=task_description,
        logger=logger,
    )
    if not block:
        return None, stats

    stats["injected_chars"] = len(block)
    return block, stats

