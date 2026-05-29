
import os
import re
import time
import json
import copy
import hashlib
import torch
import random
import argparse
from tqdm import trange
from scienceworld import ScienceWorldEnv
from data_utils.data_utils import add_current_place, add_current_objects, sanitizeStr, formalize_action
from data_utils.data_utils import compose_instance_v4
from eval_utils import load_model, load_variation, get_model_output, findValidActionWithSystem2, getFilteredValidActions, sbert_search, clean_look, is_action_failed 
from eval_utils import try_to_replace, rooms, clean_history, get_current_room, clean_obj_name, focus_on_count
from srm.srm_gate import SRMGate
from srm.action_types import normalize_action_text, parse_action
from srm.stagnation import SRMStagnationDetector
from srm.critic import (
    build_critic_prompt,
    filter_valid_actions_for_critic_with_stats,
    parse_critic_actions,
    run_critic_once,
)

from amm.config import DEFAULT_CONFIG
from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway, CallbackServerParameters
from scienceworld.constants import BASEPATH, DEBUG_MODE, ID2TASK, JAR_PATH, NAME2ID
from scienceworld.utils import infer_task
import logging
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

class MyScienceWorldEnv(ScienceWorldEnv):
    def __init__(self, taskName=None, serverPath=None, envStepLimit=100):
        serverPath = serverPath or JAR_PATH  # Use the builtin jar.

        # Launch the server and connect to the JVM.
        # Launch Java side with dynamic port and get back the port on which the
        # server was bound to.
        if DEBUG_MODE:
            import sys, time
            port = launch_gateway(
                classpath=serverPath, die_on_exit=True, cwd=BASEPATH,
                javaopts=['-agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=5005,quiet=y'],
                redirect_stdout=sys.stdout, redirect_stderr=sys.stderr)
            print("Attach debugger within the next 10 seconds")
            time.sleep(10)  # Give time for user to attach debugger
        else:
            port = launch_gateway(classpath=serverPath, die_on_exit=True, cwd=BASEPATH)

        # Connect python side to Java side with Java dynamic port and start python
        # callback server with a dynamic port
        self._gateway = JavaGateway(
            gateway_parameters=GatewayParameters(auto_field=True, port=port),
            callback_server_parameters=CallbackServerParameters(port=0, daemonize=True))

        # Retrieve the port on which the python callback server was bound to.
        python_port = self._gateway.get_callback_server().get_listening_port()

        # Tell the Java side to connect to the python callback server with the new
        # python port. Note that we use the java_gateway_server attribute that
        # retrieves the GatewayServer instance.
        self._gateway.java_gateway_server.resetCallbackClient(
            self._gateway.java_gateway_server.getCallbackClient().getAddress(),
            python_port)

        self.server = self._gateway.jvm.scienceworld.runtime.pythonapi.PythonInterface()
        logger.info(f"ScienceWorld server running on {port}" ) 

        # Keep track of the last step score, to calculate reward from score
        self.lastStepScore = 0

        # Load the script
        self.taskName = taskName
        if self.taskName:
            self.load(taskName, 0, "")

        # Set the environment step limit
        self.envStepLimit = envStepLimit

        # Clear the run histories
        self.clearRunHistories()

        # By default, set that the gold path was not generated unless the user asked for it
        self.goldPathGenerated = False


from logging import INFO, WARN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SAGE_CALLS_PER_ENV_STEP = 1
STUCK_K = 3
SAGE_COOLDOWN_STEPS = 2
MAX_CRITIC_CALLS_PER_EPISODE = 3
CRITIC_COOLDOWN_STEPS = 5
CRITIC_BACKOFF_STEPS = 10


def deterministic_fallback_action(valid_actions, current_room):
    actions = sorted(list(valid_actions)) if valid_actions else []
    if not actions:
        return None
    for preferred in ("look around", "inventory"):
        if preferred in valid_actions:
            return preferred
    room_moves = []
    for a in actions:
        lower = a.lower()
        dest = None
        if lower.startswith("go to "):
            dest = lower.replace("go to ", "", 1).strip()
        elif lower.startswith("teleport to "):
            dest = lower.replace("teleport to ", "", 1).strip()
        elif lower.startswith("open door to "):
            dest = lower.replace("open door to ", "", 1).strip()
        if dest and dest != (current_room or "").lower():
            room_moves.append((dest, a))
    if room_moves:
        room_moves.sort(key=lambda x: (x[0], x[1]))
        return room_moves[0][1]
    return actions[0]


def _inventory_signature_for_critic(inventory_text):
    lines = (inventory_text or "").splitlines()
    kept = []
    for line in lines:
        item = line.strip().lower()
        if not item:
            continue
        if item.startswith("in your inventory") or item.startswith("your inventory"):
            continue
        kept.append(item)
    kept.sort()
    return "|".join(kept)


def _critic_state_sig(room, inventory_text, obs):
    room_sig = (room or "").strip().lower()
    inv_sig = _inventory_signature_for_critic(inventory_text)
    obs_sig = sanitizeStr(obs or "").strip().lower()[:120]
    return (room_sig, inv_sig, obs_sig)


def _critic_actions_hash(actions):
    norm = [normalize_action_text(a or "").strip().lower() for a in actions]
    joined = "||".join(norm)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()

def get_file_name(args, task_num):
    if (len(args["output_path"]) > 0):
        if not args["output_path"].endswith("/"):
            args["output_path"] += "/"

        # Make path if it doesn't exist
        if not os.path.exists(args['output_path']):
            os.makedirs(args["output_path"])
  
    filenameOutPrefixSeed = args["output_path"] + "task" + str(task_num)

    return filenameOutPrefixSeed


def eval(args, task_num, logger):
    if args["compose_mode"] == "v4":
        compose_instance = compose_instance_v4
    
    demo_data = None 
    if args["demo_file"]: 
        with open(args["demo_file"]) as f:
            demo_data = json.load(f)
    
    env = MyScienceWorldEnv("", args["jar_path"], envStepLimit = args["env_step_limit"])
    taskNames = env.getTaskNames()
    taskName = taskNames[task_num]
    env.load(taskName, 0, args['simplification_str'])
    lm_model, tokenizer, sbert_model, llm = load_model(args, device)

    variations = load_variation(env, args, task_num, logger)
    filenameOutPrefixSeed = get_file_name(args, task_num)
    gpt_version = args["gpt_version"]
    scores = []

    enable_amm = bool(args.get("use_amm", False))
    amm_write_only = bool(args.get("amm_write_only", False))
    amm_retrieval_enabled = enable_amm and (not amm_write_only)
    use_sage = True
    enable_srm = not args.get("disable_srm", False)
    if amm_write_only and not enable_amm:
        logger.warning("[AMM] --amm_write_only has no effect because AMM is disabled (--use_amm not set)")

    amm_client = None
    wm = None
    if enable_amm:
        from amm.client_letta import AMMLettaClient, LettaConfig
        from amm.working_memory import WorkingMemory
        from amm.writer import write_success, write_nearmiss, write_avoidance, create_memory_record

        letta_api_token = os.getenv("LETTA_API_TOKEN")
        letta_agent_id = os.getenv("LETTA_AGENT_ID")
        
        if not letta_api_token or not letta_agent_id:
            raise ValueError(
                                "LETTA_API_TOKEN and LETTA_AGENT_ID environment variables must be set when use_amm=True. "
                                "Please set them before running the agent, or set use_amm=False for baseline mode."
            )
        
        amm_config = LettaConfig(
            api_token=letta_api_token,
            agent_id=letta_agent_id,
            agent_name="memory-agent"
        )
        amm_client = AMMLettaClient(amm_config)
        wm = WorkingMemory()
        logger.info("[AMM] Adaptive Memory Module initialized with Cloud API")
        logger.info(f"[AMM] Using agent ID: {letta_agent_id}")
    else:
        logger.info("[Baseline] Running in baseline SwiftSage mode (use_amm=False, Sage available via slow_agent)")

    for variation in variations:
        if args["debug_var"] >=0 and variation != args["debug_var"]:
            logger.info(f"Skipping the Var: {variation} because we only focus on args['debug_var'']={args['debug_var']}")
            continue 
        env.load(taskName, variation, args["simplification_str"], generateGoldPath=True)
        task_description = env.taskdescription()[18:]
        logger.info(f"task_description = {task_description}")
        
        if enable_amm and wm is not None:
            wm.reset()
            wm.pending_subgoal = task_description
            logger.info(f"[AMM] Working memory reset for new episode: {task_description[:50]}...")
            if amm_write_only:
                logger.info("[AMM] Mode=WRITE_ONLY (writes enabled, retrieval+augmentation disabled)")
            else:
                logger.info("[AMM] Mode=FULL (writes + retrieval/augmentation enabled)")

        recent_actions = ["look around"]
        recent_obs = ["N/A"]
        recent_locs = []
        recent_looks = {}
        recent_looks_flatten = []
        recent_scores = [0.0,]
        recent_reward = [0.0]
        places = []
        objects = []

        obs, info = env.reset()
        current_place = get_current_room(info['look'])        
        recent_locs.append(current_place)
        recent_looks[current_place] = info["look"]
        recent_looks_flatten.append(info["look"])

        prev_obs = 'N/A'
        prev_action = 'look around'

        done = False
        score = 0.0
        last_score = 0.0
        step = 0

        max_steps = args["env_step_limit"] * 2

        action_buffer = []
        obs_buffer = []
        buffer_owner = None  # None | "sage" | "critic"
        last_time_system2_steps = [-1]
        last_time_system2 = -1
        consecutive_system2 = 0
        focus_on_done = False
        useful_focus_on = []
        no_action_done = 0
        system_2_focused = False
        system_1_focused_trial = 0
        swift_failure_count = 0
        if enable_srm:
            gate_drop_streak = 0
            gate_drop_step = -1
            force_system2_once_reason = None
        else:
            gate_drop_streak = None
            gate_drop_step = None
            force_system2_once_reason = None
        action_source = None
        focus_limit = int(focus_on_count.get(str(task_num), 1))
        focus_used = 0
        sage_calls_this_env_step = 0
        sage_calls_step_marker = step
        same_state_sage_replan_streak = 0
        last_sage_state_sig = None
        disable_system2_until_step = -1
        no_reward_streak = 0
        repeat_obs_streak = 0
        repeat_action_streak = 0
        last_obs_text = ""
        last_action_norm = ""
        pattern = r"focus on\s+(\b\w+\b(\s+\b\w+\b)*)"
        matches = re.findall(pattern, task_description)
        to_focus = [match[0].replace("the ", " ").strip() for match in matches]
        logger.info(f"to_focus={to_focus}")
        srm_gate = SRMGate() if enable_srm else None
        if enable_srm:
            stagnation_detector = SRMStagnationDetector()
            critic_calls = 0
            last_critic_step = -10**9
            last_critic_state_sig = None
            last_critic_actions_hash = ""
            disable_critic_until_step = -1
            pending_critic_report = None
            pending_critic_step = -1
        else:
            stagnation_detector = None
            critic_calls = None
            last_critic_step = None
            last_critic_state_sig = None
            last_critic_actions_hash = None
            disable_critic_until_step = None
            pending_critic_report = None
            pending_critic_step = None
        failed_messages = []
        if enable_srm:
            logger.info(f"[FOCUS_LIMIT] task={task_num} used={focus_used}/{focus_limit}")
        while not done:           
            if step != sage_calls_step_marker:
                sage_calls_step_marker = step
                sage_calls_this_env_step = 0
 
            # Per-iteration flag: track if action came from buffer (breaks Swift failure streak)
            picked_from_buffer = False
            executed_buffer_owner = None

            no_action_done += 1 
            
            logger.info("-"*50+f"Variation: {variation}, Step: {step}"+"-"*50) 
            logger.info(f"[T1 Counter] Begin step={step} swift_failure_count={swift_failure_count}")
            logger.info(f"Action Buffer: {action_buffer}")
            logger.info(f"Guess Obs Buffer: {obs_buffer}")
            validActions = getFilteredValidActions(env, info["look"], task_id=task_num, task_desc=task_description)
            logger.info(f"look = \n {str(info['look'])}")
            logger.info(f"inventory = \n {str(env.inventory())}")
            # Truncate validActions logging to first 10 + count
            validActions_list = sorted(validActions) if isinstance(validActions, set) else list(validActions)
            n_total = len(validActions_list)
            first_10 = validActions_list[:10]
            if n_total > 10:
                logger.info(f"[ValidActions] n={n_total}, first10={first_10} (+{n_total-10} more)")
            else:
                logger.info(f"[ValidActions] n={n_total}, items={validActions_list}")
            action = None 
            executed = False

            add_current_place(obs, info['look'], places)
            add_current_objects(task_num, info['look'], objects, limit=20)

            current_place = get_current_room(info['look'])
            recent_looks[current_place] = info["look"]
            recent_looks_flatten.append(info["look"])
            
            # REMOVED: Wait bridge logic that caused loops
            # Wait should only be chosen by SRM (with gating) or Sage, not auto-injected
            
            # Try to use the actions from action buffer
            if action is None and len(action_buffer) > 0:
                action_source = "buffer"  # Track buffer actions
                scanned = 0
                max_buffer_scan_per_step = 5
                reorders_this_step = 0
                max_reorders_per_step = 3
                while action is None and action_buffer and scanned < max_buffer_scan_per_step:
                    scanned += 1
                    action_candidate_raw = action_buffer[0]
                    obs_candidate_raw = obs_buffer[0] if obs_buffer else ""
                    current_buffer_owner = buffer_owner

                    if action_candidate_raw.startswith("focus on") and focus_on_done:
                        logger.info(f"Removed {action_candidate_raw} from the buffer, because the focus on limit exceed")
                        action_buffer.pop(0)
                        if obs_buffer:
                            obs_buffer.pop(0)
                        if not action_buffer:
                            buffer_owner = None
                        continue

                    action_candidate = try_to_replace(action_candidate_raw, validActions, info['look'], info['inv'])
                    if action_candidate_raw != action_candidate:
                        logger.info(f"Replace {action_candidate_raw} --> {action_candidate}.")

                    # Validate only the current buffer head in current state.
                    if srm_gate is not None:
                        gate_state = {
                            "look": info["look"],
                            "inventory": str(env.inventory()),
                            "current_room": current_place,
                            "valid_actions": list(validActions),
                            "task_description": task_description,
                            "buffer_next_actions": action_buffer[1:],
                            "focus_limit": focus_limit,
                            "focus_used": focus_used,
                        }
                        gate_decision = srm_gate.pre_execute(action_candidate, source="SAGE_BUFFER", state=gate_state)
                        if gate_decision.kind != "ACCEPT" or (gate_decision.reason_codes or []):
                            logger.info(
                                f"[SRM Gate] src=SAGE_BUFFER raw='{action_candidate}' "
                                f"norm='{gate_decision.normalized_action}' decision={gate_decision.kind} "
                                f"reasons={gate_decision.reason_codes}"
                            )
                        if "FOCUS_LIMIT_EXCEEDED" in (gate_decision.reason_codes or []) or "FOCUS_LIMIT_REACHED" in (gate_decision.reason_codes or []):
                            logger.info(
                                f"[FOCUS_LIMIT] task={task_num} used={focus_used}/{focus_limit} "
                                f"source=SAGE_BUFFER action='{action_candidate}' decision={gate_decision.kind}"
                            )
                        if gate_decision.kind == "DROP_INVALID":
                            reason_codes = gate_decision.reason_codes or []
                            parsed_head = parse_action(normalize_action_text(action_candidate_raw))
                            is_focus_head = parsed_head is not None and parsed_head.verb == "focus"
                            if (
                                "FOCUS_TARGET_NOT_OBSERVED_YET" in reason_codes
                                and is_focus_head
                                and reorders_this_step < max_reorders_per_step
                            ):
                                deferred_action = action_buffer.pop(0)
                                deferred_obs = obs_buffer.pop(0) if obs_buffer else ""
                                teleport_index = None
                                for idx, pending_action in enumerate(action_buffer):
                                    parsed_pending = parse_action(normalize_action_text(pending_action))
                                    if parsed_pending is not None and parsed_pending.verb == "teleport":
                                        teleport_index = idx
                                        break
                                if teleport_index is not None:
                                    insert_idx = teleport_index + 1
                                    action_buffer.insert(insert_idx, deferred_action)
                                    if obs_buffer is not None:
                                        obs_buffer.insert(insert_idx, deferred_obs)
                                    reorders_this_step += 1
                                    logger.info(
                                        f"[SRM Buffer] Deferred focus action after next teleport: '{deferred_action}'"
                                    )
                                else:
                                    logger.info(
                                        f"[SRM Buffer] Dropped focus action (no teleport remaining): '{deferred_action}'"
                                    )
                                if not action_buffer:
                                    buffer_owner = None
                            else:
                                if is_focus_head and ("FOCUS_LIMIT_EXCEEDED" in reason_codes or "FOCUS_LIMIT_REACHED" in reason_codes):
                                    buffer_len_after = len(action_buffer) - 1 if len(action_buffer) > 0 else 0
                                    logger.info(
                                        f"[SRM Buffer] Dropped focus action due to FOCUS_LIMIT_EXCEEDED; buffer_len now = {buffer_len_after}"
                                    )
                                action_buffer.pop(0)
                                if obs_buffer:
                                    obs_buffer.pop(0)
                                if not action_buffer:
                                    buffer_owner = None
                            continue
                        if gate_decision.kind in ("REPAIR", "ACCEPT") and gate_decision.action_env:
                            action_candidate = gate_decision.action_env

                    if action_candidate in validActions:
                        action = action_candidate
                        action_source = "buffer"
                        executed_buffer_owner = current_buffer_owner
                        action_buffer.pop(0)
                        if obs_buffer:
                            obs_buffer.pop(0)
                        if not action_buffer:
                            buffer_owner = None
                        picked_from_buffer = True
                        break

                    # Optional head-only alias from guessed observation (still current state only).
                    action_candidate_v2 = (
                        obs_candidate_raw.lower()
                        if obs_candidate_raw and formalize_action(obs_candidate_raw.lower()) is not None
                        else None
                    )
                    if action_candidate_v2 and action_candidate_v2 != action_candidate:
                        if srm_gate is not None:
                            gate_state_v2 = {
                                "look": info["look"],
                                "inventory": str(env.inventory()),
                                "current_room": current_place,
                                "valid_actions": list(validActions),
                                "task_description": task_description,
                                "buffer_next_actions": action_buffer[1:],
                                "focus_limit": focus_limit,
                                "focus_used": focus_used,
                            }
                            gate_decision_v2 = srm_gate.pre_execute(action_candidate_v2, source="SAGE_BUFFER", state=gate_state_v2)
                            if gate_decision_v2.kind != "ACCEPT" or (gate_decision_v2.reason_codes or []):
                                logger.info(
                                    f"[SRM Gate] src=SAGE_BUFFER raw='{action_candidate_v2}' "
                                    f"norm='{gate_decision_v2.normalized_action}' decision={gate_decision_v2.kind} "
                                    f"reasons={gate_decision_v2.reason_codes}"
                                )
                            if gate_decision_v2.kind != "DROP_INVALID" and gate_decision_v2.action_env:
                                action_candidate_v2 = gate_decision_v2.action_env
                            elif gate_decision_v2.kind == "DROP_INVALID":
                                action_candidate_v2 = None

                        if action_candidate_v2 and action_candidate_v2 in validActions:
                            action = action_candidate_v2
                            action_source = "buffer"
                            executed_buffer_owner = current_buffer_owner
                            action_buffer.pop(0)
                            if obs_buffer:
                                obs_buffer.pop(0)
                            if not action_buffer:
                                buffer_owner = None
                            picked_from_buffer = True
                            break

                    # Head cannot execute now; drop it and try next head in this same step.
                    logger.info(f"Removed {action_candidate_raw} from the buffer (not executable in current state).")
                    action_buffer.pop(0)
                    if obs_buffer:
                        obs_buffer.pop(0)
                    if not action_buffer:
                        buffer_owner = None
                    
            # Reset Swift failure counter if action came from buffer (breaks consecutive Swift failure streak)
            if picked_from_buffer:
                swift_failure_count = 0
                logger.info("[T1 Counter] Reset swift_failure_count=0 because action came from action_buffer.")

            if action is None: 
                # Buffer-drain boundary: if a stagnation-triggered critic request is pending,
                # give it one chance before returning to Swift/Sage.
                if enable_srm and action_buffer == [] and pending_critic_report is not None:
                    pending_age = step - pending_critic_step
                    if pending_age <= 2:
                        cooldown_ok = (step - last_critic_step) >= CRITIC_COOLDOWN_STEPS and step >= disable_critic_until_step
                        calls_ok = critic_calls < MAX_CRITIC_CALLS_PER_EPISODE
                        if cooldown_ok and calls_ok:
                            logger.info(
                                f"[SRM Critic] RUN_AFTER_BUFFER_DRAIN step={step} pending_age={pending_age} "
                                f"reasons={pending_critic_report.reasons} buffer_owner={buffer_owner} buffer_len={len(action_buffer)}"
                            )
                            valid_actions_now = getFilteredValidActions(env, info["look"], task_id=task_num, task_desc=task_description)
                            valid_actions_list_all = sorted(list(valid_actions_now)) if isinstance(valid_actions_now, set) else list(valid_actions_now)
                            valid_actions_list_for_critic, removed_counts = filter_valid_actions_for_critic_with_stats(
                                valid_actions=valid_actions_list_all,
                                task_description=task_description,
                                look=info["look"],
                                inventory=str(info.get("inv", "")),
                                rooms=rooms,
                            )
                            if not valid_actions_list_for_critic and valid_actions_list_all:
                                valid_actions_list_for_critic = valid_actions_list_all
                                removed_counts = dict(removed_counts or {})
                                removed_counts["FILTER_EMPTY_FALLBACK"] = 1
                            removed_top = sorted((removed_counts or {}).items(), key=lambda x: (-x[1], x[0]))[:5]
                            logger.info(
                                f"[SRM Critic] valid_actions_filter: all={len(valid_actions_list_all)} "
                                f"kept={len(valid_actions_list_for_critic)} removed={len(valid_actions_list_all)-len(valid_actions_list_for_critic)} "
                                f"removed_top={removed_top}"
                            )
                            history_lines = []
                            hist_n = min(9, len(recent_actions))
                            for i in range(len(recent_actions) - hist_n, len(recent_actions)):
                                room_i = recent_locs[i] if i < len(recent_locs) else ""
                                act_i = recent_actions[i] if i < len(recent_actions) else ""
                                obs_i = recent_obs[i] if i < len(recent_obs) else ""
                                history_lines.append(
                                    f"room={room_i} | action={act_i} | obs={sanitizeStr(str(obs_i))[:160]}"
                                )
                            history_lines.append(f"room={current_place} | action={action} | obs={sanitizeStr(obs)[:160]}")
                            inventory_now = str(env.inventory())
                            critic_em_block = None
                            if enable_srm and enable_amm and amm_retrieval_enabled and amm_client is not None:
                                try:
                                    from amm.retrieve_for_critic import retrieve_memories_for_critic
                                    critic_em_block, critic_em_stats = retrieve_memories_for_critic(
                                        amm_client=amm_client,
                                        task_description=task_description,
                                        current_room=current_place,
                                        look=info["look"],
                                        inventory=inventory_now,
                                        recent_history_lines=history_lines,
                                        stagnation_reasons=pending_critic_report.reasons,
                                        stagnation_metrics=pending_critic_report.metrics,
                                        logger=logger,
                                        focus_used=focus_used,
                                        focus_limit=focus_limit,
                                    )
                                    if critic_em_block is not None and not str(critic_em_block).strip():
                                        critic_em_block = None
                                    logger.info(
                                        f"[SRM Critic][AMM] injected={bool(critic_em_block)} "
                                        f"worked={critic_em_stats.get('worked_retrieved', 0)} "
                                        f"avoid={critic_em_stats.get('avoid_retrieved', 0)} "
                                        f"chars={critic_em_stats.get('injected_chars', 0)} "
                                        "branch=deferred"
                                    )
                                except Exception as e:
                                    logger.warning(f"[SRM Critic][AMM] retrieval failed (branch=deferred): {e}")
                            if critic_em_block is not None:
                                logger.info(f"[SRM Critic][AMM] EPISODIC_BLOCK:\n{critic_em_block}")
                            else:
                                logger.info("[SRM Critic][AMM] EPISODIC_BLOCK: <NONE>")
                            prompt = build_critic_prompt(
                                task_description=task_description,
                                stagnation_reasons=pending_critic_report.reasons,
                                stagnation_metrics=pending_critic_report.metrics,
                                current_room=current_place,
                                look=info["look"],
                                inventory=inventory_now,
                                recent_history_lines=history_lines,
                                valid_actions=valid_actions_list_for_critic,
                                focus_used=focus_used,
                                focus_limit=focus_limit,
                                episodic_memories_block=critic_em_block,
                            )
                            has_ep_section = "EPISODIC_MEMORIES (non-binding evidence):" in prompt
                            if critic_em_block is not None:
                                logger.info(
                                    f"[SRM Critic][AMM] final_prompt_has_ep_section={has_ep_section} "
                                    f"block_chars={len(critic_em_block)} branch=deferred"
                                )
                            logger.info(f"[SRM Critic] FINAL_PROMPT:\n{prompt}")
                            logger.info(
                                f"[SRM Critic] prompt_meta: valid_actions_n={len(valid_actions_list_for_critic)}, "
                                f"history_n={len(history_lines)}, inv_chars={len(inventory_now)}, look_chars={len(info['look'])}"
                            )
                            critic_response = run_critic_once(llm, prompt, logger=logger.info)
                            logger.info(f"[SRM Critic] response_len={len(critic_response or '')} response='{critic_response}'")
                            logger.info(f"[SRM Critic] raw_response:\n{critic_response}")
                            parsed_actions = parse_critic_actions(critic_response, valid_actions=valid_actions_list_for_critic)
                            logger.info(f"[SRM Critic] parsed_actions(k={len(parsed_actions)}): {parsed_actions}")

                            state_sig = _critic_state_sig(current_place, inventory_now, obs)
                            actions_hash = _critic_actions_hash(parsed_actions)
                            recent_3_norm = {
                                normalize_action_text(a or "").strip().lower()
                                for a in (recent_actions[-3:] + [action])
                                if a
                            }
                            parsed_norm = [normalize_action_text(a or "").strip().lower() for a in parsed_actions if a]
                            all_duplicates_recent = bool(parsed_norm) and all(a in recent_3_norm for a in parsed_norm)
                            nav_only = bool(parsed_norm) and all(
                                a.startswith("go to ") or a.startswith("teleport to ") or a.startswith("open door to ")
                                for a in parsed_norm
                            )

                            skip_reason = None
                            if not parsed_actions:
                                skip_reason = "EMPTY_OUTPUT"
                            elif all_duplicates_recent or nav_only:
                                skip_reason = "LOW_QUALITY_OUTPUT"
                            elif state_sig == last_critic_state_sig and actions_hash == last_critic_actions_hash:
                                skip_reason = "SAME_STATE_SAME_ACTIONS"

                            pending_critic_report = None
                            pending_critic_step = -1
                            if skip_reason is not None:
                                logger.info(f"[SRM Critic] SKIP reason={skip_reason}")
                                logger.info(f"[SRM Critic] CLEAR_PENDING reason={skip_reason}")
                                if skip_reason == "SAME_STATE_SAME_ACTIONS":
                                    disable_critic_until_step = max(disable_critic_until_step, step + CRITIC_BACKOFF_STEPS)
                            else:
                                action_buffer = parsed_actions
                                obs_buffer = ["None"] * len(parsed_actions)
                                buffer_owner = "critic"
                                critic_calls += 1
                                last_critic_step = step
                                last_critic_state_sig = state_sig
                                last_critic_actions_hash = actions_hash
                                logger.info(
                                    f"[SRM Critic] injected_into_buffer: added={len(parsed_actions)} new_buffer_len={len(action_buffer)}"
                                )
                                logger.info("[SRM Critic] CLEAR_PENDING reason=RUN_AFTER_BUFFER_DRAIN")
                                continue
                        else:
                            if not calls_ok:
                                logger.info("[SRM Critic] SKIP reason=MAX_CALLS")
                            elif not cooldown_ok:
                                logger.info("[SRM Critic] SKIP reason=COOLDOWN")
                    else:
                        logger.info(
                            f"[SRM Critic] CLEAR_PENDING reason=STALE_PENDING pending_age={pending_age}"
                        )
                        pending_critic_report = None
                        pending_critic_step = -1

                if action_buffer:
                    if enable_srm:
                        logger.info(
                            f"[SRM Buffer] CLEAR reason=BUFFER_NOT_USEFUL remaining_len={len(action_buffer)} owner={buffer_owner}"
                        )
                    action_buffer = []
                    obs_buffer = []
                    buffer_owner = None
                elif buffer_owner is not None:
                    buffer_owner = None
                logger.info("Buffer is not useful. Switch to Fast Agent.")
                input_str = ""

                # Note that the agent is allowed to know the score changes.
                returns_to_go = 1.0 - float(info['score']) * 0.01
                returns_to_go = round(returns_to_go, 2)
                

                mode = args["mode"]
                logger.info("Mode: " + mode)
                
                clean_recent_actions, clean_recent_obs, clean_recent_scores, clean_recent_reward, _ = \
                    clean_history(recent_actions, recent_obs, recent_scores, recent_reward, recent_locs)
                #Creates the input string for model
                input_str, _ = compose_instance(mode=mode, step_id=step+1, task_desc=task_description, returns_to_go=returns_to_go,
                                        curr_action=None, curr_obs=obs, inventory=info['inv'], look=info['look'], 
                                        prev_action=prev_action, prev_obs=prev_obs, objects=objects, places=places, 
                                        recent_actions=clean_recent_actions, recent_obs=clean_recent_obs, 
                                        recent_scores=clean_recent_scores, recent_reward=clean_recent_reward) 
                
                
                ############
                prev_obs = obs 

                # Get valid actions at this point
                # Heuristic to change systems
                # Initialize ran_swift_this_step to track if Swift actually ran (for counter updates)
                ran_swift_this_step = False
                # Track reason for forcing System 2 (to bypass T1 when appropriate)
                force_system_2_reason = None
                if args["slow_agent"]:                    
                    force_system_2 = False 
                    force_system_1 = False 
                    force_system_2_reason = None
                    # If system 1 is stuck (no action done for 2 steps or two failed actions) switch to system 2
                    if no_action_done >= 2 or len(failed_messages) >= 2:
                        force_system_1 = False
                        force_system_2 = True            
                        force_system_2_reason = "stuck_or_failed"
                        logger.info("Force to do force_system_2")
                    # If system 1 already focused on something and system 2 did not, switch to system 2
                    if (not enable_srm) and (not system_2_focused and system_1_focused_trial >= 1):
                        force_system_1 = False
                        force_system_2 = True            
                        force_system_2_reason = "focus_gate"  # Focus gating forces System 2
                        logger.info("Force to do force_system_2")
                    # If system 2 has been used for 2 steps, switch to system 1
                    if consecutive_system2 >= 2:
                        force_system_1 = True
                        force_system_2 = False
                        force_system_2_reason = None
                        logger.info("Force to do force_system_1")
                    if step < disable_system2_until_step:
                        force_system_2 = False
                        force_system_2_reason = None
                    if enable_srm and force_system2_once_reason == "gate_drop_loop":
                        force_system_1 = False
                        force_system_2 = True
                        force_system_2_reason = "gate_drop_loop"
                        force_system2_once_reason = None
                        logger.info(f"[SRM LoopBreak] forcing System2 reason=gate_drop_loop step={step}")
                    # === BASELINE SWIFT PATH (when slow_agent=False) ===
                    if not args.get("slow_agent", False):
                        # Baseline Swift-only mode: no Sage, no AMM retrieval
                        input_str = sanitizeStr(input_str)
                        predStrs = get_model_output(args, input_str, tokenizer, lm_model, device, logger)
                        ran_swift_this_step = True  # Swift ran in baseline path
                        
                        # Paper-aligned Swift selection: Top-1 only
                        validActions = getFilteredValidActions(env, info['look'], task_id=task_num, task_desc=task_description)
                        action = try_to_replace(predStrs[0], validActions, info['look'], info['inv']).strip() if predStrs else None
                        
                        # Baseline validity check (Top-1 only)
                        found_valid_in_top = False
                        for pred in predStrs[:1]:
                            if pred.strip() in validActions:
                                found_valid_in_top = True
                                break
                        
                        used_sys2 = False
                        return_result = action
                        
                    # === ARCHITECTURE PATH (when slow_agent=True, Sage always available) ===
                    else:
                        # Use Sage agent (define use_memory_planning here for both Swift and Sage)
                        use_memory_planning = args.get("use_memory_planning", True) and enable_amm and (not amm_write_only)

                        # Determine if we should run Swift this step
                        # Run Swift if NOT forcing System 2, OR if forcing System 1
                        ran_swift_this_step = False
                        predStrs = []  # Initialize predStrs
                        if not force_system_2 or force_system_1:
                            input_str = sanitizeStr(input_str)
                            # Invokes Swift, return top predicted actions (baseline SwiftSage - no EM retrieval here)
                            # EM retrieval only happens AFTER Swift fails (T1 trigger in findValidActionWithSystem2)
                            predStrs = get_model_output(args, input_str, tokenizer, lm_model, device, logger)
                            ran_swift_this_step = True
                        else:
                            # Forcing System 2: skip Swift, set empty predictions
                            predStrs = []
                            ran_swift_this_step = False
                    
                        # Always call findValidActionWithSystem2 (it handles both Swift and Sage paths)
                        # This ensures found_valid_in_top is always defined
                        cycles_without_progress_val = wm.cycles_without_progress if (enable_amm and wm is not None) else 0
                    if sage_calls_this_env_step >= MAX_SAGE_CALLS_PER_ENV_STEP:
                        fallback_action = deterministic_fallback_action(validActions, current_place)
                        logger.info(
                            f"[LoopGuard] Sage call cap hit at step={step}. "
                            f"Using deterministic fallback action='{fallback_action}'."
                        )
                        used_sys2 = False
                        return_result = fallback_action
                        found_valid_in_top = bool(fallback_action and fallback_action in validActions)
                        action_source = "swift"
                    else:
                        used_sys2, return_result, found_valid_in_top, action_source = findValidActionWithSystem2(
                            predStrs, env, task_num, task_description, info['look'],
                            recent_actions, recent_reward, recent_obs, recent_locs, recent_looks, failed_messages,
                            demo_data, logger, sbert_model, step, last_time_system2_steps,
                            useful_focus_on, focus_on_done, force_system_1, force_system_2,
                            gpt_version, llm=llm,
                            episodic_memories=None,  # AMM will handle memory retrieval in Phase 2
                            use_memory_planning=use_memory_planning,
                            # In WRITE_ONLY mode, disable retrieval paths inside findValidActionWithSystem2
                            # by passing amm_client=None while keeping local AMM client for post-step writes.
                            amm_client=amm_client if amm_retrieval_enabled else None,
                            current_score=score,  # Current score for retrieval query
                            recent_scores=recent_scores,  # Recent scores for retrieval query
                            swift_failure_count=swift_failure_count,  # Pass swift_failure_count for T1 escalation
                                cycles_without_progress=cycles_without_progress_val,  # Pass cycles_without_progress for T2 escalation
                                force_system_2_reason=force_system_2_reason,  # Pass reason for forcing System 2 (to bypass T1 if focus_gate)
                            # Parameters for second Swift pass (T1-S2 retry)
                            args=args,
                            tokenizer=tokenizer,
                            lm_model=lm_model,
                            device=device,
                            compose_instance=compose_instance,
                            prev_action=prev_action,
                            prev_obs=prev_obs,
                            objects=objects,
                            places=places,
                            srm_gate=srm_gate
                        )
                    
                    # Track if action was filtered by focus gating (before counter update)
                    swift_action_filtered = False
                    # Track action source and update counters
                    if not used_sys2:
                        action = return_result
                        action_source = action_source if action_source else "swift"  # Default to swift if not set
                        consecutive_system2 = 0
                        same_state_sage_replan_streak = 0
                        
                        # Check if this is a focus action that will be filtered by gating
                        if (not enable_srm) and action and action.startswith("focus on") and not system_2_focused:
                            # Check if focus gating would reject this action
                            if system_1_focused_trial < 3 and not any([clean_obj_name(tf) in clean_obj_name(action) for tf in to_focus]):
                                swift_action_filtered = True
                                logger.info(f"[FocusGate] Swift proposed valid focus action but it is gated. trial={system_1_focused_trial}, matches_required={any([clean_obj_name(tf) in clean_obj_name(action) for tf in to_focus])}, decision=SKIP")
                                # No Top-2+ fallback in paper-aligned Top-1 mode.
                                logger.info("[FocusGate] Top-1 focus action gated. No alternative Swift candidate will be tried in this step.")
                    
                    # Update swift_failure_count AFTER checking for focus filtering
                    # Only update counter when Swift is actually being used (not forced System 2) and architecture is enabled
                    if use_sage and args.get("slow_agent", False):
                        if ran_swift_this_step and not force_system_2:
                            # Swift ran this step and we're not forcing System 2: update counter based on Swift result
                            if swift_action_filtered:
                                # Valid action but filtered by focus gating: do NOT increment counter
                                logger.info(f"[T1 Counter] No increment (reason=valid_but_filtered_focus_gate)")
                            elif not found_valid_in_top:
                                # True Swift failure: increment counter
                                swift_failure_count += 1
                                logger.info(f"[T1 Counter] Increment swift_failure_count -> {swift_failure_count} (reason=true_swift_failure)")
                            else:
                                # Swift succeeded: reset counter
                                # This includes both baseline Swift success and T1 retry success
                                swift_failure_count = 0
                                logger.info(f"[T1 Counter] Reset -> 0 (reason=swift_valid_action_executed)")
                        elif force_system_2:
                            # When forced to System 2, Swift failure streak should not accumulate
                            swift_failure_count = 0
                            logger.info("[T1 Counter] Reset swift_failure_count=0 because force_system_2=True (Swift not expected to operate).")
                        # If ran_swift_this_step=False and not force_system_2, counter remains unchanged (shouldn't happen in normal flow)
                    
                    if not used_sys2:
                        # Already handled above
                        pass
                    else:
                        action = None 
                        action_source = "sage"  # System 2 was used
                        sage_calls_this_env_step += 1
                        sage_state_sig = (
                            current_place or "",
                            str(info.get("look", ""))[:200],
                            str(info.get("inv", ""))[:80],
                        )
                        if sage_state_sig == last_sage_state_sig:
                            same_state_sage_replan_streak += 1
                        else:
                            same_state_sage_replan_streak = 1
                            last_sage_state_sig = sage_state_sig
                        action_buffer = return_result[0] # reset the buffer 
                        obs_buffer = return_result[1]
                        buffer_owner = "sage" if action_buffer else None
                        failed_messages = [] # reset the failed messages 
                        logger.info(f"action_buffer reset by the Slow Agent") 
                        last_time_system2 = step  
                        last_time_system2_steps.append(step)
                        consecutive_system2 += 1
                        # System 2 takeover breaks consecutive Swift failure streak
                        if use_sage:
                            swift_failure_count = 0
                            logger.info("[T1 Counter] Reset swift_failure_count=0 because System 2 was used.")
                        continue 
                        
                    # REMOVED: Invalid action wait bridge - this caused loops
                    # If action is invalid, we should have caught it earlier or escalate to Sage
                    # No automatic wait injection
                    
                    if action is None: 
                        continue 
                else:
                    # Use Swift Agent only
                    input_str = sanitizeStr(input_str)
                    logger.info("InputStr: " + input_str)
                    predStrs = get_model_output(args, input_str, tokenizer, lm_model, device, logger)
                    ran_swift_this_step = True  # Swift ran in Swift-only path
                    
                    # Check if any top prediction is valid (for swift_failure_count tracking)
                    validActions = getFilteredValidActions(env, info['look'], task_id=task_num, task_desc=task_description)
                    found_valid_in_top = False
                    for pred in predStrs[:1]:  # Check top prediction only (matching findValidActionWithSystem2 logic)
                        if pred.strip() in validActions:
                            found_valid_in_top = True
                            break
                    
                    # Update swift_failure_count based on found_valid_in_top (T1 tracks model-level invalidity)
                    if not found_valid_in_top:
                        swift_failure_count += 1
                    else:
                        swift_failure_count = 0
                    
                    action = try_to_replace(predStrs[0], validActions, info['look'], info['inv']).strip() if predStrs else None
            
 

            # Focus action was already executed (legacy path only when SRM gate disabled)
            if (not enable_srm) and action.startswith("focus on") and focus_on_done:
                logger.info(f"You have already done great focus-on action: {useful_focus_on}. Skipping this [{action}]")
                # Reset counter since this is a valid action (just already done)
                if use_sage and args.get("slow_agent", False):
                    swift_failure_count = 0
                    logger.info("[T1 Counter] Reset -> 0 (reason=valid_action_already_done)")
                continue 
            
            # Sage handled the focus on action, and we mark the flag to prevent it from trying the same subgoal
            if (not enable_srm) and action.startswith("focus on") and consecutive_system2 > 0:
                system_2_focused = True
                
            # Swift is trying to focus on, system 2 hasnt proposed focus yet
            if (not enable_srm) and action.startswith("focus on") and not system_2_focused:
                # track how many times swift tries to focus
                system_1_focused_trial += 1
                # only after 3 attempts or if the obeject its focusing on matches the task-relevant focus targets we allow it
                if system_1_focused_trial >= 3 or any([clean_obj_name(tf) in clean_obj_name(action) for tf in to_focus]):
                    logger.info(f"You have never used System 2 to focus on... but system_1 has tried multiple times... so okay with [{action}]")
                    # Valid focus action accepted: reset counter (already done in counter update logic above)
                # otherwise skip action
                else:
                    logger.info(f"[FocusGate] Swift proposed valid focus action but it is gated. trial={system_1_focused_trial}, matches_required={any([clean_obj_name(tf) in clean_obj_name(action) for tf in to_focus])}, decision=SKIP")
                    # Valid action but filtered: counter was already handled in counter update logic above
                    # Reset force flags for next iteration to avoid sticky state
                    if use_sage and args.get("slow_agent", False):
                        # Don't increment counter (already handled above), but ensure next iteration starts fresh
                        # The next iteration will check system_1_focused_trial and may force System 2, but that's intentional
                        pass
                    continue 
            
            # SRM Gate-1 (Milestone-1): final pre-execution validation for Swift/Buffer/Sage-produced action
            if action is not None and not executed and srm_gate is not None and not (action_source or "").startswith("buffer"):
                action_before_gate = action
                action = try_to_replace(action, validActions, info["look"], info.get("inv", env.inventory()))
                if action_before_gate != action:
                    logger.info(
                        f"[SRM PreGateRepair] src={(action_source or 'swift').upper()} '{action_before_gate}' -> '{action}'"
                    )
                gate_state_final = {
                    "look": info["look"],
                    "inventory": str(env.inventory()),
                    "current_room": current_place,
                    "valid_actions": list(validActions),
                    "task_description": task_description,
                    "swift_predictions": predStrs if (action_source or "swift") == "swift" else [],
                    "focus_limit": focus_limit,
                    "focus_used": focus_used,
                }
                gate_decision_final = srm_gate.pre_execute(
                    action,
                    source=(action_source or "swift").upper(),
                    state=gate_state_final,
                )
                log_swift_gate = (action_source or "swift") == "swift"
                if log_swift_gate or gate_decision_final.kind != "ACCEPT" or (gate_decision_final.reason_codes or []):
                    logger.info(
                        f"[SRM Gate] src={(action_source or 'swift').upper()} raw='{action}' "
                        f"norm='{gate_decision_final.normalized_action}' decision={gate_decision_final.kind} "
                        f"reasons={gate_decision_final.reason_codes}"
                    )
                if "FOCUS_LIMIT_EXCEEDED" in (gate_decision_final.reason_codes or []) or "FOCUS_LIMIT_REACHED" in (gate_decision_final.reason_codes or []):
                    logger.info(
                        f"[FOCUS_LIMIT] task={task_num} used={focus_used}/{focus_limit} "
                        f"source={(action_source or 'swift').upper()} action='{action}' decision={gate_decision_final.kind}"
                    )
                if gate_decision_final.kind in ("REPAIR", "ACCEPT") and gate_decision_final.action_env:
                    action = gate_decision_final.action_env
                elif gate_decision_final.kind == "DROP_INVALID":
                    if gate_drop_step != step:
                        gate_drop_step = step
                        gate_drop_streak = 0
                    gate_drop_streak += 1
                    logger.info(
                        f"[SRM LoopBreak] gate_drop step={step} streak={gate_drop_streak} "
                        f"action='{action}' reasons={gate_decision_final.reason_codes}"
                    )
                    if action_source == "swift":
                        swift_failure_count += 1
                        logger.info("[T1 Counter] gate_drop_no_step -> increment (reason=gate_drop)")
                        failed_messages.append(
                            f"\t\t Failed action: (in {current_place}) [{action}] --> GATE_INVALID:{','.join(gate_decision_final.reason_codes)}"
                        )
                    if gate_drop_streak >= 2:
                        can_force_system2 = args.get("slow_agent", False) and sage_calls_this_env_step < MAX_SAGE_CALLS_PER_ENV_STEP
                        if can_force_system2:
                            force_system2_once_reason = "gate_drop_loop"
                            logger.info(f"[SRM LoopBreak] forcing System2 reason=gate_drop_loop step={step}")
                            # Top-1 only: if gate rejects, do not try Top-2+ in this step.
                            logger.info("[SRM Gate] Action dropped/skipped before env.step.")
                            continue

                        fallback_candidates = []
                        for cand in ("look around", "inventory", "wait"):
                            if cand in validActions:
                                fallback_candidates.append(cand)
                        if not fallback_candidates:
                            fallback_pick = deterministic_fallback_action(validActions, current_place)
                            if fallback_pick:
                                fallback_candidates.append(fallback_pick)

                        fallback_exec = None
                        for cand in fallback_candidates:
                            cand_decision = srm_gate.pre_execute(
                                cand,
                                source="LOOPBREAK_FALLBACK",
                                state=gate_state_final,
                            )
                            if cand_decision.kind in ("REPAIR", "ACCEPT") and cand_decision.action_env:
                                fallback_exec = cand_decision.action_env
                                break

                        if fallback_exec is not None:
                            action = fallback_exec
                            action_source = "loopbreak_fallback"
                            logger.info(f"[SRM LoopBreak] fallback_exec action='{action}' step={step}")
                        else:
                            logger.info("[SRM Gate] Action dropped/skipped before env.step.")
                            continue
                    else:
                        # Top-1 only: if gate rejects, do not try Top-2+ in this step.
                        logger.info("[SRM Gate] Action dropped/skipped before env.step.")
                        continue
            
            # If the action was not already executed in the previous loop, execute it
            if not executed:
                obs, reward_env, done, info = env.step(action)
                if enable_srm:
                    gate_drop_streak = 0
                    gate_drop_step = -1
                if enable_srm and (action or "").lower().startswith("focus on "):
                    focus_used += 1
                    logger.info(
                        f"[FOCUS_LIMIT] task={task_num} used={focus_used}/{focus_limit} "
                        f"source={action_source or 'unknown'} action='{action}' decision=EXECUTED"
                    )
                # Log env step ground truth
                current_place_after_step = get_current_room(info['look'])
                executed_from_buffer = bool((action_source or "").startswith("buffer"))
                source_for_log = action_source or "unknown"
                if executed_from_buffer:
                    source_for_log = f"buffer(owner={executed_buffer_owner or 'unknown'})"
                logger.info(f"[ENV_STEP] source={source_for_log} executed_from_buffer={executed_from_buffer} action='{action}' reward={reward_env} score={info['score']} done={done} room={current_place_after_step} obs={obs[:160]}")
            
            # Handle ambiguous requests (resolve by choosing "0")
            if obs.startswith("Ambiguous request"):
                if srm_gate is not None:
                    gate_state_amb2 = {
                        "look": info["look"],
                        "inventory": str(env.inventory()),
                        "current_room": current_place,
                        "valid_actions": list(validActions),
                        "task_description": task_description,
                        "focus_limit": focus_limit,
                        "focus_used": focus_used,
                    }
                    gate_decision_amb2 = srm_gate.pre_execute("0", source=(action_source or "swift").upper(), state=gate_state_amb2)
                    if gate_decision_amb2.kind != "ACCEPT" or (gate_decision_amb2.reason_codes or []):
                        logger.info(
                            f"[SRM Gate] src={(action_source or 'swift').upper()} raw='0' "
                            f"norm='{gate_decision_amb2.normalized_action}' decision={gate_decision_amb2.kind} "
                            f"reasons={gate_decision_amb2.reason_codes}"
                        )
                    if gate_decision_amb2.kind in ("REPAIR", "ACCEPT") and gate_decision_amb2.action_env:
                        obs, reward_env, done, info = env.step(gate_decision_amb2.action_env)
                    else:
                        continue
                else:
                    obs, reward_env, done, info = env.step("0")
                # Log ambiguous resolution step
                current_place_after_resolve = get_current_room(info['look'])
                source_for_log = action_source or "unknown"
                if (action_source or "").startswith("buffer"):
                    source_for_log = f"buffer(owner={executed_buffer_owner or 'unknown'})"
                logger.info(f"[ENV_STEP] source={source_for_log} executed_from_buffer=False action='0' reward={reward_env} score={info['score']} done={done} room={current_place_after_resolve} obs={obs[:160]}")
            
            # Capture TRUE values from environment immediately after step
            # Reward is 0 if score doesn't increase (no negative rewards)
            # Score remains at last_score if it doesn't increase or goes negative
            score_from_env = info['score']
            if score_from_env <= last_score or score_from_env < 0:
                # Score didn't increase or went negative - reward is 0, score unchanged
                score_true = last_score
                reward_true = 0.0
            else:
                # Score increased - calculate reward as delta
                score_true = score_from_env
                reward_true = score_true - last_score
            
            # Update current place after step (may have changed)
            current_place = get_current_room(info['look'])
            if enable_srm and stagnation_detector is not None:
                stagnation_report = stagnation_detector.update(
                    step=step,
                    action=action,
                    obs=obs,
                    room=current_place,
                    inventory_text=env.inventory(),
                    score=score_true,
                )
                if stagnation_report is not None and stagnation_report.is_stagnated:
                    logger.info(
                        f"[SRM Stagnation] step={step} reasons={stagnation_report.reasons} "
                        f"room={current_place} score={score_true} "
                        f"action='{stagnation_report.last_action}' "
                        f"metrics={stagnation_report.metrics} "
                        f"obs_sig='{stagnation_report.last_obs_sig}'"
                    )
                    buffer_busy = bool(action_buffer) or (buffer_owner is not None)
                    if buffer_busy:
                        pending_critic_report = stagnation_report
                        pending_critic_step = step
                        logger.info(
                            f"[SRM Critic] PENDING step={step} reasons={stagnation_report.reasons} "
                            f"buffer_owner={buffer_owner} buffer_len={len(action_buffer)}"
                        )
                    else:
                        cooldown_ok = (step - last_critic_step) >= CRITIC_COOLDOWN_STEPS and step >= disable_critic_until_step
                        calls_ok = critic_calls < MAX_CRITIC_CALLS_PER_EPISODE
                        if cooldown_ok and calls_ok:
                            logger.info(
                                f"[SRM Critic] TRIGGER step={step} reasons={stagnation_report.reasons} "
                                f"room={current_place} score={score_true} cooldown_ok={cooldown_ok} calls={critic_calls}"
                            )
                            valid_actions_now = getFilteredValidActions(env, info["look"], task_id=task_num, task_desc=task_description)
                            valid_actions_list_all = sorted(list(valid_actions_now)) if isinstance(valid_actions_now, set) else list(valid_actions_now)
                            valid_actions_list_for_critic, removed_counts = filter_valid_actions_for_critic_with_stats(
                                valid_actions=valid_actions_list_all,
                                task_description=task_description,
                                look=info["look"],
                                inventory=str(info.get("inv", "")),
                                rooms=rooms,
                            )
                            if not valid_actions_list_for_critic and valid_actions_list_all:
                                valid_actions_list_for_critic = valid_actions_list_all
                                removed_counts = dict(removed_counts or {})
                                removed_counts["FILTER_EMPTY_FALLBACK"] = 1
                            removed_top = sorted((removed_counts or {}).items(), key=lambda x: (-x[1], x[0]))[:5]
                            logger.info(
                                f"[SRM Critic] valid_actions_filter: all={len(valid_actions_list_all)} "
                                f"kept={len(valid_actions_list_for_critic)} removed={len(valid_actions_list_all)-len(valid_actions_list_for_critic)} "
                                f"removed_top={removed_top}"
                            )
                            history_lines = []
                            hist_n = min(9, len(recent_actions))
                            for i in range(len(recent_actions) - hist_n, len(recent_actions)):
                                room_i = recent_locs[i] if i < len(recent_locs) else ""
                                act_i = recent_actions[i] if i < len(recent_actions) else ""
                                obs_i = recent_obs[i] if i < len(recent_obs) else ""
                                history_lines.append(
                                    f"room={room_i} | action={act_i} | obs={sanitizeStr(str(obs_i))[:160]}"
                                )
                            history_lines.append(f"room={current_place} | action={action} | obs={sanitizeStr(obs)[:160]}")
                            inventory_now = str(env.inventory())
                            critic_em_block = None
                            if enable_srm and enable_amm and amm_retrieval_enabled and amm_client is not None:
                                try:
                                    from amm.retrieve_for_critic import retrieve_memories_for_critic
                                    critic_em_block, critic_em_stats = retrieve_memories_for_critic(
                                        amm_client=amm_client,
                                        task_description=task_description,
                                        current_room=current_place,
                                        look=info["look"],
                                        inventory=inventory_now,
                                        recent_history_lines=history_lines,
                                        stagnation_reasons=stagnation_report.reasons,
                                        stagnation_metrics=stagnation_report.metrics,
                                        logger=logger,
                                        focus_used=focus_used,
                                        focus_limit=focus_limit,
                                    )
                                    if critic_em_block is not None and not str(critic_em_block).strip():
                                        critic_em_block = None
                                    logger.info(
                                        f"[SRM Critic][AMM] injected={bool(critic_em_block)} "
                                        f"worked={critic_em_stats.get('worked_retrieved', 0)} "
                                        f"avoid={critic_em_stats.get('avoid_retrieved', 0)} "
                                        f"chars={critic_em_stats.get('injected_chars', 0)} "
                                        "branch=immediate"
                                    )
                                except Exception as e:
                                    logger.warning(f"[SRM Critic][AMM] retrieval failed (branch=immediate): {e}")
                            if critic_em_block is not None:
                                logger.info(f"[SRM Critic][AMM] EPISODIC_BLOCK:\n{critic_em_block}")
                            else:
                                logger.info("[SRM Critic][AMM] EPISODIC_BLOCK: <NONE>")
                            prompt = build_critic_prompt(
                                task_description=task_description,
                                stagnation_reasons=stagnation_report.reasons,
                                stagnation_metrics=stagnation_report.metrics,
                                current_room=current_place,
                                look=info["look"],
                                inventory=inventory_now,
                                recent_history_lines=history_lines,
                                valid_actions=valid_actions_list_for_critic,
                                focus_used=focus_used,
                                focus_limit=focus_limit,
                                episodic_memories_block=critic_em_block,
                            )
                            has_ep_section = "EPISODIC_MEMORIES (non-binding evidence):" in prompt
                            if critic_em_block is not None:
                                logger.info(
                                    f"[SRM Critic][AMM] final_prompt_has_ep_section={has_ep_section} "
                                    f"block_chars={len(critic_em_block)} branch=immediate"
                                )
                            logger.info(f"[SRM Critic] FINAL_PROMPT:\n{prompt}")
                            logger.info(
                                f"[SRM Critic] prompt_meta: valid_actions_n={len(valid_actions_list_for_critic)}, "
                                f"history_n={len(history_lines)}, inv_chars={len(inventory_now)}, look_chars={len(info['look'])}"
                            )
                            critic_response = run_critic_once(llm, prompt, logger=logger.info)
                            logger.info(f"[SRM Critic] response_len={len(critic_response or '')} response='{critic_response}'")
                            logger.info(f"[SRM Critic] raw_response:\n{critic_response}")
                            parsed_actions = parse_critic_actions(critic_response, valid_actions=valid_actions_list_for_critic)
                            logger.info(f"[SRM Critic] parsed_actions(k={len(parsed_actions)}): {parsed_actions}")

                            state_sig = _critic_state_sig(current_place, inventory_now, obs)
                            actions_hash = _critic_actions_hash(parsed_actions)
                            recent_3_norm = {
                                normalize_action_text(a or "").strip().lower()
                                for a in (recent_actions[-3:] + [action])
                                if a
                            }
                            parsed_norm = [normalize_action_text(a or "").strip().lower() for a in parsed_actions if a]
                            all_duplicates_recent = bool(parsed_norm) and all(a in recent_3_norm for a in parsed_norm)
                            nav_only = bool(parsed_norm) and all(
                                a.startswith("go to ") or a.startswith("teleport to ") or a.startswith("open door to ")
                                for a in parsed_norm
                            )

                            skip_reason = None
                            if not parsed_actions:
                                skip_reason = "EMPTY_OUTPUT"
                            elif all_duplicates_recent or nav_only:
                                skip_reason = "LOW_QUALITY_OUTPUT"
                            elif state_sig == last_critic_state_sig and actions_hash == last_critic_actions_hash:
                                skip_reason = "SAME_STATE_SAME_ACTIONS"

                            if skip_reason is not None:
                                logger.info(f"[SRM Critic] SKIP reason={skip_reason}")
                                if skip_reason == "SAME_STATE_SAME_ACTIONS":
                                    disable_critic_until_step = max(disable_critic_until_step, step + CRITIC_BACKOFF_STEPS)
                            else:
                                action_buffer = parsed_actions
                                obs_buffer = ["None"] * len(parsed_actions)
                                buffer_owner = "critic"
                                critic_calls += 1
                                last_critic_step = step
                                last_critic_state_sig = state_sig
                                last_critic_actions_hash = actions_hash
                                logger.info(
                                    f"[SRM Critic] injected_into_buffer: added={len(parsed_actions)} new_buffer_len={len(action_buffer)}"
                                )
                        else:
                            if not calls_ok:
                                logger.info("[SRM Critic] SKIP reason=MAX_CALLS")
                            elif not cooldown_ok:
                                logger.info("[SRM Critic] SKIP reason=COOLDOWN")
            
            # Update tracking lists with TRUE values
            no_action_done = 0
            prev_action = action

            if enable_srm and reward_true > 0 and pending_critic_report is not None:
                logger.info(f"[SRM Critic] CLEAR_PENDING reason=PROGRESS reward={reward_true}")
                pending_critic_report = None
                pending_critic_step = -1

            # Shared stuck detector state (baseline/SRM/AMM fairness)
            no_reward_streak = (no_reward_streak + 1) if reward_true <= 0 else 0
            obs_norm = sanitizeStr(obs)
            action_norm = (action or "").strip().lower()
            repeat_obs_streak = (repeat_obs_streak + 1) if obs_norm == last_obs_text else 1
            repeat_action_streak = (repeat_action_streak + 1) if action_norm == last_action_norm else 1
            last_obs_text = obs_norm
            last_action_norm = action_norm

            thermometer_wait_loop = (
                repeat_action_streak >= STUCK_K
                and repeat_obs_streak >= STUCK_K
                and (
                    action_norm.startswith("wait")
                    or action_norm.startswith("use thermometer on")
                )
            )
            same_state_sage_stuck = (same_state_sage_replan_streak >= 2 and (action_source == "sage"))
            stuck_triggered = (
                (repeat_obs_streak >= STUCK_K and no_reward_streak >= STUCK_K)
                or same_state_sage_stuck
                or thermometer_wait_loop
            )
            if stuck_triggered:
                logger.info(
                    f"[LoopGuard] Stuck detected at step={step} "
                    f"(no_reward_streak={no_reward_streak}, repeat_obs_streak={repeat_obs_streak}, "
                    f"repeat_action_streak={repeat_action_streak}, same_state_sage_replan_streak={same_state_sage_replan_streak})."
                )
                action_buffer = []
                obs_buffer = []
                disable_system2_until_step = max(disable_system2_until_step, step + SAGE_COOLDOWN_STEPS)
                same_state_sage_replan_streak = 0
            
            recent_reward.append(reward_true/100)
            # Note: swift_failure_count is now tracked based on found_valid_in_top (model-level invalidity)
            # in findValidActionWithSystem2, not based on reward. Reward-based tracking is handled
            # separately via cycles_without_progress for T2 (stagnation) trigger.
            recent_scores.append(score_true/100)
            recent_actions.append(action) 
            recent_obs.append(obs)
            recent_locs.append(current_place)
            
            # Log step result
            step_source = action_source or "unknown"
            if (action_source or "").startswith("buffer"):
                step_source = f"buffer(owner={executed_buffer_owner or 'unknown'})"
            logger.info(f"[StepResult] source={step_source} action={action} reward={reward_true} score={score_true}")
            
            # === AMM HOOK: POST-STEP WRITE (BEFORE any score modifications) ===
            # Write memory with TRUE reward/score values from environment (only when use_amm=True)
            if enable_amm and amm_client is not None and wm is not None:
                try:
                    from amm.writer import write_success, write_nearmiss, write_avoidance, create_memory_record
                    from amm.config import DEFAULT_CONFIG
                    from amm.tagging import classify_episode
                    
                    # Update working memory
                    wm.record_action(action)
                    wm.update_room(current_place)
                    wm.update_inventory(env.inventory())
                    
                    # Build goal signature
                    goal_sig = task_description
                    
                    # Build rich context metadata with TRUE values
                    inventory_str = getattr(wm, "inventory_text", None) or str(env.inventory())
                    ctx_meta = {
                        "room": current_place,
                        "inventory_text": inventory_str,
                        "look": info['look'],  # Room description/look string
                        "recent_actions": recent_actions[-5:] if len(recent_actions) > 5 else recent_actions,
                        "recent_obs": [o[:100] for o in recent_obs[-5:]] if len(recent_obs) > 5 else [o[:100] for o in recent_obs],
                        "reward": reward_true,  # TRUE reward from environment
                        "score_prev": last_score,  # Score before this step
                        "score_curr": score_true,  # TRUE score from environment (not modified)
                        "done": bool(done),  # TRUE done flag from environment
                        "focus_targets": to_focus,
                    }
                    
                    # Create memory record with rich context
                    rec = create_memory_record(
                        goal_signature=goal_sig,
                        action_text=action,
                        obs_text=obs,
                        meta=ctx_meta
                    )

                    # Classify episode using the new tagging system
                    # This determines both primary tag and subtag, and which writer to call
                    try:
                        result = classify_episode(
                            action=action,
                            observation=obs,
                            reward=reward_true,  # TRUE reward
                            score_prev=last_score,
                            score_curr=score_true,  # TRUE score
                            done=done,  # TRUE done flag
                            goal_text=goal_sig,
                            milestone_threshold=DEFAULT_CONFIG.MILESTONE_THRESHOLD,
                            small_reward_threshold=DEFAULT_CONFIG.SMALL_REWARD_THRESHOLD,
                            shaping_actions=DEFAULT_CONFIG.SHAPING_ACTIONS
                        )
                        
                        # Skip writing if non-eventful or unclassifiable
                        if result is None:
                            wm.increment_cycles_without_progress()
                        else:
                            primary, subtag = result
                            
                            # Call appropriate writer based on primary tag
                            # The writer will handle embedding tags into content
                            if primary == "episodic_success":
                                write_success(amm_client, rec, meta=ctx_meta)
                                wm.reset_cycles_without_progress()
                            elif primary == "episodic_nearmiss":
                                write_nearmiss(amm_client, rec, meta=ctx_meta)
                                wm.reset_cycles_without_progress()
                            elif primary == "avoidance":
                                write_avoidance(amm_client, rec, meta=ctx_meta)
                                wm.increment_cycles_without_progress()
                            else:
                                # Fallback: should not happen, but handle gracefully
                                logger.warning(f"[AMM] Unknown primary tag: {primary}, skipping memory write")
                                wm.increment_cycles_without_progress()
                            
                    except Exception as e:
                        logger.warning(f"[AMM] Classification failed, no memory written: {e}")
                        wm.increment_cycles_without_progress()

                except Exception as e:
                    logger.error(f"[AMM] Memory writing failed: {e}")
            # ===============================
            
            # Apply score modifications AFTER memory writing (for display/evaluation only)
            # These modifications do not affect memory records which use TRUE values
            score = score_true  # Start with true score
            reward = reward_true  # Start with true reward
            
            if is_action_failed(obs):
                logger.info(f"\t\t Failed: [{action}] --> {obs}")
                failed_messages.append(f"\t\t Failed action: (in {current_place}) [{action}] --> {obs}")
            
            
            # === T3 TRIGGER: Repeated Invalid Action (Retrieval B - Avoidance EMs) ===
            # Only active when use_amm=True
            if enable_amm and amm_retrieval_enabled and amm_client is not None and wm is not None:
                from amm.config import DEFAULT_CONFIG
                from amm.retrieval import build_avoidance_retrieval_query_b, retrieve_avoidance_ems_b
                from amm.formatters import _parse_inventory_text
                
                INVALID_OBS = "No known action matches that input."
                if (
                            DEFAULT_CONFIG.enable_em_retrieval
                    and DEFAULT_CONFIG.enable_t3_retrieval
                    and len(recent_actions) >= 2
                    and len(recent_obs) >= 1
                ):
                    last_action = recent_actions[-1]
                    prev_action_t3 = recent_actions[-2]  # Local variable to avoid clobbering outer prev_action
                    last_obs = recent_obs[-1].strip()
                    
                    is_invalid_obs = (last_obs == INVALID_OBS)
                    is_repeated_action = (last_action == prev_action_t3)
                    
                    if is_invalid_obs and is_repeated_action:
                        logger.info(
                            "[T3 Trigger] Repeated invalid action '%s' with observation '%s' "
                            "→ retrieving avoidance (B) EMs",
                            last_action,
                            last_obs,
                        )
                        
                        current_room = get_current_room(info['look']) or "unknown"
                        inventory_items = _parse_inventory_text(env.inventory())
                        
                        rewards_window = recent_reward[-5:] if len(recent_reward) > 5 else recent_reward
                        actions_window = recent_actions[-5:] if len(recent_actions) > 5 else recent_actions
                        obs_window = recent_obs[-5:] if len(recent_obs) > 5 else recent_obs
                        
                        query_b = build_avoidance_retrieval_query_b(
                            task_description=task_description,
                            room_name=current_room,
                            inventory_items=inventory_items,
                            recent_rewards=rewards_window,
                            current_score=score_true,
                            look_description=info['look'],
                            recent_actions=actions_window,
                            recent_observations=obs_window,
                        )
                        
                        avoidance_ems = retrieve_avoidance_ems_b(
                            memory_agent_id=amm_client.agent_id,
                            query_text=query_b,
                            letta_client=amm_client,
                        )
                        
                        wm.set_avoidance_memories(avoidance_ems)
                        logger.info(
                            "[T3 Trigger] Stored %d avoidance EMs in WorkingMemory",
                            len(avoidance_ems),
                        )
            # ================================================================
            
            # if the focus on is useful (positive reward) we will track it
            if reward_true > 0 and action.startswith("focus on"):
                useful_focus_on.append(action)
                if len(useful_focus_on) == max(focus_on_count[str(task_num)], task_description.count("focus")):
                    focus_on_done = True 
            if srm_gate is not None:
                srm_gate.mark_focus_executed(action, obs)

            # Apply score modification logic (for display/evaluation, not for memory)
            # Note: Memory was already written with TRUE values above
            if score_true < 0 or (len(recent_reward)>=100 and sum(recent_reward[-30:])==0):
                # Note: our own solution for dealing with such cases; It is different from the official ScienceWorld evaluation script. You can find our discussion in the Issues.
                if args["no_stop"]:
                    done = True
                    score = last_score  # Modified for display only
                else:
                    done = True
                    score = 0  # Modified for display only
            
            # Update last_score for next iteration (use modified score for tracking, but memory uses TRUE values)
            last_score = score

            #logger.info("Input string: " + str(input_str))
            logger.info(f"Variation: {variation}, Step: {step}")
            logger.info(f"Action: {action}")
            logger.info("Obs: " + sanitizeStr(obs))
            logger.info(f"Score: {score}")  # Display score (may be modified for display)
            if reward_true > 0:
                logger.info(f"Reward: +{reward_true}")
            else:
                logger.info("No reward.")

            step += 1
            if (step >= max_steps) or done:
                break
  

            logger.info("Recent Actions: " + str(recent_actions))
            logger.info("Recent Observations: " + str(recent_obs))
            logger.info("Recent Reward: " + str(recent_reward))

            # Early stopping if we're in a loop
            # TODO: removed this due to "wait and checking something"
            # if len(recent_actions) >= 5 and len(set(recent_actions[-5:])) == 2:
            #     logger.info("Many recent actions in history are the same -- model is likely in a loop, stopping early.")
            #     break


        # Store results
        env.storeRunHistory(variation, notes = {'mode':args["mode"], 'lm':str(args["lm_path"])} )
        env.saveRunHistoriesBufferIfFull(filenameOutPrefixSeed, maxPerFile=args["max_episode_per_file"])

        scores.append(score)

        logger.info("Run completed...")
        logger.info("Scores: " + str(scores))
 
        time.sleep(2)

    # Episodes are finished -- manually save any last histories still in the buffer
    env.saveRunHistoriesBufferIfFull(filenameOutPrefixSeed, maxPerFile=args["max_episode_per_file"], forceSave=True)

    avg = sum(scores) / len(scores)
    logger.info("Average score: " + str(avg))

    f = open(filenameOutPrefixSeed + "-score.txt", "a")
    f.write("\n" + "Task name:" + taskName + "Scores: " + str(scores) + " Average score: " + str(avg) + " Args: " + str(args) + "\n")
    f.close()

    logger.info("Shutting down server...")
    # env.shutdown()

    logger.info("Completed.")



def parse_args():
    parser = argparse.ArgumentParser()
    debug = True 
    parser.add_argument("--jar_path", type=str) 
    parser.add_argument("--task_nums", default="11")  # use comma to split 
    parser.add_argument("--env_step_limit", type=int, default=300) # for different tasks, this should be different 
    parser.add_argument("--lm_path", default="yuchenlin/swift_sw") 
    parser.add_argument("--simplification_str", default="easy")
    parser.add_argument("--beams", type=int, default=5)
    parser.add_argument("--max_episode_per_file", type=int, default=9999)
    parser.add_argument("--mode", default="fast_system")
    parser.add_argument("--set", default="test_mini")
    parser.add_argument("--output_path", default="logs/test_fast_slow_agent_0424_debug")
    parser.add_argument("--compose_mode", default="v4")
    parser.add_argument("--model_parallelism_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_len", type=int, default=1024)
    parser.add_argument("--cut_off", action="store_true", default=True)
    parser.add_argument("--sbert", action="store_true", default=True)
    parser.add_argument("--no_stop", action="store_true", default=True) 
    parser.add_argument("--slow_agent", action="store_true", default=True) 
    parser.add_argument("--gpt_version", default="qwen2.5-1m-instruct", type=str, 
                        help="LLM model identifier (now used mainly for logging; Qwen via vLLM is default).")  
    parser.add_argument("--local_llm", default="none", type=str)  
    parser.add_argument("--demo_file", default="data_utils/demos.json", type=str)
    parser.add_argument("--debug_var", type=int, default=93)
    parser.add_argument("--use_memory_planning", action="store_true", default=True)
    # Backward-compatible runner aliases (safe no-op/profile flags).
    parser.add_argument("--baseline", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--enable-amm", dest="use_amm", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--enable-srm", dest="disable_srm", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--use_amm", dest="use_amm", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--use-amm", dest="mode_use_amm", action="store_true", default=False, help="Run baseline + AMM only (SRM disabled).")
    parser.add_argument("--use-srm", dest="mode_use_srm", action="store_true", default=False, help="Run baseline + SRM only (AMM disabled).")
    parser.add_argument("--use-full", dest="mode_use_full", action="store_true", default=False, help="Run full system (AMM + SRM).")
    parser.add_argument("--amm_write_only", "--amm-write-only", dest="amm_write_only", action="store_true", default=False, help="AMM write-only mode: keep writes enabled but disable retrieval and prompt augmentation.")
    parser.add_argument("--disable_amm_swift_injection", dest="disable_amm_swift_injection", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--disable-amm-swift-injection", dest="disable_amm_swift_injection", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--disable-swift-injection", dest="disable_amm_swift_injection", action="store_true", help="Disable AMM episodic-memory injection into Swift T1 retries.")
    parser.add_argument("--disable_srm", "--disable-srm", dest="disable_srm", action="store_true", default=True, help="Disable SRM (Self-Reflection Module).")
    args = parser.parse_args()
    mode_count = int(bool(args.mode_use_amm)) + int(bool(args.mode_use_srm)) + int(bool(args.mode_use_full))
    if mode_count > 1:
        parser.error("Mode flags are mutually exclusive: choose only one of --use-amm, --use-srm, --use-full.")
    if args.mode_use_amm:
        args.use_amm = True
        args.disable_srm = True
    elif args.mode_use_srm:
        args.use_amm = False
        args.disable_srm = False
    elif args.mode_use_full:
        args.use_amm = True
        args.disable_srm = False

    if args.disable_amm_swift_injection and not args.use_amm:
        parser.error("--disable-swift-injection requires AMM enabled (use --use-amm or --use-full, or set --use_amm).")

    params = vars(args)
    return params

#
#   Main
#

def init_logger(args, task_num, log_level=INFO):
    filenameOutPrefixSeed = get_file_name(args, task_num)
    logger = logging.getLogger()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s\t] %(message)s",
                                    datefmt='%Y-%m-%d %H:%M:%S')
    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_dir = args["output_path"]
    if logging_dir:
        os.makedirs(logging_dir, exist_ok=True)
        now = int(round(time.time() * 1000))
        timestr = time.strftime('%Y-%m-%d_%H-%M', time.localtime(now / 1000))
        filename = f"{filenameOutPrefixSeed}.log"
        print(filename)
        fh = logging.FileHandler(filename)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(fh)
    return logger

def main():
    args = parse_args()
    print(args) 
    
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed']) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    task_nums = args["task_nums"].split(",")
    for task_num in task_nums:
        logger = init_logger(args, task_num)
        logger.info(args)
        eval(args, int(task_num), logger)
        
if __name__ == "__main__":
    main()