
import os
import re
import time
import json
import copy
import torch
import random
import argparse
from tqdm import trange
from collections import defaultdict
from scienceworld import ScienceWorldEnv
from data_utils.data_utils import add_current_place, add_current_objects, sanitizeStr, formalize_action
from data_utils.data_utils import compose_instance_v1, compose_instance_v1_1, compose_instance_v2, compose_instance_v3, compose_instance_v4
from eval_utils import load_model, findValidActionNew, load_variation, get_model_output, findValidActionWithSystem2, getFilteredValidActions, sbert_search, clean_look, is_action_failed 
from eval_utils import try_to_replace, rooms, clean_history, get_current_room, clean_obj_name, focus_on_count, rank_candidates_by_common_words, gpt_select_valid


# from scienceworld
from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway, CallbackServerParameters
from scienceworld.constants import BASEPATH, DEBUG_MODE, ID2TASK, JAR_PATH, NAME2ID
from scienceworld.utils import infer_task
import logging
logger = logging.getLogger(__name__)

class MyScienceWorldEnv(ScienceWorldEnv):
    # it is only used for fixing the logging error --> logger.info(f"ScienceWorld server running on {port}") 
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



import logging
from logging import INFO, WARN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_file_name(args, task_num):
    if (len(args["output_path"]) > 0):
        if not args["output_path"].endswith("/"):
            args["output_path"] += "/"

        # Make path if it doesn't exist
        if not os.path.exists(args['output_path']):
            os.makedirs(args["output_path"])
  
    filenameOutPrefixSeed = args["output_path"] + "task" + str(task_num)

    return filenameOutPrefixSeed
  


# Example user input console, to play through a game.
def eval(args, task_num, logger):
    # if args["compose_mode"] == "v1":
    #     compose_instance = compose_instance_v1
    # elif args["compose_mode"] == "v1_1":
    #     compose_instance = compose_instance_v1_1
    # elif args["compose_mode"] == "v2":
    #     compose_instance = compose_instance_v2
    # elif args["compose_mode"] == "v3":
    #     compose_instance = compose_instance_v3
    if args["compose_mode"] == "v4":
        compose_instance = compose_instance_v4
    
    demo_data = None 
    if args["demo_file"]: 
        with open(args["demo_file"]) as f:
            demo_data = json.load(f)
    
    # Initialize environment
    # env = ScienceWorldEnv("", args["jar_path"], envStepLimit = args["env_step_limit"], threadNum = 0)
    env = MyScienceWorldEnv("", args["jar_path"], envStepLimit = args["env_step_limit"])
    taskNames = env.getTaskNames()
    taskName = taskNames[task_num]
    env.load(taskName, 0, args['simplification_str'])
    lm_model, tokenizer, sbert_model, llm = load_model(args, device)

    variations = load_variation(env, args, task_num, logger)
    filenameOutPrefixSeed = get_file_name(args, task_num)
    # plans = get_plans(args)
    gpt_version = args["gpt_version"]
    scores = []

    # === AMM INIT ===
    from amm.client_letta import AMMLettaClient, LettaConfig
    from amm.working_memory import WorkingMemory
    from amm.writer import write_success, write_nearmiss, write_avoidance, create_memory_record
    from amm.schema import MemoryRecord
    from amm.config import DEFAULT_CONFIG
    from amm.tagging import classify_episode
    from amm.retrieval import build_avoidance_retrieval_query_b, retrieve_avoidance_ems_b
    from amm.formatters import _parse_inventory_text

    # Initialize AMM client and working memory
    # Get API token and agent ID from environment or config
    letta_api_token = os.getenv("LETTA_API_TOKEN")
    letta_agent_id = os.getenv("LETTA_AGENT_ID")
    
    if not letta_api_token or not letta_agent_id:
        raise ValueError(
            "LETTA_API_TOKEN and LETTA_AGENT_ID environment variables must be set. "
            "Please set them before running the agent."
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
    # =================

    for variation in variations:
        if args["debug_var"] >=0 and variation != args["debug_var"]:
            logger.info(f"Skipping the Var: {variation} because we only focus on args['debug_var'']={args['debug_var']}")
            continue 
        # train_data = []
        env.load(taskName, variation, args["simplification_str"], generateGoldPath=True)
        task_description = env.taskdescription()[18:]
        logger.info(f"task_description = {task_description}")
        
        # === AMM HOOK: EPISODE RESET ===
        wm.reset()
        wm.pending_subgoal = task_description
        logger.info(f"[AMM] Working memory reset for new episode: {task_description[:50]}...")
        # ================================
        # task_description = env.taskdescription()  
        recent_actions = ["look around"]
        recent_obs = ["N/A"]
        recent_locs = []
        recent_looks = {}
        recent_looks_flatten = []
        recent_scores = [0.0,]
        recent_reward = [0.0]
        # recent_actions_without_open = []
        places = []
        objects = [] 
        # bad_words_ids = None
 
        obs, info = env.reset()
        current_place = get_current_room(info['look'])        
        recent_locs.append(current_place)
        recent_looks[current_place] = info["look"]
        recent_looks_flatten.append(info["look"])
        # recent_looks[current_place] = info["look"]

        prev_obs = 'N/A'
        prev_action = 'look around'
        # prev_look = ''
        # prev_inv = ''

        done = False
        score = 0.0
        last_score = 0.0
        step = 0

        # The env has an internal step count, some actions like look around are free
        # however, the t5 model only generates the action "look around", which will result in a dead loop below
        # so the max_steps here is only used to avoid the model generating the same action forever
        
        # Kill Analysis Paralysis - can be tuned as needed
        max_steps = args["env_step_limit"] * 2
 
        action_buffer = []
        obs_buffer = [] # guess_obs_list
        failed_action_trial = defaultdict(lambda: 0)
        last_time_system2_steps = [-1]
        last_time_system2 = -1
        consecutive_system2 = 0
        focus_on_done = False
        useful_focus_on = []
        no_action_done = 0
        system_2_focused = False
        system_1_focused_trial = 0
        swift_failure_count = 0
        pattern = r"focus on\s+(\b\w+\b(\s+\b\w+\b)*)"
        matches = re.findall(pattern, task_description)
        to_focus = [match[0].replace("the ", " ").strip() for match in matches]
        logger.info(f"to_focus={to_focus}")
        failed_messages = []
        while not done:           
 

            no_action_done += 1 
            
            logger.info("-"*50+f"Variation: {variation}, Step: {step}"+"-"*50) 
            logger.info(f"Action Buffer: {action_buffer}")
            logger.info(f"Guess Obs Buffer: {obs_buffer}")
            validActions = getFilteredValidActions(env, info["look"], task_id=task_num, task_desc=task_description)
            logger.info(f"look = \n {str(info['look'])}")
            logger.info(f"inventory = \n {str(env.inventory())}")
            logger.info(f"validActions= {validActions}")
            action = None 
            executed = False

            add_current_place(obs, info['look'], places)
            add_current_objects(task_num, info['look'], objects, limit=20)

            current_place = get_current_room(info['look'])
            recent_looks[current_place] = info["look"]
            recent_looks_flatten.append(info["look"])
            
            # Wait one more time for delayed actions (for e.g wait for water to boil)
            if step > 3 and recent_actions[-1] == "wait" and recent_actions[-2] != "wait":
                if recent_looks_flatten[-1] == recent_looks_flatten[-2]:
                    action = "wait"
            # Try to use the actions from action buffer
            if action is None and len(action_buffer) > 0:
                # debug  
                buffer_overall_trail = 0
                to_remove = []
                # Try to use the actions in the buffer and see if any can be executed.
                for action_ind, action_candidate in enumerate(action_buffer):
                    if buffer_overall_trail >= 2 or (buffer_overall_trail >= 1 and action_candidate.startswith("focus on")):
                        action_buffer = []
                        obs_buffer = []
                        break 
                    
                    buffer_overall_trail += 1
                    if action_candidate.startswith("focus on") and focus_on_done:
                        logger.info(f"Removed {action_candidate} from the buffer, because the focus on limit exceed")
                        to_remove.append(action_ind)
                        continue 
                    
                    if action_candidate.startswith("focus on") and any(["focus on" in a for a in action_buffer[:action_ind]]):
                        logger.info(f"Skip {action_candidate} from the buffer, because there is a previous unfinished focus on")
                        continue 

                    if action_candidate in ["examine " + r for r in rooms]:
                        logger.info(f"Removed {action_candidate} from the buffer (not useful).")
                        to_remove.append(action_ind)
                        continue 
 
                    action_candidate = try_to_replace(action_candidate, validActions, info['look'], info['inv'])
                    if action_buffer[action_ind] != action_candidate:
                        logger.info(f"Replace {action_buffer[action_ind]} --> {action_candidate}.")
                        # logger.info(f"validActions= {validActions}")

                    if failed_action_trial.get(action_candidate, 0) >= 3:
                        logger.info(f"Removed {action_candidate} from the buffer because we have tried this action for 3 times.")
                        to_remove.append(action_ind)
                        continue
                    
                    # if action_candidate.startswith("focus on") and action_candidate not in validActions:
                    #     to_remove.append(action_ind)
                    #     logger.info(f"Removed {action_candidate} from the buffer.")

                    # try to execute and see 


                    ### 1) execute the action if it is valid 

                    if action_candidate in validActions:
                        action_buffer.pop(action_ind)
                        obs_buffer.pop(action_ind)
                        action = action_candidate
                        buffer_overall_trail = 0
                        break 
                    
                    ### 2) try to execute the obs as if it is an action 
                    
                    action_candidate_v2 = obs_buffer[action_ind].lower() if formalize_action(obs_buffer[action_ind].lower()) is not None else None
                    action_candidate_v2 = None if action_candidate_v2 == action_candidate else action_candidate_v2
                    
                    if action_candidate_v2 and action_candidate_v2 in validActions:
                        action_buffer.pop(action_ind)
                        obs_buffer.pop(action_ind)
                        action = action_candidate_v2
                        buffer_overall_trail = 0
                        break  

                    ### 3) try to execute the action if v1 and v2 are both not valid 
                    action_accepted = False
                    final_action = ""
                    action_trials = [action_candidate]
                    if action_candidate_v2:
                        action_trials.append(action_candidate_v2)
                    action_trials.sort(key=lambda x: len(x), reverse=True)
                    
                    for act_cand in action_trials:
                        if not act_cand:
                            continue 
                        obs_buf, reward_buf, done_buf, info_buf = env.step(act_cand)
                        logger.info(f"Trying to execute [{act_cand}] in the buffer.")  
                        if is_action_failed(obs_buf):
                            logger.info(f"\t\t Failed: [{act_cand}] --> {obs_buf}")
                            # failed_messages.append(f"\t\t Failed action: [{act_cand}] --> {obs_buf}")
                            if act_cand == action_candidate:
                                failed_messages.append(f"\t\t Failed action: (in {current_place}) [{act_cand}] --> {obs_buf}")
                        else:
                            action_accepted = True 
                            final_action = act_cand
                            # Update obs, reward, done, info for later use
                            obs = obs_buf
                            reward_env = reward_buf
                            done = done_buf
                            info = info_buf
                            break  


                    if action_accepted:
                        logger.info(f"\t\t Success: [{final_action}] --> {obs}")
                        executed = True 
                        action_buffer.pop(action_ind)
                        obs_buffer.pop(action_ind)
                        action = final_action
                        buffer_overall_trail = 0
                        # Handle ambiguous requests for buffer-executed actions
                        if obs.startswith("Ambiguous request"):
                            obs, reward_env, done, info = env.step("0")
                        break 
                    else:
                        failed_action_trial[action_candidate] += 1

                    ### 4) use gpt to search the valid candidate 
                    if not action_candidate.startswith("focus on"):
                        candidates = rank_candidates_by_common_words(action_candidate, validActions)[:30]
                        if len(candidates) == 0: 
                            failed_action_trial[action_candidate] += 1
                        elif len(candidates) == 1:
                            action_buffer[action_ind] = candidates[0]
                        elif len(candidates) >= 2:
                            logger.info(f"searching = [{action_candidate}] with gpt")
                            selections = gpt_select_valid(action_candidate, candidates, clean_look(info['look']), info['inv'], obs_buffer[action_ind], logger.info, 1, gpt_version)
                            # # intersection = set(sbert_results) & set(edit_results)
                            # if len(intersection) == 0:
                            #     continue 
                            for s in selections:
                                if s in candidates:
                                    action = s
                                    break 
                            if action in validActions:
                                action_buffer.pop(action_ind)
                                obs_buffer.pop(action_ind)
                                buffer_overall_trail = 0
                                logger.info(f"mathced = [{action_candidate}] --> {selections}")
                                break  

                            # # if s is a new compose 
                            # action_buffer[action_ind] = selections[0]
                            # logger.info(f"mathced = [{action_candidate}] --> {selections} (update the buffer)") 
                    
                    #### 5) no matching at all.
                    to_remove.append(action_ind)
                action_buffer = [a for ind, a in enumerate(action_buffer) if ind not in to_remove] 
                obs_buffer = [o for ind, o in enumerate(obs_buffer) if ind not in to_remove] 
                    

            if action is None: 
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
                if args["slow_agent"]:                    
                    force_system_2 = False 
                    force_system_1 = False 
                    # If system 1 is stuck (no action done for 2 steps or two failed actions) switch to system 2
                    if no_action_done >= 2 or len(failed_messages) >= 2:
                        force_system_1 = False
                        force_system_2 = True            
                        logger.info("Force to do force_system_2")
                    # If system 1 already focused on something and system 2 did not, switch to system 2
                    if not system_2_focused and system_1_focused_trial >= 1:
                        force_system_1 = False
                        force_system_2 = True            
                        logger.info("Force to do force_system_2")
                    # If system 2 has been used for 2 steps, switch to system 1
                    if consecutive_system2 >= 2:
                        force_system_1 = True
                        force_system_2 = False
                        logger.info("Force to do force_system_1")
                    # if not force_system_2 or force_system_1:
                    if True:
                        input_str = sanitizeStr(input_str)
                        logger.info("InputStr: " + input_str)
                        # Invokes Swift, return top predicted actions
                        predStrs = get_model_output(args, input_str, tokenizer, lm_model, device, logger)
                    else:
                        predStrs = []
                    
                    # Use Sage agent
                    use_memory_planning = args.get("use_memory_planning", True)
                    used_sys2, return_result = findValidActionWithSystem2(
                        predStrs, env, task_num, task_description, info['look'],
                        recent_actions, recent_reward, recent_obs, recent_locs, recent_looks, failed_messages,
                        demo_data, logger, sbert_model, step, last_time_system2_steps,
                        useful_focus_on, focus_on_done, force_system_1, force_system_2,
                        gpt_version, llm=llm,
                        episodic_memories=None,  # AMM will handle memory retrieval in Phase 2
                        use_memory_planning=use_memory_planning,
                        amm_client=amm_client,  # Pass AMM client for T1 retrieval
                        current_score=score,  # Current score for retrieval query
                        recent_scores=recent_scores,  # Recent scores for retrieval query
                        swift_failure_count=swift_failure_count,  # Pass swift_failure_count for T1 escalation
                        cycles_without_progress=wm.cycles_without_progress,  # Pass cycles_without_progress for T2 escalation
                        # Parameters for second Swift pass (T1-S2 retry)
                        args=args,
                        tokenizer=tokenizer,
                        lm_model=lm_model,
                        device=device,
                        compose_instance=compose_instance,
                        prev_action=prev_action,
                        prev_obs=prev_obs,
                        objects=objects,
                        places=places
                    )  
                    if not used_sys2:
                        action = return_result
                        consecutive_system2 = 0
                        
                    else:
                        action = None 
                        action_buffer = return_result[0] # reset the buffer 
                        obs_buffer = return_result[1]
                        failed_messages = [] # reset the failed messages 
                        logger.info(f"action_buffer reset by the Slow Agent") 
                        failed_action_trial = defaultdict(lambda: 0)
                        last_time_system2 = step  
                        last_time_system2_steps.append(step)
                        consecutive_system2 += 1
                        continue 
                        
                    # action is not None but is not valid, fallback to wait
                    if action is not None and action not in validActions:
                        logger.info(f"action '{action}' is not in validActions; ") 
                        action = "wait"
                        action_buffer.append(action)
                        obs_buffer.append(action)
                        continue 

                    elif action is None: 
                        continue 
                else:
                    # Use Swift Agent only
                    input_str = sanitizeStr(input_str)
                    logger.info("InputStr: " + input_str)
                    predStrs = get_model_output(args, input_str, tokenizer, lm_model, device, logger)
                    action = findValidActionNew(predStrs, env, info['look'], recent_actions, sbert_model, logger) 
            
 

            # Focus action was already executed, continue
            if action.startswith("focus on") and focus_on_done:
                logger.info(f"You have already done great focus-on action: {useful_focus_on}. Skipping this [{action}]")
                continue 
            
            # Sage handled the focus on action, and we mark the flag to prevent it from trying the same subgoal
            if action.startswith("focus on") and consecutive_system2 > 0:
                system_2_focused = True
                
            # Swift is trying to focus on, system 2 hasnt proposed focus yet
            if action.startswith("focus on") and not system_2_focused:
                # track how many times swift tries to focus
                system_1_focused_trial += 1
                # only after 3 attempts or if the obeject its focusing on matches the task-relevant focus targets we allow it
                if system_1_focused_trial >= 3 or any([clean_obj_name(tf) in clean_obj_name(action) for tf in to_focus]):
                    logger.info(f"You have never used System 2 to focus on... but system_1 has tried multiple times... so okay with [{action}]")
                # otherwise skip action
                else:
                    logger.info(f"You have never used System 2 to focus on... so skip [{action}]")
                    continue 
            
            
            # If the action was not already executed in the previous loop, execute it
            if not executed:
                obs, reward_env, done, info = env.step(action)

            # Handle ambiguous requests (resolve by choosing "0")
            if obs.startswith("Ambiguous request"):
                obs, reward_env, done, info = env.step("0")
            
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
            
            # Update tracking lists with TRUE values
            no_action_done = 0
            prev_action = action
            recent_reward.append(reward_true/100)
            if reward_true == 0.0:
                swift_failure_count += 1
            else:
                swift_failure_count = 0
            recent_scores.append(score_true/100)
            recent_actions.append(action) 
            recent_obs.append(obs)
            recent_locs.append(current_place)
            
            # === AMM HOOK: POST-STEP WRITE (BEFORE any score modifications) ===
            # Write memory with TRUE reward/score values from environment
            try:
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
            INVALID_OBS = "No known action matches that input."
            if (
                amm_client is not None
                and DEFAULT_CONFIG.enable_em_retrieval
                and DEFAULT_CONFIG.enable_t3_retrieval
                and len(recent_actions) >= 2
                and len(recent_obs) >= 1
            ):
                last_action = recent_actions[-1]
                prev_action = recent_actions[-2]
                last_obs = recent_obs[-1].strip()
                
                is_invalid_obs = (last_obs == INVALID_OBS)
                is_repeated_action = (last_action == prev_action)
                
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
    parser.add_argument("--gpt_version", default="gpt-4", type=str)  
    parser.add_argument("--local_llm", default="none", type=str)  
    parser.add_argument("--demo_file", default="data_utils/demos.json", type=str)
    parser.add_argument("--debug_var", type=int, default=93)
    parser.add_argument("--use_memory_planning", action="store_true", default=True)
    args = parser.parse_args()
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