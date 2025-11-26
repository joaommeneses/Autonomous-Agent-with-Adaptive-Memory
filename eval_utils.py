
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoConfig
from math import ceil
import random
import re
from sentence_transformers import SentenceTransformer

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from slow_agent.utils import completion_with_backoff
from data_utils.data_utils import formalize_action, recover_action
import string 
import editdistance
import time 
import tiktoken 
from typing import List, Dict, Any

from slow_agent import local_llm

action_type_description = [
    {"action_type": "WAIT()", "desc": "wait for something to be done, for example, an object on stove to be boiled"},
    {"action_type": "TELEPORT(room)", "desc": "directly go to a room such as TELEPORT(kitchen)"},
    # {"action_type": "LOOK(object)", "desc": "look at an object"},
    {"action_type": "READ(object)", "desc": "read an object such as a recipe or a book"},
    {"action_type": "PICK(object)", "desc": "pick up an object and put it into your inventory"},
    {"action_type": "OPEN(object)", "desc": "open an object with doors before you search or put things in it. For example, OPEN(freezer), OPEN(blast furnace)."},
    {"action_type": "ACTIVATE(object)", "desc": "activate and turn on an object such as sink or stove, so that you can use it. "},
    {"action_type": "DEACTIVATE(object)", "desc": "deactivate turn off the object"},
    {"action_type": "EXAMINE(object)", "desc": "look at an object carefully. For example, EXAMINE(apple). Note that you cannot EXAMINE a location."},
    {"action_type": "CONNECT(object)", "desc": "connect two objects so that they become useful"},
    {"action_type": "MOVE(object, place)", "desc": "move/place the object to a place"},
    {"action_type": "USE(object A, object B)", "desc": "use an object A on object B, for example, USE(thermometer in inventory, water) to check the temperature of water."},
    {"action_type": "MIX(container)", "desc": "mix the objects in a container such as MIX(cup containing sugar and water)"},
    {"action_type": "DUNK(object A, object B)", "desc": "dunk object A into object B (optional)"},
    {"action_type": "DROP(object A, object B)", "desc": "drop object A into object B (optional)"},
    {"action_type": "POUR(object A, object B)", "desc": "pour the object A into the container B; For example, POUR(red paint, glass cup)"},
    {"action_type": "FOCUS(object)", "desc": "focus on an important object that are required by the task description (e.g., a substance, a plant, an animal, and so on)."},
]

focus_on_count = {
    "0": 1, "1": 1, "2": 1, "3": 1, "4": 2, "5": 1, "6":1, "7":1,
    "8": 1, "9": 1, "10": 1, "11": 1, "12": 4, "13": 4, "14":1, "15":1,
    "16": 1, "17": 1, "18": 2, "19": 1, "20": 3, "21": 3, "22":1, "23":1,   
    "24": 1, "25": 1, "26": 2, "27": 1, "28": 1, "29": 2
    
}

rooms = ["hallway", "greenhouse", "green house", "kitchen", "bathroom", "outside", "workshop", "art studio", "foundry", "bedroom", "living room"]


def is_action_failed(obs):
    return obs == "No known action matches that input." or "can't" in obs or "not" in obs or "doesn't" in obs

def find_non_alpha_index(s):
    for i, c in enumerate(s):
        if not c.isalpha() and c != ' ':
            return i
    return -1  # if no non-alpha character found 

def clean_look(look, version="not_lite"):
    
    if "You also see:" in look:
        end_ind = look.index("You also see:")
        look = look[:end_ind]

    clean_looks = []
    for line in look.splitlines():
        if not line.strip():
            continue
        if "In it, you see:"  in line:
            if version != "lite":
                clean_looks.append(line)
            continue
        if "the agent" in line or " air" in line:
            continue
        line = line.replace("substance called ", " ").strip()
        if version == "lite":
            end_ind = find_non_alpha_index(line.strip())
            if end_ind > 0:
                line = line[:end_ind].strip()
        clean_looks.append(line)
    if version == "lite":
        return ", ".join(clean_looks)        
    else:
        return "\n \t - ".join(clean_looks[:])        


 
def get_current_room(look):
    global rooms 
    first_sent = look.split(".")[0]
    for r in rooms:
        if "called the "+ r in first_sent:
            return r  
    return None 


def load_model(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args["lm_path"])
    lm_model = AutoModelForSeq2SeqLM.from_pretrained(args["lm_path"])
    lm_model.eval() 
    lm_model.to(device)
    if args["sbert"]:
        sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    else:
        sbert_model = None 
    
    if args["local_llm"] == "xgen":
        local_llm.load()
        assert local_llm.llm_model is not None 
        assert local_llm.llm_tokenizer is not None 
        print("Testing local LLM:" + args["local_llm"])
        print(local_llm.generate("Hello, who are you?")) # for testing 
        llm_model = local_llm.llm_model
    else:
        llm_model = None 
    return lm_model, tokenizer, sbert_model, llm_model




def load_variation(env, args, task_num, logger):
    variations = []
    if (args["set"] == "train"):
        variations = list(env.getVariationsTrain())
        if task_num == 26: 
            variations = variations[:int(len(variations)/10)]
        elif task_num == 29: 
            variations = variations[:int(len(variations)/2)]
    # elif (args["set"] == "test"):
    #     variations = list(env.getVariationsTest())
    #     if args["cut_off"]:
    #         test_len = min(50, len(variations))
    #         random.seed(1)
    #         random.shuffle(variations)
    #         variations = variations[:test_len]
    # New condition, that matches paper eval method
    # Use the number of variations if less than 10, use 10 if more than 10
    elif (args["set"] == "test"):               
        variations = list(env.getVariationsTest())
        if len(variations) > 5:
            # variations = variations[:5]
            variations = variations[:3] # ===== TESTING PURPOSES
    elif (args["set"] == "dev"):
        variations = list(env.getVariationsDev()) 
        variations = variations[:3]
    elif (args["set"] == "test_mini_2"):
        variations = list(env.getVariationsTest()) 
        # random.seed(1)
        # random.shuffle(variations)
        variations = variations[3:10] 
    elif (args["set"] == "test_mini"):
        variations = list(env.getVariationsTest()) 
        # random.seed(1)
        # random.shuffle(variations)
        variations = variations[:3] 
    elif (args["set"] == "test_mini_mini"):
        variations = list(env.getVariationsTest()) 
        # random.seed(1)
        # random.shuffle(variations)
        variations = variations[:1] 
    else:
        logger.info("ERROR: Unknown set to evaluate on (" + str(args["set"]) + ")")
        exit(1)
 
    logger.info(variations)
    return variations




def findValidActionNew(predictions, env, look, recent_actions, sbert_model, logger, k=5):
    global rooms
    valid_open_door = ["open door to " + i for i in rooms] 
    invalid_focus = ["focus on "+x for x in ["agent", "air"]+rooms]
    validActions = set(env.getValidActionObjectCombinations())
    validActions.update(valid_open_door)
    validActions.difference_update(invalid_focus)

    inventory = env.inventory().lower()
    
    validActions.difference_update(recent_actions[-3:]) 

    for va in list(validActions):
        if "door" in va and "open" not in va:
            validActions.remove(va)
            continue
        if va.startswith("focus on"): 
            pattern = re.compile(r"\b(?:focus|on|in|to)\b", re.IGNORECASE)
            used_objs = pattern.sub("", va).split(" ")
            valid = True
            for obj in used_objs:
                if obj not in look + " " + inventory:
                    valid = False
            if not valid:
                validActions.remove(va)
    

    # 1) if acton in top k is valid, choose it
    found_valid_in_top = False
    action = None
    for pred in predictions[:k]:
        pred = pred.replace("green house", "greenhouse") 
        if pred.strip() in validActions:
            found_valid_in_top = True
            action = pred.strip()
            break
    if found_valid_in_top:
        return action 
    else:
        logger.info(f"No valid action found in top k={k} predictions.")
        validActions = list(validActions)
        validActions.sort(key=lambda x: len(x))
        logger.info("Valid Predictions: "+ str(validActions)) 
 

    # 2) else, find most similar action

    if sbert_model:    
        pred_vectors = sbert_model.encode(predictions[:5], batch_size=5, show_progress_bar=False)
        valid_action_vectors = sbert_model.encode(validActions, batch_size=min(len(validActions), 128), show_progress_bar=False)


        # Calculate cosine similarity between each vector in pred_vectors and all vectors in valid_action_vectors
        similarity_matrix = cosine_similarity(pred_vectors, valid_action_vectors)

        # Take the sum of cosine similarities for each vector in valid_action_vectors
        sum_similarities = similarity_matrix.sum(axis=0)

        # Find the indices of the k vectors with the highest sum of cosine similarities
        N = 5 # Change this to the number of top vectors you want to retrieve
        top_indices = np.argpartition(sum_similarities, -N)[-N:]

        # Print the indices of the top vectors
        # print(f"The indices of the top {k} vectors in valid_action_vectors are: {top_indices}")
        logger.info("The most similar valid actions to the predictions:")
        for ti in top_indices:
            logger.info("\t\t - "+validActions[ti])
        action = validActions[top_indices[0]]
    else:
        # jaccard
        topValue = 0.0
        topAction = predictions[0]
        # embPred = sbert_model.encode(pred, convert_to_tensor=True)
        tokensPred = predictions[0].split(" ")
        uniqueTokensPred = set(tokensPred)

        for validAction in validActions: 
            tokensAction = validAction.split(" ")
            uniqueTokensAction = set(tokensAction)

            intersection = uniqueTokensPred.intersection(uniqueTokensAction)
            if (len(intersection) > topValue):
                topAction = validAction
                topValue = len(intersection)

        logger.info("TOP VALID ACTION: " + topAction)
        # Sanitize top action
        topAction = re.sub(r'[^A-Za-z0-9 ]+', '', topAction)
        action = topAction
    return action 
 

def getFilteredValidActions(env, look, filter=True, task_id=None, task_desc=None):
    global rooms
    valid_open_door = ["open door to " + i for i in rooms] 
    invalid_focus = ["focus on "+x for x in ["agent", "air"]+rooms]
    validActions = set(env.getValidActionObjectCombinations())
    validActions.update(valid_open_door)
    validActions.difference_update(invalid_focus)

    inventory = env.inventory()
    
    validActions.add("wait")
    validActions.add("wait1") 
    if task_id is not None and task_desc is not None: 
        if task_id not in [5,6,7,8,17,18,19,20]:
            for va in list(validActions):
                if not va.startswith("focus on"):
                    continue
                items = va.replace("focus on", "").split()
                task_desc = task_desc.translate(str.maketrans('', '', string.punctuation)).lower()
                if len(set(items) & set(task_desc.split())) == 0:
                    validActions.remove(va)
        if task_id not in [14,15,16]:
            for va in list(validActions):
                if not va.startswith("examine"):
                    continue
                items = va.replace("examine", "").split()
                task_desc = task_desc.translate(str.maketrans('', '', string.punctuation)).lower()
                if len(set(items) & set(task_desc.split())) == 0:
                    validActions.remove(va)
    for va in list(validActions):
        if not va.startswith("mix"):
            continue
        container_words = ["cup", "bowl", "metal pot", "jug"]
        if not any(["mix" + c for c in container_words]):
            validActions.remove(va)
    if not filter:
        return validActions
    for va in list(validActions):
        if "door" in va and "open" not in va:
            validActions.remove(va)
            continue
    return validActions
    
def sbert_search(action_list, validActions, sbert_model, logger, k=1, N=1, return_scores=False):
    validActions = list(validActions)
    pred_vectors = sbert_model.encode(action_list[:k], batch_size=5, show_progress_bar=False)
    valid_action_vectors = sbert_model.encode(validActions, batch_size=min(len(validActions), 128), show_progress_bar=False)

    # Calculate cosine similarity between each vector in pred_vectors and all vectors in valid_action_vectors
    similarity_matrix = cosine_similarity(pred_vectors, valid_action_vectors)

    # Take the sum of cosine similarities for each vector in valid_action_vectors
    sum_similarities = similarity_matrix.sum(axis=0)

    N = min(N, len(validActions))
    # Find the indices of the k vectors with the highest sum of cosine similarities
    # N = 10 # Change this to the number of top vectors you want to retrieve
    top_indices = np.argpartition(sum_similarities, -N)[-N:]

    # Print the indices of the top vectors
    # print(f"The indices of the top {k} vectors in valid_action_vectors are: {top_indices}")
    # logger.info("The most similar valid actions to the predictions:")
    # for ti in top_indices:
    #     logger.info("\t\t - "+validActions[ti])
    if N == 1:
        action = validActions[top_indices[0]]
        score = sum_similarities[top_indices[0]]
        if return_scores:
            return action, score
        return action
    else:
        action_list = []
        for i in range(N):
            action = validActions[top_indices[i]]
            action_list.append(action)
        return action_list



    

def find_object(action, objects_string): 
    # Find the index of the target object in the words list
    target_object = ' '.join(action.split()[2:])
    if target_object not in objects_string:
        return action 
    target_object_index = objects_string.index(target_object)
    
    # Check if the target object is inside a container
    if objects_string[target_object_index - 8:target_object_index - 1] == "called ":
        container_start_index = objects_string.rfind("(", 0, target_object_index) - 1
        container_end_index = objects_string.rfind(")", 0, target_object_index) + 1
        container = objects_string[container_start_index:container_end_index]
        action = action.replace(target_object, f"{container}")
    
    return action


def clean_obj_name(action):
    if "unknown substance" not in action:
        return action 
    for n in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        action = action.replace(f" {n}", "")
    return action 

def try_to_replace(action, validActions, look=None, inventory=None):
    if action.startswith("wait"):
        return "wait"
    if action in validActions:
        return action 
    try_action = action.replace("green house", "greenhouse") 
    try_action = try_action.replace("adult", "adult adult")
    try_action = try_action.replace("baby", "baby baby")
    if try_action in validActions:
        return try_action 
    if action.startswith("go to"):
        if action.replace("go to", "teleport to") in validActions:
            return action.replace("go to", "teleport to")
        elif action.replace("go to", "open door to") in validActions:
            return action.replace("go to", "open door to") 
    if action.startswith("pick up"):
        action = find_object(action, look)
        if action in validActions:
            return action 
        if action.replace("substance in ","") in validActions:
            return action
    if action.startswith("focus on"):
        obj = action.replace("focus on", "").strip()
        todo = "focus on substance in inventory"
        if obj in inventory and  todo in validActions:
            return todo
    if action.startswith("move") and "to" in action:
        pattern = r"move (.*?) to"
        obj = re.search(pattern, action)
        if obj is None:
            return action 
        else:
            obj = obj.group(1)
        todo = action.replace(obj, "substance in inventory")
        if obj in inventory and todo in validActions:
            return todo
    

    split_string = action.rsplit(" in ", 1) # Split the string from the last occurrence of " in "
    if split_string[0] in validActions:
        return split_string[0]

    if " unknown substance " in action:
        action = split_string[0]
        action = clean_obj_name(action)
        if action in validActions:
            return action 
        
    for r in rooms:
        action = action.replace("in " + r, "")
    return action 
        

import openai  # make sure this is at the top of the file

def rerun_swift_with_same_context(
    task_description: str,
    look: str,
    inventory: str,
    recent_actions: list,
    recent_obs: list,
    recent_locs: list,
    recent_looks: dict,
    failed_messages: list,
    logger,
    sbert_model,
    step: int,
    validActions: list,
    args: dict,
    tokenizer,
    lm_model,
    device,
    compose_instance,
    prev_action: str,
    prev_obs: str,
    objects: list,
    places: list,
    current_score: float,
    retrieved_ems: list = None,  # For future use when we inject EMs into prompt
) -> tuple:
    """Re-run the fast agent (Swift) once more and try to get a valid action.
    
    This helper rebuilds the Swift input context and calls the model again,
    then checks if any of the top predictions are valid actions.
    
    Args:
        task_description: Current task description
        look: Current room look description
        inventory: Current inventory string
        recent_actions: Recent action history
        recent_obs: Recent observation history
        recent_locs: Recent location history
        recent_looks: Recent look descriptions by location
        failed_messages: List of failed action messages
        logger: Logger instance
        sbert_model: SBERT model (not used in this helper but kept for signature consistency)
        step: Current step number
        validActions: List of valid actions for current state
        args: Arguments dict (needs 'mode', 'max_input_len', 'beams')
        tokenizer: Tokenizer for Swift model
        lm_model: Swift model (Flan-T5)
        device: Device for model inference
        compose_instance: Function to compose input string (compose_instance_v4)
        prev_action: Previous action string
        prev_obs: Previous observation string
        objects: List of objects seen
        places: List of places visited
        current_score: Current score (for returns_to_go calculation)
        retrieved_ems: List of retrieved episodic memories (for future EM injection)
    
    Returns:
        (action, found_valid_in_top) where:
        - action: The selected action (None if no valid action found)
        - found_valid_in_top: Boolean indicating if a valid action was found in top predictions
    """
    # TODO: In future, prepend a "Past Episodes" or EM block to the input_str here
    # using retrieved_ems. For now, we rebuild the same context as the original Swift call.
    
    # Calculate returns_to_go (same logic as main loop)
    returns_to_go = 1.0 - float(current_score) * 0.01
    returns_to_go = round(returns_to_go, 2)
    
    # Get mode from args
    mode = args.get("mode", "fast_system")
    
    # Clean history (same as main loop)
    recent_scores = [0.0] * len(recent_actions)  # Placeholder, not used in compose_instance
    recent_reward = [0.0] * len(recent_actions)  # Placeholder, not used in compose_instance
    clean_recent_actions, clean_recent_obs, clean_recent_scores, clean_reward_clean, _ = \
        clean_history(recent_actions, recent_obs, recent_scores, recent_reward, recent_locs)
    
    # Build input string using same compose_instance as main loop
    input_str, _ = compose_instance(
        mode=mode,
        step_id=step + 1,
        task_desc=task_description,
        returns_to_go=returns_to_go,
        curr_action=None,
        curr_obs="",  # Current obs not needed for this context
        inventory=inventory,
        look=look,
        prev_action=prev_action,
        prev_obs=prev_obs,
        objects=objects,
        places=places,
        recent_actions=clean_recent_actions,
        recent_obs=clean_recent_obs,
        recent_scores=clean_recent_scores,
        recent_reward=clean_reward_clean
    )
    
    # Sanitize input string (same as main loop)
    from data_utils.data_utils import sanitizeStr
    input_str = sanitizeStr(input_str)
    
    logger.info("[T1 Trigger] Second Swift pass - InputStr: " + input_str[:200] + "...")
    
    # Call Swift model to get predictions
    predStrs = get_model_output(args, input_str, tokenizer, lm_model, device, logger)
    
    # Apply same valid action selection logic as original Swift check
    found_valid_in_top = False
    action = None
    
    # Filter out "wait" if last action was "wait" and first prediction is "wait"
    if recent_actions and predStrs and recent_actions[-1].startswith("wait") and predStrs[0].startswith("wait"):
        predStrs = predStrs[1:]
    
    # Try top prediction
    for pred in predStrs[:1]:
        pred = try_to_replace(pred, validActions, look, inventory)
        action = pred.strip()
        if pred.strip() in validActions:
            found_valid_in_top = True
            break
    
    logger.info(f"[T1 Trigger] Second Swift pass - found_valid_in_top={found_valid_in_top} ({action})")
    
    return action, found_valid_in_top


def _build_retrieval_state(
    env, look, recent_reward, recent_scores, current_score,
    recent_actions, recent_obs
):
    """
    Helper function to build shared state for T1 and T2 retrieval queries.
    
    Returns:
        dict with keys: current_room, inventory_items, recent_rewards_window,
        current_score_val, recent_actions_window, recent_obs_window
    """
    from amm.formatters import _parse_inventory_text
    
    current_room = get_current_room(look) or "unknown"
    inventory_text = env.inventory()
    inventory_items = _parse_inventory_text(inventory_text)
    
    # Get recent rewards and scores (normalize to match expected format)
    recent_rewards_window = recent_reward[-5:] if len(recent_reward) > 5 else recent_reward
    recent_scores_window = recent_scores[-5:] if recent_scores and len(recent_scores) > 5 else (recent_scores or [])
    current_score_val = current_score if current_score is not None else (recent_scores_window[-1] * 100 if recent_scores_window else 0.0)
    
    # Get recent actions and observations
    recent_actions_window = recent_actions[-5:] if len(recent_actions) > 5 else recent_actions
    recent_obs_window = recent_obs[-5:] if len(recent_obs) > 5 else recent_obs
    
    return {
        "current_room": current_room,
        "inventory_items": inventory_items,
        "recent_rewards_window": recent_rewards_window,
        "current_score_val": current_score_val,
        "recent_actions_window": recent_actions_window,
        "recent_obs_window": recent_obs_window,
    }


def findValidActionWithSystem2(
    predictions, env, task_id, task_description, look,
    recent_actions, recent_reward, recent_obs, recent_locs, recent_looks, failed_messages,
    demo_data, logger, sbert_model, step, last_time_system2_steps,
    useful_focus_on, focus_on_done, force_system_1, force_system_2,
    gpt_version="gemini-2.5-flash-preview-04-17", llm=None,
    episodic_memories=None, use_memory_planning=True,
    amm_client=None, current_score=None, recent_scores=None,
    swift_failure_count: int = 0,
    cycles_without_progress: int = 0,  # For T2 (stagnation) trigger
    # Parameters for second Swift pass (T1-S2 retry)
    args=None, tokenizer=None, lm_model=None, device=None,
    compose_instance=None, prev_action=None, prev_obs=None,
    objects=None, places=None
):
    inventory = env.inventory()
    validActions = getFilteredValidActions(env, look, task_id=task_id, task_desc=task_description)
    enable_system2 = True

    # Fast‐agent heuristics
    found_valid_in_top = False
    action = None
    if recent_actions and predictions and recent_actions[-1].startswith("wait") and predictions[0].startswith("wait"):
        predictions = predictions[1:]
    for pred in predictions[:1]:
        pred = try_to_replace(pred, validActions, look, inventory)
        action = pred.strip()
        if pred.strip().startswith("focus on") and focus_on_done:
            break
        if pred.strip() in validActions:
            found_valid_in_top = True
            break
    logger.info(f"found_valid_in_top={found_valid_in_top} ({action})")

    # === T1 TRIGGER: Episodic Memory Retrieval (Template A, S1 → S2 → Sage) ===
    # When Swift fails to find a valid action, retrieve success EMs based on escalation level
    retrieved_ems = []
    if not found_valid_in_top and amm_client is not None:
        try:
            from amm.retrieval import (
                build_success_retrieval_query_s1, retrieve_success_ems_s1,
                build_success_retrieval_query_s2, retrieve_success_ems_s2
            )
            from amm.formatters import _parse_inventory_text
            from amm.config import DEFAULT_CONFIG
            
            # Check if EM retrieval is enabled
            if not DEFAULT_CONFIG.enable_em_retrieval:
                logger.debug("[T1 Trigger] EM retrieval is disabled (enable_em_retrieval=False), skipping retrieval")
            else:
                # Build shared state for retrieval queries
                retrieval_state = _build_retrieval_state(
                    env, look, recent_reward, recent_scores, current_score,
                    recent_actions, recent_obs
                )
                current_room = retrieval_state["current_room"]
                inventory_items = retrieval_state["inventory_items"]
                recent_rewards_window = retrieval_state["recent_rewards_window"]
                current_score_val = retrieval_state["current_score_val"]
                recent_actions_window = retrieval_state["recent_actions_window"]
                recent_obs_window = retrieval_state["recent_obs_window"]
                
                # T1 Escalation Logic: S1 → S2 → Skip (let Sage handle)
                if swift_failure_count == 0:
                    logger.info("[T1 Trigger] First Swift failure (swift_failure_count=0) → Using S1 retrieval (success-only EMs)")
                    query_text = build_success_retrieval_query_s1(
                        task_description=task_description,
                        room_name=current_room,
                        inventory_items=inventory_items,
                        recent_rewards=recent_rewards_window,
                        current_score=current_score_val,
                        look_description=look,
                        recent_actions=recent_actions_window,
                        recent_observations=recent_obs_window,
                    )
                    retrieved_ems = retrieve_success_ems_s1(
                        memory_agent_id=amm_client.agent_id,
                        query_text=query_text,
                        letta_client=amm_client,
                    )
                elif swift_failure_count == 1:
                    logger.info("[T1 Trigger] Second Swift failure (swift_failure_count=1) → Using S2 retrieval (success + partial/near-miss EMs)")
                    query_text = build_success_retrieval_query_s2(
                        task_description=task_description,
                        room_name=current_room,
                        inventory_items=inventory_items,
                        recent_rewards=recent_rewards_window,
                        current_score=current_score_val,
                        look_description=look,
                        recent_actions=recent_actions_window,
                        recent_observations=recent_obs_window,
                    )
                    retrieved_ems = retrieve_success_ems_s2(
                        memory_agent_id=amm_client.agent_id,
                        query_text=query_text,
                        letta_client=amm_client,
                    )
                    
                    # After S2 retrieval, attempt one more Swift pass before escalating to Sage
                    if retrieved_ems and args is not None and tokenizer is not None and lm_model is not None and device is not None and compose_instance is not None:
                        logger.info("[T1 Trigger] Re-running Swift once after S2 retrieval before escalating to System 2")
                        second_action, second_found_valid = rerun_swift_with_same_context(
                            task_description=task_description,
                            look=look,
                            inventory=inventory,
                            recent_actions=recent_actions,
                            recent_obs=recent_obs,
                            recent_locs=recent_locs,
                            recent_looks=recent_looks,
                            failed_messages=failed_messages,
                            logger=logger,
                            sbert_model=sbert_model,
                            step=step,
                            validActions=validActions,
                            args=args,
                            tokenizer=tokenizer,
                            lm_model=lm_model,
                            device=device,
                            compose_instance=compose_instance,
                            prev_action=prev_action if prev_action is not None else "",
                            prev_obs=prev_obs if prev_obs is not None else "",
                            objects=objects if objects is not None else [],
                            places=places if places is not None else [],
                            current_score=current_score_val,
                            retrieved_ems=retrieved_ems,  # Pass EMs for future use
                        )
                        
                        if second_found_valid and second_action is not None:
                            logger.info("[T1 Trigger] Second Swift attempt after S2 retrieval succeeded → using fast action, skipping System 2")
                            return False, second_action
                        else:
                            logger.info("[T1 Trigger] Second Swift attempt still has no valid action → falling back to existing System 2 logic")
                    elif not retrieved_ems:
                        logger.info("[T1 Trigger] S2 retrieval returned no EMs, skipping second Swift pass → falling back to existing System 2 logic")
                    else:
                        logger.debug("[T1 Trigger] Missing Swift model parameters, skipping second Swift pass → falling back to existing System 2 logic")
                else:  # swift_failure_count >= 2
                    logger.info(
                        f"[T1 Trigger] Skipping AMM retrieval; swift_failure_count={swift_failure_count} "
                        "→ will fall back to existing Sage escalation logic."
                    )
                
                logger.info(f"[T1 Trigger] Retrieved {len(retrieved_ems)} episodic memories for Swift failure")
                
                # TODO: Build augmented context for Swift with EMs (commented out for now)
                # swift_context_with_ems = build_swift_context_with_episodic_memories(
                #     base_context=base_swift_context_for_this_step,
                #     episodic_memories=retrieved_ems,
                # )
                # 
                # # Re-run Swift with EM-augmented context
                # new_actions, new_found_valid_in_top = run_swift_with_context(
                #     context=swift_context_with_ems,
                #     ...
                # )
                # 
                # if new_found_valid_in_top:
                #     # Use this new valid action
                #     ...
                # else:
                #     # Fall back to existing behavior
                #     ...
                
        except Exception as e:
            logger.warning(f"[T1 Trigger] Episodic memory retrieval failed: {e}")
            retrieved_ems = []
    # ================================================================

    # === T2 TRIGGER: Stagnation / Lack of Progress Retrieval ===
    # When Swift succeeds but agent has made no progress for several steps
    # T2 is mutually exclusive with T1 (only runs when found_valid_in_top == True)
    t2_retrieved_ems = []
    if found_valid_in_top and amm_client is not None:
        try:
            from amm.config import DEFAULT_CONFIG
            from amm.retrieval import (
                build_stagnation_retrieval_query_s1,
                build_stagnation_retrieval_query_s2,
                retrieve_success_ems_s1,
                retrieve_success_ems_s2,
            )
            from amm.formatters import _parse_inventory_text
            
            # Check if EM retrieval and T2 are enabled
            if DEFAULT_CONFIG.enable_em_retrieval and getattr(DEFAULT_CONFIG, "enable_t2_retrieval", True):
                # Build shared state for retrieval queries (reuse helper)
                retrieval_state = _build_retrieval_state(
                    env, look, recent_reward, recent_scores, current_score,
                    recent_actions, recent_obs
                )
                current_room = retrieval_state["current_room"]
                inventory_items = retrieval_state["inventory_items"]
                recent_rewards_window = retrieval_state["recent_rewards_window"]
                current_score_val = retrieval_state["current_score_val"]
                recent_actions_window = retrieval_state["recent_actions_window"]
                recent_obs_window = retrieval_state["recent_obs_window"]
                
                # Use cycles_without_progress passed in
                stagnation = cycles_without_progress or 0
                
                # T2 Escalation Logic: S1 → S2 (based on stagnation threshold)
                if stagnation == DEFAULT_CONFIG.T2_STAGNATION_THRESHOLD_S1:
                    logger.info(
                        f"[T2 Trigger] Stagnation detected (cycles_without_progress={stagnation}) "
                        "→ Using S1 retrieval (success-only EMs)"
                    )
                    # Build and retrieve S1
                    query_text = build_stagnation_retrieval_query_s1(
                        task_description=task_description,
                        room_name=current_room,
                        inventory_items=inventory_items,
                        recent_rewards=recent_rewards_window,
                        current_score=current_score_val,
                        look_description=look,
                        recent_actions=recent_actions_window,
                        recent_observations=recent_obs_window,
                        cycles_without_progress=stagnation,
                    )
                    t2_retrieved_ems = retrieve_success_ems_s1(
                        memory_agent_id=amm_client.agent_id,
                        query_text=query_text,
                        letta_client=amm_client,
                    )
                    
                elif stagnation == DEFAULT_CONFIG.T2_STAGNATION_THRESHOLD_S2:
                    logger.info(
                        f"[T2 Trigger] Stagnation persists (cycles_without_progress={stagnation}) "
                        "→ Using S2 retrieval (success + near-miss EMs)"
                    )
                    # Build and retrieve S2
                    query_text = build_stagnation_retrieval_query_s2(
                        task_description=task_description,
                        room_name=current_room,
                        inventory_items=inventory_items,
                        recent_rewards=recent_rewards_window,
                        current_score=current_score_val,
                        look_description=look,
                        recent_actions=recent_actions_window,
                        recent_observations=recent_obs_window,
                        cycles_without_progress=stagnation,
                    )
                    t2_retrieved_ems = retrieve_success_ems_s2(
                        memory_agent_id=amm_client.agent_id,
                        query_text=query_text,
                        letta_client=amm_client,
                    )
                
                if t2_retrieved_ems:
                    logger.info(f"[T2 Trigger] Retrieved {len(t2_retrieved_ems)} episodic memories for stagnation")
                    # TODO: Later merge t2_retrieved_ems with other EMs for Sage planning
                    # For now, T2 only retrieves and logs EMs; wiring into prompts comes in T4
                    
        except Exception as e:
            logger.warning(f"[T2 Trigger] Episodic memory retrieval failed: {e}")
            t2_retrieved_ems = []
    # ================================================================

    last_sys2 = last_time_system2_steps[-1] if last_time_system2_steps else -999
    if found_valid_in_top and len(recent_actions) < 10:
        enable_system2 = False
    if found_valid_in_top and (step - last_sys2) < 5:
        enable_system2 = False
    if found_valid_in_top and sum(recent_reward[-5:]) > 0:
        logger.info("Recent scores increased; skipping System 2.")
        enable_system2 = False
    if found_valid_in_top and action not in recent_actions[-3:]:
        logger.info("Action not in recent 3; skipping System 2.")
        enable_system2 = False
    if found_valid_in_top and not enable_system2 and not force_system_2:
        assert action is not None
        logger.info("Using Fast System output.")
        return False, action

    if ((not found_valid_in_top and (step - last_sys2) <= 2) or force_system_1) and not force_system_2:
        cand_preds = [try_to_replace(p, validActions, look, inventory)
                      for p in predictions if not p.startswith("focus on")]
        cand_preds = cand_preds[:3]
        trial_action = next((p for p in cand_preds if p in validActions), None)
        trial_action = trial_action or (cand_preds[0] if cand_preds else None)
        return False, trial_action

    # System 2
    assert enable_system2 or force_system_2
    fast_action = action if found_valid_in_top else None
    logger.info("Now, start using System 2: Gemini for reasoning")

    # === T4 TRIGGER: Sage Invocation (Retrieve S2 + B EMs for Planning) ===
    episodic_memories_for_planning: List[Dict[str, Any]] = []
    
    if (
        use_memory_planning
        and amm_client is not None
    ):
        from amm.config import DEFAULT_CONFIG
        from amm.retrieval import (
            build_success_retrieval_query_s2,
            retrieve_success_ems_s2,
            build_avoidance_retrieval_query_b,
            retrieve_avoidance_ems_b,
        )
        from amm.formatters import _parse_inventory_text
        
        if (
            DEFAULT_CONFIG.enable_em_retrieval
            and DEFAULT_CONFIG.enable_t4_retrieval
        ):
            logger.info("[T4 Trigger] Sage invoked → retrieving S2 EMs for planning")
            
            current_room = get_current_room(look) or "unknown"
            inventory_items = _parse_inventory_text(inventory)
            
            rewards_window = recent_reward[-5:] if len(recent_reward) > 5 else recent_reward
            scores_window = recent_scores[-5:] if (recent_scores and len(recent_scores) > 5) else (recent_scores or [])
            current_score_val = current_score if current_score is not None else (
                scores_window[-1] * 100 if scores_window else 0.0
            )
            
            actions_window = recent_actions[-5:] if len(recent_actions) > 5 else recent_actions
            obs_window = recent_obs[-5:] if len(recent_obs) > 5 else recent_obs
            
            # S2 retrieval (success + partial)
            query_s2 = build_success_retrieval_query_s2(
                task_description=task_description,
                room_name=current_room,
                inventory_items=inventory_items,
                recent_rewards=rewards_window,
                current_score=current_score_val,
                look_description=look,
                recent_actions=actions_window,
                recent_observations=obs_window,
            )
            success_ems = retrieve_success_ems_s2(
                memory_agent_id=amm_client.agent_id,
                query_text=query_s2,
                letta_client=amm_client,
            )
            logger.info(
                "[T4 Trigger] S2 retrieval returned %d episodic memories",
                len(success_ems),
            )
            episodic_memories_for_planning.extend(success_ems)
            
            avoidance_ems: List[Dict[str, Any]] = []
            # Optional B (avoidance) retrieval when there are failed/invalid actions
            if DEFAULT_CONFIG.enable_t3_retrieval and failed_messages:
                logger.info(
                    "[T4 Trigger] Failed/invalid actions detected (failed_messages non-empty) "
                    "→ retrieving B (avoidance) EMs"
                )
                query_b = build_avoidance_retrieval_query_b(
                    task_description=task_description,
                    room_name=current_room,
                    inventory_items=inventory_items,
                    recent_rewards=rewards_window,
                    current_score=current_score_val,
                    look_description=look,
                    recent_actions=actions_window,
                    recent_observations=obs_window,
                )
                avoidance_ems = retrieve_avoidance_ems_b(
                    memory_agent_id=amm_client.agent_id,
                    query_text=query_b,
                    letta_client=amm_client,
                )
                logger.info(
                    "[T4 Trigger] B retrieval returned %d episodic memories",
                    len(avoidance_ems),
                )
                episodic_memories_for_planning.extend(avoidance_ems)
            
            logger.info(
                "[T4 Trigger] Total EMs collected for planning (backbone only) = %d (S2=%d, B=%d)",
                len(episodic_memories_for_planning),
                len(success_ems),
                len(avoidance_ems),
            )
            # End of T4 retrieval block
    # ================================================================

    real_action_list = []
    try:
        # 1) Planning
        enc    = tiktoken.get_encoding("cl100k_base")
        demos  = demo_data[str(task_id)]
        prompt_to_plan = compose_prompt_to_plan(
            demos, useful_focus_on, task_description,
            recent_actions, recent_obs, recent_locs, recent_looks,
            failed_messages, look, inventory, fast_action,
            version="full"
        )
        # Memory-based augmentation (prepend Past Episodes)
        if use_memory_planning and episodic_memories:
            try:
                pe_block = format_past_episodes(episodic_memories)
            except Exception:
                pe_block = ""
            if pe_block:
                prompt_to_plan = pe_block + "\n\n" + prompt_to_plan
                logger.debug("===== AUGMENTED PROMPT TO PLAN (with Past Episodes) =====\n" + prompt_to_plan)
                logger.debug(f"Using {min(5, len(episodic_memories))} past episodes in planning.")
        # fallback to lite if too long
        if len(enc.encode(prompt_to_plan)) >= 8000:
            prompt_to_plan = compose_prompt_to_plan(
                demos, useful_focus_on, task_description,
                recent_actions, recent_obs, recent_locs, recent_looks,
                failed_messages, look, inventory, fast_action,
                version="lite"
            )

        logger.info("PROMPT TO PLAN:\n" + prompt_to_plan)
        if llm is None:
            resp = completion_with_backoff(
                model=gpt_version,
                messages=[{"parts": [{"text": prompt_to_plan}]}]
            )    
            response_plan = resp.candidates[0].content.parts[0].text
        else:
            response_plan = local_llm.generate(prompt_to_plan, logger=logger.info)

        logger.info("RESPONSE PLAN:\n" + response_plan)

        # 2) Next actions
        time.sleep(1)
        prompt_to_next_actions = compose_prompt_to_nextactions(
            demos, task_description,
            recent_actions, recent_obs, recent_locs, failed_messages,
            look, inventory, response_plan, useful_focus_on,
            k=10, version=gpt_version
        )
        logger.info("PROMPT TO NEXT ACTIONS:\n" + prompt_to_next_actions)
        if llm is None:
            resp2 = completion_with_backoff(
                model=gpt_version,
                messages=[{"parts": [{"text": prompt_to_next_actions}]}],
                temperature=0, top_p=1
            )            
            response_next_actions = resp2.candidates[0].content.parts[0].text
        else:
            response_next_actions = local_llm.generate(prompt_to_next_actions)

        # Post‐process
        def post_process(text):
            logger.info("RAW NEXT ACTIONS:\n" + text)
            lines = text.split("\n")[:5]
            ra, go = [], []
            for line in lines:
                if "repeat" in line.lower():
                    chunk = ra[-3:] if "wait" in ra[-1].lower() else ra[-2:]
                    obs_chunk = go[-3:]
                    ra += chunk * 5
                    go += obs_chunk * 5
                    if "until" in line.lower():
                        break
                    continue
                if ":" not in line or "(" not in line or ")" not in line:
                    continue
                a = line[line.index(":")+1:line.rindex(")")+1].strip()
                obs = line.split("-->")[-1].strip() if "-->" in line else "None"
                a = recover_action(a)
                if a:
                    ra.append(a)
                    go.append(obs)
            logger.info(f"Parsed actions: {ra}")
            return ra, go

        real_action_list, guess_obs_list = post_process(response_next_actions)

    except Exception as e:
        logger.info("Gemini planning error: " + str(e))
        # fallback to SBERT
        fb = try_to_replace(predictions[0], validActions, look, inventory)
        fb_action = sbert_search([fb], validActions, sbert_model, logger)
        return False, fb_action

    if not real_action_list:
        # secondary fallback
        fb = try_to_replace(predictions[0], validActions, look, inventory)
        fb_action = sbert_search([fb], validActions, sbert_model, logger)
        return False, fb_action

    return True, (real_action_list, guess_obs_list)




def compose_prompt_to_nextactions(demos, task_desc, recent_actions, recent_obs, recent_locs, failed_messages, look, inventory, response_next_subgoal, useful_focus_on, fast_action=None, k=10, version="gemini-2.5-flash-preview-04-17"):

    prompt_to_next_actions = []
    prompt_to_next_actions.append("You are an experienced teacher who always guide students to complete the science experiments. Now let's do science experiments with a sequence of actions.")
    prompt_to_next_actions.append("In this environment, there are a few locations: art studio, workshop, kitchen, living room, bedroom, bathroom, foundry, greenhouse, outside, and a hallway connecting them.")
        
    prompt_to_next_actions.append("You have done a few science experiments successfully and below are the action history of your experiments with similar tasks.")

    prompt_to_next_actions.append("Example task 1: "+ demos[0][0])
    prompt_to_next_actions += demos[0][1:]
    if len(demos) >= 2:
        prompt_to_next_actions.append("Example task 2: "+ demos[1][0])
        prompt_to_next_actions += demos[1][1:]
    # prompt_to_next_actions += ["- Action: "+ a for a in demos[1][1:]]


    prompt_to_next_actions.append("In a new science experiment that is similar to the above two, " + task_desc.replace("Your", "my"))
    
    # prompt_to_next_actions.append("Given the above completed subgoals, what should be your next subgoal to complete for finishing the task?")
    
    prompt_to_next_actions.append(f"My previous {k} actions and observations are as follows:")

    recent_actions, recent_obs, _, _, recent_locs = clean_history(recent_actions, recent_obs, [-1]*len(recent_actions), [-1]*len(recent_actions), recent_locs)
        

    history = []
    repeat = 0    
    for ind, (l, a, o) in enumerate(zip(recent_locs[:], recent_actions[:], recent_obs[:])):
        if o == "N/A":
            continue 
        fa = formalize_action(a)
        if "(" not in fa:
            continue
        at = fa[:fa.index("(")]
        if at not in "\n".join(demos[0][1:]):
            # Skipping the actions with types not in the demos
            continue
        to_add = f"- (in {l}) Action: {fa} --> {o}"
        if ind+1 < len(recent_actions) and a in recent_actions[max(0, ind-5):ind] and a in recent_actions[ind+1:min(len(recent_actions), ind+5)]:
            repeat += 1
            continue 
        
        history.append(to_add) 
        if repeat > 0:
            history.append(f"Repeat the above action for {repeat} times.")             
            repeat = 0
    # prompt_to_next_actions.append()
    prompt_to_next_actions += history[-k:]

    if useful_focus_on:
        prompt_to_next_actions.append("Importantly, I have FOCUS on these things already: " + ", ".join([fo.replace("focus on", "") for fo in  useful_focus_on]))
    else:
        prompt_to_next_actions.append("Importantly, I have FOCUS on nothing yet.")

    pattern = r"focus on\s+(\b\w+\b(\s+\b\w+\b)*)"
    matches = re.findall(pattern, task_desc)
    to_focus = [match[0].replace("the ", " ").strip() for match in matches]
    pattern = r"find\s+(\b\w+\b(\s+\b\w+\b)*)"
    matches = re.findall(pattern, task_desc.replace("a(n)", "a"))
    to_focus_v2 = [match[0].replace("the ", " ").strip() for match in matches]

    # prompt_to_next_actions.append("You have completed these subgoals:")
    # prompt_to_next_actions.append(response_previous_subgoals)
    prompt_to_next_actions.append("However, my actions so far cannot complete the task now. I do not know what to do for the next steps.")
    if failed_messages:
        failed_messages = set(failed_messages)
        prompt_to_next_actions.append("There are some error messages about my previous actions:")
        prompt_to_next_actions += failed_messages
    prompt_to_next_actions.append("I asked my teacher for advice and the teacher told me these advice:")
    prompt_to_next_actions.append(response_next_subgoal.replace("Question", "Answer").replace("Answer", "Advice")) 
    prompt_to_next_actions.append("")
    prompt_to_next_actions.append("In current environment: " + clean_look(look) + "\n" + inventory)
    prompt_to_next_actions.append("What should be my next actions to complete the next subgoal in the current environment? ")
    prompt_to_next_actions.append("If any of the suggested next subgoals need knowledge to make decisions (e.g., determining or comparing the properties of objects and animals), please do that for me.")
    prompt_to_next_actions.append("The ONLY allowed action types are:")
    for ai in action_type_description:
        at = ai['action_type']
        at = at[:at.index("(")]
        if at not in "\n".join(demos[0][1:] + demos[0][2:]):
            continue
        prompt_to_next_actions.append(f"- {ai['action_type']} : {ai['desc']} ")   

    prompt_to_next_actions.append(f"Important! You can only use FOCUS actions on these items: {', '.join(to_focus)} . ") # (Hint: {','.join(to_focus_v2)})
    prompt_to_next_actions.append("You cannot FOCUS on any other things. Please only use FOCUS as required by the task description. Also, please FOCUS more directly, try not to focus on the container.")

    prompt_to_next_actions.append("Please use the above mentioned action types to convert the unfinished subgoal to a short sequence of concrete actions.  DO NOT USER OTHER TYPES OF ACTIONS. Follow the report of the two example tasks shown to you previously.")    
    prompt_to_next_actions.append("Please do not try to look for books or computers to look up information. You will need to use your own commonsense knowledge to make decisions (e.g., determining properties of objects and animals).")
    prompt_to_next_actions.append("Note that I can only do actions with available objects in the current location or inventory!!") 
    prompt_to_next_actions.append("Please use the below format to organize the response.")
    prompt_to_next_actions.append("Action 1: [...] -->  \n Action 2: [...] --> \n ...")
    return "\n".join(prompt_to_next_actions)

 
def compose_prompt_to_plan(demos, useful_focus_on, task_desc, recent_actions, recent_obs, recent_locs, recent_looks, failed_messages, look, inventory, fast_action, version="full"):
    clean_obs = []
    assert len(recent_obs) == len(recent_locs)
    repeat = 0 
    for i, obs in enumerate(recent_obs[1:]):
        # if obs.startswith("This room is called"):
        #     end_index = obs.index("In it")
        #     obs = obs[:end_index]
        if obs.startswith("You move to the") or obs.startswith("You go to the") or obs.startswith("You teleport to the"):
            obs = obs.replace("go to", "move to").replace("teleport to", "move to")
        if obs == "The door is already open.":
            continue
        # if obs.startswith("a substance called"): 
        if f"In {recent_locs[i+1]}, {obs}" in clean_obs:
            continue
        
        if recent_actions[i+1] in recent_actions[i+1-5:i+1] and recent_actions[i+1] in recent_actions[i+2:i+2+5]:
            repeat += 1
            continue
        if "move to the" in obs:
            clean_obs.append(f"{obs}")
        else:
            if version == "lite":
                clean_obs.append(f"In {recent_locs[i+1]}, {obs}")
            else:
                clean_obs.append(f"In {recent_locs[i+1]}, {recent_actions[i+1]} --> {obs}")

        if repeat > 0: 
            clean_obs.append(f"Repeat the above {repeat} times.")        
            repeat = 0
    final_obs = []
    for i, co in enumerate(clean_obs):
        if i+1 < len(clean_obs) and "move to the" in clean_obs[i] and "move to the" in clean_obs[i+1]:
            continue
        final_obs.append(co.replace("a substance called", "there is a"))
    prev_obs = [f"- {j+1}. {o}" for j, o in enumerate(final_obs)]

    
    prompt_to_plan  = []

    prompt_to_plan.append("You are an experienced teacher who always guides students to complete the science experiments by giving executable advice and instructions with world knowledge.")

    prompt_to_plan.append("You have done a science experiment successfully and below is the action history of your experiment.")

    prompt_to_plan.append("Example task: "+ demos[0][0])
    clean_actions = []
    for history in demos[0][1:]:
        if "Action: " not in history:
            continue
        start_ind = history.index("Action: ") + len("Action: ")
        end_ind = history.index(" -->")
        action = history[start_ind:end_ind]
        action = recover_action(action)
        if action is not None:
            clean_actions.append(history[:start_ind] + action + history[end_ind:])
    prompt_to_plan += clean_actions

    prompt_to_plan.append("In a new science experiment that is similar to the above one, " + task_desc.replace("Your", "my")) 
    prompt_to_plan.append("In this environment, there are a few rooms: art studio, workshop, kitchen, living room, bedroom, bathroom, foundry, greenhouse, outside, and a hallway connecting them.")
    prompt_to_plan.append("To complete this task, I have done some actions and the observations are listed here:")
    if version == "lite":
        prev_obs = prev_obs[-15:]
    prompt_to_plan += prev_obs
    # print(recent_looks)
    # print(recent_locs)
    if len(recent_looks) >= 2 and version != "lite":
        prompt_to_plan.append("In some previously visited locations:")    
        for location, look_round in recent_looks.items():
            if location != recent_locs[-1]:
                prompt_to_plan.append(f"In {location}: " + clean_look(look_round, version="lite"))
    prompt_to_plan.append("* Current location *: " + clean_look(look)) # + look.replace(" egg", " ").replace(" adult ", " ").replace(" baby ", " ")
    prompt_to_plan.append(inventory.replace("Your ", "My "))
    if useful_focus_on:
        prompt_to_plan.append("Importantly, I have FOCUS on these things already: " + ", ".join([fo.replace("focus on", "") for fo in  useful_focus_on]))
    else:
        prompt_to_plan.append("Importantly, I have FOCUS on nothing yet.")
    # prompt_to_plan.append("However, my actions so far cannot complete the task. I do not know what to do for the next steps.")
    prompt_to_plan.append("However, I do not know what to do for the next steps.")
    if fast_action:
        prompt_to_plan.append(f"My instinct tells me that it might be reasonable to {fast_action} now but I'm not so sure.")
    if failed_messages:
        failed_messages = set(failed_messages)
        failed_messages = set(failed_messages)
        prompt_to_plan.append("There are some error messages about my previous actions:")
        prompt_to_plan += failed_messages
    prompt_to_plan.append("Please review the task description and the previous observations and then answer the following questions to help me plan for efficiently completing the next subgoal.")
    prompt_to_plan.append("Question 1: To efficiently complete the task, what substance and objects do I need to collect? Please list them and their possible locations one by one. Please ignore protective gears because I have them already.")
    prompt_to_plan.append("Question 2: Based on your answer to Question 1, are there any substance or objects that are not in my inventory now and I should keep looking for?" + \
                          " If so, which rooms are they likely to be? " + \
                          "Note that some of your suggested items might not exist in the rooms. In that case, let's try to use the similar ones in the environment." + \
                          " Note that I cannot do actions without them if they are not collected yet. ")
   

    pattern = r"focus on\s+(\b\w+\b(\s+\b\w+\b)*)"
    matches = re.findall(pattern, task_desc)
    to_focus = [match[0].replace("the ", " ").strip() for match in matches]

    prompt_to_plan.append("Question 3: To most efficiently complete the task, what will be the important subgoals to finish? Please list up to five subgoals." + \
                          f" Importantly, please include the subgoals about 'focus on' as required in the task description. Remember that it is ONLY possible focus on these items: {', '.join(to_focus)}! You should NOT focus on other things!! If you list a subgoal of focusing on, make sure that is mentioned and required by the task.")
    prompt_to_plan.append("Question 4: In these subgoals, what have I already completed based on the previous observations? And which subgoals should I aim to do right now?" + \
                          " These subgoals may need additional common knowledge to make decisions. Please recall the knowledge about the properties of objects or animals. Think step by step, and list the facts that are useful. And then use them for determining or comparing if needed. Finally, list the next subgoals based on the knowledge and current observations.")
    prompt_to_plan.append("Question 5: Based on the observations, did I make any mistakes that prevent me from efficiently finishing the next subgoals? Did I forget to go to a location to pick up thing? Or did I forget to open/activate/move something? Did I repeat any actions too many times? If so, how should I fix it?")
    prompt_to_plan.append("Please do not try to look for books or computers to look up information. You will need to use your own commonsense knowledge to make decisions (e.g., determining properties of objects and animals).")
    prompt_to_plan.append("Please read the task description carefully, and think step by step to answer these questions one by one. Please be concise. Thank you very much.")
    return '\n'.join(prompt_to_plan)


# ================= Adaptive Memory: Past Episodes Utilities =================
def parse_episode(content):
    """Parse a SUCCESS-style episodic memory string.

    Returns dict with keys: room, action, observation, reward_delta, task.
    On failure, returns None.
    """
    if not content or not isinstance(content, str):
        return None
    text = content.strip()
    try:
        # Task
        task_match = re.search(r'While\s+working\s+on\s+the\s+task:\s*"([\s\S]*?)"', text, re.IGNORECASE)
        task = task_match.group(1).strip() if task_match else None

        # Room/location (" at <room>," pattern)
        room_match = re.search(r'\bat\s+([A-Za-z\s]+?),\s*\n?', text, re.IGNORECASE)
        room = room_match.group(1).strip().lower() if room_match else None

        # Action and Observation (the action 'X' caused 'Y'.)
        action_match = re.search(r"the\s+action\s+'([^']+)'\s+caused\s+'([^']+)'", text, re.IGNORECASE)
        action = action_match.group(1).strip() if action_match else None
        observation = action_match.group(2).strip() if action_match else None

        # Reward delta (This resulted in a reward: <num>)
        reward_match = re.search(r'This\s+resulted\s+in\s+a\s+reward:\s*([-+]?\d+(?:\.\d+)?)', text, re.IGNORECASE)
        reward_delta = float(reward_match.group(1)) if reward_match else None

        if not (room and action and observation and task is not None and reward_delta is not None):
            return None

        return {
            "room": room,
            "action": action,
            "observation": observation,
            "reward_delta": reward_delta,
            "task": task,
        }
    except Exception:
        return None


def format_past_episodes(mem_list):
    """Format a compact Past Episodes block from episodic memories.

    mem_list: list of dicts with at least {timestamp, content}
    Returns a string or empty string if nothing to show.
    """
    if not mem_list:
        return ""

    parsed = []
    for m in mem_list[:10]:  # attempt parse beyond 5 in case of skips
        content = m.get("content", "")
        info = parse_episode(content)
        if not info:
            continue
        parsed.append(info)
        if len(parsed) >= 5:
            break

    if not parsed:
        return ""

    lines = ["Past Episodes (most related first, max 5)"]

    # Build bullets
    for p in parsed:
        room = p["room"]
        action = p["action"]
        observation = p["observation"][:120]
        rd = p["reward_delta"]
        rd_str = ("+" if rd is not None and rd > 0 else "") + (f"{int(rd)}" if rd is not None and float(rd).is_integer() else f"{rd}")
        # shorten task
        task = p["task"].replace("\n", " ").strip()
        if len(task) > 80:
            task = task[:77] + "..."
        bullet = f"• [{room}] — [{action}] → [{observation}] (Δscore: {rd_str}; task: \"{task}\")"
        lines.append(bullet)

    # Optional hint if same action in same room appears multiple times
    freq = {}
    for p in parsed:
        key = (p["room"], p["action"])
        freq[key] = freq.get(key, 0) + 1
    common = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    if common and common[0][1] >= 2:
        (room, action), _ = common[0]
        lines.append(f"Hint: actions that helped in [{room}]: [{action}].")

    block = "\n".join(lines)
    # Hard cap length to ~1200 chars
    if len(block) > 1200:
        block = block[:1197] + "..."
    return block

def clean_history(recent_actions, recent_obs, recent_score, recent_reward, recent_locs):
    assert len(recent_actions) == len(recent_obs) == len(recent_score) == len(recent_reward) == len(recent_locs)
    N = len(recent_actions)
    inds_to_remove = []
    for ind in range(N):
        if recent_actions[ind].startswith("examine"):
            inds_to_remove.append(ind)
        if recent_actions[ind].startswith("teleport to") and recent_score[ind] >= 0:
            recent_actions[ind] = recent_actions[ind].replace("teleport", "go")
            recent_obs[ind] = recent_obs[ind].replace("teleport", "go")
        if recent_actions[ind].startswith("go to") and recent_score[ind] < 0:
            recent_actions[ind] = recent_actions[ind].replace("go", "teleport")
            recent_obs[ind] = recent_obs[ind].replace("go", "teleport")
        if recent_actions[ind].startswith("open door") and recent_score[ind] < 0:
            inds_to_remove.append(ind)
        if recent_actions[ind] in recent_actions[ind+1: min(ind+3, N)] and recent_score[ind] >= 0 :
            inds_to_remove.append(ind)
    
    recent_actions = [item for idx, item in enumerate(recent_actions) if idx not in inds_to_remove]
    recent_obs = [item for idx, item in enumerate(recent_obs) if idx not in inds_to_remove]
    recent_score = [item for idx, item in enumerate(recent_score) if idx not in inds_to_remove]
    recent_reward = [item for idx, item in enumerate(recent_reward) if idx not in inds_to_remove]
    recent_locs = [item for idx, item in enumerate(recent_locs) if idx not in inds_to_remove]
    return recent_actions, recent_obs, recent_score, recent_reward, recent_locs

    
def get_model_output(args, input_str, tokenizer, lm_model, device, logger): 
    input_ids = tokenizer(input_str, return_tensors="pt", max_length=args["max_input_len"] , truncation=True).input_ids

    sample_outputs = lm_model.generate(
        input_ids.to(device),
        max_length=16,
        num_return_sequences=args['beams'],
        num_beams=args['beams'],
    )
 
    lm_pred = sample_outputs

    # Take the first prediction that is not "look around"
    logger.info("Top N Predictions:")
    predStrs = []
    for i, pred in enumerate(lm_pred):
        text = tokenizer.decode(pred)
        text = post_process_generation(text)
        logger.info("\t" + str(i) + "\t" + str(text) )
        predStrs.append(text)

    return predStrs


def post_process_generation(raw_pred):
    ans_match = re.match(r".*<extra_id_0>(.*)<extra_id_1>.*", raw_pred)
    if ans_match is not None:
        result = ans_match.group(1)
    else:
        result = raw_pred

    # remove extra <*>'s left in
    result = result.replace("<", " <")
    out = ""
    for token in result.split(" "):
        if (len(token.strip()) > 0):
            if (token[0] != "<"):
                out += token + " "
    result = out

    return result.strip()


def gpt_select_valid(action, candidates, look, inventory, goal, logger, n=1, gpt_version="gpt-4", llm=None):
    prompt_to_search = []
    prompt_to_search.append("Let's play a text game.")
    prompt_to_search.append(clean_look(look, version="all"))
    prompt_to_search.append(inventory)
    prompt_to_search.append("There are some action candidates as follows:")
    for ac in candidates:
        prompt_to_search.append(f"- {ac}")
    prompt_to_search.append(f"\n I want to achieve this goal: {goal} but my action '{action}' is not in the candidate list.")
    prompt_to_search.append(f"Please consider the objects in the room and inventory and my goal. Think carefully, and then select the best replacement from the list. If no one in the list is a good replacement, return 'none'.")
    prompt_to_search.append(f"Selected action:") 

    prompt_to_search = "\n".join(prompt_to_search)
    logger("-"*30 + "prompt_to_search" + "-"*30)
    logger("\n"+prompt_to_search)
    logger("-"*35 + "-"*35)
    if llm is None:
        responses = completion_with_backoff(model=gpt_version,
               messages=[{"parts": [{"text": prompt_to_search}]}])

        selections = [responses.candidates[i].content.parts[0].text for i in range(n)]
    else:    
        selections = local_llm.generate(prompt_to_search)
    return selections


def rank_candidates_by_common_words(query, candidates):
    """
    Rank the candidates based on their edit distance to the query.
    """

    # the first word must be the same 
    candidates = [va for va in candidates if va.split()[0] == query.split()[0]]
    
    # Compute the edit distance between each candidate and the query
    num_commons = [len(set(query.split()) & set(candidate.split())) for candidate in candidates]
    
    # Sort the candidates based on their distance to the query
    ranked_candidates = [candidate for _, candidate in sorted(zip(num_commons, candidates), reverse=True)]
    
    return ranked_candidates

if __name__ == "__main__":  
    print()