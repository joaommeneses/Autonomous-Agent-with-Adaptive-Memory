import argparse
import os
import re
import time
import random
from scienceworld import ScienceWorldEnv
import json
from data_utils.data_utils import sanitizeStr
from eval_utils import findValidActionNew, load_variation
from slow_agent.utils import completion_with_backoff
import tiktoken
# import openai
from scienceworld.constants import BASEPATH, DEBUG_MODE, ID2TASK, JAR_PATH, NAME2ID
from scienceworld.utils import infer_task
from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway, CallbackServerParameters

# openai.api_key = os.environ["OPENAI_API_KEY"]
from google import genai
from dotenv import load_dotenv

load_dotenv()
# instantiate a single client with your API key (or read from ENV)
genai_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
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

def clean(s):
    clean_toks = ['\n', '\t']
    for tok in clean_toks:
        s = s.replace(tok, ' ')
    return s

# Call language model
def llm_gpt(prompt, stop=["\n"], gpt_version="gemini-2.5-flash-preview-04-17"):
    resp = completion_with_backoff(
                model=gpt_version,
                messages=[{"parts": [{"text": prompt}]}]
            )    
    return resp.candidates[0].content.parts[0].text

# def llm(prompt, stop=["\n"]):
#     response = openai.Completion.create(
#       model="text-davinci-002",
#       prompt=prompt,
#       temperature=0,
#       max_tokens=50,
#       top_p=1,
#       frequency_penalty=0.0,
#       presence_penalty=0.0,
#       stop=stop
#     )
#     return response["choices"][0]["text"]

def get_file_name(args, task_num):
    if (len(args["output_path"]) > 0):
        args["output_path"] = args["output_path"] + "/"

        # Make path if it doesn't exist
        if (not os.path.exists(args['output_path'])):
            os.makedirs(args["output_path"])

    # filenameOutPrefix = args["output_path"] + "transformer-" + args["mode"] + "-eval-" + str(args["lm_path"].split('/')[-1]) + "-task" + str(task_num)
    filenameOutPrefixSeed = args["output_path"] + "task" + str(task_num)

    return filenameOutPrefixSeed
  


# Example user input console, to play through a game.
def eval(args, task_num, logger):

    # Initialize environment
    # env = ScienceWorldEnv("", args["jar_path"], envStepLimit = args["env_step_limit"], threadNum = 0)
    env = MyScienceWorldEnv("", args["jar_path"], envStepLimit = args["env_step_limit"])
    taskNames = env.getTaskNames()
    taskName = taskNames[task_num]
    env.load(taskName, 0, args['simplification_str'])
    variations = load_variation(env, args, task_num, logger)
    filenameOutPrefixSeed = get_file_name(args, task_num)

    # Load init prompt
    with open(args["prompt_file"], 'r') as f:
        d = json.load(f)
    
    # Load encoding tool to count token numbers
    encoding = tiktoken.get_encoding("cl100k_base")
    # plans = get_plans(args)

    scores = []

    for variation in variations:

        # train_data = []
        env.load(taskName, variation, args["simplification_str"], generateGoldPath=True)
        task_description = env.taskdescription()[18:]
        # task_description = env.taskdescription()  
        recent_actions = ["look around"]
 
        obs, info = env.reset()

        done = False
        score = 0.0
        last_score = 0.0
        step = 0

        # The env has an internal step count, some actions like look around are free
        # however, the t5 model only generates the action "look around", which will result in a dead loop below
        # so the max_steps here is only used to avoid the model generating the same action forever
        max_steps = args["env_step_limit"] * 2
 
        init_prompt = 'Interact with a household to solve a task. Here is an example.\n' + d[str(task_num)]
        prompt = '\n\nHere is the task.\n' + clean(obs) + '\n' + task_description + '\n>'

        # Different models have different maximun token numbers
        if args["model_name"] == "gpt-3.5-turbo":
            max_len = 4096
        elif args["model_name"] == "gpt-4":
            max_len = 8192
        else:
            max_len = 20000

        while not done:        

            # Cut the prompt to make it shorter than maximun token numbers
            while len(encoding.encode(init_prompt + prompt)) > max_len - 60:
                index1 = init_prompt.find('>')

                # If init prompt doesn't have actions, cut game prompt
                if index1 == -1:
                    index1_prompt = prompt.find('>')
                    index2_prompt = prompt.find('>', index1_prompt+1)
                    prompt = prompt[:index1_prompt] + prompt[index2_prompt:]

                # Cut initial prompt
                else:
                    index2 = init_prompt.find('>', index1+1)
                    if index2 == -1:
                        init_prompt = init_prompt[:index1]
                    else:
                        init_prompt = init_prompt[:index1] + init_prompt[index2:]

            logger.info("Prompt: " + init_prompt + prompt)
            # action = llm(init_prompt + prompt, stop=['\n']).strip()
            action = llm_gpt(init_prompt + prompt, stop=['\n'], gpt_version="gemini-2.5-flash-preview-04-17").strip()

            # Don't need to actually do think actions
            if action.startswith('think:'):
                obs = 'OK.'
            else:
                # Get valid actions at this point
                action = findValidActionNew([action], env, info['look'], recent_actions, None, logger)
                obs, reward, done, info = env.step(action)

                score = info['score']

                if score < 0:
                    # Our own solution for dealing with such cases
                    if args["no_stop"]:
                        done = True
                        score = last_score
                    else:
                        done = True
                        score = 0
                last_score = score
            
            obs = clean(obs)

            # Add action and observaton to game prompt
            prompt += f' {action}\n{obs}\n>'
            
            recent_actions.append(action) 
            
            #logger.info("Input string: " + str(input_str))
            logger.info(f"Variation: {variation}, Step: {step}, Action: {action}")
            logger.info("Obs: " + obs)
            logger.info(f"Score: {score}")
            logger.info("")

            step += 1
            if (step >= max_steps) or done:
                break
  

            logger.info("Recent Actions: " + str(recent_actions))

            # Early stopping if we're in a loop
            if len(recent_actions) >= 5 and len(set(recent_actions[-5:])) == 2:
                logger.info("Many recent actions in history are the same -- model is likely in a loop, stopping early.")
                break


        # Store results
        env.storeRunHistory(variation, notes = {'mode':"react_baseline", 'lm': None} )
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
    parser.add_argument("--jar_path", type=str) 
    parser.add_argument("--task_nums", default="0")  # use comma to split 
    parser.add_argument("--env_step_limit", type=int, default=100)
    parser.add_argument("--simplification_str", default="easy")
    parser.add_argument("--max_episode_per_file", type=int, default=9999)
    parser.add_argument("--set", default="test_mini")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--no_stop", action="store_true", default=False)
    parser.add_argument("--prompt_file", default="ReAct_baseline/prompt.jsonl")
    parser.add_argument("--model_name", default="gemini-2.5-flash-preview-04-17")

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

    task_nums = args["task_nums"].split(",")
    for task_num in task_nums:
        logger = init_logger(args, task_num)
        logger.info(args)
        eval(args, int(task_num), logger)
        
if __name__ == "__main__":
    main()