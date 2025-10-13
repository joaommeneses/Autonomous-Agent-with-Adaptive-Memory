export CUDA_VISIBLE_DEVICES=1

for task in 25
do
    python eval_agent_react.py \
        --task_nums $task \
        --set test \
        --no_stop \
        --env_step_limit 100 \
        --simplification_str easy \
        --prompt_file ReAct_baseline/prompt.jsonl \
        --output_path ReAct_logs/gemini-2.5-flash-preview-04-17 \
        --model_name gemini-2.5-flash-preview-04-17
done