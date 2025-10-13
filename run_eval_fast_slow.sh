num_gpus=$1  
seed=42
split="test"
gpt_version="gemini-2.5-flash-preview-04-17"
# gpt_version="gpt-4"


if [ $num_gpus -eq 1 ]; then
    task_nums=("6")
    L=1
elif [ $num_gpus -eq 4 ]; then
    task_nums=("0,12,20,16" "26,13,2,28" "22,17,3,10" "1,4,5,29" "18,14,11,15" "25,6,27,24" "19,8,9" "21,23,7")
    L=8
fi 

output_path="fast_slow_logs/${split}_all_0512_${gpt_version}/"
mkdir -p $output_path
echo "---> $output_path" 
 
cp eval_agent_fast_slow.py $output_path/
cp eval_utils.py $output_path/
cp data_utils/demos.json $output_path/
cp data_utils/data_utils.py $output_path/

for ((i=0; i<L; i++)); do
    task_num=${task_nums[$i]}
    # ((gpu=i%num_gpus)) # the number of gpus
    gpu=$i
    echo $task_num "on" $gpu    
    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu python eval_agent_fast_slow.py \
        --task_nums $task_num \
        --set ${split} \
        --seed $seed \
        --debug_var -1 \
        --gpt_version $gpt_version \
        --output_path $output_path & # > /dev/null 2>&1 &
    sleep 10
done
