num_gpus=1
if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    num_gpus="$1"
    shift
elif [ $# -gt 0 ]; then
    args_arr=("$@")
    last_idx=$((${#args_arr[@]} - 1))
    last_arg="${args_arr[$last_idx]}"
    if [[ "$last_arg" =~ ^[0-9]+$ ]]; then
        num_gpus="$last_arg"
        unset 'args_arr[$last_idx]'
        set -- "${args_arr[@]}"
    fi
fi

seed=42
split="test"
gpt_version="qwen2.5-1m-instruct"

mode="baseline"
disable_swift_injection=false
amm_write_only=false

pass_through_args=()
for arg in "$@"; do
    case "$arg" in
        --use-amm)
            if [ "$mode" != "baseline" ]; then
                echo "Error: mode flags are mutually exclusive. Use only one of --use-amm, --use-srm, --use-full."
                exit 2
            fi
            mode="use-amm"
            ;;
        --use-srm)
            if [ "$mode" != "baseline" ]; then
                echo "Error: mode flags are mutually exclusive. Use only one of --use-amm, --use-srm, --use-full."
                exit 2
            fi
            mode="use-srm"
            ;;
        --use-full)
            if [ "$mode" != "baseline" ]; then
                echo "Error: mode flags are mutually exclusive. Use only one of --use-amm, --use-srm, --use-full."
                exit 2
            fi
            mode="use-full"
            ;;
        --disable-swift-injection|--disable_amm_swift_injection)
            disable_swift_injection=true
            ;;
        --amm-write-only|--amm_write_only) amm_write_only=true ;;
        *) pass_through_args+=("$arg") ;;
    esac
done

if [ "$num_gpus" != "1" ] && [ "$num_gpus" != "2" ] && [ "$num_gpus" != "4" ]; then
    echo "Error: num_gpus must be one of 1, 2, or 4 (got '$num_gpus')."
    exit 2
fi

if [ "$disable_swift_injection" = true ] && [ "$mode" != "use-amm" ] && [ "$mode" != "use-full" ]; then
    echo "Error: --disable-swift-injection requires AMM mode (--use-amm or --use-full)."
    exit 2
fi

extra_flags=""
if [ "$mode" = "use-amm" ]; then
    extra_flags="$extra_flags --use_amm --disable_srm"
elif [ "$mode" = "use-srm" ]; then
    extra_flags="$extra_flags --enable-srm"
elif [ "$mode" = "use-full" ]; then
    extra_flags="$extra_flags --use_amm --enable-srm"
else
    # baseline-only (rely on python defaults: AMM OFF, SRM OFF)
    extra_flags="$extra_flags"
fi
if [ "$amm_write_only" = true ]; then
    extra_flags="$extra_flags --amm_write_only"
fi
if [ "$disable_swift_injection" = true ]; then
    extra_flags="$extra_flags --disable_amm_swift_injection"
fi

if [ "$mode" = "use-amm" ]; then
    profile="baseline+AMM"
elif [ "$mode" = "use-srm" ]; then
    profile="baseline+SRM"
elif [ "$mode" = "use-full" ]; then
    if [ "$disable_swift_injection" = true ]; then
        profile="full-no-swift-injection"
    else
        profile="full"
    fi
else
    profile="baseline-only"
fi
echo "Config profile: $profile (flags: $extra_flags)"

if [ $num_gpus -eq 1 ]; then
    task_nums=("25,26,27,28,29")
    L=1
elif [ $num_gpus -eq 2 ]; then
    task_nums=("10", "12,14")
    L=2
elif [ $num_gpus -eq 4 ]; then
    task_nums=("0,12,20,16" "26,13,2,28" "22,17,3,10" "1,4,5,29" "18,14,11,15" "25,6,27,24" "19,8,9" "21,23,7")
    L=8
fi 

output_path="fast_slow_logs/${split}_all_0512_${gpt_version}_${profile}/"
mkdir -p $output_path
echo "---> $output_path" 
 
cp eval_agent_fast_slow.py $output_path/
cp eval_utils.py $output_path/
cp data_utils/demos.json $output_path/
cp data_utils/data_utils.py $output_path/

for ((i=0; i<L; i++)); do
    task_num=${task_nums[$i]}
    #((gpu=i%num_gpus)) # the number of gpus
    gpu=1
    echo $task_num "on" $gpu    
    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu python eval_agent_fast_slow.py \
        --task_nums $task_num \
        --set ${split} \
        --seed $seed \
        --debug_var -1 \
        --gpt_version $gpt_version \
        --output_path $output_path \
        $extra_flags \
        "${pass_through_args[@]}" & # > /dev/null 2>&1 &
    sleep 10
done
