#!/usr/bin/env bash

cd /root/autodl-tmp/V-Droid
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export ANDROID_ENV_A11Y_GRPC_PORT=20000

export OPENAI_ENDPOINT="https://poloai.top/v1"
export OPENAI_MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-44wgJ1trMWo3RlW3qDpshWPtItDwfFthY5HZA2tPoKu8rTSw

# default configs for the eveluation on androidworld benchmark
# Use 127.0.0.1:5555 when connecting to a local emulator through SSH reverse forwarding.
emulator_name="127.0.0.1:15555"
console_port=15554 # the port for the console communications
grpc=18554 # the port for the accessibility and grpc service
#adb_path="/root/autodl-tmp/android-sdk/platform-tools/adb"
adb_path="/root/autodl-tmp/android-sdk/platform-tools/adb_wrap"
setup=False # whether to perform the emulator and evelaution env setup

agent_name="VDroid" # the agent name used for the eveluation, e.g., default t3a, m3a in androidworld or VDroid
base_model="/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
lora_name="/root/autodl-tmp/V-Droid/V-Droid-8B-0323" # the name of the folder where the lora weights of VDroid is saved
summary=llm # the mode for the working memory construction
llm_name="gpt-4o-mini" # the llm used for the action completion and working memory construction
service_name="openai" # the name of the service used for calling the llm above
closed_loop=False # set True to enable planner/subgoal/reflection closed-loop execution
max_replans=2 # the maximum number of replans after subgoal failure
subgoal_step_limit=4 # default max execution steps for each subgoal
num_gpus=1
task_range="${1:-}" # optional task id range from android_world_tasks.txt, e.g. 1-5 or 6-10
task_list_file="./android_world_tasks.txt"
tasks_arg=()

if [[ -n "$task_range" ]]; then
    if [[ ! "$task_range" =~ ^[0-9]+-[0-9]+$ ]]; then
        echo "Invalid task range: $task_range. Use a range like 1-5 or 6-10."
        exit 1
    fi

    task_start="${task_range%-*}"
    task_end="${task_range#*-}"
    task_start=$((10#$task_start))
    task_end=$((10#$task_end))

    if (( task_start < 1 || task_end < task_start )); then
        echo "Invalid task range: $task_range. The start id must be >= 1 and <= end id."
        exit 1
    fi

    selected_tasks=()
    while IFS= read -r line; do
        if [[ "$line" =~ ^0*([0-9]+)\.\ (.+)$ ]]; then
            task_id=$((10#${BASH_REMATCH[1]}))
            if (( task_id >= task_start && task_id <= task_end )); then
                selected_tasks+=("${BASH_REMATCH[2]}")
            fi
        fi
    done < "$task_list_file"

    if (( ${#selected_tasks[@]} == 0 )); then
        echo "No tasks found for range ${task_start}-${task_end} in $task_list_file."
        exit 1
    fi

    tasks=$(IFS=,; echo "${selected_tasks[*]}")
    task_range_label="${task_start}-${task_end}"
    tasks_arg=(--tasks="$tasks")
    save_name="Round110k_try_${task_range_label}_$(date +%m%d_%H%M%S)" # the saved file name for exp data
else
    save_name="Round110k_try_$(date +%m%d_%H%M%S)" # the saved file name for exp data
fi

mkdir -p "$(dirname "$text_name")"

echo "device_name=$emulator_name console_port=$console_port grpc=$grpc adb_path=$adb_path"
if [[ -n "$task_range" ]]; then
    echo "task_range=$task_range_label tasks=$tasks"
else
    echo "task_range=all tasks=all"
fi
echo "save_name=$save_name"

python run_suite.py \
    --agent_name=$agent_name \
    --base_model=$base_model \
    --lora_dir=$lora_name \
    --llm_name=$llm_name \
    --service_name=$service_name \
    --summary=$summary \
    --num_gpus=$num_gpus \
    --closed_loop=$closed_loop \
    --max_replans=$max_replans \
    --subgoal_step_limit=$subgoal_step_limit \
    --save_name=$save_name \
    "${tasks_arg[@]}" \
    --device_name=$emulator_name \
    --console_port=$console_port \
    --grpc_port=$grpc \
    --perform_emulator_setup=$setup \
    --adb_path=$adb_path