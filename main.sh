cd /root/autodl-tmp/V-Droid
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export ANDROID_ENV_A11Y_GRPC_PORT=20000

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
llm_name="gpt-4o" # the llm used for the action completion and working memory construction
service_name="trapi" # the name of the service used for calling the llm above
closed_loop=False # set True to enable planner/subgoal/reflection closed-loop execution
max_replans=2 # the maximum number of replans after subgoal failure
subgoal_step_limit=4 # default max execution steps for each subgoal
num_gpus=1
save_name="Round110k_try_$(date +%m%d_%H%M%S)" # the saved file name for exp data

mkdir -p "$(dirname "$text_name")"

echo "device_name=$emulator_name console_port=$console_port grpc=$grpc adb_path=$adb_path"

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
    --device_name=$emulator_name \
    --console_port=$console_port \
    --grpc_port=$grpc \
    --perform_emulator_setup=$setup \
    --adb_path=$adb_path