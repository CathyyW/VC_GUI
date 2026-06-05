#!/usr/bin/env bash
# Collect ~70 raw AndroidWorld trajectories (framework.md schema) with V-Droid verifier scores.

set -euo pipefail
cd /root/autodl-tmp/V-Droid

export OPENAI_ENDPOINT="https://poloai.top/v1"
export OPENAI_MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-44wgJ1trMWo3RlW3qDpshWPtItDwfFthY5HZA2tPoKu8rTSw

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/autodl-tmp/huggingface/hub}"
export ANDROID_ENV_A11Y_GRPC_PORT="${ANDROID_ENV_A11Y_GRPC_PORT:-20000}"

# Emulator via SSH reverse tunnel (see exp.md)
emulator_name="${EMULATOR_NAME:-127.0.0.1:15555}"
console_port="${CONSOLE_PORT:-15554}"
grpc_port="${GRPC_PORT:-18554}"
adb_path="${ADB_PATH:-/root/autodl-tmp/android-sdk/platform-tools/adb_wrap}"
setup="${EMULATOR_SETUP:-False}"

base_model="${BASE_MODEL:-/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-bnb-4bit}"
lora_name="${LORA_DIR:-/root/autodl-tmp/V-Droid/V-Droid-8B-0323}"
llm_name="${LLM_NAME:-gpt-4o-mini}"
service_name="${SERVICE_NAME:-openai}"
num_gpus="${NUM_GPUS:-1}"

task_range="${1:-1-70}"
max_trajectories="${MAX_TRAJECTORIES:-70}"
output_dir="${OUTPUT_DIR:-./collected_trajectories/run_$(date +%m%d_%H%M%S)}"
save_name="collect_${task_range}_$(date +%m%d_%H%M%S)"

mkdir -p "$output_dir"

echo "device=$emulator_name output_dir=$output_dir task_range=$task_range max=$max_trajectories"

python run_collect.py \
  --adb_path="$adb_path" \
  --device_name="$emulator_name" \
  --console_port="$console_port" \
  --grpc_port="$grpc_port" \
  --perform_emulator_setup="$setup" \
  --base_model="$base_model" \
  --lora_dir="$lora_name" \
  --llm_name="$llm_name" \
  --service_name="$service_name" \
  --num_gpus="$num_gpus" \
  --iteration=1 \
  --task_range="$task_range" \
  --max_trajectories="$max_trajectories" \
  --output_dir="$output_dir" \
  --save_name="$save_name"

echo "Done. Manifest: $output_dir/manifest.jsonl"
