# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collect raw AndroidWorld trajectories with V-Droid verifier scores."""

from collections.abc import Sequence
import os
import re
import subprocess

from absl import app
from absl import flags
from absl import logging
from transformers import set_seed

from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.agents import base_agent
from android_world.agents import vdroid
from android_world.env import env_launcher
from android_world.env import interface

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'NONE'
os.environ['GRPC_TRACE'] = 'none'

set_seed(42)


def _find_adb_directory() -> str:
    potential_paths = [
        os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
        os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
        os.path.expanduser('/root/autodl-tmp/android-sdk/platform-tools/adb'),
    ]
    for path in potential_paths:
        if os.path.isfile(path):
            return path
    raise EnvironmentError('adb not found in common Android SDK paths.')


def _load_tasks_from_list_file(
    list_file: str,
    start_id: int,
    end_id: int,
) -> list[str]:
    """Parse numbered entries like '001. TaskName' from android_world_tasks.txt."""
    tasks = []
    with open(list_file, encoding='utf-8') as f:
        for line in f:
            match = re.match(r'^0*(\d+)\.\s+(.+)$', line.strip())
            if not match:
                continue
            task_id = int(match.group(1))
            if start_id <= task_id <= end_id:
                tasks.append(match.group(2))
    return tasks


_ADB_PATH = flags.DEFINE_string('adb_path', _find_adb_directory(), 'Path to adb.')
_EMULATOR_SETUP = flags.DEFINE_boolean('perform_emulator_setup', False, 'Run emulator setup.')
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer('console_port', 5554, 'Emulator console port.')
_DEVICE_NAME = flags.DEFINE_string('device_name', 'emulator-5554', 'ADB device name.')
_GRPC_PORT = flags.DEFINE_integer('grpc_port', 8554, 'gRPC port for accessibility service.')

_SUITE_FAMILY = flags.DEFINE_enum(
    'suite_family',
    registry.TaskRegistry.ANDROID_WORLD_FAMILY,
    [registry.TaskRegistry.ANDROID_WORLD_FAMILY],
    'Suite family (android_world only for collection).',
)
_TASK_RANDOM_SEED = flags.DEFINE_integer('task_random_seed', 30, 'Random seed for task params.')
_TASKS = flags.DEFINE_list('tasks', None, 'Explicit task names to run.')
_TASK_LIST_FILE = flags.DEFINE_string(
    'task_list_file',
    './android_world_tasks.txt',
    'Numbered task list file.',
)
_TASK_RANGE = flags.DEFINE_string(
    'task_range',
    '1-70',
    'Inclusive task id range from task_list_file, e.g. 1-70.',
)
_N_TASK_COMBINATIONS = flags.DEFINE_integer('n_task_combinations', 1, 'Instances per task.')

_AGENT_NAME = flags.DEFINE_string('agent_name', 'VDroid', 'Agent name.')
_LLM_NAME = flags.DEFINE_string('llm_name', 'gpt-4o-mini', 'LLM for action completion.')
_SERVICE_NAME = flags.DEFINE_string('service_name', 'openai', 'LLM service.')
_SAVE_NAME = flags.DEFINE_string('save_name', 'collect', 'Run label under ./saved/.')
_LORA_DIR = flags.DEFINE_string('lora_dir', 'V-Droid-8B-0323', 'Verifier LoRA path.')
_BASE_MODEL = flags.DEFINE_string(
    'base_model',
    'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
    'Base model path or repo id.',
)
_ITERATION = flags.DEFINE_string(
    'iteration',
    '1',
    'MCTS outer iterations (same default as run_suite.py / eval).',
)
_SUMMARY = flags.DEFINE_string('summary', 'llm', 'Working memory summary mode.')
_NUM_GPUS = flags.DEFINE_integer('num_gpus', 1, 'GPUs for verifier scoring.')
_VC_LOOP = flags.DEFINE_boolean(
    'vc_loop',
    False,
    'Enable VC closed loop with post-action critic feedback.',
)
_CRITIC_BASE_MODEL = flags.DEFINE_string(
    'critic_base_model',
    None,
    'Base model for the post-action critic.',
)
_CRITIC_ADAPTER_DIR = flags.DEFINE_string(
    'critic_adapter_dir',
    None,
    'LoRA adapter for the post-action critic.',
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    './collected_trajectories',
    'Directory for raw trajectory JSON + manifest.jsonl.',
)
_MAX_TRAJECTORIES = flags.DEFINE_integer(
    'max_trajectories',
    70,
    'Stop after collecting this many trajectories.',
)


def disable_key_board(emulator_id: str = 'emulator-5554'):
    commands = [
        f'adb -s {emulator_id} shell pm disable-user com.google.android.inputmethod.latin',
        f'adb -s {emulator_id} shell pm disable-user com.google.android.tts',
    ]
    for command in commands:
        try:
            subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError:
            print(f'Command failed: {command}')


def _resolve_tasks() -> list[str] | None:
    if _TASKS.value:
        return _TASKS.value
    if not _TASK_RANGE.value:
        return None
    match = re.match(r'^(\d+)-(\d+)$', _TASK_RANGE.value.strip())
    if not match:
        raise ValueError(f'Invalid task_range: {_TASK_RANGE.value}')
    start_id, end_id = int(match.group(1)), int(match.group(2))
    tasks = _load_tasks_from_list_file(_TASK_LIST_FILE.value, start_id, end_id)
    if not tasks:
        raise ValueError(
            f'No tasks found for range {start_id}-{end_id} in {_TASK_LIST_FILE.value}'
        )
    return tasks


def _get_agent(env: interface.AsyncEnv, family: str) -> base_agent.EnvironmentInteractingAgent:
    agent = vdroid.VDroidAgent(
        env,
        _BASE_MODEL.value,
        adapter_dir=_LORA_DIR.value,
        llm_name=_LLM_NAME.value,
        service_name=_SERVICE_NAME.value,
        n_iters=int(_ITERATION.value),
        family=family,
        summary_mode=_SUMMARY.value,
        num_actors=_NUM_GPUS.value,
        closed_loop=False,
        collect_trajectory=True,
        vc_loop=_VC_LOOP.value,
        critic_base_model=_CRITIC_BASE_MODEL.value,
        critic_adapter_dir=_CRITIC_ADAPTER_DIR.value,
    )
    # Step budget (max_n_steps+10) is set in episode_runner, same as eval.
    agent.trajectory_output_dir = _OUTPUT_DIR.value
    agent.max_trajectories = _MAX_TRAJECTORIES.value
    agent.collected_trajectory_count = 0
    agent.name = _AGENT_NAME.value
    return agent


def _main() -> None:
    tasks = _resolve_tasks()
    os.makedirs(_OUTPUT_DIR.value, exist_ok=True)

    env = env_launcher.load_and_setup_env(
        console_port=_DEVICE_CONSOLE_PORT.value,
        emulator_setup=_EMULATOR_SETUP.value,
        adb_path=_ADB_PATH.value,
        grpc_port=_GRPC_PORT.value,
        device_name=_DEVICE_NAME.value,
        family=_SUITE_FAMILY.value,
    )
    if _EMULATOR_SETUP.value:
        disable_key_board(emulator_id=_DEVICE_NAME.value)
    env_launcher.verify_api_level(env)

    task_registry = registry.TaskRegistry()
    suite = suite_utils.create_suite(
        task_registry.get_registry(family=_SUITE_FAMILY.value),
        n_task_combinations=_N_TASK_COMBINATIONS.value,
        seed=_TASK_RANDOM_SEED.value,
        tasks=tasks,
    )
    suite.suite_family = _SUITE_FAMILY.value

    agent = _get_agent(env, _SUITE_FAMILY.value)
    agent.transition_pause = 3.0

    checkpoint_dir = os.path.join('./saved', f'{agent.name}_{_SAVE_NAME.value}', 'task_info')
    print(
        f'Collecting up to {_MAX_TRAJECTORIES.value} trajectories into {_OUTPUT_DIR.value}'
    )
    print(f'Tasks: {len(suite)} templates, checkpoint: {checkpoint_dir}')

    suite_utils.run(
        suite,
        agent,
        checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
        demo_mode=False,
        save_name=_SAVE_NAME.value,
    )

    print(
        f'Collection finished. Saved {getattr(agent, "collected_trajectory_count", 0)} '
        f'trajectories under {_OUTPUT_DIR.value}'
    )
    env.close()


def main(argv: Sequence[str]) -> None:
    del argv
    _main()


if __name__ == '__main__':
    app.run(main)
