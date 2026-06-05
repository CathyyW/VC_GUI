from absl import logging
from MCTS.mcts_node import MCTSNode
from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a_utils
from android_world.agents.reward_model import MAX_LENGTH
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils
from typing import Generic, TypeVar, Optional, NamedTuple, Callable, Hashable
import numpy as np
from abc import ABC, abstractmethod
import math
import os
import re
import json
from PIL import Image
from tqdm import trange
from copy import deepcopy
from math import ceil
from typing import Any, Type
import torch

from android_world.task_evals import task_eval
from html_representation.bbox_representation import turn_tree_to_group_bounding_boxes
from html_representation.html_representation import extract_actions_with_display_id, html_truncate, turn_tree_to_html_input, extract_actions_with_display_id_v2, turn_tree_to_html_input_v2
from util import ActionStack, entropy_estimation, generate_step_summary, obtain_reversed_action, polish_summary, polish_action, polish_reason
from prompt_template import *
from typing import Union

import ray

from android_world.agents.trajectory_collector import build_step_record

Node_info = TypeVar("Node_info")
Action = TypeVar("Action")
Example = TypeVar("Example")

LARGE_SCORE_SHIFT = 1e3
SOFT_SCORE_SHIFT = 1e2


def _ui_element_text(element: representation_utils.UIElement) -> str:
    parts = [
        element.text,
        element.content_description,
        element.resource_id,
        element.class_name,
    ]
    return " ".join(str(part) for part in parts if part)


def _ui_elements_to_html(ui_elements: list[representation_utils.UIElement]) -> str:
    lines = ["<div>"]
    for index, element in enumerate(ui_elements):
        text = _ui_element_text(element).replace('"', "'")
        tag = "input" if element.is_editable else "button" if element.is_clickable else "p"
        attrs = [f"id={index}"]
        if text:
            attrs.append(f'text="{text}"')
        if element.is_scrollable:
            attrs.append('scrollable="true"')
        lines.append(f"  <{tag} {' '.join(attrs)}></{tag}>")
    lines.append("</div>")
    return "\n".join(lines)


def _ui_elements_to_actions(ui_elements: list[representation_utils.UIElement]) -> list[str]:
    actions = []
    for index, element in enumerate(ui_elements):
        if element.is_clickable or element.is_checkable:
            actions.append(f'{{"action_type": "click", "index": {index}}}')
            actions.append(f'{{"action_type": "long_press", "index": {index}}}')
        if element.is_editable:
            actions.append(
                f'{{"action_type": "input_text", "text": "<text_input>", "index": {index}}}'
            )
            actions.append(f'{{"action_type": "clear_text", "index": {index}}}')
        if element.is_scrollable:
            for direction in ("up", "down", "left", "right"):
                actions.append(
                    f'{{"action_type": "scroll", "direction": "{direction}", "index": {index}}}'
                )

    actions.extend([
        '{"action_type": "navigate_home"}',
        '{"action_type": "navigate_back"}',
        '{"action_type": "open_app", "app_name": "<name>"}',
        '{"action_type": "wait"}',
        '{"action_type": "status", "goal_status": "complete"}',
        '{"action_type": "answer", "text": "<answer_text>"}',
    ])
    return actions


def _state_html_and_actions(ui_state, family: str) -> tuple[str, list[str]]:
    if ui_state.forest is not None:
        return (
            turn_tree_to_html_input(ui_state.forest),
            extract_actions_with_display_id_v2(
                ui_state.forest,
                refine_a11y_tree=family == "android_lab",
            ),
        )
    return (
        _ui_elements_to_html(ui_state.ui_elements),
        _ui_elements_to_actions(ui_state.ui_elements),
    )


@ray.remote(num_gpus=1)
class ModelActor:
    def __init__(self, service_name, model_name, lora_dir, acc_design, temperature=0.2):
        from android_world.agents.infer import Gpt4_Llama_Mix_Wrapper
        reward_type = "probs" if acc_design == "policy" else "score"
        prefix_share = False if acc_design in ["no_prefix", "policy"] else True

        self.llm = Gpt4_Llama_Mix_Wrapper(
            'gpt-4',
            service_name,
            model_name,
            temperature=temperature,
            adapter_dir=lora_dir,
            reward_type=reward_type,
            prefix_sharing=prefix_share
        )

    def predict(self, prompts):
        # policy-like generation
        with torch.no_grad():
            return self.llm.predict(prompts)

    def predict_scores_batch(self, prompts):
        with torch.no_grad():
            return self.llm.predict_scores_batch(prompts)

    def reinit_llm(self, model_name, service_name, lora_dir, acc_design, temperature=0.2):
        """
        Clear old model, re-init a new one. 
        Equivalent to your 'if count == 10' logic for clearing cache.
        """
        from android_world.agents.infer import Gpt4_Llama_Mix_Wrapper
        del self.llm
        torch.cuda.empty_cache()

        reward_type = "probs" if acc_design == "policy" else "score"
        prefix_share = False if acc_design in ["no_prefix", "policy"] else True

        self.llm = Gpt4_Llama_Mix_Wrapper(
            'gpt-4',
            service_name,
            model_name,
            temperature=temperature,
            adapter_dir=lora_dir,
            reward_type=reward_type,
            prefix_sharing=prefix_share
        )

    def warm_up(self):
        warm_up_text = PROMPT_PREFIX_V2 + \
            'The (overall) user goal/request is: '
        self.predict_scores_batch([warm_up_text])
        return True

    def get_llm(self):
        return self.llm

    def ping(self):
        return True


class VDroidAgent(base_agent.EnvironmentInteractingAgent):
    """V-Droid for mobile task automation"""

    def __init__(
        self,
        env: interface.AsyncEnv,
        local_model_name,
        adapter_dir,
        llm_name,
        service_name,
        save_dir: str = None,
        goal: str = None,
        task: Type[task_eval.TaskEval] = None,
        name: str = 'VDroid',
        wait_after_action_seconds: float = 30.0,
        calc_q: Callable[[list[float]], float] = np.mean,
        n_iters: int = 5,
        output_strategy: str = 'max_reward',
        uct_with_fast_reward: bool = True,
        depth_limit: int = 30,
        simulate_strategy: str | Callable[[list[float]], int] = 'max',
        cum_reward: Callable[[list[float]], float] = sum,
        goal_reward_default: float = 0.,
        goal_reached_reward: float = 10.,
        reward_alpha: float = 0.5,
        w_exp: float = 1.,
        explore_step_count_limit: int = 9,
        if_store_screen: bool = True,
        input_type: str = "html",
        family: str = "android_world",
        summary_mode: str = 'llm',
        num_actors: int = 2,
        closed_loop: bool = False,
        max_replans: int = 2,
        subgoal_step_limit: int = 4,
        collect_trajectory: bool = False,
    ):
        """Initializes a M3A Agent.

        Args:
        env: The environment.
        llm: The multimodal LLM wrapper.
        name: The agent name.
        wait_after_action_seconds: Seconds to wait for the screen to stablize
            after executing an action
        :param cum_reward: the way to calculate the cumulative reward from each step. Defaults: sum
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        :param uct_with_fast_reward: if True, use fast_reward instead of reward for unvisited children in UCT
                                 Otherwise, visit the *unvisited* children with maximum fast_reward first
        :param simulate_strategy: simulate strategy. Options: 'max', 'sample', 'random', or use a custom function
        :param cum_reward: the way to calculate the cumulative reward from each step. Defaults: sum
        :param w_exp: the weight of exploration in UCT
        :param explore_step_count_limit: the step count limit for simulation
        """
        super().__init__(env, name)

        if not ray.is_initialized():
            ray.init()

        actors = []
        ModelClass = ModelActor.options(num_gpus=1)
        for _ in range(num_actors):
            actor = ModelClass.remote(
                service_name, local_model_name, adapter_dir, "dynamic_batch")
            actors.append(actor)

        ray.get([act.ping.remote() for act in actors])

        # llm used for action completion and working memory construction
        self.llm = infer.Gpt4Wrapper(llm_name, service_name, temperature=0.2)

        self.iter_idx = 0
        self.additional_guidelines = None
        self.wait_after_action_seconds = wait_after_action_seconds
        self.cum_reward = cum_reward
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward
        self.reward_alpha = reward_alpha
        self.w_exp = w_exp
        self.explore_step_count_limit = explore_step_count_limit
        self.calc_q = calc_q
        self.n_iters = n_iters
        self.if_store_screen = if_store_screen

        self.goal = goal
        self.task = task
        self.save_dir = save_dir
        if save_dir != None:
            self.save_path = os.path.join(save_dir, f"screen_shot/")
            os.makedirs(self.save_path, exist_ok=True)

        assert output_strategy in ['max_reward', 'follow_max',
                                   'max_visit', 'max_iter', 'last_iter', 'last_terminal_iter']

        self.output_strategy = output_strategy
        self.uct_with_fast_reward = uct_with_fast_reward
        self.depth_limit = depth_limit

        default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            'max': lambda x: np.argmax(x),
            'sample': lambda x: np.random.choice(len(x), p=x),
            'random': lambda x: np.random.choice(len(x)),
        }

        self.simulate_choice: Callable[[list[float]], int] = default_simulate_strategies.get(simulate_strategy,
                                                                                             simulate_strategy)
        self.input_type = input_type
        if self.input_type == "image":
            self.add_image_desc = True
        else:
            self.add_image_desc = False

        self.history = []
        self.history_traj = []

        self.family = family
        self.summary_mode = summary_mode
        self.closed_loop = closed_loop
        self.max_replans = max_replans
        self.subgoal_step_limit = subgoal_step_limit
        self.closed_loop_trace = []

        self.collect_trajectory = collect_trajectory
        self.trajectory_steps: list[dict[str, Any]] = []
        self._trajectory_record_enabled = False
        self._trajectory_task_id: str | None = None
        self._trajectory_instance_id: int = 0
        self._trajectory_seed: int | None = None

        warmup_futs = [act.warm_up.remote() for act in actors]
        ray.get(warmup_futs)
        self.num_actors = num_actors
        self.actors = actors

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        self.additional_guidelines = task_guidelines

    def start_trajectory_episode(
        self,
        task_id: str,
        goal: str,
        instance_id: int = 0,
        seed: int | None = None,
    ) -> None:
        """Reset per-episode trajectory buffer (collector mode only)."""
        if not self.collect_trajectory:
            return
        self.trajectory_steps = []
        self._trajectory_task_id = task_id
        self._trajectory_instance_id = instance_id
        self._trajectory_seed = seed
        self._trajectory_episode_goal = goal

    def finish_trajectory_episode(
        self,
        task_success: bool,
        agent_indicated_done: bool,
    ) -> dict[str, Any]:
        """Package collected steps into one trajectory dict."""
        if not self.collect_trajectory:
            return {}
        return {
            "task_id": self._trajectory_task_id,
            "instance_id": self._trajectory_instance_id,
            "seed": self._trajectory_seed,
            "goal": getattr(self, "_trajectory_episode_goal", self.goal),
            "task_success": task_success,
            "agent_indicated_done": agent_indicated_done,
            "steps": self.trajectory_steps,
        }

    def _trajectory_history_summaries(self) -> list[str]:
        return [
            step_info.get("summary", "")
            for step_info in self.history
            if step_info.get("summary")
        ]

    def _try_record_trajectory_step(self, node: MCTSNode) -> None:
        if not self.collect_trajectory or not self._trajectory_record_enabled:
            return
        parent = node.parent
        if parent is None:
            return
        expand_data = getattr(parent, "collector_expand_data", None)
        if not expand_data:
            return

        step_record = build_step_record(
            task_id=self._trajectory_task_id or "unknown",
            goal=getattr(self, "_trajectory_episode_goal", self.goal),
            step_id=len(self.trajectory_steps),
            history=self._trajectory_history_summaries(),
            before_ui_html=parent.node_info.get("html_desc"),
            after_ui_html=node.node_info.get("html_desc"),
            action_space=expand_data["action_space"],
            scores=expand_data["scores"],
            selected_action=node.action or expand_data.get("selected_action", ""),
            summary=node.node_info.get("summary"),
        )
        # Same filenames as eval store_screen (written right after step() returns).
        depth_label = node.depth + 1
        iter_label = self.iter_idx if self.iter_idx else 1
        step_record["screenshot"] = f"iter_{iter_label}_step{depth_label}.jpg"
        step_record["screenshot_ann"] = (
            f"iter_{iter_label}_step{depth_label}_ann.jpg"
        )
        self.trajectory_steps.append(step_record)

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        # Hide the coordinates on screen which might affect the vision model.
        self.env.hide_automation_ui()
        if self.history:
            self.history_traj.append(self.history)

        self.history = []

    def step(self, node: MCTSNode, converted_action,):
        logical_screen_size = self.env.logical_screen_size
        physical_frame_boundary = self.env.physical_frame_boundary
        orientation = self.env.orientation

        if converted_action.action_type == 'answer':
            print('Agent answered with: ' + converted_action.text)

        try:
            self.env.execute_action(converted_action)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print('Failed to execute action.')
            print(str(e))

            node.node_info['summary'] = (
                'Can not execute the action, make sure to select the action with'
                ' the required parameters (if any) in the correct JSON format!'
            )
            node.node_info['ui_elements'] = node.parent.node_info['ui_elements'].copy()
            if node.parent.node_info['html_desc']:
                node.node_info['html_desc'] = node.parent.node_info['html_desc']
            state = node.parent.state.copy()
            return state
        
        ui_state = self.get_post_transition_state()
        try:
            html_desc, available_actions = _state_html_and_actions(ui_state, self.family)
            node.node_info['html_desc'] = html_desc
        except:
            logging.error("Extract html_desc wrong")
            html_desc = node.parent.node_info.get('html_desc')
            available_actions = _ui_elements_to_actions(ui_state.ui_elements)

        state = {
            'screenshot_raw': None,
            'screenshot_som': None,
            'orientation': None,
            'physical_frame_boundary': None,
            'logical_screen_size': None,
            'available_actions': available_actions,
            'raw_ui_state': None,
        }

        logical_screen_size = self.env.logical_screen_size
        orientation = self.env.orientation
        physical_frame_boundary = self.env.physical_frame_boundary
        after_ui_elements = ui_state.ui_elements
        after_screenshot = ui_state.pixels.copy()

        state['screenshot_raw'] = after_screenshot.copy()
        for index, ui_element in enumerate(after_ui_elements):
            if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
                m3a_utils.add_ui_element_mark(
                    after_screenshot,
                    ui_element,
                    index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                    add_image_desc=self.add_image_desc
                )

        group_bounding_boxes = (
            turn_tree_to_group_bounding_boxes(
                orientation, logical_screen_size, physical_frame_boundary, ui_state.forest)
            if ui_state.forest is not None else {}
        )
        if self.add_image_desc:
            m3a_utils.apply_group_bouding_boxes(
                after_screenshot,
                group_bounding_boxes
            )

        state['screenshot_som'] = after_screenshot.copy()
        state['orientation'] = orientation
        state['physical_frame_boundary'] = physical_frame_boundary
        state['logical_screen_size'] = logical_screen_size
        state['raw_ui_state'] = ui_state

        node.node_info['ui_elements'] = after_ui_elements

        if self.summary_mode == "llm":
            summary_prompt = summarize_prompt(
                node.action,
                node.node_info["reason"],
                self.goal,
                node.parent.node_info['html_desc'],
                node.node_info['html_desc'],
                self.input_type,
            )

            summary, is_safe, raw_response = self.llm.predict_mm(
                summary_prompt, [],)
            summary = polish_summary(summary)
            node.node_info['summary_prompt'] = summary_prompt
            node.node_info['summary'] = f'Action selected: {node.action}. {summary}'
            node.node_info['summary_raw_response'] = raw_response

        elif self.summary_mode == "skip":
            summary = "None"
            node.node_info['summary_prompt'] = "we skipp summary for acceleration."
            node.node_info['summary'] = f'Action selected: {node.action}.'
            node.node_info['summary_raw_response'] = "None"

        elif self.summary_mode == "rule":
            last_step_action = json.loads(node.action)
            summary = generate_step_summary(
                last_step_action, node.parent.node_info["ui_elements"], node.node_info["ui_elements"],)
            node.node_info['summary_prompt'] = "Use rule-based summary."
            node.node_info['summary'] = f'Action selected: {node.action}. {summary}'
            node.node_info['summary_raw_response'] = "Use rule-based summary."

        print('Summary: ' + summary)
        logging.warning('Summary: ' + summary)
        if self._trajectory_record_enabled:
            self._try_record_trajectory_step(node)
        return state

    def iterate(self, node: MCTSNode, iter: int) -> list[MCTSNode]:
        path = self._select(node, iter)
        if not self._is_terminal_with_depth_limit(path[-1]):
            self._expand_verifier(path[-1], iter)
            self._simulate(path, iter)

        cum_reward = self._back_propagate(path)
        if self.output_strategy == 'max_iter' and path[-1].is_terminal and cum_reward > self._output_cum_reward:
            self._output_cum_reward = cum_reward
            self._output_iter = path
        if self.output_strategy == 'last_iter':
            self._output_cum_reward = cum_reward
            self._output_iter = path
        if self.output_strategy == 'last_terminal_iter' and path[-1].is_terminal:
            self._output_cum_reward = cum_reward
            self._output_iter = path

        if hasattr(self.llm, "local_model_name") and "deepseek" in self.llm.local_model_name.lower():
            torch.cuda.empty_cache()
        return path

    def _back_propagate(self, path: list[MCTSNode]):
        rewards = []
        cum_reward = -math.inf
        for node in reversed(path):
            if node.reward:
                rewards.append(node.reward)
            else:
                rewards.append(node.score)
            cum_reward = self.cum_reward(rewards[::-1])
            node.cum_rewards.append(cum_reward)
        # how do they obtain the final reward?
        # for node in reversed(path):
        return

    def _is_terminal_with_depth_limit(self, node: MCTSNode):
        return node.is_terminal or node.depth >= self.depth_limit

    def _is_terminal_with_simulate_depth_limit(self, explore_step_count):
        return explore_step_count >= self.explore_step_count_limit

    def _select(self, node: MCTSNode, iter: int) -> list[MCTSNode]:
        path = []
        while True:
            path.append(node)
            if node.children is None or len(node.children) == 0 or self._is_terminal_with_depth_limit(node):
                self.repeat_corresponding_actions(path, iter)
                return path
            node = self._uct_select(node)
            node = self._action_completion(node)
            print(f"Select action {node.action}")
            logging.warn(f"Select action {node.action}")

    def repeat_corresponding_actions(self, path: list[MCTSNode], iter: int):
        for node in path[:-1]:
            if node.action is None:
                if node.depth == 0:
                    self.store_screen(node, iter)
                    pass
                continue
            else:
                try:
                    converted_action = json_action.JSONAction(
                        **agent_utils.extract_json(node.action),
                    )
                    node.node_info['action_output_json'] = converted_action
                    action_index = converted_action.index
                    num_ui_elements = len(node.parent.node_info['ui_elements'])
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print('Failed to convert the output to a valid action.')
                    print(str(e))

                if converted_action is not None:
                    if converted_action.action_type == 'status':
                        if converted_action.goal_status == 'infeasible':
                            print(
                                'Agent stopped since it thinks mission impossible.')
                        node.node_info['summary'] = 'Agent thinks the request has been completed.'
                        node.state = node.parent.state.copy()
                        node.is_terminal = True
                        node.node_info['ui_elements'] = node.parent.node_info['ui_elements'].copy(
                        )
                        if node.parent.node_info['html_desc']:
                            node.node_info['html_desc'] = node.parent.node_info['html_desc']
                        node.reward, node.score_details = self.reward(
                            node.score_details['self_eval'], node.is_terminal)
                        node.score = node.reward
                        logging.warning(
                            f"The final reward for the finished node is {node.reward} ")
                    else:
                        take_a_step = True
                        if (converted_action.action_type
                                in ['click', 'long_press', 'input_text', 'scroll']
                                and action_index is not None
                                ):
                            if action_index >= num_ui_elements:
                                print(
                                    f'Index out of range, prediction index is {action_index}, but the'
                                    f' UI element list only has {num_ui_elements} elements.'
                                )

                                node.state = node.parent.state.copy()
                                node.node_info['summary'] = (
                                    'The parameter index is out of range. Remember the index must be in'
                                    ' the UI element list!'
                                )
                                node.node_info['ui_elements'] = node.parent.node_info['ui_elements'].copy(
                                )
                                if node.parent.node_info['html_desc']:
                                    node.node_info['html_desc'] = node.parent.node_info['html_desc']

                                take_a_step = False
                            else:
                                m3a_utils.add_ui_element_mark(
                                    node.parent.state['screenshot_raw'],
                                    node.parent.node_info['ui_elements'][action_index],
                                    action_index,
                                    node.parent.state["logical_screen_size"],
                                    node.parent.state["physical_frame_boundary"],
                                    node.parent.state["orientation"],
                                    add_image_desc=self.add_image_desc
                                )

                        if take_a_step:
                            node_state = self.step(node, converted_action,)
                            node.state = node_state

                    self.store_screen(node, iter)
                    self.history.append(node.node_info)
        return

    def _uct(self, node: MCTSNode) -> float:
        value = node.Q + self.w_exp * \
            np.sqrt(np.log(len(node.parent.cum_rewards) + 1) /
                    max(0.5, len(node.cum_rewards)))
        print(value)
        return value

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        if self.uct_with_fast_reward or all(x.state is not None for x in node.children):
            return max(node.children, key=self._uct)
        else:
            unvisited_children = filter(
                lambda x: x.state is None, node.children)
            return max(unvisited_children, key=lambda x: x.score)

    def _simulate(self, path: list[MCTSNode], iter: int):
        node = path[-1]
        simulate_count = 0
        while True:
            if self._is_terminal_with_depth_limit(node) or len(node.children) == 0 or self._is_terminal_with_simulate_depth_limit(simulate_count):
                return

            scores = [child.score for child in node.children]
            node = node.children[self.simulate_choice(scores)]

            node = self._action_completion(node)

            simulate_count += 1
            path.append(node)
            self._expand_verifier(node, iter)

    def query_llm_for_action_completion(self, node: MCTSNode):

        step_summary = ['Step ' + str(i+1) + '- ' + step_info['summary']
                        for i, step_info in enumerate(self.history)]

        action_prompt = action_completion_prompt(
            self.goal,
            node.action,
            step_summary,
            node.parent.node_info['html_desc'],
        )

        action_output, is_safe, raw_response = self.llm.predict_mm(
            action_prompt, [],)

        if is_safe == False:  # pylint: disable=singleton-comparison
            #  is_safe could be None
            action_output = """Reason: Triggered LLM safety classifier.
            Action: {"action_type": "status", "goal_status": "infeasible"}"""

        if raw_response is None:
            raise RuntimeError('Error calling LLM in action selection phase.')

        action = m3a_utils.parse_action_output(action_output)
        if action == None:  # we fall back to the origin action so the program can proceed.
            action = node.action

        print(f"The incomplete action {node.action} is modified to {action}")
        logging.warning(
            f"The incomplete action {node.action} is modified to {action}")
        return action

    def _action_completion(self, node: MCTSNode):
        actions_requiring_completion = {
            "input_text": "text_input",
            "open_app": "name",
            'answer': "answer_text",
        }
        action = node.action
        if action:
            for key, missing_field in actions_requiring_completion.items():
                if key in action and f"<{missing_field}>" in action:
                    # Query LLM to complete the action
                    action = self.query_llm_for_action_completion(node)
                    # Replace the action with the LLM's response
                    node.action = action
        return node

    def _scoring_with_verifier_by_batch(self, actions: list[str], history, ui_desc):
        """
        For a list of actions, build the prompts and call predict_scores_batch in parallel
        across self.actors. Then flatten the results in the right order and return them.
        """

        input_prompts = []
        for action in actions:
            # Prepare your prompt as you do normally
            input_prompt = action_selection_prompt_with_verifier(
                action,
                history,
                self.goal,
                ui_desc,
            )
            input_prompts.append(input_prompt)

        num_actors = self.num_actors
        subset_size = len(input_prompts) // num_actors
        remainder = len(input_prompts) % num_actors

        prompt_subsets = []
        st = 0
        for i in range(num_actors):
            extra = 1 if i < remainder else 0
            en = st + subset_size + extra
            subset = input_prompts[st:en]
            prompt_subsets.append(subset)
            st = en

        futures = []
        for i, actor in enumerate(self.actors):
            if len(prompt_subsets[i]) == 0:
                continue

            fut = actor.predict_scores_batch.remote(prompt_subsets[i])
            futures.append(fut)

        results_list = ray.get(futures)

        combined_results = []
        for res in results_list:
            combined_results.extend(res)

        rewards = []
        for i, (_, _, quality_output) in enumerate(combined_results):
            rewards.append(quality_output[0])
        return rewards

    def calculate_reward(self, self_eval, goal_reached=None) -> float:
        if goal_reached is None:
            goal_reward = self.goal_reward_default
        elif goal_reached:
            goal_reward = self.goal_reached_reward
        return self_eval * self.reward_alpha + goal_reward * (1 - self.reward_alpha)

    def reward(self, self_eval: float = None,
               goal_reached: tuple[bool, float] = None) -> float:
        assert self_eval is not None, "self_eval is required to calculate reward in this search config, consider passing it in fast_reward"
        assert goal_reached is not None, "goal_reached is required to calculate reward in this search config, consider passing it in world model's step"
        return (self.calculate_reward(self_eval, goal_reached),
                {'self_eval': self_eval, 'goal_reached': goal_reached})

    def _expand_verifier(self, node: MCTSNode, iter: int):
        if node.state is None:
            try:
                converted_action = json_action.JSONAction(
                    **agent_utils.extract_json(node.action),
                )
                node.node_info['action_output_json'] = converted_action
                action_index = converted_action.index

            except Exception as e:  # pylint: disable=broad-exception-caught
                converted_action = None
                print('Failed to convert the output to a valid action.')
                print(str(e))

                node.state = node.parent.state.copy()
                node.node_info['summary'] = (
                    'Can not parse the output to a valid action. Please make sure to pick'
                    ' the action from the list with required parameters (if any) in the'
                    ' correct JSON format!'
                )
                node.node_info['ui_elements'] = node.parent.node_info['ui_elements'].copy(
                )
                if node.parent.node_info['html_desc']:
                    node.node_info['html_desc'] = node.parent.node_info['html_desc']
                self._assign_action_failure_penalty(node)

            if converted_action:
                if converted_action.action_type == 'status':
                    node.node_info['summary'] = 'Agent thinks the request has been completed.'
                    node.state = node.parent.state.copy()
                    node.is_terminal = True
                    node.node_info['ui_elements'] = node.parent.node_info['ui_elements'].copy(
                    )
                    if node.parent.node_info['html_desc']:
                        node.node_info['html_desc'] = node.parent.node_info['html_desc']

                    node.score = 10
                    node.reward = 100
                    return base_agent.AgentInteractionResult(
                        True,
                        node.node_info,
                    )
                else:
                    try:
                        self._trajectory_record_enabled = self.collect_trajectory
                        node.state = self.step(node, converted_action,)
                        self._trajectory_record_enabled = False
                    except:
                        print(
                            f'Error when taking a step.'
                        )
                        node.state = node.parent.state.copy()
                        node.node_info['summary'] = (
                            f'The action {node.parent.action} is not executable. Remember to follow the correct action format.'
                        )
                        node.node_info['ui_elements'] = node.parent.node_info['ui_elements'].copy(
                        )
                        if node.parent.node_info['html_desc']:
                            node.node_info['html_desc'] = node.parent.node_info['html_desc']
                        self._assign_action_failure_penalty(node)

            self.store_screen(node, iter)
            self.history.append(node.node_info)

        child_node_info = {
            'ui_elements': None,
            'reason': None,
            'action_prompt': None,
            'action_output': None,
            'action_output_json': None,
            'action_reason': None,
            'action_raw_response': None,
            'summary_prompt': None,
            'summary': None,
            'summary_raw_response': None,
        }

        available_actions = node.state["available_actions"]
        logging.warning(available_actions)

        step_summary = ['Step ' + str(i+1) + '- ' + step_info['summary']
                        for i, step_info in enumerate(self.history)]

        best_child = self.score_by_batch(
            node, available_actions, step_summary, child_node_info)

        node.children = best_child
        return base_agent.AgentInteractionResult(
            False,
            node.node_info,
        )

    def score_by_batch(self, node: MCTSNode, available_actions, memory, child_node_info, batch_size=16, cot_verify=False):
        num_actors = self.num_actors  # integer

        batch_sizes = [num_actors] + [batch_size] * ceil(
            (len(available_actions) - num_actors) / batch_size
        )
        start_idx = 0

        best_child = None
        best_reward = float("-inf")
        all_actions: list[str] = []
        all_scores: list[float] = []

        for current_batch_size in batch_sizes:
            end_idx = start_idx + current_batch_size
            action_batch = available_actions[start_idx:end_idx]

            scores = self._scoring_with_verifier_by_batch(
                action_batch,
                memory,
                node.node_info['html_desc']
            )
            all_actions.extend(action_batch)
            all_scores.extend(scores)

            for i, action in enumerate(action_batch):
                logging.warning(
                    f"For action {action}, the score is {scores[i]}.")
                print(f"For action {action}, the score is {scores[i]}.")

                if scores[i] > best_reward:
                    best_reward = scores[i]
                    # Create a new child node (overwrites the old best child)
                    best_child = MCTSNode(
                        state=None,
                        node_info=child_node_info,
                        action=action,
                        parent=node,
                        is_terminal=False,
                        score=scores[i],
                        score_details=None,
                        calc_q=self.calc_q,
                    )

            start_idx = end_idx

        if self.collect_trajectory and best_child is not None:
            node.collector_expand_data = {
                "action_space": all_actions,
                "scores": all_scores,
                "selected_action": best_child.action,
            }

        return [best_child]

    def _reset_and_construct_root(self):
        self.step_idx = 0
        self.history = []
        state = {
            'screenshot_raw': None,
            'screenshot_som': None,
            'orientation': None,
            'physical_frame_boundary': None,
            'logical_screen_size': None,
            'raw_ui_state': None,
        }
        node_info = {
            'ui_elements': None,
            'reason': None,
            'action_prompt': None,
            'action_output': None,
            'action_output_json': None,
            'action_reason': None,
            'action_raw_response': None,
            'summary_prompt': None,
            'summary': None,
            'summary_raw_response': None,
            'html_desc': None,
        }

        ui_state = self.env.get_state(wait_to_stabilize=False)
        orientation = self.env.orientation
        physical_frame_boundary = self.env.physical_frame_boundary

        ui_elements = ui_state.ui_elements
        logical_screen_size = self.env.logical_screen_size
        state['screenshot_raw'] = ui_state.pixels.copy()
        state['raw_ui_state'] = ui_state
        before_screenshot = ui_state.pixels.copy()

        if self.input_type == "html":
            html_desc, available_actions = _state_html_and_actions(ui_state, self.family)
        elif self.input_type == "image":
            html_desc = None
            available_actions = _ui_elements_to_actions(ui_state.ui_elements)
        node_info['html_desc'] = html_desc

        for index, ui_element in enumerate(ui_elements):
            if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
                m3a_utils.add_ui_element_mark(
                    before_screenshot,
                    ui_element,
                    index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                    add_image_desc=self.add_image_desc
                )

        group_bounding_boxes = (
            turn_tree_to_group_bounding_boxes(
                orientation, logical_screen_size, physical_frame_boundary, ui_state.forest)
            if ui_state.forest is not None else {}
        )
        if self.add_image_desc:
            m3a_utils.apply_group_bouding_boxes(
                before_screenshot,
                group_bounding_boxes
            )

        state['screenshot_som'] = before_screenshot.copy()
        state['orientation'] = orientation
        state['physical_frame_boundary'] = physical_frame_boundary
        state['logical_screen_size'] = logical_screen_size
        state['available_actions'] = available_actions

        node_info['ui_elements'] = ui_elements

        self.root = MCTSNode(state=state, node_info=node_info,
                             action=None, parent=None, calc_q=self.calc_q)
        self.store_screen(self.root, iter=1)
        return

    def store_screen(self, node: MCTSNode, iter: int):
        if self.if_store_screen:
            pixels = node.state['screenshot_raw']
            img = Image.fromarray(np.uint8(pixels))
            img.save(self.save_path +
                     f"iter_{iter}_step{node.depth + 1}.jpg", 'JPEG')

            pixels_ann = node.state['screenshot_som']
            img_ann = Image.fromarray(np.uint8(pixels_ann))
            img_ann.save(self.save_path +
                         f"iter_{iter}_step{node.depth + 1}_ann.jpg", 'JPEG')

    def _dfs_max_reward(self, path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
        cur = path[-1]
        if cur.is_terminal:
            return self.cum_reward([node.score for node in path[1:]]), path
        if cur.children is None:
            return -math.inf, path
        visited_children = [x for x in cur.children if x.node_info is not None]
        if len(visited_children) == 0:
            return -math.inf, path
        return max((self._dfs_max_reward(path + [child]) for child in visited_children), key=lambda x: x[0])

    def _check_task_finish(self, view_client) -> bool:
        event = None
        finished = False
        if self.event_queue.empty():
            finished = self.current_task.check_finish(view_client, None)

        while not self.event_queue.empty():
            event = self.event_queue.get_nowait()  # get also pops the element
            finished = self.current_task.check_finish(view_client, event)
            if finished:
                break

        print("-----------------------")
        print(f"Task Finished? {finished}")
        print("-----------------------")
        return finished

    def _assign_action_failure_penalty(self, node: MCTSNode):
        node.score = node.score - 10  # assign penalty to it
        return

    def _parse_json_response(self, response: str) -> dict[str, Any] | None:
        if not response:
            return None
        response = response.strip()
        fence_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        if fence_match:
            response = fence_match.group(1).strip()
        try:
            parsed = json.loads(response)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        start = response.find('{')
        end = response.rfind('}')
        if start == -1 or end == -1 or start >= end:
            return None
        try:
            parsed = json.loads(response[start:end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _normalize_subgoals(self, plan: dict[str, Any] | list[Any] | None) -> list[dict[str, Any]]:
        if plan is None:
            return []
        if isinstance(plan, dict):
            subgoals = plan.get('subgoals') or plan.get('revised_subgoals') or []
        else:
            subgoals = plan

        normalized = []
        for idx, subgoal in enumerate(subgoals):
            if isinstance(subgoal, str):
                subgoal = {
                    'subgoal': subgoal,
                    'expected_outcome': subgoal,
                }
            if not isinstance(subgoal, dict):
                continue
            item = dict(subgoal)
            item.setdefault('id', f'sg{idx + 1}')
            item.setdefault('subgoal', item.get('step', ''))
            item.setdefault('expected_outcome', item.get('expect_outcome', item.get('outcome', item['subgoal'])))
            item.setdefault('max_steps', self.subgoal_step_limit)
            item.setdefault('failure_hint', '')
            normalized.append(item)
        return normalized

    def _fallback_plan(self, goal: str, reason: str) -> dict[str, Any]:
        return {
            'task_status': 'in_progress',
            'overall_strategy': f'Fallback to the original V-Droid goal because {reason}.',
            'subgoals': [{
                'id': 'sg1',
                'subgoal': goal,
                'expected_outcome': 'The original user task is completed.',
                'max_steps': self.subgoal_step_limit,
                'failure_hint': 'Use the original V-Droid verifier behavior.',
            }],
        }

    def generate_plan(self, goal: str, screen_html: str, memory: list[str] | None = None) -> dict[str, Any]:
        prompt = planner_prompt(goal, screen_html, memory or [])
        response, _, raw_response = self.llm.predict_mm(prompt, [])
        plan = self._parse_json_response(response)
        if plan is None:
            plan = self._fallback_plan(goal, 'the planner response was not valid JSON')
        plan['planner_prompt'] = prompt
        plan['planner_raw_response'] = response
        plan['planner_call_succeeded'] = raw_response is not None
        return plan

    def replan(
        self,
        goal: str,
        previous_plan: dict[str, Any],
        completed_subgoals: list[dict[str, Any]],
        failed_subgoal: dict[str, Any],
        memory: list[str],
        screen_html: str,
    ) -> dict[str, Any]:
        prompt = replan_prompt(
            goal=goal,
            previous_plan=previous_plan,
            completed_subgoals=completed_subgoals,
            failed_subgoal=failed_subgoal,
            expected_outcome=failed_subgoal.get('expected_outcome', ''),
            memory=memory,
            screen_html=screen_html,
        )
        response, _, raw_response = self.llm.predict_mm(prompt, [])
        plan = self._parse_json_response(response)
        if plan is None:
            plan = self._fallback_plan(goal, 'the replanner response was not valid JSON')
        plan['replan_prompt'] = prompt
        plan['replan_raw_response'] = response
        plan['replan_call_succeeded'] = raw_response is not None
        return plan

    def format_subgoal_goal(self, overall_goal: str, subgoal: dict[str, Any]) -> str:
        return subgoal_aware_goal(
            overall_goal,
            subgoal.get('subgoal', ''),
            subgoal.get('expected_outcome', ''),
        )

    def _closed_loop_memory_from_output(self, output: list[dict[str, Any]]) -> list[str]:
        memory = []
        for idx, step_info in enumerate(output):
            summary = step_info.get('summary')
            if summary:
                memory.append(f'Step {idx + 1}- {summary}')
        return memory

    def _annotate_closed_loop_output(
        self,
        output: list[dict[str, Any]],
        plan_version: int,
        subgoal: dict[str, Any],
    ) -> None:
        for step_info in output:
            step_info['closed_loop_plan_version'] = plan_version
            step_info['current_subgoal_id'] = subgoal.get('id')
            step_info['current_subgoal'] = subgoal.get('subgoal')
            step_info['expected_outcome'] = subgoal.get('expected_outcome')

    def closed_loop_search(self):
        original_goal = self.goal
        self.closed_loop_trace = []

        self._reset_and_construct_root()
        plan = self.generate_plan(
            original_goal,
            self.root.node_info.get('html_desc'),
            memory=[],
        )
        self.closed_loop_trace.append({'event': 'plan', 'plan': plan})

        if plan.get('task_status') == 'complete':
            self.root.node_info['summary'] = 'Planner judged the task already complete.'
            self.root.node_info['closed_loop_plan'] = plan
            return True, [self.root.node_info]
        if plan.get('task_status') == 'infeasible':
            self.root.node_info['summary'] = 'Planner judged the task infeasible.'
            self.root.node_info['closed_loop_plan'] = plan
            return True, [self.root.node_info]

        subgoals = self._normalize_subgoals(plan)
        if not subgoals:
            plan = self._fallback_plan(original_goal, 'the planner returned no subgoals')
            subgoals = self._normalize_subgoals(plan)

        all_output = []
        completed_subgoals = []
        previous_plan = plan
        plan_version = 0
        replan_count = 0
        subgoal_idx = 0

        while subgoal_idx < len(subgoals):
            subgoal = subgoals[subgoal_idx]
            execution_goal = self.format_subgoal_goal(original_goal, subgoal)
            step_limit = int(subgoal.get('max_steps') or self.subgoal_step_limit)

            is_done, output = self._search_once_for_goal(
                execution_goal,
                step_limit=step_limit,
            )
            self._annotate_closed_loop_output(output, plan_version, subgoal)
            all_output.extend(output)

            event = {
                'event': 'subgoal_result',
                'plan_version': plan_version,
                'subgoal': subgoal,
                'is_done': is_done,
                'steps': len(output),
            }
            self.closed_loop_trace.append(event)

            if is_done:
                completed_subgoals.append(subgoal)
                subgoal_idx += 1
                continue

            if replan_count >= self.max_replans:
                break

            memory = self._closed_loop_memory_from_output(all_output)
            current_html = None
            if output:
                current_html = output[-1].get('html_desc')
            if not current_html and hasattr(self, 'root'):
                current_html = self.root.node_info.get('html_desc')

            previous_plan = self.replan(
                original_goal,
                previous_plan,
                completed_subgoals,
                subgoal,
                memory,
                current_html,
            )
            self.closed_loop_trace.append({'event': 'replan', 'plan': previous_plan})

            revised_subgoals = self._normalize_subgoals(previous_plan)
            if not revised_subgoals:
                break
            subgoals = revised_subgoals
            plan_version += 1
            replan_count += 1
            subgoal_idx = 0

        is_complete = len(completed_subgoals) == len(subgoals) and len(subgoals) > 0
        if all_output:
            all_output[-1]['closed_loop_trace'] = self.closed_loop_trace
        return is_complete, all_output

    def search(self,):
        if self.closed_loop:
            return self.closed_loop_search()
        return self._search_once_for_goal(self.goal)

    def _search_once_for_goal(self, execution_goal: str, step_limit: int | None = None):
        original_goal = self.goal
        original_depth_limit = self.depth_limit
        original_explore_step_count_limit = self.explore_step_count_limit
        if step_limit is not None:
            self.depth_limit = step_limit
            self.explore_step_count_limit = step_limit
        self.goal = execution_goal

        try:
            self._output_cum_reward = -math.inf
            self._output_iter = None

            self._reset_and_construct_root()

            terminal_iter = False
            path = [self.root]
            # we can allow the agents to try multiple times in emulator but v-droid only try once.
            for idx in trange(self.n_iters, disable=True, desc='search iteration', leave=False):
                self.iter_idx += 1
                path = self.iterate(self.root, self.iter_idx)
                for idx, node in enumerate(path):
                    if node.is_terminal:
                        terminal_iter = True
                        break

                    if idx == len(path) - 1 and node.children is not None:
                        scores = [child.score for child in node.children]
                        child = node.children[self.simulate_choice(scores)]

                        try:
                            converted_action = json_action.JSONAction(
                                **agent_utils.extract_json(child.action),
                            )
                            node.node_info['action_output_json'] = converted_action

                        except Exception as e:
                            converted_action = None
                            print('Failed to convert the output to a valid action.')
                            print(str(e))

                            node.state = node.parent.state.copy()
                            node.node_info['summary'] = (
                                'Can not parse the output to a valid action. Please make sure to pick'
                                ' the action from the list with required parameters (if any) in the'
                                ' correct JSON format!'
                            )
                            node.node_info['ui_elements'] = node.parent.node_info['ui_elements'].copy(
                            )
                            if node.parent.node_info['html_desc']:
                                node.node_info['html_desc'] = node.parent.node_info['html_desc']
                            # we dont do penalty here to avoid double penalty
                            self._assign_action_failure_penalty(node)

                        if converted_action is not None:
                            if converted_action.action_type == 'status':
                                if converted_action.goal_status == 'infeasible':
                                    print(
                                        'Agent stopped since it thinks mission impossible.')
                                node.node_info['summary'] = 'Agent thinks the request has been completed.'
                                node.state = node.parent.state.copy()
                                node.is_terminal = True
                                node.node_info['ui_elements'] = node.parent.node_info['ui_elements'].copy(
                                )
                                if node.parent.node_info['html_desc']:
                                    node.node_info['html_desc'] = node.parent.node_info['html_desc']
                                node.reward, node.score_details = self.reward(
                                    node.score_details['self_eval'], node.is_terminal)
                                node.score = node.reward
                                logging.warning(
                                    f"The final reward for the finished node is {node.reward} ")

                                terminal_iter = True
                                break

                if terminal_iter:
                    break

            if self.output_strategy == 'max_reward':
                self._output_cum_reward, self._output_iter = self._dfs_max_reward([
                                                                                  self.root])
                if self._output_cum_reward == -math.inf:
                    self._output_iter = None

            is_done = False
            cal_path = None
            if self._output_iter is not None:
                cal_path = self._output_iter
            else:
                cal_path = path

            output = []
            images = []
            for node in cal_path:
                if node.is_terminal == True:
                    is_done = True
                node.node_info["step_number"] = node.depth
                output.append(node.node_info)
                images.append(node.state["screenshot_raw"])

            model_name = self.llm.model_name.lower()
            if 'llama-3.2' in model_name or 'llama-3.1' in model_name or 'deepseek' in model_name:
                torch.cuda.empty_cache()
            return is_done, output
        finally:
            self.goal = original_goal
            self.depth_limit = original_depth_limit
            self.explore_step_count_limit = original_explore_step_count_limit
