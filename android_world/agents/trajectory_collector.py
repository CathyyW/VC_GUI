"""Utilities for collecting raw AndroidWorld trajectories (framework.md schema)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np

from android_world.agents import agent_utils


def to_json_serializable(obj: Any) -> Any:
    """Convert numpy scalars/arrays and other non-JSON types for json.dump."""
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return to_json_serializable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def action_string_to_dict(action: str) -> dict[str, Any] | None:
    """Parse a JSON action string from the action space into a dict."""
    if not action:
        return None
    parsed = agent_utils.extract_json(action)
    if parsed is None:
        try:
            parsed = json.loads(action)
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def normalize_action_space(action_space: list[str]) -> list[dict[str, Any] | str]:
    """Convert action strings to dicts where possible; keep raw string on failure."""
    normalized = []
    for action in action_space:
        parsed = action_string_to_dict(action)
        normalized.append(parsed if parsed is not None else action)
    return normalized


def build_auto_human_label(
    action_space: list[str],
    scores: list[float],
    selected_action: str,
    negative_score_threshold: float = 0.15,
) -> dict[str, Any]:
    """Heuristic positives/negatives from verifier scores (for offline P3 later)."""
    best_action = action_string_to_dict(selected_action)
    bad_actions = []
    selected_parsed = action_string_to_dict(selected_action)

    for action, score in zip(action_space, scores):
        if score >= negative_score_threshold:
            continue
        parsed = action_string_to_dict(action)
        if parsed == selected_parsed:
            continue
        bad_actions.append(parsed if parsed is not None else action)

    return {
        "best_action": best_action if best_action is not None else selected_action,
        "bad_actions": bad_actions,
        "human_corrected": False,
    }


def build_step_record(
    *,
    task_id: str,
    goal: str,
    step_id: int,
    history: list[str],
    before_ui_html: str | None,
    after_ui_html: str | None,
    action_space: list[str],
    scores: list[float],
    selected_action: str,
    summary: str | None = None,
) -> dict[str, Any]:
    """One step in the raw trajectory format from framework.md."""
    record = {
        "task_id": task_id,
        "goal": goal,
        "step_id": step_id,
        "history": history,
        "before_ui_html": before_ui_html,
        "after_ui_html": after_ui_html,
        "action_space": normalize_action_space(action_space),
        "scores": [float(s) for s in scores],
        "selected_action": (
            action_string_to_dict(selected_action)
            if action_string_to_dict(selected_action) is not None
            else selected_action
        ),
        "human_label": build_auto_human_label(action_space, scores, selected_action),
    }
    if summary:
        record["summary"] = summary
    return record


def save_trajectory(
    output_dir: str,
    trajectory: dict[str, Any],
) -> str:
    """Write one trajectory JSON file and append to manifest.jsonl."""
    os.makedirs(output_dir, exist_ok=True)
    trajs_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(trajs_dir, exist_ok=True)

    task_id = trajectory.get("task_id", "unknown")
    instance_id = trajectory.get("instance_id", 0)
    seed = trajectory.get("seed", 0)
    filename = f"{task_id}_inst{instance_id}_seed{seed}.json"
    filepath = os.path.join(trajs_dir, filename)

    payload = to_json_serializable(trajectory)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    manifest_entry = to_json_serializable({
        "path": filepath,
        "task_id": task_id,
        "instance_id": instance_id,
        "seed": seed,
        "task_success": trajectory.get("task_success"),
        "num_steps": len(trajectory.get("steps", [])),
        "collected_at": trajectory.get("collected_at"),
    })
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

    return filepath


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
