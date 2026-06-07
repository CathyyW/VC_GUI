import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROMPT_PREFIX = (
    'You are an agent capable of operating an Android phone on behalf of a user. Your task is to assist with user requests or goals by:\n'
    '1. Answering questions or chat-like messages, such as "What is my schedule for today?".\n'
    "2. Performing tasks step-by-step on the phone based on the user's instructions.\n\n"
    "For each step, you will be provided:\n"
    "- A history of actions you have taken so far.\n"
    "- Critic feedback about the outcomes of previous actions.\n"
    "- The current screenshot's HTML description.\n\n"
)

SELF_EVAL_TEMPLATE_VERIFIER_TRAINING_V3 = (
    "{prompt_prefix}"
    + "The (overall) user goal/request is: {goal}\n\n"
    "Here is the history of actions taken:\n{history}\n\n"
    "Here is the critic feedback from previous actions:\n{critic_summary}\n\n"
    "Here is the detailed information about the UI elements in the current screenshot:\n{before_ui_html}\n"
    "\n"
    "Your task: \n"
    "- Determine if the action is helpful for completing the user's task.\n"
    '- Respond with **"Yes"** if the action is helpful, even if it does not directly complete the task.\n'
    '- Respond with **"No"** if the action is not helpful for the task.\n'
    + "\n"
    "Is {action} helpful for completing the task?\n"
    "Answer:"
)

SPLIT_FILES = {
    "train": "critic_distill_train.jsonl",
    "val": "critic_distill_val.jsonl",
    "test": "critic_distill_test.jsonl",
}

RECOVERY_ACTION_TYPES = {
    "navigate_back": {"navigate_back"},
    "clear_text": {"clear_text"},
    # The collected action space does not contain a dedicated close_dialog action.
    # navigate_back is the closest generic recovery action for overlays/menus.
    "close_dialog": {"navigate_back"},
    "complete_task": {"status", "answer"},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def base_id(sample_id: str) -> str:
    return sample_id.split("__", 1)[0]


def step_id(trajectory: dict[str, Any], step: dict[str, Any]) -> str:
    return (
        f"{trajectory['task_id']}_inst{trajectory['instance_id']}"
        f"_seed{trajectory['seed']}_step{step['step_id']}"
    )


def action_key(action: dict[str, Any] | None) -> tuple[Any, ...]:
    if not action:
        return ("",)
    action_type = action.get("action_type", "")
    if action_type in {"click", "long_press", "clear_text"}:
        return (action_type, action.get("index"))
    if action_type == "input_text":
        return (action_type, action.get("index"))
    if action_type == "scroll":
        return (action_type, action.get("direction"), action.get("index"))
    if action_type in {"open_app", "answer"}:
        # Candidate action spaces use placeholders for these fields.
        return (action_type,)
    if action_type == "status":
        return (action_type, action.get("goal_status"))
    return (action_type,)


def same_action(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    return action_key(left) == action_key(right)


def action_to_text(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, sort_keys=True)


def history_to_text(history: Any) -> str:
    if not history:
        return "(empty)"
    if isinstance(history, list):
        return "\n".join(str(item) for item in history)
    return str(history)


def render_prompt(
    goal: str,
    history: Any,
    critic_summary: str,
    before_ui_html: str,
    action: dict[str, Any],
) -> str:
    return SELF_EVAL_TEMPLATE_VERIFIER_TRAINING_V3.format(
        prompt_prefix=PROMPT_PREFIX,
        goal=goal or "",
        history=history_to_text(history),
        critic_summary=critic_summary or "",
        before_ui_html=before_ui_html or "",
        action=action_to_text(action),
    )


def load_critic_data(critic_data_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    critic_by_id: dict[str, dict[str, Any]] = {}
    split_by_id: dict[str, str] = {}
    for split, filename in SPLIT_FILES.items():
        path = critic_data_dir / filename
        for row in read_jsonl(path):
            row_id = base_id(row["id"])
            critic_by_id.setdefault(row_id, row["output"])
            split_by_id.setdefault(row_id, split)
    return critic_by_id, split_by_id


def ensure_critic_prefix(label: dict[str, Any]) -> dict[str, Any]:
    label = dict(label)
    summary = (label.get("summary_to_history") or "").strip()
    if summary and not summary.startswith("[Critic]"):
        summary = f"[Critic] {summary}"
    label["summary_to_history"] = summary
    return label


def load_critic_labels(labels_file: Path) -> dict[str, dict[str, Any]]:
    labels = {}
    for row in read_jsonl(labels_file):
        labels[base_id(row["id"])] = ensure_critic_prefix(row["label"])
    return labels


def load_corrections(corrections_file: Path | None) -> dict[str, dict[str, Any]]:
    if not corrections_file:
        return {}
    corrections = json.loads(corrections_file.read_text(encoding="utf-8-sig"))
    if not isinstance(corrections, dict):
        raise ValueError("Corrections file must be a JSON object keyed by transition id.")
    return {base_id(row_id): ensure_critic_prefix(label) for row_id, label in corrections.items()}


def split_name(trajectory_id: str) -> str:
    bucket = int(hashlib.sha1(trajectory_id.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def trajectory_id_of(trajectory: dict[str, Any]) -> str:
    return f"{trajectory['task_id']}_inst{trajectory['instance_id']}_seed{trajectory['seed']}"


def group_balanced_splits(
    trajectories: list[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, str]:
    groups = []
    for trajectory in trajectories:
        trajectory_id = trajectory_id_of(trajectory)
        groups.append(
            {
                "trajectory_id": trajectory_id,
                "size": len(trajectory.get("steps", [])),
                "order": hashlib.sha1(trajectory_id.encode("utf-8")).hexdigest(),
            }
        )
    groups.sort(key=lambda item: item["order"])

    total = sum(group["size"] for group in groups)
    targets = {
        "train": total * train_ratio,
        "val": total * val_ratio,
        "test": total * (1 - train_ratio - val_ratio),
    }
    sizes = {"train": 0, "val": 0, "test": 0}
    assignments = {}

    def score(after_sizes: dict[str, int]) -> float:
        return sum(((after_sizes[split] - targets[split]) / max(1, total)) ** 2 for split in sizes)

    for group in groups:
        best_split = None
        best_score = float("inf")
        for split in ("train", "val", "test"):
            proposed = dict(sizes)
            proposed[split] += group["size"]
            proposed_score = score(proposed)
            if proposed_score < best_score:
                best_score = proposed_score
                best_split = split
        sizes[best_split] += group["size"]
        assignments[group["trajectory_id"]] = best_split
    return assignments


def load_trajectories(root: Path) -> list[dict[str, Any]]:
    files = sorted(root.glob("**/trajectories/*.json"))
    trajectories = []
    for path in files:
        trajectory = json.loads(path.read_text(encoding="utf-8-sig"))
        trajectory["_source_path"] = str(path)
        trajectories.append(trajectory)
    return trajectories


def candidate_rows(step: dict[str, Any]) -> list[dict[str, Any]]:
    actions = step.get("action_space") or []
    scores = step.get("scores") or []
    rows = []
    for index, action in enumerate(actions):
        score = scores[index] if index < len(scores) else None
        rows.append({"action": action, "score": score, "index": index})
    return rows


def find_recovery_action(
    candidates: list[dict[str, Any]],
    suggested_recovery: str,
    previous_action: dict[str, Any] | None,
) -> dict[str, Any] | None:
    allowed_types = RECOVERY_ACTION_TYPES.get(suggested_recovery, set())
    if not allowed_types:
        return None

    matches = [row for row in candidates if row["action"].get("action_type") in allowed_types]
    if suggested_recovery == "clear_text" and previous_action:
        previous_index = previous_action.get("index")
        same_index = [row for row in matches if row["action"].get("index") == previous_index]
        if same_index:
            matches = same_index

    if suggested_recovery == "complete_task":
        complete = [
            row
            for row in matches
            if row["action"].get("action_type") == "status"
            and row["action"].get("goal_status") == "complete"
        ]
        if complete:
            matches = complete

    if not matches:
        return None
    return max(matches, key=lambda row: row["score"] if row["score"] is not None else float("-inf"))["action"]


def low_score_rejections(
    candidates: list[dict[str, Any]],
    chosen: dict[str, Any],
    max_rejected: int,
) -> list[dict[str, Any]]:
    rejected = [row for row in candidates if not same_action(row["action"], chosen)]
    rejected.sort(key=lambda row: row["score"] if row["score"] is not None else float("-inf"))
    if max_rejected > 0:
        rejected = rejected[:max_rejected]
    return [row["action"] for row in rejected]


def top_non_matching_action(
    candidates: list[dict[str, Any]],
    avoid: dict[str, Any] | None,
) -> dict[str, Any] | None:
    available = [row for row in candidates if not same_action(row["action"], avoid)]
    if not available:
        return None
    return max(available, key=lambda row: row["score"] if row["score"] is not None else float("-inf"))["action"]


def choose_actions(
    step: dict[str, Any],
    previous_step: dict[str, Any] | None,
    previous_critic: dict[str, Any] | None,
    max_rejected: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    candidates = candidate_rows(step)
    if not candidates:
        return None, [], "skip_no_action_space"

    selected = step.get("selected_action")
    previous_action = previous_step.get("selected_action") if previous_step else None
    previous_outcome = (previous_critic or {}).get("outcome")
    previous_recovery = (previous_critic or {}).get("suggested_recovery")

    if previous_outcome in {None, "progress"}:
        chosen = selected
        rule = "initial_or_previous_progress_selected"
    elif previous_outcome == "failure":
        chosen = find_recovery_action(candidates, previous_recovery or "", previous_action)
        rule = f"previous_failure_recovery_{previous_recovery}"
        if chosen is None:
            chosen = selected
            rule += "_fallback_selected"
    elif previous_outcome == "no_effect":
        chosen = selected
        if same_action(chosen, previous_action):
            chosen = top_non_matching_action(candidates, previous_action)
            rule = "previous_no_effect_top_non_repeat"
        else:
            rule = "previous_no_effect_selected_non_repeat"
    elif previous_outcome == "done":
        chosen = find_recovery_action(candidates, "complete_task", previous_action)
        rule = "previous_done_complete_task"
        if chosen is None:
            return None, [], "skip_previous_done_no_complete_action"
    else:
        chosen = selected
        rule = f"unknown_previous_outcome_{previous_outcome}_fallback_selected"

    if chosen is None:
        return None, [], "skip_no_chosen"

    rejections = []
    if previous_outcome == "failure" and selected and not same_action(selected, chosen):
        rejections.append(selected)
    if previous_outcome == "no_effect" and previous_action:
        current_equivalent = [row["action"] for row in candidates if same_action(row["action"], previous_action)]
        rejections.extend(current_equivalent)
    if previous_outcome == "done" and selected and not same_action(selected, chosen):
        rejections.append(selected)

    seen = {action_key(action) for action in rejections if not same_action(action, chosen)}
    rejections = [action for action in rejections if not same_action(action, chosen)]
    for action in low_score_rejections(candidates, chosen, 0):
        key = action_key(action)
        if key in seen:
            continue
        seen.add(key)
        rejections.append(action)
        if max_rejected > 0 and len(rejections) >= max_rejected:
            break

    if max_rejected > 0:
        rejections = rejections[:max_rejected]
    return chosen, rejections, rule


def write_markdown_examples(path: Path, debug_rows: list[dict[str, Any]], limit: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Critic-Augmented P3 Examples", ""]
    for row in debug_rows[:limit]:
        lines.extend(
            [
                f"## {row['id']}",
                "",
                f"- split: `{row['split']}`",
                f"- rule: `{row['metadata']['construction_rule']}`",
                f"- previous outcome: `{row['metadata'].get('previous_outcome')}`",
                f"- current outcome: `{row['metadata'].get('current_outcome')}`",
                "",
                "chosen action:",
                "```json",
                json.dumps(row["chosen_input"]["action"], ensure_ascii=False, indent=2),
                "```",
                "",
                "rejected action:",
                "```json",
                json.dumps(row["rejected_input"]["action"], ensure_ascii=False, indent=2),
                "```",
                "",
                "critic summary:",
                "```text",
                row["chosen_input"]["critic_summary"],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build critic-augmented P3 prompt pairs for V-Droid verifier tuning.")
    parser.add_argument("--trajectories-root", required=True)
    parser.add_argument(
        "--critic-data-dir",
        help="Optional critic SFT split directory. Provides labels and/or split map from critic_distill_train/val/test.jsonl.",
    )
    parser.add_argument(
        "--critic-labels-file",
        help="Optional full reviewed critic labels JSONL. Prefer this for P3 so all raw trajectory steps have labels.",
    )
    parser.add_argument("--corrections-file", help="Optional JSON corrections keyed by transition id.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split-strategy",
        choices=["group_balanced", "trajectory_hash", "critic_data"],
        default="group_balanced",
        help=(
            "group_balanced keeps a whole trajectory in one split while approximating 80/10/10; "
            "trajectory_hash keeps a whole trajectory with hash buckets; "
            "critic_data reuses critic SFT split ids when available."
        ),
    )
    parser.add_argument(
        "--max-rejected-per-step",
        type=int,
        default=0,
        help="0 means use all eligible rejected actions; positive values cap rejections per chosen action.",
    )
    parser.add_argument(
        "--skip-bad-selected-chosen",
        action="store_true",
        help=(
            "If the construction rule chooses the original selected_action, skip the step "
            "unless that selected_action's post-action critic label is progress or done."
        ),
    )
    parser.add_argument(
        "--skip-unverified-back-chosen",
        action="store_true",
        help=(
            "Skip navigate_back as chosen when it was not the executed selected_action and "
            "the previous critic did not explicitly recommend navigate_back/close_dialog."
        ),
    )
    args = parser.parse_args()
    if not args.critic_data_dir and not args.critic_labels_file:
        raise ValueError("Provide --critic-labels-file, --critic-data-dir, or both.")

    trajectories = load_trajectories(Path(args.trajectories_root))
    trajectory_split_by_id = group_balanced_splits(trajectories)
    critic_by_id: dict[str, dict[str, Any]] = {}
    split_by_id: dict[str, str] = {}
    if args.critic_data_dir:
        critic_by_id, split_by_id = load_critic_data(Path(args.critic_data_dir))
        critic_by_id = {row_id: ensure_critic_prefix(label) for row_id, label in critic_by_id.items()}
    if args.critic_labels_file:
        critic_by_id.update(load_critic_labels(Path(args.critic_labels_file)))
    critic_by_id.update(load_corrections(Path(args.corrections_file) if args.corrections_file else None))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step_inputs_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompt_pairs_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    debug_pairs_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    stats = {
        "source": {
            "trajectories_root": args.trajectories_root,
            "critic_data_dir": args.critic_data_dir,
            "critic_labels_file": args.critic_labels_file,
            "corrections_file": args.corrections_file,
        },
        "split_strategy": args.split_strategy,
        "max_rejected_per_step": args.max_rejected_per_step,
        "skip_bad_selected_chosen": args.skip_bad_selected_chosen,
        "skip_unverified_back_chosen": args.skip_unverified_back_chosen,
        "trajectories": len(trajectories),
        "steps_seen": 0,
        "steps_with_pairs": Counter(),
        "pairs": Counter(),
        "rules": Counter(),
        "previous_outcomes": Counter(),
        "current_outcomes": Counter(),
        "skipped": skipped,
    }

    for trajectory in trajectories:
        steps = sorted(trajectory.get("steps", []), key=lambda item: item.get("step_id", 0))
        for position, step in enumerate(steps):
            stats["steps_seen"] += 1
            current_id = step_id(trajectory, step)
            trajectory_id = trajectory_id_of(trajectory)
            if args.split_strategy == "critic_data" and current_id in split_by_id:
                split = split_by_id[current_id]
            elif args.split_strategy == "trajectory_hash":
                split = split_name(trajectory_id)
            else:
                split = trajectory_split_by_id[trajectory_id]
            current_critic = critic_by_id.get(current_id)
            if not current_critic:
                skipped["missing_current_critic"] += 1
                continue

            previous_step = steps[position - 1] if position else None
            previous_id = step_id(trajectory, previous_step) if previous_step else None
            previous_critic = critic_by_id.get(previous_id) if previous_id else None
            critic_summary = (previous_critic or {}).get("summary_to_history", "")

            chosen_action, rejected_actions, rule = choose_actions(
                step,
                previous_step,
                previous_critic,
                args.max_rejected_per_step,
            )
            if not chosen_action or not rejected_actions:
                skipped[rule if rule.startswith("skip") else "no_rejected_actions"] += 1
                continue
            chosen_matches_selected = same_action(chosen_action, step.get("selected_action"))
            previous_recovery = (previous_critic or {}).get("suggested_recovery")
            if (
                args.skip_bad_selected_chosen
                and chosen_matches_selected
                and current_critic.get("outcome") not in {"progress", "done"}
            ):
                skipped["bad_selected_action_used_as_chosen"] += 1
                continue
            if (
                args.skip_unverified_back_chosen
                and chosen_action.get("action_type") == "navigate_back"
                and not chosen_matches_selected
                and previous_recovery not in {"navigate_back", "close_dialog"}
            ):
                skipped["unverified_navigate_back_used_as_chosen"] += 1
                continue

            step_input = {
                "id": current_id,
                "trajectory_id": trajectory_id,
                "task_id": trajectory["task_id"],
                "step_id": step.get("step_id"),
                "split": split,
                "goal": step.get("goal") or trajectory.get("goal"),
                "history": step.get("history") or [],
                "critic_summary": critic_summary,
                "previous_critic": previous_critic,
                "current_critic": current_critic,
                "before_ui_html": step.get("before_ui_html") or "",
                "selected_action": step.get("selected_action"),
                "action_space_size": len(step.get("action_space") or []),
            }
            step_inputs_by_split[split].append(step_input)
            stats["steps_with_pairs"][split] += 1
            stats["previous_outcomes"][(previous_critic or {}).get("outcome", "<none>")] += 1
            stats["current_outcomes"][current_critic.get("outcome", "<missing>")] += 1
            stats["rules"][rule] += 1

            for pair_index, rejected_action in enumerate(rejected_actions):
                pair_id = f"{current_id}_pair{pair_index}"
                chosen_input = {
                    "goal": step_input["goal"],
                    "history": step_input["history"],
                    "critic_summary": step_input["critic_summary"],
                    "before_ui_html": step_input["before_ui_html"],
                    "action": chosen_action,
                }
                rejected_input = {
                    "goal": step_input["goal"],
                    "history": step_input["history"],
                    "critic_summary": step_input["critic_summary"],
                    "before_ui_html": step_input["before_ui_html"],
                    "action": rejected_action,
                }
                prompt_pair = {
                    "chosen": render_prompt(**chosen_input),
                    "rejected": render_prompt(**rejected_input),
                }
                debug_pair = {
                    "id": pair_id,
                    "split": split,
                    "chosen_input": chosen_input,
                    "rejected_input": rejected_input,
                    "metadata": {
                        "trajectory_id": step_input["trajectory_id"],
                        "task_id": step_input["task_id"],
                        "step_id": step_input["step_id"],
                        "previous_step_id": previous_step.get("step_id") if previous_step else None,
                        "previous_outcome": (previous_critic or {}).get("outcome"),
                        "previous_failure_type": (previous_critic or {}).get("failure_type"),
                        "previous_suggested_recovery": (previous_critic or {}).get("suggested_recovery"),
                        "current_outcome": current_critic.get("outcome"),
                        "current_failure_type": current_critic.get("failure_type"),
                        "current_suggested_recovery": current_critic.get("suggested_recovery"),
                        "construction_rule": rule,
                        "chosen_matches_selected_action": chosen_matches_selected,
                    },
                }
                prompt_pairs_by_split[split].append(prompt_pair)
                debug_pairs_by_split[split].append(debug_pair)
                stats["pairs"][split] += 1

    for split in ("train", "val", "test"):
        write_jsonl(output_dir / f"{split}.jsonl", prompt_pairs_by_split[split])
        write_jsonl(output_dir / f"{split}_debug.jsonl", debug_pairs_by_split[split])
        write_jsonl(output_dir / f"step_inputs_{split}.jsonl", step_inputs_by_split[split])

    serializable_stats = {
        **stats,
        "steps_with_pairs": dict(stats["steps_with_pairs"]),
        "pairs": dict(stats["pairs"]),
        "rules": dict(stats["rules"]),
        "previous_outcomes": dict(stats["previous_outcomes"]),
        "current_outcomes": dict(stats["current_outcomes"]),
        "skipped": dict(stats["skipped"]),
    }
    (output_dir / "stats.json").write_text(
        json.dumps(serializable_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    all_debug = []
    for split in ("train", "val", "test"):
        all_debug.extend(debug_pairs_by_split[split])
    write_markdown_examples(output_dir / "examples.md", all_debug)
    print(json.dumps(serializable_stats, ensure_ascii=False, indent=2))
    print(f"Wrote critic-augmented P3 data to {output_dir}")


if __name__ == "__main__":
    main()
