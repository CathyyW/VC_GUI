import argparse
import json
from pathlib import Path


def iter_trajectory_files(dataset_dir: Path):
    yield from sorted(dataset_dir.glob("run_*/trajectories/*.json"))


def selected_score(step):
    selected = step.get("selected_action")
    if selected is None:
        return None
    for action, score in zip(step.get("action_space", []), step.get("scores", [])):
        if action == selected:
            return score
    return None


def main():
    parser = argparse.ArgumentParser(description="Flatten VC raw trajectories into transition JSONL.")
    parser.add_argument("--dataset-dir", required=True, help="Path to cloned cathyww/VC_raw_traj_ADW dataset.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--max-history", type=int, default=5, help="Keep only the last N history items.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8") as f:
        for path in iter_trajectory_files(dataset_dir):
            with path.open("r", encoding="utf-8") as tf:
                traj = json.load(tf)

            steps = traj.get("steps", [])
            task_id = traj.get("task_id")
            instance_id = traj.get("instance_id")
            seed = traj.get("seed")
            trajectory_id = f"{task_id}_inst{instance_id}_seed{seed}"

            for step in steps:
                step_id = step.get("step_id")
                history = step.get("history", [])
                if args.max_history > 0:
                    history = history[-args.max_history:]

                record = {
                    "id": f"{trajectory_id}_step{step_id}",
                    "trajectory_id": trajectory_id,
                    "source_file": str(path).replace("\\", "/"),
                    "task_id": task_id,
                    "instance_id": instance_id,
                    "seed": seed,
                    "goal": traj.get("goal") or step.get("goal"),
                    "trajectory_success": traj.get("task_success"),
                    "agent_indicated_done": traj.get("agent_indicated_done"),
                    "step_id": step_id,
                    "num_steps": len(steps),
                    "is_last_step": step_id == len(steps) - 1,
                    "history": history,
                    "before_ui_html": step.get("before_ui_html", ""),
                    "selected_action": step.get("selected_action"),
                    "after_ui_html": step.get("after_ui_html", ""),
                    "existing_summary": step.get("summary", ""),
                    "human_label": step.get("human_label", {}),
                    "action_space_size": len(step.get("action_space", [])),
                    "selected_score": selected_score(step),
                    "screenshot": step.get("screenshot"),
                    "screenshot_ann": step.get("screenshot_ann"),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} transitions to {output}")


if __name__ == "__main__":
    main()
