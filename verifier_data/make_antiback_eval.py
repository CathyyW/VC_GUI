import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_critic_augmented_p3 import render_prompt


BACK_LIKE_RECOVERIES = {"navigate_back", "close_dialog"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def write_json(path: Path, rows: list[dict[str, Any]] | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def action_type(row: dict[str, Any], side: str) -> str | None:
    return row[f"{side}_input"]["action"].get("action_type")


def pair_row(row: dict[str, Any], critic_override: str | None = None) -> dict[str, Any]:
    chosen_input = dict(row["chosen_input"])
    rejected_input = dict(row["rejected_input"])
    if critic_override is not None:
        chosen_input["critic_summary"] = critic_override
        rejected_input["critic_summary"] = critic_override
    chosen = row.get("chosen") or render_prompt(**chosen_input)
    rejected = row.get("rejected") or render_prompt(**rejected_input)
    if critic_override is not None:
        chosen = render_prompt(**chosen_input)
        rejected = render_prompt(**rejected_input)
    return {
        "id": row.get("id"),
        "chosen": chosen,
        "rejected": rejected,
        "metadata": row.get("metadata", {}),
        "chosen_action": chosen_input["action"],
        "rejected_action": rejected_input["action"],
        "critic_summary": chosen_input.get("critic_summary", ""),
    }


def pairs_only(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"chosen": row["chosen"], "rejected": row["rejected"]} for row in rows]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prev_outcome = Counter()
    recovery = Counter()
    rules = Counter()
    current_outcome = Counter()
    chosen_types = Counter()
    rejected_types = Counter()
    steps = set()

    for row in rows:
        metadata = row.get("metadata", {})
        prev_outcome[metadata.get("previous_outcome", "<none>")] += 1
        recovery[metadata.get("previous_suggested_recovery", "<none>")] += 1
        rules[metadata.get("construction_rule", "<none>")] += 1
        current_outcome[metadata.get("current_outcome", "<none>")] += 1
        chosen_types[row.get("chosen_action", {}).get("action_type", "<none>")] += 1
        rejected_types[row.get("rejected_action", {}).get("action_type", "<none>")] += 1
        steps.add((metadata.get("trajectory_id"), metadata.get("step_id")))

    return {
        "pairs": len(rows),
        "unique_steps": len(steps),
        "previous_outcomes": dict(prev_outcome),
        "previous_suggested_recoveries": dict(recovery),
        "construction_rules": dict(rules),
        "current_outcomes": dict(current_outcome),
        "chosen_action_types": dict(chosen_types),
        "rejected_action_types": dict(rejected_types),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build anti-back verifier evaluation sets from critic-augmented P3 debug rows."
        )
    )
    parser.add_argument(
        "--debug-jsonl",
        type=Path,
        default=Path(
            "data/verifier_p3/androidworld_critic_augmented_p3_20260606/train_debug.jsonl"
        ),
        help="Path to train_debug.jsonl from the P3 data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/verifier_p3/anti_back_eval_20260607"),
        help="Directory for anti-back evaluation files.",
    )
    args = parser.parse_args()

    debug_rows = read_jsonl(args.debug_jsonl)
    anti_back_rows: list[dict[str, Any]] = []
    anti_back_empty_rows: list[dict[str, Any]] = []
    valid_back_rows: list[dict[str, Any]] = []
    valid_back_empty_rows: list[dict[str, Any]] = []

    for row in debug_rows:
        metadata = row.get("metadata", {})
        recovery = metadata.get("previous_suggested_recovery")
        chosen_type = action_type(row, "chosen")
        rejected_type = action_type(row, "rejected")

        if rejected_type == "navigate_back" and recovery not in BACK_LIKE_RECOVERIES:
            anti_back_rows.append(pair_row(row))
            anti_back_empty_rows.append(pair_row(row, critic_override=""))

        if chosen_type == "navigate_back" and recovery in BACK_LIKE_RECOVERIES:
            valid_back_rows.append(pair_row(row))
            valid_back_empty_rows.append(pair_row(row, critic_override=""))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "anti_back_reject.jsonl", anti_back_rows)
    write_json(args.output_dir / "anti_back_reject_pairs.json", pairs_only(anti_back_rows))
    write_jsonl(args.output_dir / "anti_back_reject_empty_critic.jsonl", anti_back_empty_rows)
    write_json(
        args.output_dir / "anti_back_reject_empty_critic_pairs.json",
        pairs_only(anti_back_empty_rows),
    )
    write_jsonl(args.output_dir / "valid_back_chosen.jsonl", valid_back_rows)
    write_json(args.output_dir / "valid_back_chosen_pairs.json", pairs_only(valid_back_rows))
    write_jsonl(args.output_dir / "valid_back_chosen_empty_critic.jsonl", valid_back_empty_rows)
    write_json(
        args.output_dir / "valid_back_chosen_empty_critic_pairs.json",
        pairs_only(valid_back_empty_rows),
    )

    summary = {
        "source_debug_jsonl": str(args.debug_jsonl),
        "anti_back_reject": summarize(anti_back_rows),
        "anti_back_reject_empty_critic": summarize(anti_back_empty_rows),
        "valid_back_chosen": summarize(valid_back_rows),
        "valid_back_chosen_empty_critic": summarize(valid_back_empty_rows),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
