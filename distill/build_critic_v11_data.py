import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FILES = {
    "train": "critic_distill_train.jsonl",
    "val": "critic_distill_val.jsonl",
    "test": "critic_distill_test.jsonl",
}
RECOVERY_ACTIONS = {"navigate_back", "clear_text", "close_dialog"}
FAILURE_TYPES = (
    "no_effect",
    "text_error",
    "wrong_page",
    "wrong_target",
    "popup_blocking",
    "premature_complete",
)


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
    return sample_id.split("__repeat", 1)[0].split("__v11", 1)[0]


def load_corrections(path: Path | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, list):
        value = {item["id"]: item["output"] for item in value}
    if not isinstance(value, dict):
        raise ValueError("Correction file must be a dict or a list of {id, output} items.")
    return value


def apply_corrections(rows: list[dict[str, Any]], corrections: dict[str, dict[str, str]]) -> int:
    changed = 0
    for row in rows:
        corrected = corrections.get(base_id(row["id"]))
        if corrected:
            row["output"] = corrected
            changed += 1
    return changed


def action_type(row: dict[str, Any]) -> str:
    try:
        return json.loads(row["input"].get("action", "{}")).get("action_type", "")
    except (json.JSONDecodeError, AttributeError):
        return ""


def critic_bucket(row: dict[str, Any]) -> str:
    if action_type(row) in RECOVERY_ACTIONS:
        return "self_correction"
    output = row["output"]
    if output.get("outcome") in {"failure", "no_effect"} or output.get("failure_type") != "none":
        return "failure"
    return "success"


def unique_by_base_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = {}
    for row in rows:
        unique.setdefault(base_id(row["id"]), row)
    return [unique[key] for key in sorted(unique)]


def parse_targets(spec: str) -> dict[str, int]:
    targets = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, value = chunk.split("=", 1)
        name = name.strip()
        if name not in FAILURE_TYPES:
            raise ValueError(f"Unknown failure type {name}; expected one of {FAILURE_TYPES}")
        targets[name] = int(value)
    return targets


def sample_with_replacement(
    rows: list[dict[str, Any]],
    target: int,
    rng: random.Random,
    repeat_prefix: str,
) -> list[dict[str, Any]]:
    if target <= 0 or not rows:
        return []

    rows = [dict(row) for row in rows]
    rng.shuffle(rows)
    selected = []
    repeat_counts = Counter()

    for index in range(target):
        source = rows[index % len(rows)]
        row = dict(source)
        original_id = base_id(source["id"])
        repeat_counts[original_id] += 1
        if repeat_counts[original_id] == 1:
            row["id"] = original_id
        else:
            row["id"] = f"{original_id}__{repeat_prefix}{repeat_counts[original_id] - 1}"
        selected.append(row)

    selected.sort(key=lambda item: item["id"])
    return selected


def train_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = Counter(critic_bucket(row) for row in rows)
    failure_types = Counter(
        row["output"].get("failure_type")
        for row in rows
        if critic_bucket(row) == "failure"
    )
    action_types = Counter(action_type(row) for row in rows)
    unique_failure_types = Counter(
        row["output"].get("failure_type")
        for row in unique_by_base_id(rows)
        if critic_bucket(row) == "failure"
    )
    return {
        "total": len(rows),
        "buckets": dict(sorted(buckets.items())),
        "failure_types": dict(sorted(failure_types.items())),
        "unique_failure_types": dict(sorted(unique_failure_types.items())),
        "action_types": dict(sorted(action_types.items())),
    }


def build_train_v11(
    train_rows: list[dict[str, Any]],
    success_target: int,
    self_target: int,
    failure_targets: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    unique_rows = unique_by_base_id(train_rows)
    by_bucket = defaultdict(list)
    by_failure_type = defaultdict(list)

    for row in unique_rows:
        bucket = critic_bucket(row)
        by_bucket[bucket].append(row)
        if bucket == "failure":
            by_failure_type[row["output"].get("failure_type")].append(row)

    rebuilt = []
    rebuilt.extend(sample_with_replacement(by_bucket["success"], success_target, rng, "v11repeat"))
    rebuilt.extend(sample_with_replacement(by_bucket["self_correction"], self_target, rng, "v11repeat"))

    for failure_type in FAILURE_TYPES:
        target = failure_targets.get(failure_type, 0)
        sampled = sample_with_replacement(
            by_failure_type[failure_type],
            target,
            rng,
            f"v11_{failure_type}_repeat",
        )
        rebuilt.extend(sampled)

    rebuilt.sort(key=lambda item: (critic_bucket(item), item["output"].get("failure_type", ""), item["id"]))
    return rebuilt


def main() -> None:
    parser = argparse.ArgumentParser(description="Build C-v1.1 critic SFT data with corrected labels and failure-type resampling.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corrections-file")
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--success-target", type=int, default=227)
    parser.add_argument("--self-target", type=int, default=12)
    parser.add_argument(
        "--failure-type-targets",
        default="no_effect=70,text_error=90,wrong_page=20,wrong_target=20,popup_blocking=12,premature_complete=12",
        help="Comma-separated train targets for failure subtypes.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    corrections = load_corrections(Path(args.corrections_file) if args.corrections_file else None)
    failure_targets = parse_targets(args.failure_type_targets)

    loaded = {split: read_jsonl(input_dir / filename) for split, filename in FILES.items()}
    correction_counts = {
        split: apply_corrections(rows, corrections)
        for split, rows in loaded.items()
    }

    original_train_stats = train_stats(loaded["train"])
    loaded["train"] = build_train_v11(
        loaded["train"],
        args.success_target,
        args.self_target,
        failure_targets,
        args.seed,
    )
    rebuilt_train_stats = train_stats(loaded["train"])

    for split, filename in FILES.items():
        write_jsonl(output_dir / filename, loaded[split])

    metadata = {
        "source": str(input_dir),
        "seed": args.seed,
        "corrections_file": args.corrections_file,
        "correction_counts": correction_counts,
        "success_target": args.success_target,
        "self_target": args.self_target,
        "failure_type_targets": failure_targets,
        "original_train": original_train_stats,
        "rebuilt_train": rebuilt_train_stats,
        "val": train_stats(loaded["val"]),
        "test": train_stats(loaded["test"]),
    }
    (output_dir / "v11_resample_stats.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Wrote C-v1.1 SFT data to {output_dir}")


if __name__ == "__main__":
    main()
