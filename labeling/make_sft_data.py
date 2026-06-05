import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a post-action critic for Android GUI automation. "
    "Given goal, recent history, before UI, selected action, and after UI, "
    "output strict JSON with outcome, failure_type, suggested_recovery, and summary_to_history."
)

BUCKETS = ("success", "failure", "self_correction")
SPLITS = ("train", "val", "test")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def compact_text(text: str) -> str:
    return "\n".join(line.strip() for line in (text or "").splitlines() if line.strip())


def normalize_ui(text: str) -> str:
    return " ".join((text or "").split())


def action_type(transition: dict) -> str:
    action = transition.get("selected_action") or {}
    return action.get("action_type", "")


def same_ui_before_after(transition: dict) -> bool:
    return normalize_ui(transition.get("before_ui_html")) == normalize_ui(transition.get("after_ui_html"))


def distill_bucket(transition: dict, label: dict) -> str:
    """Three-way grouping used only for split/balance control."""
    act_type = action_type(transition)
    if act_type in {"navigate_back", "clear_text", "close_dialog"} or (act_type == "navigate_back" and same_ui_before_after(transition)):
        return "self_correction"
    if label.get("outcome") in {"failure", "no_effect"} or label.get("failure_type") != "none":
        return "failure"
    return "success"


def split_name(trajectory_id: str) -> str:
    bucket = int(hashlib.sha1(trajectory_id.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def build_user_content(t: dict) -> str:
    history = t.get("history") or []
    history_text = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(history)) or "(empty)"
    return f"""Goal:
{t.get("goal")}

Recent history:
{history_text}

Before UI HTML:
{compact_text(t.get("before_ui_html", ""))}

Selected action:
{json.dumps(t.get("selected_action"), ensure_ascii=False)}

After UI HTML:
{compact_text(t.get("after_ui_html", ""))}
"""


def build_input_output_sample(sample_id: str, t: dict, label: dict) -> dict:
    return {
        "id": sample_id,
        "input": {
            "goal": t.get("goal"),
            "history": t.get("history") or [],
            "before_ui": compact_text(t.get("before_ui_html", "")),
            "action": json.dumps(t.get("selected_action"), ensure_ascii=False),
            "after_ui": compact_text(t.get("after_ui_html", "")),
        },
        "output": label,
    }


def build_chat_sample(sample_id: str, t: dict, label: dict) -> dict:
    return {
        "id": sample_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(t)},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)},
        ],
    }


def split_counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_n = round(n * train_ratio)
    val_n = round(n * val_ratio)
    if train_n + val_n > n:
        overflow = train_n + val_n - n
        val_n = max(0, val_n - overflow)
    test_n = n - train_n - val_n
    return train_n, val_n, test_n


def stratified_split(records: list[dict], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_bucket = defaultdict(list)
    for record in records:
        by_bucket[record["bucket"]].append(record)

    splits = {"train": [], "val": [], "test": []}
    for bucket in BUCKETS:
        items = sorted(by_bucket[bucket], key=lambda item: item["id"])
        rng.shuffle(items)
        train_n, val_n, _ = split_counts(len(items), train_ratio, val_ratio)
        splits["train"].extend(items[:train_n])
        splits["val"].extend(items[train_n : train_n + val_n])
        splits["test"].extend(items[train_n + val_n :])

    for split_items in splits.values():
        split_items.sort(key=lambda item: item["id"])
    return splits


def trajectory_hash_split(records: list[dict]) -> dict[str, list[dict]]:
    splits = {split: [] for split in SPLITS}
    for record in records:
        splits[split_name(record["transition"]["trajectory_id"])].append(record)
    return splits


def group_stratified_split(records: list[dict], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_trajectory = defaultdict(list)
    for record in records:
        by_trajectory[record["transition"]["trajectory_id"]].append(record)

    groups = []
    for trajectory_id, items in sorted(by_trajectory.items()):
        groups.append(
            {
                "trajectory_id": trajectory_id,
                "records": items,
                "total": len(items),
                "buckets": Counter(item["bucket"] for item in items),
            }
        )

    split_ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": 1 - train_ratio - val_ratio,
    }
    total_buckets = Counter(item["bucket"] for item in records)
    total_n = len(records)
    target_total = {split: total_n * split_ratios[split] for split in SPLITS}
    target_buckets = {
        split: {bucket: total_buckets[bucket] * split_ratios[split] for bucket in BUCKETS}
        for split in SPLITS
    }

    def evaluate(assignment: list[str]) -> tuple[float, dict[str, Counter], dict[str, int]]:
        bucket_counts = {split: Counter() for split in SPLITS}
        split_sizes = {split: 0 for split in SPLITS}
        for group, split in zip(groups, assignment):
            bucket_counts[split].update(group["buckets"])
            split_sizes[split] += group["total"]

        score = 0.0
        for split in SPLITS:
            score += 12 * ((split_sizes[split] - target_total[split]) / total_n) ** 2
            for bucket in BUCKETS:
                # Val/test need enough error cases to make validation meaningful.
                weight = 8 if split != "train" and bucket == "failure" else 3 if split != "train" else 1
                denom = max(1, total_buckets[bucket])
                score += weight * ((bucket_counts[split][bucket] - target_buckets[split][bucket]) / denom) ** 2

        min_failure = max(1, round(total_buckets["failure"] * split_ratios["val"] * 0.8))
        for split in ("val", "test"):
            if bucket_counts[split]["failure"] < min_failure:
                score += (min_failure - bucket_counts[split]["failure"]) * 0.08
        return score, bucket_counts, split_sizes

    def random_initial_assignment() -> list[str]:
        order = list(range(len(groups)))
        rng.shuffle(order)
        assignment = ["train"] * len(groups)
        split_sizes = {split: 0 for split in SPLITS}
        for index in order:
            candidates = sorted(SPLITS, key=lambda split: split_sizes[split] - target_total[split])
            split = candidates[0] if rng.random() > 0.25 else rng.choice(SPLITS)
            assignment[index] = split
            split_sizes[split] += groups[index]["total"]
        return assignment

    best_assignment = None
    best_score = float("inf")
    restarts = 120
    steps = 1200
    for _ in range(restarts):
        assignment = random_initial_assignment()
        current_score = evaluate(assignment)[0]
        temperature = 0.05
        for _ in range(steps):
            index = rng.randrange(len(groups))
            old_split = assignment[index]
            new_split = rng.choice([split for split in SPLITS if split != old_split])
            assignment[index] = new_split
            new_score = evaluate(assignment)[0]
            if new_score < current_score or rng.random() < math.exp((current_score - new_score) / max(temperature, 1e-9)):
                current_score = new_score
            else:
                assignment[index] = old_split
            temperature *= 0.998
        final_score = evaluate(assignment)[0]
        if final_score < best_score:
            best_score = final_score
            best_assignment = list(assignment)

    splits = {split: [] for split in SPLITS}
    for group, split in zip(groups, best_assignment or []):
        splits[split].extend(group["records"])
    for split_items in splits.values():
        split_items.sort(key=lambda item: item["id"])
    return splits


def balance_train(records: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_bucket = defaultdict(list)
    for record in records:
        by_bucket[record["bucket"]].append(record)

    target = max((len(items) for items in by_bucket.values()), default=0)
    balanced = []
    for bucket in BUCKETS:
        items = list(by_bucket[bucket])
        if not items:
            continue
        balanced.extend(items)
        needed = target - len(items)
        shuffled = list(items)
        rng.shuffle(shuffled)
        for index in range(needed):
            clone = dict(shuffled[index % len(shuffled)])
            clone["repeat_index"] = index + 1
            balanced.append(clone)

    balanced.sort(key=lambda item: (item["bucket"], item["id"], item.get("repeat_index", 0)))
    return balanced


def parse_bucket_ratios(spec: str) -> dict[str, float]:
    ratios = {}
    for chunk in spec.split(","):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid ratio item: {chunk}")
        name, value = chunk.split("=", 1)
        name = name.strip()
        if name not in BUCKETS:
            raise ValueError(f"Unknown bucket {name}; expected one of {BUCKETS}")
        ratios[name] = float(value)
    missing = [bucket for bucket in BUCKETS if bucket not in ratios]
    if missing:
        raise ValueError(f"Missing ratio(s): {missing}")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Bucket ratios must sum to a positive number.")
    return {bucket: ratios[bucket] / total for bucket in BUCKETS}


def resample_train_to_ratios(records: list[dict], ratios: dict[str, float], seed: int) -> list[dict]:
    """Resample train to target bucket ratios.

    The target total is anchored on the larger non-self-correction bucket so that
    the self-correction bucket can be capped instead of forcing huge oversampling.
    """
    rng = random.Random(seed)
    by_bucket = defaultdict(list)
    for record in records:
        by_bucket[record["bucket"]].append(record)

    non_self = [bucket for bucket in ("success", "failure") if ratios[bucket] > 0 and by_bucket[bucket]]
    if non_self:
        target_total = max(len(by_bucket[bucket]) / ratios[bucket] for bucket in non_self)
    else:
        target_total = len(records)

    targets = {}
    for bucket in BUCKETS:
        target = round(target_total * ratios[bucket])
        if ratios[bucket] > 0 and by_bucket[bucket]:
            target = max(1, target)
        targets[bucket] = target

    resampled = []
    for bucket in BUCKETS:
        items = list(by_bucket[bucket])
        rng.shuffle(items)
        target = targets[bucket]
        if target <= 0 or not items:
            continue
        if len(items) >= target:
            selected = items[:target]
            for item in selected:
                item = dict(item)
                item["repeat_index"] = 0
                resampled.append(item)
            continue

        for item in items:
            item = dict(item)
            item["repeat_index"] = 0
            resampled.append(item)
        needed = target - len(items)
        for index in range(needed):
            clone = dict(items[index % len(items)])
            clone["repeat_index"] = index + 1
            resampled.append(clone)

    resampled.sort(key=lambda item: (item["bucket"], item["id"], item.get("repeat_index", 0)))
    return resampled


def split_stats(splits: dict[str, list[dict]]) -> dict:
    stats = {}
    for split, items in splits.items():
        bucket_counts = Counter(item["bucket"] for item in items)
        outcome_counts = Counter(item["label"]["outcome"] for item in items)
        stats[split] = {
            "total": len(items),
            "buckets": {bucket: bucket_counts.get(bucket, 0) for bucket in BUCKETS},
            "outcomes": dict(outcome_counts),
        }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build C-model distillation JSONL from reviewed critic labels.")
    parser.add_argument("--transitions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--format",
        choices=["input_output", "chat"],
        default="input_output",
        help="input_output matches the VC critic distillation schema; chat is useful for some SFT frameworks.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["trajectory_hash", "stratified", "group_stratified"],
        default="trajectory_hash",
        help=(
            "trajectory_hash keeps all steps from one trajectory in the same split; "
            "stratified controls the three critic buckets at transition level; "
            "group_stratified controls buckets while keeping each trajectory in one split."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balanced-train",
        action="store_true",
        help="Oversample minority critic buckets in train only; val/test keep their natural stratified counts.",
    )
    parser.add_argument(
        "--train-bucket-ratios",
        help=(
            "Optional train-only target ratios, e.g. "
            "success=0.4875,failure=0.4875,self_correction=0.025. "
            "This supersedes --balanced-train."
        ),
    )
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train-ratio and --val-ratio must leave a positive test split.")

    transitions = {row["id"]: row for row in read_jsonl(Path(args.transitions))}
    labels = list(read_jsonl(Path(args.labels)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for row in labels:
        t = transitions.get(row["id"])
        if not t:
            continue
        label = row["label"]
        assistant_json = {
            "outcome": label["outcome"],
            "failure_type": label["failure_type"],
            "suggested_recovery": label["suggested_recovery"],
            "summary_to_history": label["summary_to_history"],
        }
        records.append(
            {
                "id": row["id"],
                "transition": t,
                "label": assistant_json,
                "bucket": distill_bucket(t, assistant_json),
                "repeat_index": 0,
            }
        )

    if args.split_strategy == "stratified":
        splits = stratified_split(records, args.train_ratio, args.val_ratio, args.seed)
    elif args.split_strategy == "group_stratified":
        splits = group_stratified_split(records, args.train_ratio, args.val_ratio, args.seed)
    else:
        splits = trajectory_hash_split(records)

    raw_stats = split_stats(splits)
    train_bucket_ratios = parse_bucket_ratios(args.train_bucket_ratios) if args.train_bucket_ratios else None
    if train_bucket_ratios:
        splits["train"] = resample_train_to_ratios(splits["train"], train_bucket_ratios, args.seed)
    elif args.balanced_train:
        splits["train"] = balance_train(splits["train"], args.seed)

    prefix = "critic_distill" if args.format == "input_output" else "critic_sft"
    files = {
        "train": (output_dir / f"{prefix}_train.jsonl").open("w", encoding="utf-8"),
        "val": (output_dir / f"{prefix}_val.jsonl").open("w", encoding="utf-8"),
        "test": (output_dir / f"{prefix}_test.jsonl").open("w", encoding="utf-8"),
    }
    counts = {name: 0 for name in files}

    try:
        for split, split_records in splits.items():
            for record in split_records:
                sample_id = record["id"]
                if record.get("repeat_index"):
                    sample_id = f"{sample_id}__repeat{record['repeat_index']}"
                if args.format == "input_output":
                    sample = build_input_output_sample(sample_id, record["transition"], record["label"])
                else:
                    sample = build_chat_sample(sample_id, record["transition"], record["label"])
                files[split].write(json.dumps(sample, ensure_ascii=False) + "\n")
                counts[split] += 1
    finally:
        for f in files.values():
            f.close()

    final_stats = split_stats(splits)
    stats = {
        "split_strategy": args.split_strategy,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1 - args.train_ratio - args.val_ratio,
        "balanced_train": args.balanced_train,
        "train_bucket_ratios": train_bucket_ratios,
        "raw_before_balance": raw_stats,
        "final": final_stats,
    }
    with (output_dir / "split_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Wrote SFT data to {output_dir}")
    print(counts)
    print(json.dumps(final_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
