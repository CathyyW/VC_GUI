import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


BEGIN_TOKEN = "<|begin_of_text|>"


def read_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows.append({"chosen": row["chosen"], "rejected": row["rejected"]})
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def filter_rows(
    rows: list[dict[str, str]],
    tokenizer: Any,
    max_length: int,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    kept: list[dict[str, str]] = []
    stats = {
        "input": len(rows),
        "kept": 0,
        "dropped_identical_after_truncation": 0,
        "dropped_action_missing_after_truncation": 0,
    }

    for row in rows:
        chosen_ids = tokenizer(
            BEGIN_TOKEN + row["chosen"],
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        rejected_ids = tokenizer(
            BEGIN_TOKEN + row["rejected"],
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        if chosen_ids == rejected_ids:
            stats["dropped_identical_after_truncation"] += 1
            continue

        chosen_tail = tokenizer.decode(chosen_ids[-120:])
        rejected_tail = tokenizer.decode(rejected_ids[-120:])
        if (
            "Is {" not in chosen_tail
            or "Answer:" not in chosen_tail
            or "Is {" not in rejected_tail
            or "Answer:" not in rejected_tail
        ):
            stats["dropped_action_missing_after_truncation"] += 1
            continue
        kept.append(row)

    stats["kept"] = len(kept)
    return kept, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert critic-augmented P3 JSONL splits into V-Droid JSON arrays."
    )
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2800)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    sources = {
        "train": args.train_jsonl,
        "val": args.val_jsonl,
    }
    if args.test_jsonl:
        sources["test"] = args.test_jsonl

    all_stats: dict[str, dict[str, int]] = {}
    for split, source in sources.items():
        rows = read_jsonl(source)
        kept, stats = filter_rows(rows, tokenizer, args.max_length)
        write_json(args.output_dir / f"{split}.json", kept)
        all_stats[split] = stats
        print(split, stats)

    write_json(args.output_dir / "prepare_stats.json", all_stats)


if __name__ == "__main__":
    main()
