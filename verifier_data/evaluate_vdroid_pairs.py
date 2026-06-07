import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any

import torch


BEGIN_TOKEN = "<|begin_of_text|>"
ACTION_PREFIX = "\nIs "
ACTION_SUFFIX = " helpful for completing the task?\nAnswer:"


def load_pairs(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def prepare_prompt(text: str, prepend_begin_token: bool) -> str:
    if prepend_begin_token and not text.startswith(BEGIN_TOKEN):
        return BEGIN_TOKEN + text
    return text


def extract_action(row: dict[str, Any], side: str) -> dict[str, Any] | None:
    action = row.get(f"{side}_action")
    if isinstance(action, dict):
        return action

    prompt = row.get(side)
    if not isinstance(prompt, str):
        return None
    start = prompt.rfind(ACTION_PREFIX)
    end = prompt.rfind(ACTION_SUFFIX)
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        parsed = json.loads(prompt[start + len(ACTION_PREFIX) : end])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def truncated_valid(tokenizer: Any, chosen: str, rejected: str, max_length: int) -> bool:
    chosen_ids = tokenizer(chosen, truncation=True, max_length=max_length)["input_ids"]
    rejected_ids = tokenizer(rejected, truncation=True, max_length=max_length)["input_ids"]
    if chosen_ids == rejected_ids:
        return False
    chosen_tail = tokenizer.decode(chosen_ids[-120:])
    rejected_tail = tokenizer.decode(rejected_ids[-120:])
    return "Is {" in chosen_tail and "Answer:" in chosen_tail and "Is {" in rejected_tail and "Answer:" in rejected_tail


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score V-Droid pairwise prompts and report chosen-vs-rejected margins."
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--vdroid-root", type=Path, default=Path("/root/autodl-tmp/V-Droid"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 means evaluate all pairs.")
    parser.add_argument("--max-length", type=int, default=2800)
    parser.add_argument(
        "--no-prepend-begin-token",
        action="store_true",
        help="Disable the begin token prefix used by the official training dataloader.",
    )
    args = parser.parse_args()

    if str(args.vdroid_root) not in sys.path:
        sys.path.insert(0, str(args.vdroid_root))

    from android_world.agents.reward_model import LlamaRewardModel

    rows = load_pairs(args.pairs)
    if args.limit > 0:
        rows = rows[: args.limit]

    prepend_begin_token = not args.no_prepend_begin_token
    model = LlamaRewardModel(
        args.base_model,
        lora_path=args.lora_path,
        reward_type="score",
        train_from_scratch=0,
        if_train=False,
        kv_cache=False,
    ).cuda().eval()
    tokenizer = model.tokenizer

    kept_rows: list[dict[str, Any]] = []
    skipped_truncated = 0
    for row in rows:
        chosen = prepare_prompt(row["chosen"], prepend_begin_token)
        rejected = prepare_prompt(row["rejected"], prepend_begin_token)
        if not truncated_valid(tokenizer, chosen, rejected, args.max_length):
            skipped_truncated += 1
            continue
        kept = dict(row)
        kept["_chosen_prompt_for_scoring"] = chosen
        kept["_rejected_prompt_for_scoring"] = rejected
        kept_rows.append(kept)

    scored: list[dict[str, Any]] = []
    diffs: list[float] = []

    with torch.no_grad():
        for start in range(0, len(kept_rows), args.batch_size):
            batch = kept_rows[start : start + args.batch_size]
            chosen_scores = (
                model.get_next_token_score_from_prompt(
                    [row["_chosen_prompt_for_scoring"] for row in batch]
                )
                .view(-1)
                .float()
                .cpu()
            )
            rejected_scores = (
                model.get_next_token_score_from_prompt(
                    [row["_rejected_prompt_for_scoring"] for row in batch]
                )
                .view(-1)
                .float()
                .cpu()
            )
            for row, chosen_score, rejected_score in zip(batch, chosen_scores, rejected_scores):
                diff = float(chosen_score - rejected_score)
                diffs.append(diff)
                chosen_action = extract_action(row, "chosen")
                rejected_action = extract_action(row, "rejected")
                output_row = {
                    "id": row.get("id"),
                    "chosen_score": float(chosen_score),
                    "rejected_score": float(rejected_score),
                    "margin": diff,
                    "chosen_wins": diff > 0,
                    "metadata": row.get("metadata", {}),
                    "chosen_action": chosen_action,
                    "rejected_action": rejected_action,
                    "critic_summary": row.get("critic_summary"),
                }
                scored.append(output_row)
            print(f"done {min(start + args.batch_size, len(kept_rows))} / {len(kept_rows)}", flush=True)

    wins = sum(diff > 0 for diff in diffs)
    losses = [row for row in scored if not row["chosen_wins"]]
    rejected_action_types = Counter(
        (row.get("rejected_action") or {}).get("action_type", "unknown") for row in scored
    )
    loss_rejected_action_types = Counter(
        (row.get("rejected_action") or {}).get("action_type", "unknown") for row in losses
    )
    rejected_back_losses = loss_rejected_action_types.get("navigate_back", 0)
    summary = {
        "input_pairs": str(args.pairs),
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "evaluated": len(diffs),
        "skipped_truncated_or_identical": skipped_truncated,
        "accuracy_chosen_gt_rejected": wins / len(diffs) if diffs else None,
        "chosen_not_preferred_or_tied_count": len(losses),
        "rejected_navigate_back_preferred_or_tied_count": rejected_back_losses,
        "rejected_action_types": dict(rejected_action_types),
        "wrong_or_tie_rejected_action_types": dict(loss_rejected_action_types),
        "mean_margin": sum(diffs) / len(diffs) if diffs else None,
        "min_margin": min(diffs) if diffs else None,
        "p10_margin": quantile(diffs, 0.10),
        "p50_margin": quantile(diffs, 0.50),
        "p90_margin": quantile(diffs, 0.90),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "scores.jsonl", scored)
    write_jsonl(args.output_dir / "wrong_or_tie_cases.jsonl", losses)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
