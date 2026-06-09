import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from train_critic_lora import build_prompt, read_jsonl


FIELDS = ["outcome", "failure_type", "suggested_recovery"]
OUTPUT_FIELDS = FIELDS + ["summary_to_history"]
ALLOWED_VALUES = {
    "outcome": {"progress", "failure", "done", "no_effect"},
    "failure_type": {
        "none",
        "wrong_page",
        "wrong_target",
        "text_error",
        "popup_blocking",
        "premature_complete",
        "no_effect",
    },
    "suggested_recovery": {
        "continue",
        "navigate_back",
        "clear_text",
        "close_dialog",
        "retry",
        "complete_task",
    },
}


def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def percentage(value: int, total: int) -> float:
    return round(100 * value / total, 2) if total else 0.0


def critic_bucket(row: dict[str, Any]) -> str:
    action = row["input"].get("action", "")
    try:
        action_type = json.loads(action).get("action_type", "")
    except (json.JSONDecodeError, AttributeError):
        action_type = ""
    if action_type in {"navigate_back", "clear_text", "close_dialog"}:
        return "self_correction"
    gold = row["output"]
    if gold.get("outcome") in {"failure", "no_effect"} or gold.get("failure_type") != "none":
        return "failure"
    return "success"


def main():
    parser = argparse.ArgumentParser(description="Evaluate critic predictions on a held-out JSONL split.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = args.adapter or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = read_jsonl(Path(args.data_file))
    if args.limit is not None:
        rows = rows[: args.limit]

    valid_json = 0
    valid_schema = 0
    exact_three_fields = 0
    exact_four_fields = 0
    summary_prefix = 0
    field_correct = Counter()
    per_gold = {field: defaultdict(Counter) for field in FIELDS}
    per_bucket = defaultdict(Counter)
    predictions = []

    for index, row in enumerate(rows, 1):
        prompt = build_prompt(row)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        prediction = extract_json(text)
        gold = row["output"]
        bucket = critic_bucket(row)
        per_bucket[bucket]["total"] += 1

        if prediction is not None:
            valid_json += 1
            per_bucket[bucket]["valid_json"] += 1
            if (
                set(prediction) == set(OUTPUT_FIELDS)
                and all(prediction.get(field) in allowed for field, allowed in ALLOWED_VALUES.items())
                and isinstance(prediction.get("summary_to_history"), str)
            ):
                valid_schema += 1
                per_bucket[bucket]["valid_schema"] += 1
            for field in OUTPUT_FIELDS:
                if prediction.get(field) == gold.get(field):
                    field_correct[field] += 1
            if all(prediction.get(field) == gold.get(field) for field in FIELDS):
                exact_three_fields += 1
                per_bucket[bucket]["exact_three_fields"] += 1
            if all(prediction.get(field) == gold.get(field) for field in OUTPUT_FIELDS):
                exact_four_fields += 1
            if str(prediction.get("summary_to_history", "")).startswith("[Critic]"):
                summary_prefix += 1

        for field in FIELDS:
            predicted_value = prediction.get(field, "<invalid_json>") if prediction else "<invalid_json>"
            per_gold[field][str(gold.get(field))][str(predicted_value)] += 1

        predictions.append(
            {
                "id": row.get("id"),
                "gold": gold,
                "prediction": prediction,
                "raw_prediction": text,
                "bucket": bucket,
                "three_fields_correct": bool(
                    prediction and all(prediction.get(field) == gold.get(field) for field in FIELDS)
                ),
            }
        )
        print(f"[{index}/{len(rows)}] {row.get('id')} valid_json={prediction is not None}")

    total = len(rows)
    metrics = {
        "model": args.model,
        "adapter": args.adapter,
        "data_file": args.data_file,
        "total": total,
        "valid_json": {"count": valid_json, "percent": percentage(valid_json, total)},
        "valid_schema": {"count": valid_schema, "percent": percentage(valid_schema, total)},
        "exact_three_fields": {
            "count": exact_three_fields,
            "percent": percentage(exact_three_fields, total),
        },
        "exact_four_fields_including_summary_text": {
            "count": exact_four_fields,
            "percent": percentage(exact_four_fields, total),
        },
        "summary_critic_prefix": {
            "count": summary_prefix,
            "percent": percentage(summary_prefix, total),
        },
        "field_accuracy": {
            field: {"count": field_correct[field], "percent": percentage(field_correct[field], total)}
            for field in OUTPUT_FIELDS
        },
        "confusion_by_gold_value": {
            field: {
                gold: dict(predicted_counts)
                for gold, predicted_counts in sorted(gold_counts.items())
            }
            for field, gold_counts in per_gold.items()
        },
        "bucket_metrics": {
            bucket: {
                "total": counts["total"],
                "valid_json_percent": percentage(counts["valid_json"], counts["total"]),
                "valid_schema_percent": percentage(counts["valid_schema"], counts["total"]),
                "exact_three_fields_percent": percentage(
                    counts["exact_three_fields"],
                    counts["total"],
                ),
            }
            for bucket, counts in sorted(per_bucket.items())
        },
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved evaluation to {output_dir}")


if __name__ == "__main__":
    main()
