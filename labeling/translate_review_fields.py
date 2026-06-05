import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import List

from pydantic import BaseModel


class ReviewTranslation(BaseModel):
    goal_zh: str
    history_zh: List[str]
    selected_action_zh: str
    summary_to_history_zh: str
    quick_judgement_zh: str


SYSTEM_PROMPT = """You translate Android GUI automation review records into concise Chinese for human review.

Translate only the non-UI fields. Do not translate raw UI HTML trees.
Keep action_type, index numbers, file names, app names, and UI labels recognizable.
Use plain Chinese. Be concise.
Return strict JSON only.
"""


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_review_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_done_ids(path: Path):
    if not path.exists():
        return set()
    return {row["id"] for row in read_jsonl(path)}


def build_prompt(review_row: dict, transition: dict) -> str:
    payload = {
        "id": review_row.get("id"),
        "task_id": review_row.get("task_id"),
        "goal": transition.get("goal"),
        "history": transition.get("history") or [],
        "selected_action": transition.get("selected_action"),
        "ai_label": {
            "outcome": review_row.get("outcome"),
            "failure_type": review_row.get("failure_type"),
            "suggested_recovery": review_row.get("suggested_recovery"),
            "summary_to_history": review_row.get("summary_to_history"),
        },
        "review_reason": review_row.get("review_reason"),
    }
    return f"""Translate this review record into Chinese.

Fields to produce:
- goal_zh: Chinese translation of goal.
- history_zh: Chinese translations of each history item, same order and count.
- selected_action_zh: Explain the selected action in Chinese, preserving action_type and index/app/text values.
- summary_to_history_zh: Chinese translation of summary_to_history.
- quick_judgement_zh: One concise Chinese sentence explaining what the AI label means for this step.

Record:
{json.dumps(payload, ensure_ascii=False)}
"""


def main():
    parser = argparse.ArgumentParser(description="Translate review-page non-UI fields into Chinese.")
    parser.add_argument("--transitions", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output", required=True, help="Output JSONL translations.")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible API base URL, can also be set with OPENAI_BASE_URL.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Set it first, then rerun.")

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The openai package is not installed. Run: python -m pip install -r labeling\\requirements.txt"
        ) from exc

    transitions = {row["id"]: row for row in read_jsonl(Path(args.transitions))}
    review_rows = load_review_rows(Path(args.review_csv))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output)

    client_kwargs = {}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    wrote = 0
    with output.open("a", encoding="utf-8") as out:
        for row in review_rows:
            rid = row["id"]
            if rid in done_ids:
                continue
            if args.limit and wrote >= args.limit:
                break
            transition = transitions.get(rid)
            if not transition:
                continue

            prompt = build_prompt(row, transition)
            for attempt in range(3):
                try:
                    completion = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                        response_format={"type": "json_object"},
                    )
                    content = completion.choices[0].message.content
                    translation = ReviewTranslation.model_validate_json(content).model_dump()
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise
                    wait = 2 ** attempt
                    print(f"Retrying {rid} after error: {exc}. Waiting {wait}s.")
                    time.sleep(wait)

            out.write(json.dumps({"id": rid, "translation": translation}, ensure_ascii=False) + "\n")
            out.flush()
            wrote += 1
            print(f"[{wrote}] translated {rid}")
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Wrote {wrote} new translations to {output}")


if __name__ == "__main__":
    main()
