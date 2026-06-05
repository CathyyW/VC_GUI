import argparse
import json
import os
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


Outcome = Literal["progress", "failure", "done", "no_effect"]
FailureType = Literal[
    "none",
    "wrong_page",
    "wrong_target",
    "text_error",
    "popup_blocking",
    "premature_complete",
    "no_effect",
]
Recovery = Literal["continue", "navigate_back", "clear_text", "close_dialog", "retry", "complete_task"]


class CriticLabel(BaseModel):
    outcome: Outcome
    failure_type: FailureType
    suggested_recovery: Recovery
    summary_to_history: str = Field(
        description="One short sentence for the next verifier step."
    )


SYSTEM_PROMPT = """You are labeling one transition from an Android GUI automation trajectory.

Your job is to judge whether the selected action made progress toward the goal by comparing before_ui and after_ui.

Important rules:
1. Judge this transition locally. Do not mark every step in a failed trajectory as failure.
2. outcome=progress means the action usefully moves toward the goal, but the whole task is not finished yet.
3. outcome=done means after_ui already satisfies the user goal or the action correctly completed the task.
4. outcome=no_effect means the UI barely changed or the action did not matter.
5. outcome=failure means the action moved away from the goal, chose the wrong target, entered a wrong page, made a text error, blocked progress, or completed too early.
6. If a permission dialog appears and the next right action is to grant permission, the action that opened the dialog is usually progress, not failure.
7. If selected_action is navigate_back and before_ui is effectively the same as after_ui, this is a no-effect loop: use outcome=no_effect, failure_type=no_effect, suggested_recovery=retry, and the summary must tell the verifier to avoid repeating navigate_back.
8. The summary_to_history is for the next verifier step, not for a human report. It must be one concise sentence, no line breaks, preferably under 30 words.
9. The summary_to_history must summarize the first three labels in natural language: what happened, why it matters, and what to do next.
10. Do not add extra fields. Return exactly the four required keys.

Good summary examples:
- "The previous action opened the wrong page; navigate back and choose the target file list instead."
- "The previous action made useful progress; continue from the current screen."
- "The previous navigate_back did not change the UI; avoid repeating it and retry with a task-relevant action."
"""


def compact_text(text: str, limit: int) -> str:
    text = "\n".join(line.strip() for line in (text or "").splitlines() if line.strip())
    if limit <= 0 or len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return text[:half] + "\n...[TRUNCATED]...\n" + text[-half:]


def build_user_prompt(record: dict, max_ui_chars: int) -> str:
    history = record.get("history") or []
    history_text = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(history)) or "(empty)"
    selected_action = json.dumps(record.get("selected_action"), ensure_ascii=False)
    metadata = {
        "task_id": record.get("task_id"),
        "step_id": record.get("step_id"),
        "num_steps": record.get("num_steps"),
        "is_last_step": record.get("is_last_step"),
        "trajectory_success": record.get("trajectory_success"),
        "agent_indicated_done": record.get("agent_indicated_done"),
    }
    return f"""Metadata:
{json.dumps(metadata, ensure_ascii=False)}

Goal:
{record.get("goal")}

Recent history:
{history_text}

Before UI HTML:
{compact_text(record.get("before_ui_html", ""), max_ui_chars)}

Selected action:
{selected_action}

After UI HTML:
{compact_text(record.get("after_ui_html", ""), max_ui_chars)}
"""


def build_json_mode_prompt(record: dict, max_ui_chars: int) -> str:
    return (
        build_user_prompt(record, max_ui_chars)
        + """

Return only one valid JSON object with exactly these keys:
{
  "outcome": "progress | failure | done | no_effect",
  "failure_type": "none | wrong_page | wrong_target | text_error | popup_blocking | premature_complete | no_effect",
  "suggested_recovery": "continue | navigate_back | clear_text | close_dialog | retry | complete_task",
  "summary_to_history": "one concise sentence for the next verifier step"
}
"""
    )


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_label(label: dict) -> dict:
    label["summary_to_history"] = " ".join((label.get("summary_to_history") or "").split())
    return label


def load_done_ids(output: Path):
    if not output.exists():
        return set()
    done = set()
    for row in read_jsonl(output):
        done.add(row["id"])
    return done


def main():
    parser = argparse.ArgumentParser(description="Annotate transitions with GPT-4o critic labels.")
    parser.add_argument("--input", required=True, help="Transition JSONL from flatten_transitions.py.")
    parser.add_argument("--output", required=True, help="Output labeled JSONL.")
    parser.add_argument("--model", default="gpt-4o-2024-08-06", help="OpenAI model name.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible API base URL, for example https://xxx.example.com/v1. "
        "Can also be set with OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Use normal JSON mode instead of Structured Outputs. Use this if your proxy does not support parse/json_schema.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only annotate this many new rows. 0 means all.")
    parser.add_argument("--max-ui-chars", type=int, default=20000, help="Max chars for each before/after UI.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between requests.")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Set it first, then rerun.")

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The openai package is not installed. Run: python -m pip install -r labeling\\requirements.txt"
        ) from exc

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(output_path)
    client_kwargs = {}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)
    wrote = 0

    with output_path.open("a", encoding="utf-8") as out:
        for record in read_jsonl(input_path):
            if record["id"] in done_ids:
                continue
            if args.limit and wrote >= args.limit:
                break

            user_prompt = build_user_prompt(record, args.max_ui_chars)
            for attempt in range(3):
                try:
                    if args.json_mode:
                        completion = client.chat.completions.create(
                            model=args.model,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": build_json_mode_prompt(record, args.max_ui_chars)},
                            ],
                            temperature=0,
                            response_format={"type": "json_object"},
                        )
                        content = completion.choices[0].message.content
                        label = CriticLabel.model_validate_json(content).model_dump()
                    else:
                        completion = client.chat.completions.parse(
                            model=args.model,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0,
                            response_format=CriticLabel,
                        )
                        parsed = completion.choices[0].message.parsed
                        label = parsed.model_dump()
                    label = normalize_label(label)
                    break
                except Exception as exc:
                    if "invalid_api_key" in str(exc) or "Error code: 401" in str(exc):
                        raise SystemExit(
                            "OpenAI API key is invalid. Please create/copy a valid key, set OPENAI_API_KEY again, "
                            "then rerun this command."
                        ) from exc
                    if attempt == 2:
                        raise
                    wait = 2 ** attempt
                    print(f"Retrying {record['id']} after error: {exc}. Waiting {wait}s.")
                    time.sleep(wait)

            row = {
                "id": record["id"],
                "trajectory_id": record.get("trajectory_id"),
                "task_id": record.get("task_id"),
                "step_id": record.get("step_id"),
                "model": args.model,
                "label": label,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            wrote += 1
            print(f"[{wrote}] {record['id']} -> {label['outcome']} / {label['failure_type']}")
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Annotated {wrote} new transitions. Output: {output_path}")


if __name__ == "__main__":
    main()
