import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def preview(text: str, limit: int = 240) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + "..."


def normalize_ui(text: str) -> str:
    return " ".join((text or "").split())


def summary_word_count(label: dict) -> int:
    return len((label.get("summary_to_history") or "").split())


def action_type(transition: dict) -> str:
    action = transition.get("selected_action") or {}
    return action.get("action_type", "")


def same_ui_before_after(transition: dict) -> bool:
    return normalize_ui(transition.get("before_ui_html")) == normalize_ui(transition.get("after_ui_html"))


def review_bucket(transition: dict, label: dict) -> str:
    act_type = action_type(transition)
    same_ui = same_ui_before_after(transition)
    if act_type in {"navigate_back", "clear_text", "close_dialog"} or (act_type == "navigate_back" and same_ui):
        return "self_correction"
    if label.get("outcome") in {"failure", "no_effect"} or label.get("failure_type") != "none":
        return "failure"
    return "success"


def review_decision(transition: dict, label: dict, first_error_ids: set[str]) -> tuple[str, list[str]]:
    reasons = []
    act_type = action_type(transition)
    same_ui = same_ui_before_after(transition)
    failed_trajectory = transition.get("trajectory_success") is False

    if transition["id"] in first_error_ids:
        reasons.append("first_error_candidate")
    if act_type == "navigate_back" and same_ui:
        reasons.append("navigate_back_no_effect")
    if act_type == "status" and not transition.get("trajectory_success"):
        reasons.append("premature_or_wrong_complete")
    if label.get("failure_type") != "none":
        reasons.append("failure_type_not_none")
    if label.get("outcome") in {"failure", "no_effect"}:
        reasons.append(f"outcome_{label.get('outcome')}")
    if label.get("suggested_recovery") in {"navigate_back", "retry", "complete_task"}:
        reasons.append(f"recovery_{label.get('suggested_recovery')}")
    if transition.get("is_last_step") and failed_trajectory:
        reasons.append("failed_trajectory_last_step")
    if summary_word_count(label) > 35:
        reasons.append("summary_too_long")
    if "\n" in (label.get("summary_to_history") or ""):
        reasons.append("summary_has_newline")
    if act_type in {"navigate_back", "navigate_home", "wait", "answer", "clear_text"}:
        reasons.append(f"special_action_{act_type}")

    if any(
        reason in reasons
        for reason in [
            "first_error_candidate",
            "navigate_back_no_effect",
            "premature_or_wrong_complete",
            "failure_type_not_none",
            "outcome_failure",
            "outcome_no_effect",
        ]
    ):
        return "P0", reasons
    if reasons:
        return "P1", reasons
    return "P2", ["spot_check"]


def needs_review(transition: dict, label: dict, first_error_ids: set[str], include_p2: bool) -> bool:
    priority, _ = review_decision(transition, label, first_error_ids)
    return include_p2 or priority in {"P0", "P1"}


def main():
    parser = argparse.ArgumentParser(description="Validate critic labels and create a review CSV.")
    parser.add_argument("--transitions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument(
        "--include-p2",
        action="store_true",
        help="Include ordinary success/progress rows for spot checking. Default only writes P0/P1 rows.",
    )
    args = parser.parse_args()

    transitions = {row["id"]: row for row in read_jsonl(Path(args.transitions))}
    labels = list(read_jsonl(Path(args.labels)))
    labels_by_id = {row["id"]: row for row in labels}
    first_error_ids = set()
    trajectories = {}
    for tid, transition in transitions.items():
        trajectories.setdefault(transition.get("trajectory_id"), []).append(transition)
    for trajectory_id, items in trajectories.items():
        items.sort(key=lambda item: item.get("step_id", 0))
        if not items or items[0].get("trajectory_success") is not False:
            continue
        for transition in items:
            row = labels_by_id.get(transition["id"])
            if not row:
                continue
            label = row["label"]
            if label.get("outcome") in {"failure", "no_effect"} or label.get("failure_type") != "none":
                first_error_ids.add(transition["id"])
                break

    missing = sorted(set(transitions) - {row["id"] for row in labels})
    outcome_counts = Counter(row["label"]["outcome"] for row in labels)
    failure_counts = Counter(row["label"]["failure_type"] for row in labels)
    recovery_counts = Counter(row["label"]["suggested_recovery"] for row in labels)
    bucket_counts = Counter(review_bucket(transitions[row["id"]], row["label"]) for row in labels if row["id"] in transitions)
    priority_counts = Counter(
        review_decision(transitions[row["id"]], row["label"], first_error_ids)[0]
        for row in labels
        if row["id"] in transitions
    )

    print(f"Transitions: {len(transitions)}")
    print(f"Labels: {len(labels)}")
    print(f"Missing labels: {len(missing)}")
    print(f"Review bucket counts: {dict(bucket_counts)}")
    print(f"Review priority counts: {dict(priority_counts)}")
    print(f"Outcome counts: {dict(outcome_counts)}")
    print(f"Failure type counts: {dict(failure_counts)}")
    print(f"Recovery counts: {dict(recovery_counts)}")
    print(f"First-error candidates: {len(first_error_ids)}")

    review_path = Path(args.review_csv)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in labels:
        transition = transitions.get(row["id"])
        if not transition:
            continue
        label = row["label"]
        if not needs_review(transition, label, first_error_ids, args.include_p2):
            continue
        priority, reasons = review_decision(transition, label, first_error_ids)
        rows.append(
            {
                "id": row["id"],
                "trajectory_id": transition.get("trajectory_id"),
                "task_id": row.get("task_id"),
                "step_id": row.get("step_id"),
                "goal": transition.get("goal"),
                "selected_action": json.dumps(transition.get("selected_action"), ensure_ascii=False),
                "action_type": action_type(transition),
                "trajectory_success": transition.get("trajectory_success"),
                "is_last_step": transition.get("is_last_step"),
                "same_ui_before_after": same_ui_before_after(transition),
                "review_priority": priority,
                "review_reason": ";".join(reasons),
                "review_bucket": review_bucket(transition, label),
                "first_error_candidate": row["id"] in first_error_ids,
                "outcome": label.get("outcome"),
                "failure_type": label.get("failure_type"),
                "suggested_recovery": label.get("suggested_recovery"),
                "summary_to_history": label.get("summary_to_history"),
                "summary_word_count": summary_word_count(label),
                "before_ui_preview": preview(transition.get("before_ui_html")),
                "after_ui_preview": preview(transition.get("after_ui_html")),
                "corrected_outcome": "",
                "corrected_failure_type": "",
                "corrected_suggested_recovery": "",
                "corrected_summary_to_history": "",
                "reviewer_note": "",
            }
        )

    with review_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Review candidates: {len(rows)}")
    print(f"Review CSV: {review_path}")


if __name__ == "__main__":
    main()
