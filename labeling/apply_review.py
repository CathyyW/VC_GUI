import argparse
import csv
import json
from pathlib import Path


ALLOWED = {
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


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Apply manual CSV corrections to critic label JSONL.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    corrections = {}
    with Path(args.review_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = row.get("id")
            if not cid:
                continue
            update = {}
            mapping = {
                "corrected_outcome": "outcome",
                "corrected_failure_type": "failure_type",
                "corrected_suggested_recovery": "suggested_recovery",
                "corrected_summary_to_history": "summary_to_history",
            }
            for source, target in mapping.items():
                value = (row.get(source) or "").strip()
                if value:
                    if target in ALLOWED and value not in ALLOWED[target]:
                        allowed = ", ".join(sorted(ALLOWED[target]))
                        raise SystemExit(
                            f"Invalid correction in {cid}: {source}={value!r}. "
                            f"Allowed values for {target}: {allowed}"
                        )
                    update[target] = value
            note = (row.get("reviewer_note") or "").strip()
            if note:
                update["reviewer_note"] = note
            if update:
                corrections[cid] = update

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    applied = 0
    with output.open("w", encoding="utf-8") as out:
        for row in read_jsonl(Path(args.labels)):
            update = corrections.get(row["id"])
            if update:
                row["label"].update({k: v for k, v in update.items() if k != "reviewer_note"})
                row["human_reviewed"] = True
                if "reviewer_note" in update:
                    row["reviewer_note"] = update["reviewer_note"]
                applied += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Applied corrections: {applied}")
    print(f"Wrote reviewed labels: {output}")


if __name__ == "__main__":
    main()
