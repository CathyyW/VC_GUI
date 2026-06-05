import argparse
import csv
from pathlib import Path


CORRECTION_FIELDS = [
    "corrected_outcome",
    "corrected_failure_type",
    "corrected_suggested_recovery",
    "corrected_summary_to_history",
    "reviewer_note",
]


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Sync corrected_* fields from subset review CSVs into a master CSV.")
    parser.add_argument("--target", required=True, help="Master review CSV to update, e.g. review_needed.csv.")
    parser.add_argument("--sources", nargs="+", required=True, help="Subset CSVs containing edits.")
    args = parser.parse_args()

    target_path = Path(args.target)
    target_rows = read_csv(target_path)
    if not target_rows:
        raise SystemExit("Target CSV has no rows.")
    fieldnames = list(target_rows[0].keys())
    by_id = {row["id"]: row for row in target_rows}

    updated = 0
    for source_name in args.sources:
        for source_row in read_csv(Path(source_name)):
            target_row = by_id.get(source_row.get("id"))
            if not target_row:
                continue
            changed = False
            for field in CORRECTION_FIELDS:
                if field in source_row and field in target_row and target_row.get(field, "") != source_row.get(field, ""):
                    target_row[field] = source_row.get(field, "")
                    changed = True
            if changed:
                updated += 1

    write_csv(target_path, target_rows, fieldnames)
    print(f"Synced correction fields for {updated} rows into {target_path}")


if __name__ == "__main__":
    main()
