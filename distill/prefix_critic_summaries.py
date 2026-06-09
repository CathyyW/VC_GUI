import argparse
import json
from pathlib import Path


FILES = [
    "critic_distill_train.jsonl",
    "critic_distill_val.jsonl",
    "critic_distill_test.jsonl",
]


def add_prefix(summary: str) -> str:
    summary = (summary or "").strip()
    if summary.startswith("[Critic]"):
        return summary
    return f"[Critic] {summary}"


def convert_file(src: Path, dst: Path) -> tuple[int, int]:
    total = 0
    changed = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8-sig") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            output = row.get("output") or {}
            old = output.get("summary_to_history", "")
            new = add_prefix(old)
            if new != old:
                changed += 1
            output["summary_to_history"] = new
            row["output"] = output
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
    return total, changed


def main():
    parser = argparse.ArgumentParser(description="Add [Critic] prefix to SFT summary_to_history fields.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    for name in FILES:
        total, changed = convert_file(input_dir / name, output_dir / name)
        print(f"{name}: total={total}, prefixed={changed}")

    stats = input_dir / "split_stats.json"
    if stats.exists():
        (output_dir / "split_stats.json").write_text(stats.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
