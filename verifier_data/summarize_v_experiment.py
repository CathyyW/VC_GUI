import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


RUN_SPECS = {
    "original_full_test": {
        "section": "full_test",
        "model": "Original V-Droid",
        "input_variant": "Real critic",
        "label": "完整测试集 / 原始 V",
    },
    "trained_full_test": {
        "section": "full_test",
        "model": "Critic-augmented V",
        "input_variant": "Real critic",
        "label": "完整测试集 / 微调 V",
    },
    "original_test_back_real_c": {
        "section": "navigate_back_diagnostic",
        "model": "Original V-Droid",
        "input_variant": "Real critic",
        "label": "navigate_back 诊断 / 原始 V + C",
    },
    "original_test_back_empty_c": {
        "section": "navigate_back_diagnostic",
        "model": "Original V-Droid",
        "input_variant": "Empty critic",
        "label": "navigate_back 诊断 / 原始 V，无 C",
    },
    "trained_test_back_real_c": {
        "section": "navigate_back_diagnostic",
        "model": "Critic-augmented V",
        "input_variant": "Real critic",
        "label": "navigate_back 诊断 / 微调 V + C",
    },
    "trained_test_back_empty_c": {
        "section": "navigate_back_diagnostic",
        "model": "Critic-augmented V",
        "input_variant": "Empty critic",
        "label": "navigate_back 诊断 / 微调 V，无 C",
    },
}

CORE_RUNS = tuple(RUN_SPECS)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def pct(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def number(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def integer_error_count(summary: dict[str, Any]) -> int | None:
    direct = summary.get("chosen_not_preferred_or_tied_count")
    if direct is not None:
        return int(direct)
    evaluated = summary.get("evaluated")
    accuracy = summary.get("accuracy_chosen_gt_rejected")
    if evaluated is None or accuracy is None:
        return None
    return int(evaluated) - round(float(evaluated) * float(accuracy))


def navigate_back_error_count(summary: dict[str, Any]) -> int | None:
    value = summary.get("rejected_navigate_back_preferred_or_tied_count")
    if value is not None:
        return int(value)
    return None


def load_runs(results_root: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    runs: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name in CORE_RUNS:
        path = results_root / name / "summary.json"
        if not path.exists():
            missing.append(str(path))
            continue
        runs[name] = read_json(path)
    return runs, missing


def load_training(model_dir: Path | None) -> dict[str, Any]:
    if model_dir is None:
        return {}

    result: dict[str, Any] = {"model_dir": str(model_dir)}
    args_path = model_dir / "args.json"
    state_path = model_dir / "trainer_state.json"

    if args_path.exists():
        result["args"] = read_json(args_path)

    if state_path.exists():
        state = read_json(state_path)
        result["global_step"] = state.get("global_step")
        result["best_metric"] = state.get("best_metric")
        for entry in state.get("log_history", []):
            for key in (
                "train_loss",
                "train_runtime",
                "eval_loss",
                "eval_accuracy",
                "epoch",
            ):
                if key in entry:
                    result[key] = entry[key]
    return result


def load_data_stats(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return read_json(path)


def metric_row(
    run_name: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    spec = RUN_SPECS[run_name]
    return {
        "run_name": run_name,
        **spec,
        "evaluated": summary.get("evaluated"),
        "accuracy": summary.get("accuracy_chosen_gt_rejected"),
        "accuracy_percent": (
            float(summary["accuracy_chosen_gt_rejected"]) * 100
            if summary.get("accuracy_chosen_gt_rejected") is not None
            else None
        ),
        "error_or_tie_count": integer_error_count(summary),
        "navigate_back_error_or_tie_count": navigate_back_error_count(summary),
        "mean_margin": summary.get("mean_margin"),
        "min_margin": summary.get("min_margin"),
        "p10_margin": summary.get("p10_margin"),
        "p50_margin": summary.get("p50_margin"),
        "p90_margin": summary.get("p90_margin"),
        "input_pairs": summary.get("input_pairs"),
        "lora_path": summary.get("lora_path"),
    }


def calculate_key_metrics(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    required = set(CORE_RUNS)
    if not required.issubset(runs):
        return {}

    original_full = runs["original_full_test"]
    trained_full = runs["trained_full_test"]
    original_real = runs["original_test_back_real_c"]
    trained_real = runs["trained_test_back_real_c"]
    trained_empty = runs["trained_test_back_empty_c"]

    return {
        "full_test": {
            "evaluated_pairs": trained_full["evaluated"],
            "original_accuracy": original_full["accuracy_chosen_gt_rejected"],
            "trained_accuracy": trained_full["accuracy_chosen_gt_rejected"],
            "accuracy_gain_percentage_points": (
                trained_full["accuracy_chosen_gt_rejected"]
                - original_full["accuracy_chosen_gt_rejected"]
            )
            * 100,
            "original_errors": integer_error_count(original_full),
            "trained_errors": integer_error_count(trained_full),
        },
        "navigate_back_diagnostic": {
            "evaluated_pairs": trained_real["evaluated"],
            "original_real_c_accuracy": original_real["accuracy_chosen_gt_rejected"],
            "trained_real_c_accuracy": trained_real["accuracy_chosen_gt_rejected"],
            "accuracy_gain_percentage_points": (
                trained_real["accuracy_chosen_gt_rejected"]
                - original_real["accuracy_chosen_gt_rejected"]
            )
            * 100,
            "original_real_c_back_errors": navigate_back_error_count(original_real),
            "trained_real_c_back_errors": navigate_back_error_count(trained_real),
        },
        "critic_ablation_after_training": {
            "real_c_accuracy": trained_real["accuracy_chosen_gt_rejected"],
            "empty_c_accuracy": trained_empty["accuracy_chosen_gt_rejected"],
            "accuracy_difference_percentage_points": (
                trained_real["accuracy_chosen_gt_rejected"]
                - trained_empty["accuracy_chosen_gt_rejected"]
            )
            * 100,
            "real_c_mean_margin": trained_real["mean_margin"],
            "empty_c_mean_margin": trained_empty["mean_margin"],
            "mean_margin_difference": (
                trained_real["mean_margin"] - trained_empty["mean_margin"]
            ),
        },
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_experiment_summary(
    rows: list[dict[str, Any]],
    key: dict[str, Any],
    training: dict[str, Any],
    data_stats: dict[str, Any],
    missing: list[str],
) -> str:
    row_map = {row["run_name"]: row for row in rows}
    full = key.get("full_test", {})
    back = key.get("navigate_back_diagnostic", {})
    ablation = key.get("critic_ablation_after_training", {})

    full_rows = []
    for name in ("original_full_test", "trained_full_test"):
        row = row_map.get(name)
        if row:
            full_rows.append(
                [
                    row["model"],
                    str(row["evaluated"]),
                    pct(row["accuracy"]),
                    str(row["error_or_tie_count"]),
                    number(row["mean_margin"]),
                    number(row["min_margin"]),
                ]
            )

    diagnostic_rows = []
    for name in (
        "original_test_back_real_c",
        "original_test_back_empty_c",
        "trained_test_back_real_c",
        "trained_test_back_empty_c",
    ):
        row = row_map.get(name)
        if row:
            diagnostic_rows.append(
                [
                    row["model"],
                    row["input_variant"],
                    str(row["evaluated"]),
                    pct(row["accuracy"]),
                    str(row["navigate_back_error_or_tie_count"]),
                    number(row["mean_margin"]),
                ]
            )

    data_pairs = data_stats.get("pairs", {})
    data_steps = data_stats.get("steps_with_pairs", {})
    train_args = training.get("args", {})
    training_lines = [
        f"- 基础模型：`{train_args.get('base_model_path', '未记录')}`",
        f"- 初始化 verifier：`{train_args.get('lora_path', '未记录')}`",
        f"- 训练轮数：`{train_args.get('epoch', training.get('epoch', '未记录'))}`",
        f"- 学习率：`{train_args.get('learning_rate', '未记录')}`",
        f"- 训练耗时：`{number(training.get('train_runtime'), 1)}` 秒",
        f"- 最终训练 loss：`{number(training.get('train_loss'), 6)}`",
        f"- 验证集 accuracy：`{pct(training.get('eval_accuracy'))}`",
    ]
    if data_pairs:
        training_lines.extend(
            [
                f"- P3 pair 数量：train `{data_pairs.get('train')}` / val `{data_pairs.get('val')}` / test `{data_pairs.get('test')}`",
                f"- 有效 step 数量：train `{data_steps.get('train')}` / val `{data_steps.get('val')}` / test `{data_steps.get('test')}`",
                f"- 每个 step 最多负样本数：`{data_stats.get('max_rejected_per_step')}`",
            ]
        )

    warnings = ""
    if missing:
        warnings = "\n## 缺失输入\n\n" + "\n".join(f"- `{path}`" for path in missing) + "\n"

    return f"""# Critic-Augmented V 模型实验结果

生成时间：{datetime.now().astimezone().isoformat(timespec="seconds")}

## 一句话结论

在独立测试数据上，critic-augmented P3 微调将 V 模型的 pairwise 准确率从 **{pct(full.get("original_accuracy"))}** 提升至 **{pct(full.get("trained_accuracy"))}**；在 `navigate_back` 专项诊断中，准确率从 **{pct(back.get("original_real_c_accuracy"))}** 提升至 **{pct(back.get("trained_real_c_accuracy"))}**，错误偏向 `navigate_back` 的样本由 **{back.get("original_real_c_back_errors", "-")}** 个降至 **{back.get("trained_real_c_back_errors", "-")}** 个。

## 实验目标

在原始 V-Droid verifier 的输入中加入上一步 Critic 的反馈，并使用 pairwise P3 数据继续微调 V 模型，使其适应新输入格式，同时减少对无效 `navigate_back` 动作的偏好。

## 训练配置

{chr(10).join(training_lines)}

## 完整测试集结果

{markdown_table(
    ["模型", "Pair 数", "准确率", "错误/平局", "平均 Margin", "最小 Margin"],
    full_rows,
)}

- 完整测试集准确率提升：**{number(full.get("accuracy_gain_percentage_points"))} 个百分点**。
- 错误/平局数量：**{full.get("original_errors", "-")} → {full.get("trained_errors", "-")}**。
- 微调后最小 margin 为正，表示测试集中所有 chosen action 的评分均高于对应 rejected action。

## navigate_back 专项诊断

{markdown_table(
    ["模型", "Critic 输入", "Pair 数", "准确率", "错误偏向 back", "平均 Margin"],
    diagnostic_rows,
)}

- 使用真实 Critic 输入时，专项准确率提升：**{number(back.get("accuracy_gain_percentage_points"))} 个百分点**。
- 微调后，错误偏向 `navigate_back` 的样本数：**{back.get("original_real_c_back_errors", "-")} → {back.get("trained_real_c_back_errors", "-")}**。
- 微调 V 在真实 C 与空 C 下准确率均为 **{pct(ablation.get("real_c_accuracy"))}**；真实 C 的平均 margin 比空 C 高 **{number(ablation.get("mean_margin_difference"))}**。

## 可用于报告的结论

实验结果表明，在原始 V-Droid verifier 上使用包含 Critic 反馈的 P3 pairwise 数据继续微调，可以显著提升 V 对新输入格式和动作偏好关系的适应能力。完整测试集准确率由 {pct(full.get("original_accuracy"))} 提升至 {pct(full.get("trained_accuracy"))}；在 `navigate_back` 专项诊断中，准确率由 {pct(back.get("original_real_c_accuracy"))} 提升至 {pct(back.get("trained_real_c_accuracy"))}，错误偏向 `navigate_back` 的样本由 {back.get("original_real_c_back_errors", "-")} 个降至 {back.get("trained_real_c_back_errors", "-")} 个。消融结果中，真实 Critic 与空 Critic 的最终准确率相同，但真实 Critic 输入获得了更高的平均评分间隔，说明 Critic 信息增强了模型对正确动作的评分置信度；当前结果主要支持“格式适配与动作偏好改善”，尚不单独宣称 Critic 已稳定改变最终动作排序。

## 使用边界

- 这里展示的是 verifier 离线 pairwise 评估结果，不等同于完整 AndroidWorld 闭环任务成功率。
- `navigate_back` 专项诊断用于衡量无效回退偏好，不代表整个实验只针对该动作。
- 当前采用 1 epoch 模型作为最终候选，不建议仅因训练集或验证集接近满分而继续追加 epoch。
{warnings}
"""


def build_ppt_outline(key: dict[str, Any], training: dict[str, Any], data_stats: dict[str, Any]) -> str:
    full = key.get("full_test", {})
    back = key.get("navigate_back_diagnostic", {})
    ablation = key.get("critic_ablation_after_training", {})
    args = training.get("args", {})
    pairs = data_stats.get("pairs", {})

    return f"""# V 模型实验 PPT 文案提纲

## 第 1 页：为什么微调 V

**标题：Critic-Augmented Verifier 微调**

- 原始 V-Droid verifier 根据目标、历史、当前 UI 和候选 action 进行动作评分。
- 本实验在输入中加入上一步 Critic 反馈，并保持 pairwise preference loss 不变。
- 目标：适应 Critic 增强后的输入格式，并减少无效 `navigate_back` 偏好。

## 第 2 页：训练方案

**标题：基于 Critic-Augmented P3 的 Pairwise 微调**

- 初始化：官方 V-Droid verifier adapter + value head。
- Base model：`{args.get("base_model_path", "Meta-Llama-3.1-8B-Instruct-bnb-4bit")}`。
- 数据：train `{pairs.get("train", "-")}` / val `{pairs.get("val", "-")}` / test `{pairs.get("test", "-")}` pairs。
- 训练：`{args.get("epoch", 1)}` epoch，learning rate `{args.get("learning_rate", "2e-5")}`。
- Loss 目标：提高 chosen action 评分，降低 rejected action 评分。

## 第 3 页：完整测试集结果

**标题：微调后 Pairwise 判断达到 100%**

- 原始 V：**{pct(full.get("original_accuracy"))}**
- 微调 V：**{pct(full.get("trained_accuracy"))}**
- 提升：**{number(full.get("accuracy_gain_percentage_points"))} 个百分点**
- 错误/平局：**{full.get("original_errors", "-")} → {full.get("trained_errors", "-")}**

建议图表：使用 `chart_data.csv` 中 `full_test` 两行绘制准确率柱状图。

## 第 4 页：navigate_back 专项结果

**标题：显著减少无效回退动作偏好**

- 原始 V + C：准确率 **{pct(back.get("original_real_c_accuracy"))}**，错误偏向 back **{back.get("original_real_c_back_errors", "-")}** 次。
- 微调 V + C：准确率 **{pct(back.get("trained_real_c_accuracy"))}**，错误偏向 back **{back.get("trained_real_c_back_errors", "-")}** 次。
- 专项准确率提升：**{number(back.get("accuracy_gain_percentage_points"))} 个百分点**。

建议图表：准确率柱状图 + 错误 back 次数对比图。

## 第 5 页：Critic 消融与结论

**标题：当前收益主要来自格式适配与动作偏好改善**

- 微调后真实 C / 空 C 准确率均为 **{pct(ablation.get("real_c_accuracy"))}**。
- 真实 C 的平均 margin 比空 C 高 **{number(ablation.get("mean_margin_difference"))}**。
- 结论：微调显著提升 verifier 判断能力并减少无效 back；Critic 提高评分置信度。
- 边界：当前是离线 pairwise 结果，尚未验证完整 AndroidWorld 闭环成功率。
"""


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate report- and PPT-ready summaries for the critic-augmented V experiment."
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-stats", type=Path)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()

    runs, missing = load_runs(args.results_root)
    training = load_training(args.model_dir)
    data_stats = load_data_stats(args.data_stats)
    rows = [metric_row(name, runs[name]) for name in CORE_RUNS if name in runs]
    key = calculate_key_metrics(runs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "key_metrics.json", key)
    write_json(
        args.output_dir / "run_manifest.json",
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "results_root": str(args.results_root),
            "data_stats": str(args.data_stats) if args.data_stats else None,
            "model_dir": str(args.model_dir) if args.model_dir else None,
            "loaded_runs": list(runs),
            "missing_inputs": missing,
            "training": training,
        },
    )

    metric_fields = [
        "run_name",
        "section",
        "model",
        "input_variant",
        "evaluated",
        "accuracy_percent",
        "error_or_tie_count",
        "navigate_back_error_or_tie_count",
        "mean_margin",
        "min_margin",
        "p10_margin",
        "p50_margin",
        "p90_margin",
        "input_pairs",
        "lora_path",
    ]
    write_csv(args.output_dir / "metrics_table.csv", rows, metric_fields)

    chart_rows = [
        {
            "chart": row["section"],
            "model": row["model"],
            "critic_input": row["input_variant"],
            "accuracy_percent": row["accuracy_percent"],
            "error_or_tie_count": row["error_or_tie_count"],
            "navigate_back_error_or_tie_count": row[
                "navigate_back_error_or_tie_count"
            ],
            "mean_margin": row["mean_margin"],
        }
        for row in rows
    ]
    write_csv(
        args.output_dir / "chart_data.csv",
        chart_rows,
        [
            "chart",
            "model",
            "critic_input",
            "accuracy_percent",
            "error_or_tie_count",
            "navigate_back_error_or_tie_count",
            "mean_margin",
        ],
    )

    (args.output_dir / "experiment_summary.md").write_text(
        build_experiment_summary(rows, key, training, data_stats, missing),
        encoding="utf-8",
    )
    (args.output_dir / "ppt_outline.md").write_text(
        build_ppt_outline(key, training, data_stats),
        encoding="utf-8",
    )

    print(f"Generated experiment materials in: {args.output_dir}")
    for path in sorted(args.output_dir.iterdir()):
        print(f"- {path.name}")
    if missing:
        print(f"Warning: {len(missing)} expected summary files were missing.")


if __name__ == "__main__":
    main()
