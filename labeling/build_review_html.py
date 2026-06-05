import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def compact(text: str, limit: int = 7000) -> str:
    text = "\n".join(line.strip() for line in (text or "").splitlines() if line.strip())
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n\n...[TRUNCATED]...\n\n" + text[-half:]


def esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


OUTCOME_ZH = {
    "progress": "有进展",
    "failure": "出错",
    "done": "已完成",
    "no_effect": "无效果",
}

FAILURE_ZH = {
    "none": "没有错误",
    "wrong_page": "进错页面/应用",
    "wrong_target": "点错目标",
    "text_error": "文本错误",
    "popup_blocking": "弹窗阻挡",
    "premature_complete": "过早完成",
    "no_effect": "无有效变化",
}

RECOVERY_ZH = {
    "continue": "继续",
    "navigate_back": "返回上一页",
    "clear_text": "清空文本",
    "close_dialog": "关闭弹窗",
    "retry": "换个相关动作重试",
    "complete_task": "结束任务",
}

BUCKET_ZH = {
    "success": "正常进展",
    "failure": "失败/错误",
    "self_correction": "自我纠错",
}

REASON_ZH = {
    "first_error_candidate": "失败轨迹里疑似第一处错误",
    "navigate_back_no_effect": "navigate_back 后界面没变化",
    "premature_or_wrong_complete": "可能过早/错误 complete",
    "failure_type_not_none": "failure_type 不是 none",
    "outcome_failure": "AI 判断这步失败",
    "outcome_no_effect": "AI 判断这步无效果",
    "recovery_retry": "AI 建议重试",
    "recovery_navigate_back": "AI 建议返回",
    "recovery_complete_task": "AI 建议完成任务",
    "failed_trajectory_last_step": "失败轨迹的最后一步",
    "summary_too_long": "summary 太长",
    "summary_has_newline": "summary 有换行",
    "special_action_navigate_back": "特殊动作 navigate_back",
    "special_action_clear_text": "特殊动作 clear_text",
    "special_action_wait": "特殊动作 wait",
    "spot_check": "普通抽查",
}


def zh_value(kind: str, value: str) -> str:
    maps = {
        "outcome": OUTCOME_ZH,
        "failure_type": FAILURE_ZH,
        "suggested_recovery": RECOVERY_ZH,
        "bucket": BUCKET_ZH,
    }
    text = maps.get(kind, {}).get(value, "")
    return f"{value}（{text}）" if text else value


def badge(text: str, kind: str = "") -> str:
    return f"<span class='badge {esc(kind)}'>{esc(text)}</span>"


def load_review_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_translations(path: Path | None):
    if not path or not path.exists():
        return {}
    translations = {}
    for row in read_jsonl(path):
        translations[row["id"]] = row.get("translation", {})
    return translations


def sort_key(row: dict):
    priority_order = {"P0": 0, "P1": 1, "P2": 2}
    first_error = 0 if row.get("first_error_candidate") == "True" else 1
    return (
        priority_order.get(row.get("review_priority"), 9),
        first_error,
        row.get("task_id", ""),
        int(row.get("step_id") or 0),
    )


def correction_hint(row: dict) -> str:
    action_type = row.get("action_type")
    same_ui = row.get("same_ui_before_after") == "True"
    if action_type == "navigate_back" and same_ui:
        return (
            "如果前后界面确实没变，通常改成：outcome=no_effect，"
            "failure_type=no_effect，suggested_recovery=retry，并在 summary 里提醒 V 不要重复 navigate_back。"
        )
    if row.get("first_error_candidate") == "True":
        return "重点检查：这是失败轨迹里疑似第一处出错的 step。"
    if row.get("suggested_recovery") == "complete_task":
        return "确认任务真的已经完成，才能接受 complete_task。"
    return "如果 AI 标注没问题，就不用改 CSV 里的 corrected_* 列。"


def translate_reasons(reason_text: str) -> str:
    reasons = [item for item in (reason_text or "").split(";") if item]
    if not reasons:
        return "无"
    return "；".join(REASON_ZH.get(item, item) for item in reasons)


def translated_block(title: str, zh_text: str, original_title: str, original_text: str, pre: bool = False) -> str:
    if not zh_text:
        content = f"<pre>{esc(original_text)}</pre>" if pre else f"<p>{esc(original_text)}</p>"
        return f"""
<section class="block">
  <h3>{esc(title)}</h3>
  {content}
</section>
"""
    zh_content = f"<pre class='zh'>{esc(zh_text)}</pre>" if pre else f"<p class='zh'>{esc(zh_text)}</p>"
    original_content = f"<pre>{esc(original_text)}</pre>" if pre else f"<p>{esc(original_text)}</p>"
    return f"""
<section class="block">
  <h3>{esc(title)}</h3>
  {zh_content}
  <details>
    <summary>{esc(original_title)}</summary>
    {original_content}
  </details>
</section>
"""


def render_card(index: int, row: dict, transition: dict, translation: dict) -> str:
    label_badges = [
        badge(f"{row.get('review_priority')} {'必看' if row.get('review_priority') == 'P0' else '建议看'}", row.get("review_priority", "").lower()),
        badge(zh_value("bucket", row.get("review_bucket")), row.get("review_bucket")),
        badge(f"第 {row.get('step_id')} 步"),
        badge(row.get("action_type"), "action"),
    ]
    if row.get("first_error_candidate") == "True":
        label_badges.append(badge("疑似首错", "warn"))
    if row.get("same_ui_before_after") == "True":
        label_badges.append(badge("前后界面相同", "warn"))
    if row.get("trajectory_success") == "False":
        label_badges.append(badge("失败轨迹", "fail"))

    history = transition.get("history") or []
    history_html = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(history)) or "(empty)"
    history_zh_items = translation.get("history_zh") or []
    history_zh = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(history_zh_items)) if history_zh_items else ""
    selected_action = json.dumps(transition.get("selected_action"), ensure_ascii=False, indent=2)
    selected_action_zh = translation.get("selected_action_zh", "")

    label_rows = [
        ("outcome", zh_value("outcome", row.get("outcome"))),
        ("failure_type", zh_value("failure_type", row.get("failure_type"))),
        ("suggested_recovery", zh_value("suggested_recovery", row.get("suggested_recovery"))),
        ("summary_to_history", row.get("summary_to_history")),
    ]
    summary_zh = translation.get("summary_to_history_zh", "")
    if summary_zh:
        label_rows.append(("summary 中文", summary_zh))
    label_table = "".join(
        f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in label_rows
    )
    correction_rows = [
        ("corrected_outcome", row.get("corrected_outcome")),
        ("corrected_failure_type", row.get("corrected_failure_type")),
        ("corrected_suggested_recovery", row.get("corrected_suggested_recovery")),
        ("corrected_summary_to_history", row.get("corrected_summary_to_history")),
        ("reviewer_note", row.get("reviewer_note")),
    ]
    filled_corrections = [(k, v) for k, v in correction_rows if v]
    correction_table = ""
    if filled_corrections:
        correction_table = (
            "<div class='correction-box'><h4>已填写的人工修正</h4><table>"
            + "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in filled_corrections)
            + "</table></div>"
        )

    data_attrs = " ".join(
        [
            f"data-priority='{esc(row.get('review_priority'))}'",
            f"data-bucket='{esc(row.get('review_bucket'))}'",
            f"data-task='{esc(row.get('task_id'))}'",
            f"data-text='{esc(' '.join([row.get('id',''), row.get('task_id',''), row.get('review_reason',''), row.get('summary_to_history','')]).lower())}'",
        ]
    )

    return f"""
<article class="card" {data_attrs}>
  <div class="card-head">
    <div>
      <h2>#{index} {esc(row.get('task_id'))}</h2>
      <div class="id">{esc(row.get('id'))}</div>
    </div>
    <div class="badges">{''.join(label_badges)}</div>
  </div>

  {translated_block("任务目标", translation.get("goal_zh", ""), "英文原文 Goal", row.get('goal') or "")}

  <div class="grid two">
    {translated_block("最近历史", history_zh, "英文原文 Recent History", history_html, pre=True)}
    {translated_block("本步动作", selected_action_zh, "动作 JSON 原文", selected_action, pre=True)}
  </div>

  <div class="grid two">
    <section class="block">
      <h3>动作前界面 Before UI</h3>
      <pre>{esc(compact(transition.get('before_ui_html', '')))}</pre>
    </section>
    <section class="block">
      <h3>动作后界面 After UI</h3>
      <pre>{esc(compact(transition.get('after_ui_html', '')))}</pre>
    </section>
  </div>

  <div class="grid two">
    <section class="block">
      <h3>AI 标注结果</h3>
      <table>{label_table}</table>
    </section>
    <section class="block guidance">
      <h3>人工审查提示</h3>
      <p><strong>为什么要看：</strong>{esc(translate_reasons(row.get('review_reason')))}</p>
      <p><strong>怎么判断：</strong>{esc(correction_hint(row))}</p>
      <p><strong>AI 标签中文解释：</strong>{esc(translation.get('quick_judgement_zh', '暂无翻译'))}</p>
      <p>如果 AI 标错，就到 CSV 里找到这个 id，只填写 corrected_* 列。</p>
      {correction_table}
    </section>
  </div>
</article>
"""


def main():
    parser = argparse.ArgumentParser(description="Build a readable HTML review page from review CSV.")
    parser.add_argument("--transitions", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--translations", default="", help="Optional JSONL from translate_review_fields.py.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    transitions = {row["id"]: row for row in read_jsonl(Path(args.transitions))}
    rows = load_review_rows(Path(args.review_csv))
    translations = load_translations(Path(args.translations) if args.translations else None)
    rows.sort(key=sort_key)

    priority_counts = Counter(row.get("review_priority") for row in rows)
    bucket_counts = Counter(row.get("review_bucket") for row in rows)
    task_counts = Counter(row.get("task_id") for row in rows)

    cards = []
    for index, row in enumerate(rows, start=1):
        transition = transitions.get(row["id"])
        if transition:
            cards.append(render_card(index, row, transition, translations.get(row["id"], {})))

    task_options = "\n".join(
        f"<option value='{esc(task)}'>{esc(task)} ({count})</option>"
        for task, count in sorted(task_counts.items())
    )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VC Critic 中文审查页</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --card: #ffffff;
      --ink: #172033;
      --muted: #687084;
      --line: #dce2ea;
      --blue: #2563eb;
      --red: #dc2626;
      --amber: #b45309;
      --green: #15803d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 16px 24px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .intro {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
      max-width: 1100px;
      font-size: 14px;
    }}
    .stats, .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 10px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 10px;
      background: #f9fafb;
      font-size: 13px;
    }}
    input, select {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: white;
      font-size: 14px;
    }}
    input {{ min-width: 320px; }}
    main {{ padding: 20px 24px 48px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 0 0 18px;
      padding: 18px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      border-bottom: 1px solid var(--line);
      padding-bottom: 12px;
      margin-bottom: 12px;
    }}
    h2 {{ margin: 0; font-size: 18px; }}
    h3 {{ margin: 0 0 8px; font-size: 13px; color: var(--muted); text-transform: uppercase; }}
    .id {{ margin-top: 4px; color: var(--muted); font-family: Consolas, monospace; font-size: 12px; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      background: #eef2ff;
      color: #3730a3;
    }}
    .badge.p0, .badge.fail {{ background: #fee2e2; color: var(--red); }}
    .badge.p1, .badge.warn {{ background: #fef3c7; color: var(--amber); }}
    .badge.p2, .badge.success {{ background: #dcfce7; color: var(--green); }}
    .badge.self_correction {{ background: #e0f2fe; color: #0369a1; }}
    .badge.failure {{ background: #fee2e2; color: var(--red); }}
    .block {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      margin-top: 10px;
      background: #fcfdff;
    }}
    .grid {{
      display: grid;
      gap: 12px;
    }}
    .grid.two {{ grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 360px;
      overflow: auto;
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
      line-height: 1.45;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{
      border-top: 1px solid var(--line);
      padding: 8px;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{ width: 170px; color: var(--muted); }}
    .guidance p {{ margin: 0 0 8px; line-height: 1.5; }}
    .correction-box {{
      margin-top: 12px;
      border: 1px solid #86efac;
      background: #f0fdf4;
      border-radius: 6px;
      padding: 10px;
    }}
    .correction-box h4 {{
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--green);
    }}
    .hidden {{ display: none; }}
    @media (max-width: 900px) {{
      .grid.two {{ grid-template-columns: 1fr; }}
      input {{ min-width: 100%; }}
      .card-head {{ flex-direction: column; }}
      .badges {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>VC Critic 中文审查页</h1>
    <p class="intro">看每张卡片时，只问一个问题：这一步动作执行后，after UI 是否比 before UI 更接近任务目标？如果是，就是有进展；如果走偏、没变化、过早完成，就需要修正。</p>
    <div class="stats">
      <span class="stat">总行数: {len(rows)}</span>
      <span class="stat">已有中文翻译: {sum(1 for row in rows if row.get('id') in translations)}</span>
      <span class="stat">P0 必看: {priority_counts.get('P0', 0)}</span>
      <span class="stat">P1 建议看: {priority_counts.get('P1', 0)}</span>
      <span class="stat">正常进展: {bucket_counts.get('success', 0)}</span>
      <span class="stat">失败/错误: {bucket_counts.get('failure', 0)}</span>
      <span class="stat">自我纠错: {bucket_counts.get('self_correction', 0)}</span>
    </div>
    <div class="filters">
      <input id="search" placeholder="搜索 task / id / reason / summary">
      <select id="priority">
        <option value="">全部优先级</option>
        <option value="P0">P0</option>
        <option value="P1">P1</option>
        <option value="P2">P2</option>
      </select>
      <select id="bucket">
        <option value="">全部类型</option>
        <option value="success">正常进展 success</option>
        <option value="failure">失败/错误 failure</option>
        <option value="self_correction">自我纠错 self_correction</option>
      </select>
      <select id="task">
        <option value="">全部任务</option>
        {task_options}
      </select>
      <span id="visibleCount" class="stat"></span>
    </div>
  </header>
  <main id="cards">
    {''.join(cards)}
  </main>
  <script>
    const cards = Array.from(document.querySelectorAll('.card'));
    const search = document.getElementById('search');
    const priority = document.getElementById('priority');
    const bucket = document.getElementById('bucket');
    const task = document.getElementById('task');
    const visibleCount = document.getElementById('visibleCount');
    function applyFilters() {{
      const q = search.value.trim().toLowerCase();
      let count = 0;
      for (const card of cards) {{
        const okSearch = !q || card.dataset.text.includes(q);
        const okPriority = !priority.value || card.dataset.priority === priority.value;
        const okBucket = !bucket.value || card.dataset.bucket === bucket.value;
        const okTask = !task.value || card.dataset.task === task.value;
        const show = okSearch && okPriority && okBucket && okTask;
        card.classList.toggle('hidden', !show);
        if (show) count += 1;
      }}
      visibleCount.textContent = `当前显示: ${{count}}`;
    }}
    [search, priority, bucket, task].forEach(el => el.addEventListener('input', applyFilters));
    applyFilters();
  </script>
</body>
</html>
"""

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    print(f"Wrote review HTML: {output}")


if __name__ == "__main__":
    main()
