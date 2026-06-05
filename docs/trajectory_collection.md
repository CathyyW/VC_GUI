# AndroidWorld 原始轨迹收集

与 **eval（`run_suite.py` / `main.sh`）完全相同的执行路径**：`n_iters=1`、`_simulate` 贪心连走、步数上限 `max_n_steps + 10`（由 `10 × task.complexity + 10` 决定）。

收集模式 **只多写** 每步的轨迹 JSON（`action_space`、`scores`、前后 UI 等），不改变选动作/执行逻辑。

## 每步额外记录字段

| 字段 | 说明 |
|------|------|
| `task_id` / `goal` / `step_id` | 任务与步序号 |
| `history` | 此前步骤 summary 列表 |
| `before_ui_html` / `after_ui_html` | 执行前后 accessibility HTML |
| `action_space` | 候选动作列表 |
| `scores` | verifier 对每个候选的分数 |
| `selected_action` | 实际执行的动作 |
| `screenshot` / `screenshot_ann` | 与 eval 相同文件名，见下 |
| `human_label` | 自动启发式（`human_corrected: false`），可后续人工改 |

Episode 级：`task_success`、`instance_id`、`seed`、`collected_at`。

## 运行

```bash
./collect_trajectories.sh          # 默认 task 1-70
./collect_trajectories.sh 1-40
```

参数与 `main.sh` 对齐：`--iteration=1`，不设单独的 `explore_step_limit`。

## 输出

```
collected_trajectories/<run>/
  manifest.jsonl
  trajectories/<Task>_inst0_seed30.json

saved/VDroid_collect_*/record/<TaskName>/screen_shot/
  iter_1_step1.jpg          # 与 eval 相同命名
  iter_1_step1_ann.jpg
  iter_1_step2.jpg
  ...
```

轨迹 JSON 里的 `screenshot` 字段指向上述 eval 风格文件名。

## 与 eval 的差异（仅数据）

| | eval | 收集 |
|--|------|------|
| 执行 | `n_iters=1` + `_simulate` | **相同** |
| 步数上限 | `episode_runner`: `max_n_steps+10` | **相同** |
| 截图 | `iter_1_step*.jpg` | **相同** |
| 额外输出 | checkpoint / pkl | `collected_trajectories/*.json` |

## 后续（未实现）

- 离线转 P³ `chosen/rejected`
- Critic teacher 蒸馏
