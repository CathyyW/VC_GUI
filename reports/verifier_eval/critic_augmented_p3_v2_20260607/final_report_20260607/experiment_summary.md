# Critic-Augmented V 模型实验结果

生成时间：2026-06-07T16:12:29+08:00

## 一句话结论

在独立测试数据上，critic-augmented P3 微调将 V 模型的 pairwise 准确率从 **89.94%** 提升至 **100.00%**；在 `navigate_back` 专项诊断中，准确率从 **68.00%** 提升至 **98.00%**，错误偏向 `navigate_back` 的样本由 **16** 个降至 **1** 个。

## 实验目标

在原始 V-Droid verifier 的输入中加入上一步 Critic 的反馈，并使用 pairwise P3 数据继续微调 V 模型，使其适应新输入格式，同时减少对无效 `navigate_back` 动作的偏好。

## 训练配置

- 基础模型：`/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
- 初始化 verifier：`/root/autodl-tmp/models/V-Droid-8B-0323`
- 训练轮数：`1`
- 学习率：`2e-05`
- 训练耗时：`8696.5` 秒
- 最终训练 loss：`0.083698`
- 验证集 accuracy：`100.00%`
- P3 pair 数量：train `1335` / val `165` / test `159`
- 有效 step 数量：train `445` / val `55` / test `53`
- 每个 step 最多负样本数：`3`

## 完整测试集结果

| 模型 | Pair 数 | 准确率 | 错误/平局 | 平均 Margin | 最小 Margin |
| --- | --- | --- | --- | --- | --- |
| Original V-Droid | 159 | 89.94% | 16 | 44.04 | -27.75 |
| Critic-augmented V | 159 | 100.00% | 0 | 43.11 | 8.76 |

- 完整测试集准确率提升：**10.06 个百分点**。
- 错误/平局数量：**16 → 0**。
- 微调后最小 margin 为正，表示测试集中所有 chosen action 的评分均高于对应 rejected action。

## navigate_back 专项诊断

| 模型 | Critic 输入 | Pair 数 | 准确率 | 错误偏向 back | 平均 Margin |
| --- | --- | --- | --- | --- | --- |
| Original V-Droid | Real critic | 50 | 68.00% | 16 | 21.38 |
| Original V-Droid | Empty critic | 50 | 70.00% | 15 | 20.64 |
| Critic-augmented V | Real critic | 50 | 98.00% | 1 | 28.54 |
| Critic-augmented V | Empty critic | 50 | 98.00% | 1 | 24.16 |

- 使用真实 Critic 输入时，专项准确率提升：**30.00 个百分点**。
- 微调后，错误偏向 `navigate_back` 的样本数：**16 → 1**。
- 微调 V 在真实 C 与空 C 下准确率均为 **98.00%**；真实 C 的平均 margin 比空 C 高 **4.38**。

## 可用于报告的结论

实验结果表明，在原始 V-Droid verifier 上使用包含 Critic 反馈的 P3 pairwise 数据继续微调，可以显著提升 V 对新输入格式和动作偏好关系的适应能力。完整测试集准确率由 89.94% 提升至 100.00%；在 `navigate_back` 专项诊断中，准确率由 68.00% 提升至 98.00%，错误偏向 `navigate_back` 的样本由 16 个降至 1 个。消融结果中，真实 Critic 与空 Critic 的最终准确率相同，但真实 Critic 输入获得了更高的平均评分间隔，说明 Critic 信息增强了模型对正确动作的评分置信度；当前结果主要支持“格式适配与动作偏好改善”，尚不单独宣称 Critic 已稳定改变最终动作排序。

## 使用边界

- 这里展示的是 verifier 离线 pairwise 评估结果，不等同于完整 AndroidWorld 闭环任务成功率。
- `navigate_back` 专项诊断用于衡量无效回退偏好，不代表整个实验只针对该动作。
- 当前采用 1 epoch 模型作为最终候选，不建议仅因训练集或验证集接近满分而继续追加 epoch。

