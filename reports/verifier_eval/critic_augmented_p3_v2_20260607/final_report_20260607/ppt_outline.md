# V 模型实验 PPT 文案提纲

## 第 1 页：为什么微调 V

**标题：Critic-Augmented Verifier 微调**

- 原始 V-Droid verifier 根据目标、历史、当前 UI 和候选 action 进行动作评分。
- 本实验在输入中加入上一步 Critic 反馈，并保持 pairwise preference loss 不变。
- 目标：适应 Critic 增强后的输入格式，并减少无效 `navigate_back` 偏好。

## 第 2 页：训练方案

**标题：基于 Critic-Augmented P3 的 Pairwise 微调**

- 初始化：官方 V-Droid verifier adapter + value head。
- Base model：`/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-bnb-4bit`。
- 数据：train `1335` / val `165` / test `159` pairs。
- 训练：`1` epoch，learning rate `2e-05`。
- Loss 目标：提高 chosen action 评分，降低 rejected action 评分。

## 第 3 页：完整测试集结果

**标题：微调后 Pairwise 判断达到 100%**

- 原始 V：**89.94%**
- 微调 V：**100.00%**
- 提升：**10.06 个百分点**
- 错误/平局：**16 → 0**

建议图表：使用 `chart_data.csv` 中 `full_test` 两行绘制准确率柱状图。

## 第 4 页：navigate_back 专项结果

**标题：显著减少无效回退动作偏好**

- 原始 V + C：准确率 **68.00%**，错误偏向 back **16** 次。
- 微调 V + C：准确率 **98.00%**，错误偏向 back **1** 次。
- 专项准确率提升：**30.00 个百分点**。

建议图表：准确率柱状图 + 错误 back 次数对比图。

## 第 5 页：Critic 消融与结论

**标题：当前收益主要来自格式适配与动作偏好改善**

- 微调后真实 C / 空 C 准确率均为 **98.00%**。
- 真实 C 的平均 margin 比空 C 高 **4.38**。
- 结论：微调显著提升 verifier 判断能力并减少无效 back；Critic 提高评分置信度。
- 边界：当前是离线 pairwise 结果，尚未验证完整 AndroidWorld 闭环成功率。
