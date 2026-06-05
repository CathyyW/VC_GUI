
## Ecoagent

![[Pasted image 20260603110101.png]]
### pros: closed-loop
### problem：
1）这个闭环的反馈质量完全依赖 observer 和 executor 的小模型能力: failure analysis->visual grounding表现差
2）Planner 和 Executor 之间存在严重 model gap：GPT-4o + ShowUI 2B / OS-Atlas-Pro 4B + Qwen2-VL-2B
语义-动作落差
此时agent 越多，接口不稳定性越大
3）多 agent 串行协作调用 + 环境等待 + 错误恢复成本->带来了较高延时，使得其实际部署效果受限
4）EcoAgent 有 semantic / planning-level self-correction，但 correction 的前提是 observer 能发现错；correction 的上限受 executor grounding 能力限制；它能重新规划，但不能真正学会。
### conclusion:
EcoAgent 把移动端自动化拆成 planner / executor / observer 的闭环协作，但它的闭环可靠性受限于边端小模型的 grounding 和 verification 能力；一旦 executor 点不准、observer 判不准，planner 的高层推理优势就很难真正转化成高 SR。

我们的期中的实验可证明，实际部署十分受限，sr随机性很大，sr↓->延时↑

## V-Droid 
### pros
1）verification 比 generation 更可靠：V-Droid 不让 LLM 直接生成动作，而是先抽取候选动作，再让 verifier 判断每个 action 是否 helpful，最后选择分数最高的动作。这个设计把无限连续动作空间变成了有限候选动作空间，因此对 8B 级别小模型更友好。
P³ pair-wise preference training->sr↑
2）prefilling-only (not decoding)->batch scoring+prefix caching 延时↓

### cons
3）因为action extractor （accessibility tree）->环境必须能被文本化、结构化，视觉密集型 app，上限会受限
4）self-correction training dataset具有正负作用
在 GUI 任务里，错误状态数量天然远多于正确状态。因为每一步正确动作通常只有一个或少数几个，但错误动作很多。
-self-correction training 把 AndroidWorld SR 从 52.2% 提升到 59.5%，平均轨迹长度从 10.3 增加到 11.6，因为 agent 会多做一些探索和纠正动作。
-但如果错误状态样本太多，模型会过度学习“navigate back”，这过度依赖错误/正确数据集比例data ratio（原论文提到了这个，说限制在2.5%以内）

实验作证，很多任务在中间陷入了死循环，每一轮action都是navigateback 打分最高，于是ui一直维持home，因为一直选择navigation back，performance collapse
![[Pasted image 20260603170149.png|470]]



## 架构 pre-action verifier+post-action critic


引入critic的原因：
修正 complete task 误判
Progress 判断
Failure reason 诊断
Recovery suggestion

### dataset collection 无开源

#### 收集
可以写一个 collector，在 V-Droid 每一步执行前后把信息都存下来。
每一步记录：
```
task_id
goal
step_id
history
before_ui current_html / accessibility tree
action_space
verifier 对每个 action 的 score
selected_action
after_ui current_html / accessibility tree
是否 task success
是否人工修正
```

原始数据可以长这样：
```
{
  "task_id": "files_delete_001",
  "goal": "Delete q2a8_fancy_banana.mp3 from Notifications folder",
  "step_id": 0,
  "history": [],
  "before_ui_html": "...",
  "after_ui_html": "...",
  "action_space": [
    {"action_type": "open_app", "app_name": "Files"},
    {"action_type": "click", "index": 5},
    {"action_type": "click", "index": 13},
    {"action_type": "navigate_back"}
  ],
  "scores": [0.91, 0.77, 0.08, 0.02],
  "selected_action": {"action_type": "open_app", "app_name": "Files"},
  "human_label": {
    "best_action": {"action_type": "open_app", "app_name": "Files"},
    "bad_actions": [
      {"action_type": "click", "index": 13},
      {"action_type": "navigate_back"}
    ]
  }
}
```
然后再离线转换成 P3：

```
chosen = prompt(goal, history, ui_html, best_action)  
rejected = prompt(goal, history, ui_html, bad_action)
```
#### 标注

1）第一类：成功轨迹里的正负样本

当 V-Droid 成功完成一个任务时，这条轨迹大概率是有用的。

对每一步：

```
selected_action = positive
同一 action_space 里明显无关动作 = negative
```

例如任务是删文件，首页状态下：

```
chosen: open_app Files
rejected: open Gmail / long_press Search / open YouTube
```

2）第二类：失败轨迹里的人工修正样本

当 V-Droid 失败时，不要整条丢掉。最有价值的是找到第一处错误 step。

比如：

```
step 0 正确step 1 正确step 2 点错了文件
```

只需要人工标 step 2：

```
rejected = V-Droid 当时选的错误动作
chosen = 人工认为应该选的动作
```

3）第三类：少量 self-correction 样本

比如执行错误动作后进入了错误页面，你可以构造：

```
chosen = navigate_back / clear_text / close_dialog
rejected = 继续乱点 / complete_task / 错误输入
```

但这里一定要少量。V-Droid 论文自己就提到，错误状态比正确状态多得多，过多 self-correcting pairs 会让模型持续输出 `navigate_back`，所以他们把 self-correcting training pairs 随机采样到整个训练集约 2.5%。这个点非常重要。

可以先设：

```
self-correction pairs 占比：1% - 3%不要超过 5%
```


得到上述收集的数据后，可以为下面两个模型做准备
### 微调verification actor（vdroid）

```
input:
{
chosen = 当前状态下正确 action
rejected = 当前状态下错误 action
}
原论文loss函数
```

### 蒸馏critic model

用 GPT-4o / Claude / Gemini / Qwen-VL-Max 作为 teacher
对收集的 trajectories 自动打标签
再人工检查一小部分
最后蒸馏到小 critic model

```
input = goal + history + before_ui + action + after_ui  
output:
{
  "completed": false,
  "progress": "failure",
  "failure_type": "wrong_target",
  "suggested_recovery": "navigate_back"
}
```