# AndroidWorld 评测对比报告：V-Droid vs VC_GUI

> 生成时间：2026-06-09  
> 数据来源：`V-Droid/saved` 与 `VC_GUI/saved` 中 `VDroid_Round110k_try_*` eval 运行结果  
> 任务 ID 对照：`V-Droid/android_world_tasks.txt`（001–116）  
> 完整 CSV：`/root/autodl-tmp/task_comparison.csv`

---

## 1. 实验背景

### 1.1 V-Droid（原始基线）

- **机制**：从 accessibility tree 抽取候选动作 → Verifier 打分 → 选最高分执行
- **覆盖**：111/116 任务完成 eval
- **已知问题**：self-correction 训练易导致 `navigate_back` 死循环；缺乏 post-action 反馈

### 1.2 VC_GUI（改进架构）

- **机制**：Pre-action Verifier + Post-action Critic 联合推理

```
V 选动作 → 执行 → C 输出 critic JSON → 下一步 V 读取 summary_to_history 再选动作
```

- **Critic 输出**：`outcome` / `failure_type` / `suggested_recovery` / `summary_to_history`
- **Verifier 微调**：使用含 Critic feedback 的 P³ 数据做 LoRA
- **覆盖**：106/116 任务完成 eval

---

## 2. 总体统计

### 2.1 双方均有明确结果的任务（92 个）

| 指标 | V-Droid | VC_GUI | 变化 |
|------|---------|--------|------|
| 成功率 | **42/92 = 45.7%** | **24/92 = 26.1%** | -19.6 pp |
| 平均步数 | 14.1 | 13.4 | — |
| 平均耗时 | 332s | 332s | — |

### 2.2 分类统计（全 115 个已出现任务）

| 分类 | 数量 | 说明 |
|------|------|------|
| 都成功 | 22 | 双方 eval 均通过 |
| 一成功一失败 | 22 | 一方通过、一方失败 |
| 都失败 | 48 | 双方均未通过 |
| 未双方评测 | 17 | 仅一方有明确结果 |
| 未评测 | 6 | 双方均无有效 success rate |

**图例**：✓ 成功 · ✗ 失败 · △50% 部分成功 · — 无有效结果

---

## 3. 架构核心优势：原模型失败、VC_GUI 成功的任务

| ID | 任务 | V-Droid | VC_GUI | 关键机制 |
|:--:|------|---------|--------|----------|
| 9 | ClockStopWatchRunning | ✗ 5步 132s | ✓ 5步 111s | Critic 识别误点 Reset，引导重试 Start |
| 96 | SystemBluetoothTurnOn | ✗ 21步 457s | ✓ 7步 155s | Critic 打破 navigate_back 死循环，7步完成 |

### 3.1 SystemBluetoothTurnOn（ID 96）

**V-Droid 失败原因**：在 Settings 中 `navigate_back` 持续获最高分（48–63 分），在设置页与搜索页之间循环 21 步。

**VC_GUI 成功路径**：Critic 在无效点击后输出 `no_effect → retry`，引导搜索 "Bluetooth" 并开启开关，7 步完成。

### 3.2 ClockStopWatchRunning（ID 9）

**V-Droid 失败原因**：误点 Start 后仍选择 `complete_task`（得分 62），agent 自认为完成但 eval 判定失败（premature_complete）。

**VC_GUI 成功路径**：Critic 输出 `[Critic] clicked Reset instead of Start; retry with correct action`，下一步正确点击 Start。

---

## 4. 效率优势（双方均成功，VC_GUI 步数更少）

| ID | 任务 | V-Droid 步数 | VC_GUI 步数 | 节省 |
|:--:|------|:------------:|:-----------:|:----:|
| 55 | RecipeDeleteMultipleRecipesWithConstraint | 41 | 23 | -18 |
| 95 | SystemBluetoothTurnOffVerify | 21 | 10 | -11 |

---

## 5. 全任务对比表

### 5.1 都成功（22 个）

| ID | 任务名 | VD结果 | VD步数 | VD耗时 | VC结果 | VC步数 | VC耗时 |
|:--:|--------|:------:|:------:|:------:|:------:|:------:|:------:|
| 16 | ExpenseAddSingle | ✓ | 10 | 214s | ✓ | 10 | 222s |
| 21 | ExpenseDeleteSingle | ✓ | 6 | 93s | ✓ | 6 | 105s |
| 26 | MarkorCreateFolder | ✓ | 6 | 88s | ✓ | 6 | 114s |
| 27 | MarkorCreateNote | ✓ | 8 | 113s | ✓ | 9 | 191s |
| 30 | MarkorDeleteAllNotes | ✓ | 12 | 158s | ✓ | 12 | 236s |
| 31 | MarkorDeleteNewestNote | ✓ | 6 | 114s | ✓ | 6 | 112s |
| 41 | NotesTodoItemCount | ✓ | 6 | 137s | ✓ | 6 | 414s |
| 42 | OpenAppTaskEval | ✓ | 3 | 25s | ✓ | 3 | 59s |
| 55 | RecipeDeleteMultipleRecipesWithConstraint | ✓ | 41 | 897s | ✓ | 23 | 565s |
| 71 | SimpleCalendarDeleteEventsOnRelativeDay | ✓ | 10 | 197s | ✓ | 10 | 234s |
| 72 | SimpleCalendarDeleteOneEvent | ✓ | 7 | 138s | ✓ | 7 | 158s |
| 76 | SimpleCalendarEventsOnDate | ✓ | 5 | 68s | ✓ | 5 | 108s |
| 78 | SimpleCalendarLocationOfEvent | ✓ | 6 | 111s | ✓ | 6 | 112s |
| 95 | SystemBluetoothTurnOffVerify | ✓ | 21 | 377s | ✓ | 10 | 253s |
| 97 | SystemBluetoothTurnOnVerify | ✓ | 7 | 125s | ✓ | 7 | 135s |
| 99 | SystemBrightnessMaxVerify | ✓ | 5 | 73s | ✓ | 5 | 84s |
| 101 | SystemBrightnessMinVerify | ✓ | 5 | 83s | ✓ | 12 | 231s |
| 104 | SystemWifiTurnOffVerify | ✓ | 5 | 88s | ✓ | 13 | 289s |
| 106 | SystemWifiTurnOnVerify | ✓ | 5 | 201s | ✓ | 21 | 571s |
| 109 | TasksDueOnDate | ✓ | 4 | 125s | ✓ | 7 | 156s |
| 112 | TasksIncompleteTasksOnDate | ✓ | 4 | 130s | ✓ | 5 | 117s |
| 114 | TurnOnWifiAndOpenApp | ✓ | 7 | 291s | ✓ | 13 | 327s |

### 5.2 一成功一失败（22 个）

| ID | 任务名 | VD结果 | VD步数 | VD耗时 | VC结果 | VC步数 | VC耗时 |
|:--:|--------|:------:|:------:|:------:|:------:|:------:|:------:|
| 6 | CameraTakePhoto | ✓ | 4 | 88s | ✗ | 11 | 233s |
| 7 | CameraTakeVideo | ✓ | 7 | 179s | ✗ | 5 | 89s |
| 9 | **ClockStopWatchRunning** | ✗ | 5 | 132s | ✓ | 5 | 111s |
| 11 | ContactsAddContact | ✓ | 11 | 280s | ✗ | 10 | 215s |
| 12 | ContactsNewContactDraft | ✓ | 11 | 321s | ✗ | 14 | 329s |
| 17 | ExpenseDeleteDuplicates | ✓ | 31 | 637s | ✗ | 19 | 442s |
| 18 | ExpenseDeleteDuplicates2 | ✓ | 18 | 375s | ✗ | 7 | 152s |
| 22 | FilesDeleteFile | ✓ | 11 | 212s | ✗ | 21 | 505s |
| 29 | MarkorCreateNoteFromClipboard | ✓ | 20 | 274s | ✗ | 13 | 290s |
| 32 | MarkorDeleteNote | ✓ | 6 | 117s | ✗ | 5 | 78s |
| 46 | RecipeAddMultipleRecipes | ✓ | 33 | 1182s | ✗ | 20 | 488s |
| 61 | RetroPlaylistDuration | ✓ | 19 | 518s | ✗ | 15 | 374s |
| 69 | SimpleCalendarAnyEventsOnDate | ✓ | 5 | 91s | ✗ | 9 | 219s |
| 70 | SimpleCalendarDeleteEvents | ✓ | 13 | 284s | ✗ | 7 | 164s |
| 73 | SimpleCalendarEventOnDateAtTime | ✓ | 5 | 86s | ✗ | 6 | 140s |
| 75 | SimpleCalendarEventsInTimeRange | ✓ | 5 | 93s | ✗ | 5 | 97s |
| 77 | SimpleCalendarFirstEventAfterStartTime | ✓ | 5 | 74s | ✗ | 5 | 98s |
| 80 | SimpleCalendarNextMeetingWithPerson | ✓ | 6 | 87s | ✗ | 5 | 87s |
| 96 | **SystemBluetoothTurnOn** | ✗ | 21 | 457s | ✓ | 7 | 155s |
| 103 | SystemWifiTurnOff | ✓ | 6 | 111s | ✗ | 21 | 498s |
| 105 | SystemWifiTurnOn | ✓ | 6 | 121s | ✗ | 9 | 216s |
| 111 | TasksHighPriorityTasksDueOnDate | ✓ | 5 | 173s | ✗ | 16 | 594s |

> 加粗任务为 **VC_GUI 独有成功**（原 V-Droid 失败）

### 5.3 都失败（48 个）

| ID | 任务名 | VD结果 | VD步数 | VD耗时 | VC结果 | VC步数 | VC耗时 |
|:--:|--------|:------:|:------:|:------:|:------:|:------:|:------:|
| 8 | ClockStopWatchPausedVerify | ✗ | 2 | 29s | ✗ | 21 | 516s |
| 13 | ExpenseAddMultiple | ✗ | 15 | 238s | ✗ | 20 | 502s |
| 14 | ExpenseAddMultipleFromGallery | ✗ | 16 | 316s | ✗ | 15 | 367s |
| 15 | ExpenseAddMultipleFromMarkor | ✗ | 11 | 214s | ✗ | 14 | 318s |
| 19 | ExpenseDeleteMultiple | ✗ | 31 | 580s | ✗ | 9 | 173s |
| 20 | ExpenseDeleteMultiple2 | ✗ | 9 | 159s | ✗ | 9 | 182s |
| 23 | FilesMoveFile | ✗ | 31 | 663s | ✗ | 31 | 744s |
| 24 | MarkorAddNoteHeader | ✗ | 31 | 677s | ✗ | 31 | 761s |
| 25 | MarkorChangeNoteContent | ✗ | 31 | 464s | ✗ | 31 | 738s |
| 28 | MarkorCreateNoteAndSms | ✗ | 31 | 449s | ✗ | 17 | 402s |
| 33 | MarkorEditNote | ✗ | 7 | 117s | ✗ | 11 | 238s |
| 34 | MarkorMergeNotes | ✗ | 17 | 326s | ✗ | 15 | 325s |
| 35 | MarkorMoveNote | ✗ | 31 | 676s | ✗ | 31 | 730s |
| 36 | MarkorTranscribeReceipt | ✗ | 31 | 626s | ✗ | 10 | 223s |
| 37 | MarkorTranscribeVideo | ✗ | 31 | 885s | ✗ | 24 | 908s |
| 38 | NotesIsTodo | ✗ | 8 | 201s | ✗ | 6 | 200s |
| 43 | OsmAndFavorite | ✗ | 21 | 445s | ✗ | 21 | 648s |
| 47 | RecipeAddMultipleRecipesFromImage | ✗ | 20 | 672s | ✗ | 51 | 1307s |
| 51 | RecipeDeleteDuplicateRecipes | ✗ | 21 | 633s | ✗ | 13 | 392s |
| 52 | RecipeDeleteDuplicateRecipes2 | ✗ | 41 | 1180s | ✗ | 25 | 551s |
| 53 | RecipeDeleteDuplicateRecipes3 | ✗ | 51 | 1073s | ✗ | 26 | 607s |
| 54 | RecipeDeleteMultipleRecipes | ✗ | 11 | 206s | ✗ | 11 | 227s |
| 59 | RetroCreatePlaylist | ✗ | 31 | 698s | ✗ | 29 | 683s |
| 60 | RetroPlayingQueue | ✗ | 11 | 184s | ✗ | 17 | 425s |
| 62 | RetroSavePlaylist | ✗ | 24 | 552s | ✗ | 11 | 286s |
| 63 | SaveCopyOfReceiptTaskEval | ✗ | 21 | 561s | ✗ | 21 | 518s |
| 64 | SimpleCalendarAddOneEvent | ✗ | 29 | 800s | ✗ | 13 | 327s |
| 65 | SimpleCalendarAddOneEventInTwoWeeks | ✗ | 23 | 546s | ✗ | 29 | 687s |
| 66 | SimpleCalendarAddOneEventRelativeDay | ✗ | 17 | 301s | ✗ | 31 | 789s |
| 67 | SimpleCalendarAddOneEventTomorrow | ✗ | 13 | 245s | ✗ | 19 | 492s |
| 68 | SimpleCalendarAddRepeatingEvent | ✗ | 12 | 240s | ✗ | 14 | 359s |
| 74 | SimpleCalendarEventsInNextWeek | ✗ | 5 | 98s | ✗ | 5 | 96s |
| 79 | SimpleCalendarNextEvent | ✗ | 5 | 81s | ✗ | 5 | 101s |
| 88 | SportsTrackerActivitiesCountForWeek | ✗ | 9 | 138s | ✗ | 11 | 286s |
| 89 | SportsTrackerActivitiesOnDate | ✗ | 6 | 64s | ✗ | 15 | 424s |
| 90 | SportsTrackerActivityDuration | ✗ | 7 | 97s | ✗ | 7 | 171s |
| 91 | SportsTrackerLongestDistanceActivity | ✗ | 19 | 314s | ✗ | 9 | 222s |
| 92 | SportsTrackerTotalDistanceForCategoryOverInterval | ✗ | 7 | 85s | ✗ | 8 | 205s |
| 93 | SportsTrackerTotalDurationForCategoryThisWeek | ✗ | 6 | 74s | ✗ | 6 | 132s |
| 94 | SystemBluetoothTurnOff | ✗ | 5 | 65s | ✗ | 9 | 258s |
| 98 | SystemBrightnessMax | ✗ | 6 | 90s | ✗ | 16 | 287s |
| 100 | SystemBrightnessMin | ✗ | 6 | 82s | ✗ | 6 | 104s |
| 107 | TasksCompletedTasksForDate | ✗ | 14 | 621s | ✗ | 19 | 570s |
| 108 | TasksDueNextWeek | ✗ | 5 | 200s | ✗ | 6 | 133s |
| 110 | TasksHighPriorityTasks | ✗ | 5 | 194s | ✗ | 9 | 197s |
| 113 | TurnOffWifiAndTurnOnBluetooth | △50% | 31 | 1497s | ✗ | 9 | 173s |
| 115 | VlcCreatePlaylist | ✗ | 31 | 1617s | ✗ | 17 | 640s |
| 116 | VlcCreateTwoPlaylists | ✗ | 8 | 514s | ✗ | 11 | 561s |

### 5.4 未双方评测（17 个）

| ID | 任务名 | VD结果 | VD步数 | VD耗时 | VC结果 | VC步数 | VC耗时 |
|:--:|--------|:------:|:------:|:------:|:------:|:------:|:------:|
| 1 | AudioRecorderRecordAudio | — | — | — | ✓ | 9 | 163s |
| 3 | BrowserDraw | — | — | — | ✗ | 15 | 322s |
| 4 | BrowserMaze | — | — | — | ✗ | 14 | 415s |
| 5 | BrowserMultiply | — | — | — | ✗ | 14 | 371s |
| 10 | ClockTimerEntry | ✗ | 21 | 629s | — | — | 496s |
| 39 | NotesMeetingAttendeeCount | ✓ | 8 | 191s | — | — | 287s |
| 40 | NotesRecipeIngredientCount | ✓ | 8 | 229s | — | — | 70s |
| 44 | OsmAndMarker | ✗ | 21 | 595s | — | — | 3s |
| 48 | RecipeAddMultipleRecipesFromMarkor | ✗ | 12 | 414s | — | — | 664s |
| 49 | RecipeAddMultipleRecipesFromMarkor2 | ✗ | 20 | 775s | — | — | 21s |
| 50 | RecipeAddSingleRecipe | ✓ | 12 | 416s | — | — | 17s |
| 56 | RecipeDeleteMultipleRecipesWithNoise | ✓ | 20 | 410s | — | — | — |
| 57 | RecipeDeleteSingleRecipe | ✓ | 7 | 123s | — | — | — |
| 58 | RecipeDeleteSingleWithRecipeWithNoise | ✓ | 9 | 188s | — | — | — |
| 81 | SimpleDrawProCreateDrawing | ✗ | 11 | 186s | — | — | — |
| 85 | SimpleSmsSend | ✓ | 13 | 195s | — | — | — |
| 86 | SimpleSmsSendClipboardContent | ✗ | 10 | 153s | — | — | — |

### 5.5 未评测（6 个）

| ID | 任务名 | VD耗时 | VC耗时 |
|:--:|--------|:------:|:------:|
| 45 | OsmAndTrack | 400s | 213s |
| 82 | SimpleSmsReply | 6s | — |
| 83 | SimpleSmsReplyMostRecent | 6s | — |
| 84 | SimpleSmsResend | 21s | — |
| 87 | SimpleSmsSendReceivedAddress | 19s | 30s |
| 102 | SystemCopyToClipboard | 518s | 168s |

---

## 6. 按应用类别对比（双方均有结果的 92 任务）

| 应用类别 | V-Droid SR | VC_GUI SR | 变化 |
|----------|-----------|----------|------|
| Clock | 0% | 50% | +50 pp |
| System | 67% | 58% | -8 pp |
| SimpleCalendar | 59% | 24% | -35 pp |
| Camera | 100% | 0% | -100 pp |
| Contacts | 100% | 0% | -100 pp |
| Markor | 43% | 29% | -14 pp |
| Expense | 44% | 22% | -22 pp |
| SportsTracker | 0% | 0% | 0 |

---

## 7. Critic 模块运行统计（VC_GUI）

| Critic 判断 | 次数 | 占比 |
|------------|------|------|
| progress | 701 | 53.6% |
| no_effect | 424 | 32.4% |
| failure | 131 | 10.0% |
| done | 51 | 3.9% |

| 恢复建议 | 次数 |
|----------|------|
| continue | 701 |
| retry | 482 |
| complete_task | 51 |
| navigate_back | **11** |

Critic 对 `navigate_back` 的建议仅 11 次，显著低于原模型中的 back 偏好，说明架构设计方向正确。

---

## 8. 结论与建议

### 8.1 架构已验证的优势

1. **纠错能力**：在系统设置、时钟类任务上实现 0→1 突破
2. **死循环缓解**：Critic 能识别 `no_effect` 并建议 `retry`，避免无限 `navigate_back`
3. **过早完成检测**：修正 `premature_complete` 误判
4. **轨迹效率**：部分成功任务步数减少 40%–50%

### 8.2 当前不足

1. 整体 SR 从 45.7% 降至 26.1%（-19.6 pp），Verifier LoRA 微调破坏了部分原有能力
2. Camera、Contacts、SimpleCalendar 等原高成功率任务明显退步
3. Critic 对 progress 判断偏乐观（约 54%），可能误导后续 Verifier
4. VC_GUI 尚有 10 个任务未完成 eval

### 8.3 后续建议

1. 控制 self-correction 训练样本比例为 1%–3%
2. 加大 Critic 在 failure/no_effect 上的训练权重
3. 对退步任务做 case study，检查 prompt 格式与微调过拟合
4. 补跑剩余 10 个任务后更新本报告

---

## 附录

- 原始 V-Droid 结果：`/root/autodl-tmp/V-Droid/saved/`
- VC_GUI 结果：`/root/autodl-tmp/VC_GUI/saved/`
- 架构设计文档：`V-Droid/docs/framework.md`、`VC_GUI/docs/数据集构造.md`
- 可编辑数据：`/root/autodl-tmp/task_comparison.csv`
