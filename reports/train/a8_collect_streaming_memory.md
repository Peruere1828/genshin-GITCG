# A8：`train.collect` 父进程内存修复（流式落盘）

> 日期：2026-10-12｜作者：**A8 子代理**｜口径：工程/数据平面（不影响强度验收）
> 触发：A5 报告 `reports/train/bc_scale_datafirst.md` §1.3/§6 —— `train.collect` 父进程把每局样本
> 攒在 RAM，约 500 局/进程 ≈ 5 GB，14 个并行作业把 30 GB 机器打到 **swap 死**；A5 只能用
> `--max-tasks 50` 分块重起进程绕过。A8 把父进程改为**边到边流式消费**，消除该放量瓶颈。

---

## 0. 结论速览

- **根因**：`envs.rollout.run_task_shard` 无论是否传 `on_result`，都把 worker 的**每条结果 dict 追加进返回列表**
  （`results.append`/`results.extend`），`train.collect.collect()` 虽已用 `on_result` 流式落盘，但返回列表
  全程驻留父进程 → 父进程 RSS 随局数**线性增长**。
- **修复**：新增 `envs.rollout.stream_task_shard`（不保留结果，只把每条结果交给 `on_result` 后即弃），
  `train.collect` 改走该入口。`run_task_shard` 行为**逐字节不变**（rollout/arena/npc 不受影响）。
- **实测**（1 deck × 1 对手 × 500 局，8 worker）：父进程峰值 RSS = **583 MB / 0.57 GB**（旧实现同规模外推 ≈ 5 GB），
  **0 error**、**0 swap**，吞吐 **3,299 局/h**；50 局峰值 559 MB，即 **10× 局数只多 ~24 MB**（内存与局数解耦）。详见 §3。
- **不变量全保**：可续跑（`task_index`）、断行容错、`manifest.json` 指纹护栏、逐局流式落盘（C8）均未变，
  现有测试全绿。详见 §2/§4。

---

## 1. 根因与改动

### 1.1 根因（父进程汇聚）

`train.collect.collect()` 原先：

```python
run_task_shard(pending, worker_module="train.collect_worker", workers=workers, on_result=_on_result)
```

`on_result` 确实逐局 `dumps` 落盘，但 `run_task_shard` 内部：

- `_run_shard_in_subprocess._drain_stdout`：`results.append(match)`（每条结果都留一份）；
- `run_task_shard` 多 worker 路径：`results.extend(future.result())` 再合并。

于是一局 ~1.4 MB 的编码样本，500 局就在父进程留下 ~700 MB/worker 的 `results`（A5 实测单 worker 194 局
≈ 2 GB，含 Python 对象放大），多进程并行直接打满 30 GB 机器。

### 1.2 改动（`envs/rollout.py`）

- 把 `_run_shard_in_subprocess` 的主体抽成 `_drive_worker(..., retain_results: bool = True)`，返回
  `(results, received)`：`received` 始终计数（用于检测 worker 崩溃/截断），`retain_results=False` 时
  **不向 `results` 追加**，每条结果只经 `on_result` 后释放。
- `_run_shard_in_subprocess` 保留为**向后兼容薄包装**（retain=True，返回完整列表）——现有调用方与
  `tests/test_rollout_subprocess.py` 不受影响。
- `run_task_shard` 新增 `retain_results: bool = True` 形参（默认行为不变），内部改调 `_drive_worker`。
- 新增公开入口 **`stream_task_shard(shard, *, worker_module, workers=None, on_result) -> int`**：
  不保留任何结果，返回收到条数。多 worker 分片复用同一套死锁安全的双管排空逻辑。
- 抽出 `_resolve_shard_workers` / `_shard_exec_env` 小工具，三处入口共享（纯重构，无行为变化）。

### 1.3 改动（`train/collect.py`）

- `from envs.rollout import stream_task_shard`（替换 `run_task_shard`）；
- `collect()` 的 pending 分片改走 `stream_task_shard(..., on_result=_on_result)`。
- `_on_result` 逻辑不变（加锁 → 逐局 `append` → 立即释放）；补注释说明这是内存有界的关键。

> 未改动 `train/data.py`（格式/指纹语义）、`npc/`、`web/`、`engine-overlay/`、`PLAN.md`、`AGENTS.md`。

---

## 2. 保持的不变量

| 不变量 | 状态 | 说明 |
|---|---|---|
| 可续跑（按 `task_index`） | ✅ | `_scan_samples` 流式扫描完成局、`pending` 过滤逻辑未动 |
| 断行容错 | ✅ | `ensure_trailing_newline` + `_scan_samples`/`iter_task_records` 跳过 torn 行未动；新增测试覆盖 |
| `manifest.json` 指纹护栏 | ✅ | `CollectSpec.fingerprint()` 与 manifest 读写未动；新增漂移拒绝测试 |
| 逐局流式落盘（C8） | ✅ | 每条结果到达即 append（`on_result`），中断保留已完成局 |
| `run_task_shard` 旧行为 | ✅ | 默认 `retain_results=True`，返回合并排序后的完整列表 |

---

## 3. 大采集实测（1 deck × 1 对手 × 500 局，8 worker）

命令（`/usr/bin/time -v` 记录父进程峰值 RSS）：

```bash
PY=/home/ubuntuaaa/miniconda3/envs/gitcg/bin/python
/usr/bin/time -v $PY -m train.collect \
    --deck superconduct_aggro --opponents natlan_battleship \
    --seed-start 1000000 --seed-count 500 --workers 8 --tag a8_memtest
```

| 指标 | 值 |
|---|---|
| 局数 | 500（`ran_this_call=500`，`samples.jsonl` 500 行） |
| 父进程峰值 RSS（`Maximum resident set size`） | **583,432 kB ≈ 583 MB（0.57 GB）** |
| 墙钟（`Elapsed`） | **9:05.68 = 545.7 s** |
| 吞吐 | **3,299 局/h**（8 worker；含启动 + 收尾全量扫描） |
| CPU | 847%（8 worker 近满载） |
| error | **0** |
| swap | **0**（`Swaps: 0`；旧实现此处打满 swap） |
| 落盘 | **28,277** 样本 / **826 MB**（jsonl，≈ 1.65 MB/局） |
| exit | 0 |

> 对照：旧实现同规模父进程 RSS 外推 ≈ **5 GB**（A5 实测单 worker 194 局 ≈ 2 GB；14 并行 → ~28 GB → swap 死）；
> 修复后父进程内存**与局数解耦**。

### 3.1 父进程内存与局数解耦（真实采集，同 spec 不同规模）

| 规模 | 父进程峰值 RSS | 墙钟 | 吞吐 |
|---:|---:|---:|---:|
| 仅导入（无对局，`train.collect` + encoder 基线） | 358.8 MB | – | – |
| 50 局 | 559.1 MB | 63.4 s | 2,839 局/h |
| 500 局 | 583.4 MB | 545.7 s | 3,299 局/h |

**10× 局数只增加 ~24 MB（0.6 GB 的 ~4%）**，峰值基本 = 固定基线（~359 MB，Python+numpy+encoder 导入）
+ 有界的 in-flight 缓冲。旧实现则随局数线性增长（每局 ~1.4 MB 编码样本 + Python 对象放大，全部驻留）。

### 3.2 父进程内存对照实验（合成 worker，可复现）

用 mock worker（每局 128 KiB 载荷，400 局，4 worker）在**同一进程**内对比两种路径的
`tracemalloc` 峰值（`tests/test_collect.py::test_stream_task_shard_does_not_accumulate_results` 同源）：

| 路径 | 结果条数 | 峰值内存 | 流经总量 |
|---|---:|---:|---:|
| `stream_task_shard`（新） | 400 | **1.86 MB** | 52.4 MB |
| `run_task_shard(retain=True)`（旧） | 400 | **53.04 MB** | 52.4 MB |

即流式路径峰值 ≈ 流经总量的 **3.5%**（仅 in-flight 行），旧路径峰值 ≈ 流经总量（全量驻留）。
测试断言 `peak < streamed // 4`（13.1 MB），对两类行为有 7× 余量区分。

---

## 4. 测试

新增 5 个快速测试到 `tests/test_collect.py`：

- `test_stream_task_shard_does_not_accumulate_results`：mock worker 400 局 × 128 KiB，`tracemalloc`
  断言峰值 < 流经总量 /4（内存有界）；
- `test_run_task_shard_retain_results_is_backward_compatible`：默认 retain 返回完整列表；`retain_results=False`
  返回空表但全部流经 `on_result`；`stream_task_shard` 返回计数；
- `test_collect_streams_and_resumes_exactly_once`：`collect` 走流式入口 + 续跑 0 重跑 + 文件无重复 index；
- `test_collect_redoes_torn_trailing_game`：写入 torn 尾行后，续跑只重做该局、不重复已完成局；
- `test_collect_fingerprint_guard_refuses_drift`：spec 漂移时 `SystemExit`（指纹护栏）。

结果：

```bash
$PY -m pytest tests/test_collect.py -q
# 13 passed（含 2 个真实引擎慢测：续跑 + search teacher）
$PY -m pytest tests/test_collect.py tests/test_rollout_subprocess.py tests/test_rollout_workers.py -m "not slow" -q
# 14 passed（流式入口 + 死锁回归 + worker 解析）
$PY -m pytest -m "not slow" -q        # 全量快速套件（envs/rollout.py 被 rollout/arena 共用）
# 234 passed, 61 deselected
```

---

## 5. 与旧 `--max-tasks` 分块绕法对比

| 维度 | 旧：`--max-tasks 50` 块循环 | 新：流式 `collect` |
|---|---|---|
| 父进程 RSS | 每块封顶 ~50 局（~70 MB）+ 进程重启 | **与局数解耦**（500 局实测峰值 583 MB，10× 局数仅 +24 MB） |
| 操作 | 需外层 `for` 循环反复起进程（A5 §1.3） | **一条命令跑完**，无外层编排 |
| 启动开销 | 每块重启（导入 + 引擎初始化）× N 块 | 一次启动 |
| 续跑/指纹 | 依赖同 tag 指纹，OK | 同机制，OK |
| 长作业健壮性 | 块间自然回收 | 单进程长跑，但落盘逐局、崩溃可续跑 |

> 结论：旧绕法只是把"内存线性增长"截断在 50 局，代价是 N 次进程重启与手工编排；新实现把瓶颈本身移除。

---

## 6. 未决问题

1. **`on_result` 抛错时原异常被吞**（reader 线程死亡 → 主线程以 `received != len(shard)` 报错，丢失
   原始 traceback）。这是 `run_task_shard` 既有行为，非本次引入；可另开小修（把回调异常转存并重抛）。
2. **写盘为每局 open/append/close**：为断点耐久性保留；若后续追求极致吞吐，可在"块内缓冲 + 块末 flush"
   与耐久性之间取平衡（当前 8 worker 下写盘非瓶颈）。
3. **`npc/eval.py` 仍用 `run_task_shard` 并忽略返回列表**（同样全量驻留）。本次按纪律未改 `npc/`；
   若 NPC 评测也放量，可一行切到 `stream_task_shard`。
4. 峰值 RSS 实测受同机其它会话争用影响；§3 数字为本次单作业实测，趋势（与局数解耦）与 §3.1 对照实验一致。

---

## 7. 产物索引

- 代码：`envs/rollout.py`（`_drive_worker`/`stream_task_shard`）、`train/collect.py`（改走流式入口）
- 测试：`tests/test_collect.py`（A8 段）
- 报告：本文件 `reports/train/a8_collect_streaming_memory.md`
- 关联：`reports/train/bc_scale_datafirst.md` §1.3/§6（问题来源）、`reports/train/collect_throughput_notes.md`（A6）
- 原始数据（gitignore）：`data/replays/a8_memtest/`
