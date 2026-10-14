# 采集吞吐与紧凑载荷（A6）：体积 / 微基准 / 降种子口径

> 结论先行：新增**可选** `--format compact` 把观测里的稠密 float 数组换成 base64 fp16 buffer，
> 单局落盘 **≈ 0.54×**（`--no-privileged-state` **≈ 0.45×**），编码/解码各快 **~2.6× / ~2×**；
> 默认行为与现有 JSONL replay **完全不变**。种子口径建议 **~50–100 seeds/对手**，把预算投 **更多 deck 的多样性**。

日期：2026-10-12 · 相关代码：`train/data.py`、`train/collect.py`、`scripts/bench_payload_format.py`

---

## 1. 实测背景（为什么要做）

逐字段统计（`data/replays/datafirst_bc_sa500`，200 局 / 11,355 决策）：

| 字段 | 占比 | 类型 |
|---|---:|---|
| `token_features` | 35.7% | 稠密 float（14×65） |
| `option_features` | 34.6% | 稠密 float（K×68） |
| `privileged_state` | 18.9% | 稠密 float（1022 维） |
| `belief_target` | 6.4% | 稠密 float 直方图（152+152+10+10） |
| masks / codes / envelope | ~4.5% | bool / int，小 |

即 **~95% 的体积是把稠密 float 逐值写成 JSON 文本**（`"0.0, "`），既撑大落盘，也让父进程
`json.dumps`/`json.loads` 反复造/析几十万个 Python float。A6 只处理这一部分。

---

## 2. 紧凑载荷格式（additive，默认不变）

- 新增 CLI：`--format jsonl|compact`（默认 `jsonl`）、`--compact-dtype float16|float32`（默认 `float16`）、
  `--no-privileged-state`。
- `compact` 下把 `token_features`/`option_features`/`privileged_state`/belief 直方图编码为
  `{"__dense__": "float16", "shape": [..], "data": "<base64>"}`；整数/布尔字段与「一局一行」信封不变。
- **读取透明**：`train.data.load_samples` / `observation_from_payload` 按字段自动识别两种格式，
  旧 JSONL 目录照常读（无需知晓文件格式）。平铺 JSONL 与 replay 目录两条路径都兼容。
- **指纹护栏**：`format` / `compact_dtype` / `include_privileged_state` 进入 `CollectSpec.fingerprint()`
  与 manifest `spec` —— 换格式 = 换指纹，续跑会被拒绝，须用新 `--tag`。
- fp16 有损但受控：本轮实测 `option_features` 最大绝对误差 **2.3e-4**（特征多为 0/1 指示量与
  归一化标量，落在训练噪声以下）。需要精确时用 `--compact-dtype float32`（体积不减，纯提速）。
- 边界：行内矩阵若不等宽（异常）自动回退成 list，不会写坏数据；空向量仍写成 `[]`。

**未改动**（纪律）：默认 `jsonl` 路径逐字节等价于旧格式（`observation_to_payload` 构造顺序与字段完全保留），
另一会话正在读的 `data/replays/**` JSONL 不受影响。

---

## 3. 体积对比

### 3.1 端到端采集（同 spec、6 局，`superconduct_aggro` vs `natlan_battleship`）

| 格式 | `samples.jsonl` | MB/局 | 压缩比 |
|---|---:|---:|---:|
| `jsonl`（默认） | 9,776,933 B | 1.63 | 1.000 |
| `compact` fp16 | 5,260,233 B | 0.88 | **0.538** |
| `compact` fp16 + `--no-privileged-state` | 4,357,667 B | 0.73 | **0.446** |

### 3.2 逐决策归一（`scripts/bench_payload_format`，60 局）

| 格式 | dtype | privileged | B/决策 | 相对 jsonl |
|---|---|---:|---:|---:|
| `jsonl` | – | 保留 | 30,869 | 1.000 |
| `compact` | float32 | 保留 | 31,018 | 1.005 |
| `compact` | float16 | 保留 | 16,603 | **0.538** |
| `compact` | float16 | 丢弃 | 13,800 | **0.447** |

> fp32 compact **体积不省**（base64 的 4B/float ≈ JSON 文本长度），它的价值在 CPU（见 §4）。
> 整目录推算（`datafirst_bc_sa500`，4.24 GB / 3004 局 ≈ 1.41 MB/局）：compact fp16 ≈ **0.76 MB/局**、
> `--no-privileged-state` ≈ **0.63 MB/局**。

---

## 4. 序列化 / 反序列化微基准

进程内单线程，`--repeats 3` 取最优（本机另有一会话在采集，吞吐为**受竞争下的指示值**，体积数字不受影响）：

| 格式 | 编码 MB/s | 编码 决策/s | 解码 MB/s | 解码 决策/s |
|---|---:|---:|---:|---:|
| `jsonl` | 42.5 | 1,377 | 30.0 | 972 |
| `compact` float32 | 85.2 | 2,747 | 55.9 | 1,802 |
| `compact` float16 | 60.3 | **3,632** | 31.8 | **1,915** |
| `compact` float16 no-priv | 58.0 | **4,203** | 32.2 | **2,333** |

- 编码 ≈ **2.6–3.0×** 快（numpy `tobytes`+base64 是 C 层，远快于 `json.dumps` 几万个 float）。
- 解码 ≈ **2.0–2.4×** 快（`np.frombuffer` 近零拷贝）。
- 采集侧收益叠加：worker 序列化 + 父进程 `dumps`/`loads` 都要过一遍稠密数组，compact 让**两边同时变小变快**。

命令：
```bash
python -m scripts.bench_payload_format --replays data/replays/datafirst_bc_sa500 \
    --games 60 --repeats 3 --json reports/train/payload_bench.json
```

> 注（2026-10-14）：`reports/train/payload_bench.json` **应该**用下面命令重新生成；jsonl 行**应该**标 `n/a`（jsonl writer 忽略 `compact_dtype`，`bench_payload_format.py` 已按此输出）。旧文件已清理，避免误读标签。

---

## 5. 降种子采样口径（建议）

现状与硬事实：

- **A3/A4（PLAN D22/D23）**：M3-CPU 未过（~0.50）；search teacher **无增益且显著有害**（全池 Δ=−0.066/−0.114）→
  改排产「数据先行」；**BC 天花板 = 专家对池 0.561**。搜索重开需价值头校准 + 非任意候选裁剪 + 门控式覆盖。
- 采集吞吐本机上限 **~4,900 局/h**（12 逻辑核，SMT 无加速）；**每对手每种子 = 1 局**。

**建议口径：每个对手 50–100 seeds，预算优先给「更多 deck 的多样性」。**

理由：

1. **统计功效**：单组胜率的 95% Wilson/正态半宽 ≈ `1.96·sqrt(p(1-p)/n)`。`n=100` 时约 ±9.8%，
   `n=400`（4 对手 × 100）时约 ±4.9%。**50–100 seeds/对手** 足以把「某 deck 对池整体」的评估噪声
   压到与 0.50↔0.561 的差距同量级；再堆到几百/上千 seeds 边际收益很小。
2. **多样性 > 深度**：20 套 deck × 50 seeds = 1,000 局（~12 分钟），覆盖 20 个对手的分布；
   同样 1,000 局若只压 1 个对手，对「对池整体」几乎无信息。BC 上限由**专家覆盖到的状态分布**决定，
   配牌/对手越广，覆盖越好。
3. **与 BC 天花板对齐**：目标是先让纯 BC 逼近 0.561，而不是用搜索硬补。~50–100 seeds/对手 × 几十套 deck
   即可把数据规模推到 1e4–1e5 决策量级（数据先行档）。

**推荐命令**（紧凑格式 + 省掉用不到的 privileged_state）：
```bash
# 单 deck 对全池，100 seeds/对手
python -m train.collect --deck superconduct_aggro --all-opponents \
    --seed-start 0 --seed-count 100 --workers 12 --tag bc_sa_c100 \
    --format compact --no-privileged-state

# 多 deck 语料：对每套 deck 重复（tag 带 deck 名），并把它们一起喂训练
for deck in $(python -c "from envs.decks import deck_names; print(' '.join(deck_names()))"); do
  python -m train.collect --deck "$deck" --all-opponents \
      --seed-start 0 --seed-count 50 --workers 12 --tag "bc_${deck}_c50" \
      --format compact --no-privileged-state
done
python -m train.pipeline --replays data/replays/bc_*_c50 --epochs 30 --tag bc_multi
```

> 注：`data/` 不入库；`--tag` 一格式一目录，jsonl/compact 不混。续跑（同命令重跑）自动跳过已完成局。

---

## 6. 传入 A800 的收益估算

窗口约束（D21）：A800 只做 GPU 训练/推理，数据本地采集后传入，**传输预算 ~37 GB**。

以 `datafirst_bc_sa500` 的 1.41 MB/局（jsonl）为基准：

| 格式 | MB/局 | 37 GB 可容纳局数 | 相对 jsonl |
|---|---:|---:|---:|
| `jsonl` | 1.41 | ~26,000 | 1.0× |
| `compact` fp16 | 0.76 | ~49,000 | **1.9×** |
| `compact` fp16 + no-priv | 0.63 | ~59,000 | **2.2×** |

即**同样 37 GB 预算能多带 ~1.9–2.2× 的对局**（或同样数据量传输时间近乎减半）；训练侧读盘也快 ~2×。

---

## 7. 改动文件与测试

- `train/data.py`：`observation_to_payload` 新增 `format`/`compact_dtype`/`include_privileged_state`；
  dense 编解码 helper（`is_dense_payload`/`_pack_dense`/`_unpack_dense`）；`observation_from_payload`
  自动识别两格式；`sample_to_payload`/`write_samples_jsonl` 透传参数。
- `train/collect.py`：`CollectTask`/`CollectSpec` 新增 `payload_format`/`compact_dtype`/`include_privileged_state`；
  `_payload_for` 让 worker 直接按格式序列化；指纹与 manifest `spec` 纳入格式；CLI 三个新参数。
- `scripts/bench_payload_format.py`：体积 + 微基准（真实 replay 或 `--synthetic`）。
- `tests/test_payload_format.py`：round-trip（fp16 容差 / fp32 紧）、旧 JSONL 兼容、自动识别、
  不等宽回退、指纹/任务序列化、compact 目录读盘、断行容错、非法 format/dtype 报错。

测试：`python -m pytest tests/test_payload_format.py tests/test_collect.py` → **20 passed**（含两处引擎慢测）。

---

## 8. 下一步

1. **让流式读盘也吃 compact**：`scripts/train_bc_scale.py`（另一会话的大规模流式 BC）目前直接把
   `obs["token_features"]` 当 list 用——compact 下需走 `train.data` 的解码 helper。**建议与那会话协调后改**，
   本次不动他人文件。
2. **可选把 mask/整数 code 也打包**（再省 ~4%，收益小）。
3. **compact 版 `--store-cache`**：把解码后的张量直接 `.pt` 落盘（训练完全免 JSON）。
4. 训练/评估沿用现有 JSONL 直读路径无回归；新采集默认走 compact。
