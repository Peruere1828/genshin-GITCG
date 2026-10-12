# A800 机会窗口 Runbook（PLAN §5.1 / D14）

> 对象：**A800 80G + 128 核机会资源**（申请制、时间受限）的操作执行。
> 不是南大 LSF 集群（那是 `scripts/cluster/README.md` 的 LSF 路线，仍暂缓）。
> 纪律（D7/D14）：**无常驻服务**、**一切长任务可断点续跑**、长作业挂 `tmux/setsid` 流式落盘；
> 窗口内优先级 **训练 > 评测 > 采集 > LLM 分析**；里程碑不依赖本窗口。
> 结果只回传**聚合报告 + checkpoint**；原始 replay 留 A800 存储（本机只放代码与小体积报告）。

本 runbook 只包含**现在就绪、不依赖未完成模块**的工作。搜索蒸馏（S-C 后半）落地后会再补一节（见 §5）。

---

## 0. 环境准备（一次性）

```bash
# 代码（记录 commit hash 到报告）
git clone git@github.com:Peruere1828/genshin-GITCG.git && cd genshin-GITCG
git rev-parse HEAD

# Python 环境（3.12；conda 或 venv 均可）
conda create -y -n gitcg python=3.12 && conda activate gitcg
pip install -e ".[train,dev]"        # 装 gitcg>=0.21.0 + torch + pytest
python -c "import gitcg, torch; print(gitcg.__file__); print(torch.__version__, torch.cuda.is_available())"

# 自检（应 ~162 passed）；注意用 env 内 python，别用会被吞输出的包装器
python -m pytest -m "not slow" -q
```

- **资产元数据**：首次跑对局时 `agents/scripted/assets.py` 会联网拉卡牌元数据
  （host `gi-tcg-assets-api-hf.guyutongxue.site`）。**A800 无外网**则从本机拷贝
  `data/assets/expert_system_assets_chs_latest.json`，或设 `GITCG_ASSETS_OFFLINE=1` + `GITCG_ASSETS_CACHE=<path>`。
- **记录**：GPU 型号/显存、torch/cuda 版本、`nproc`、commit hash、`configs/engine.lock*`（若有）。

---

## 1. 基准测量（先做，~1h；结果决定后续规模）

```bash
python scripts/bench_ops.py --device cuda          # 算子 + CVPN 前向/训练吞吐 → reports/explore/*.json
python scripts/bench_device.py --devices cuda      # CVPN 训练吞吐
# CPU 采集吞吐小测（换算 局/h）
python -m train.collect --deck superconduct_aggro --opponents natlan_battleship \
    --seed-count 40 --workers 120 --tag a800_bench
```
回传：`reports/explore/` 下新 json + 采集 summary（含耗时/局数）。

---

## 2. 大规模采集（CPU 128 核；数据瓶颈，最该先喂饱）

采集即 BC 语料：`teacher = expert:<deck>`（我方脚本专家）、`opponent = 池内 20 套`；一行一局、
可续跑、`manifest.json` 指纹护栏。多套 meta deck 各开一个 tag 增多样性：

```bash
for d in superconduct_aggro unyielding_geo double_geo_navia dvalin_bonk natlan_battleship; do
  python -m train.collect --deck "$d" --all-opponents \
      --seed-start 0 --seed-count 500 --workers 120 --tag "a800_bc_$d"
done
```
- 规模参考：本机 12 worker ≈ 4,900 局/h ⇒ 128 核理论 ~50k 局/h（含调度损耗）。500 seed × 20 对手 = **1 万局/deck**；要 1e5+ 局就加 seed 或多 deck。
- **断点续跑**：同命令重跑即续（完成的局跳过）；长作业 `setsid`/`tmux` + 日志。
- 若已有 checkpoint，可用 `--teacher-spec neural:<ckpt>` 做一次「自博弈/DAgger 式」扩充（可选）。

---

## 3. GPU 训练放大（A800 80G）★窗口首要

> **已 push**：`train.pipeline --distill --d-model ...`（蒸馏 + belief 头 + 网络规模参数）与 `train.learning_curve --replays --d-models ...`。开跑前先 `git pull` 并 `--help` 核对。

```bash
# 数据量 × 网络规模 的 scale 关系（M3 用；可扫 d_model，committed 已支持）
python -m train.learning_curve --replays data/replays/a800_bc_* \
    --sizes 2000 8000 32000 --d-models 128 256 512 --device cuda --precollate --epochs 30

# 从 replay 训练出一个 candidate checkpoint（data/checkpoints/<run>.pt，gitignore）
python -m train.pipeline --replays data/replays/a800_bc_* --device cuda --epochs 50 --precollate --tag a800_bc
```
- 显存 80G 充足：可上 d_model 256–512、4–8 层、bf16（以 `bench_ops` 结论为准）。
- 每步记录 train/val loss 与 acc；产出 checkpoint 别覆盖历史，只追加。
- 训练脚本/规模以 `reports/` 报告形式回传。

---

## 4. 评测认证（CPU 128 核；对 §3 产出的 checkpoint）

```bash
# M3-CPU / M2 口径：纯策略 vs 池内全部对手（配对种子 + cluster CI）
python -m scripts.eval_protocol --contestant neural:data/checkpoints/<ckpt>.pt \
    --seeds 50 --workers 120 --strict-ci --pooled-threshold 0.50 --tag a800_eval
# 合并门禁（候选 vs 上一 checkpoint 的配对 A/B 非劣）
python -m scripts.eval_protocol --contestant neural:<new>.pt --baseline neural:<old>.pt \
    --seeds 20 --workers 120 --tag a800_ab
```
回传 `reports/explore/<tag>_*.json`（聚合）与门槛判定。

---

## 5. 搜索蒸馏数据采集（S-C 已交付；M3 关键，窗口内优先）

`train.collect --search` 用 MCCFR 搜索当 teacher，逐搜索决策记录**根平均策略 / 根价值 / belief 目标**（S-C 蒸馏数据）：

```bash
python -m train.collect --deck superconduct_aggro --all-opponents --search \
    --iterations 8 --depth 1 --candidate-limit 4 --max-searches 8 --belief-samples 2 \
    --seed-start 0 --seed-count 100 --workers 120 --tag a800_distill
```
- MCCFR + 子进程 fork **很贵**：先小样本量吞吐（用 `--max-searches/--iterations/--depth` 控预算），再按实测吞吐放量。
- 蒸馏训练：
```bash
python -m train.pipeline --distill --replays data/replays/a800_distill \
    --device cuda --d-model 256 --epochs 50 --tag a800_distill
```
- `--search` 有 spec 指纹护栏与断点续跑；**换搜索参数会改指纹**，重跑请换 `--tag`。

---

## 6. 回传清单（窗口结束前）

- `reports/`（benchmark / learning_curve / eval / train 小报告，聚合）；
- candidate **checkpoint**（`data/checkpoints/*.pt`，scp 或共享盘；**不入库**）；
- 每步的**命令 + commit hash + 机器/GPU 信息**；
- 原始 replay（`data/replays/a800_*`）留 A800 存储即可（大）。

## 7. 纪律速查

- 长任务：`setsid ... 2>&1 | tee <log> &` 或 tmux；断点续跑（采集/评测已内建）。
- 不覆盖历史 checkpoint/数据/配置，只追加。
- 不改 `refs/`（只读）；本窗口不碰引擎 overlay 构建。
- 任何异常先看原始行数（C8）：`wc -l data/replays/<tag>/samples.jsonl` 与 `manifest.json` 指纹。
