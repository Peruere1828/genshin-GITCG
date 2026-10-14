# A800 机会窗口 Runbook（GPU 为主；CPU 配额 ~10 核，D21）

> 对象：**A800 80G 机会资源**（申请制、时间受限）的操作执行；非南大 LSF 集群（见同目录 `README.md`）。
> 纪律（D14）：**无常驻服务**、**一切长任务可断点续跑**、长作业挂 `tmux/setsid` 流式落盘。
> **窗口结果只回传聚合报告 + checkpoint**；原始 replay 留本机/WSL（不入库）。

> ⚠️ **前提（D21，2026-10-12 实测）**：窗口**只有 GPU 可用，CPU 配额仅 ~10 核**（与本机 12 核同级）。
> 因此 **CPU 密集的采集/评测不放窗口**——A800 **只做 GPU 训练/推理**。
> 数据流固定为：**本机/WSL `train.collect` 采集 → 数据传入 A800 → `train.pipeline --replays/--distill` GPU 训练 → checkpoint 回流本机 → 本机 `scripts/eval_protocol.py` 认证**。
> （若窗口实际能申请到更多 CPU，回来更新本 runbook 与 PLAN D21。）

---

## 0. 环境准备（一次性）

```bash
git clone git@github.com:Peruere1828/genshin-GITCG.git && cd genshin-GITCG
git rev-parse HEAD                      # 记录 commit hash
conda create -y -n gitcg python=3.12 && conda activate gitcg
pip install -e ".[train,dev]"           # gitcg>=0.21.0 + torch(CUDA) + pytest
python -c "import gitcg, torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest -m "not slow" -q       # 应 ~186 passed
```
- **资产元数据**：首次跑对局会联网拉卡牌元数据（`gi-tcg-assets-api-hf.guyutongxue.site`）；无外网时从本机拷
  `data/assets/expert_system_assets_chs_latest.json`，或设 `GITCG_ASSETS_OFFLINE=1` + `GITCG_ASSETS_CACHE=<path>`。
- **记录**：GPU 型号/显存、torch/cuda、`nproc`（预期 ~10）、commit hash、engine lock。

---

## 1. GPU 基准（先做，~30 min）

```bash
python scripts/bench_ops.py --device cuda       # 算子 + CVPN 前向/训练吞吐 → reports/explore/*.json
python scripts/bench_device.py --devices cuda   # CVPN 训练吞吐
```
- **重点观测**：在只有 ~10 核 CPU 时，**训练 dataloader 是否成瓶颈**（GPU 利用率）。若瓶颈明显，优先用
  `--precollate`（一次性 pad/collate 后按设备切片），并控制 DataLoader 进程数别超过可用核。
- 回传 `reports/explore/*.json` + `nvidia-smi`/日志要点。

---

## 2. 数据准备（在**本机/WSL**做，再传入 A800；不在窗口采集）

**本地采集**（可续跑、指纹护栏；长作业 `setsid`）：
```bash
# (a) BC 冷启动语料（便宜：expert:<deck> 当 teacher）
for d in superconduct_aggro unyielding_geo double_geo_navia dvalin_bonk; do
  python -m train.collect --deck "$d" --all-opponents --seed-count 500 --workers 12 --tag "bc_$d"
done
# (b) 搜索 teacher 语料（贵：MCCFR；用预算参数控成本）
python -m train.collect --deck superconduct_aggro --all-opponents --search \
    --iterations 8 --depth 1 --candidate-limit 4 --max-searches 8 --belief-samples 2 \
    --seed-start 0 --seed-count 100 --workers 12 --tag search_distill
```

**传输到 A800**（记录大小；热数据可压缩）：
```bash
tar -C data/replays -czf /tmp/replays.tgz bc_* search_distill   # 或 rsync -avz
scp /tmp/replays.tgz <a800>:~/genshin-GITCG/data/replays/ && ssh <a800> 'cd ~/genshin-GITCG/data/replays && tar xzf replays.tgz'
```

---

## 3. GPU 训练放大（A800 首要任务）

```bash
# (a) 蒸馏训练（S-C 主目标：policy CE + value 回归 + belief 评分；d_model 放大）
python -m train.pipeline --distill --replays data/replays/search_distill data/replays/bc_* \
    --device cuda --d-model 256 --epochs 60 --tag a800_distill
# (b) 数据量 × 网络规模 scale 关系（M3 用）
python -m train.learning_curve --replays data/replays/bc_* \
    --sizes 4000 16000 64000 --d-models 128 256 512 --device cuda --precollate --epochs 40
```
- 80G 显存充足：可上 d_model 256–512、4–8 层、bf16（以 §1 结论为准）。
- **~10 核注意**：用 `--precollate` 或小 batch + 少量 DataLoader worker；随时看 GPU 利用率，别让 CPU 喂不饱卡。
- 产出 checkpoint（`data/checkpoints/*.pt`）**只追加**；训练曲线/报告落 `reports/train/`。

---

## 4. 认证（**在本机**做；不在窗口）

```bash
# M3-CPU/M2 口径：纯策略 vs 池内全部对手（配对种子 + cluster CI）
python -m scripts.eval_protocol --contestant neural:data/checkpoints/<回流ckpt>.pt \
    --seeds 50 --workers 12 --strict-ci --pooled-threshold 0.50 --tag a3_eval
# 合并门禁：候选 vs 上一 checkpoint 的配对 A/B 非劣
python -m scripts.eval_protocol --contestant neural:<new>.pt --baseline neural:<old>.pt --seeds 20 --workers 12 --tag a3_ab
```

---

## 5. 搜索蒸馏（采集在本地、训练在 A800）

见 §2(b) 采集（本地，`--search`）→ §3(a) 蒸馏（A800）。
- 搜索 teacher 数据是 M3 蒸馏的关键；`--search` 的 spec 指纹护栏会随参数变化，**换参数请换 `--tag`**。
- 预算：`--max-searches/--iterations/--depth` 控制每局搜索成本；先小样本量吞吐再放量。

---

## 6. 回传清单（窗口结束前）

- `reports/`（benchmark / learning_curve / train 小报告，聚合）；
- candidate **checkpoint**（`data/checkpoints/*.pt`，scp 回流；**不入库**）；
- 每步**命令 + commit hash + GPU/机器信息**；
- 原始 replay / 大数据留本机或 A800 存储，不入库。

## 7. 纪律速查

- 长任务：`setsid ... 2>&1 | tee <log> &` 或 tmux；训练可断点续跑。
- 不覆盖历史 checkpoint/数据/配置，只追加。
- 不改 `refs/`（只读）；不碰引擎 overlay 构建（那是本机 B 线的活）。
- 异常先核对原始行数/指纹（C8）：`wc -l data/replays/<tag>/samples.jsonl`、`manifest.json`。
