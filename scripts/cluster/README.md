# 超算（南大，LSF）作业模板与说明（PLAN.md §5 / §10 step 7）

工程约束（D7）：**无常驻服务**、**一切长任务可断点续跑**、热数据在作业节点本地盘、归档走大容量存储。

## 目录

- `bootstrap.sh`：一次性环境准备（conda/venv + `gitcg` + `pytest` + `torch` + clone `refs/` + 拉取卡池资产缓存 +
  冒烟测试）。可重复执行；不触碰 `data/`（只追加）。
- `lsf/*.lsf`：LSF 切片作业模板（`bsub < xxx.lsf`）。均假设已 `source` 好环境或有 `$GITCG_PY`。

## 队列对应（PLAN.md §5）

| 用途 | 队列示例 | 说明 |
|------|----------|------|
| 自博弈 / arena（纯 CPU） | `cpu1`（小作业）、`7702ib`(128C)、`9242opa`(96C)、`6140ib`(72C)、`6330ib`(56C) | 主力；横向扩 rollout |
| CVPN 训练 / batched inference | `945090ib`(5090 32G)、`734090ib`/`75434090ib`(4090 24G)、`83a100ib`(A100 40G)、`62v100ib`(V100 32G) | 小网络+bf16 任意一张够 |

## 关键做法

- **海量作业模式**：公共共享队列不要长占；把大任务切成许多短作业（每个作业跑一批种子），
  结果 append 到各自 JSONL，天然断点续跑。
- **资产离线**：作业里设 `GITCG_ASSETS_OFFLINE=1`，资产缓存随代码同步或由 `bootstrap.sh` 预拉。
- **确定性**：`envs/match.py` 已保证同种子可复现（预洗牌 + codebook reset），跨节点一致（PYTHONHASHSEED 无关，已实测）。
- **GPU 不常驻**：`batched_inference`（待实现）按作业拉起/销毁；LLM 服务同理，客户端强制降级。
- **产物归属**：checkpoint/数据只追加；小报告回 `reports/` 用 git 同步，大 JSONL 留超算存储（不入库）。

## 示例

```bash
bsub < scripts/cluster/lsf/benchmark.lsf
bsub -J "arena[1-20]" < scripts/cluster/lsf/arena_array.lsf   # LSF 数组作业(视集群语法)
```
