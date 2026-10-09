# reports — 评测曲线、复盘报告、hypothesis 台账（小体积文件入库）

## 结构

- `reports/arena/<run_id>.json`：一次 arena 运行的汇总（contestant 总胜率 + 每对局矩阵 + Elo + 元数据）。
- `reports/arena/<run_id>.pairings.csv`：每对局一行（含 Wilson CI），便于画图/导入表格。
- `reports/benchmarks/bench_<ts>.json`：吞吐基准（games/h/core、decisions/s/core、engine.lock 元数据）。
- `reports/engine/probe_snapshot_<ts>.json`：引擎桥快照实测（往返无损/克隆确定/活体复现率；L5.4、I9）。
- `reports/LOCAL_BASELINE.md`：本地先行（L0/L1）实测基线与验收换算。

原始逐局记录（大体积）写入 `data/arena/<run_id>.jsonl`（gitignore，不入库）。

命令：
```bash
python -m eval.arena --smoke --seeds 3 --workers 12
python -m eval.arena --seeds 50 --workers 12          # 全 20 套对手矩阵，隔夜
python -m envs.benchmark --games 12 --workers 1       # 单核吞吐基线
```
