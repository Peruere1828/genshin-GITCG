# reports/npc — NPC PvE 调试系统（WS7/D17）报告

聚合摘要目录（入库）；逐局原始 JSONL 在 `data/npc/`（gitignore）。

| 文件 | 内容 |
|---|---|
| `npc_pve_smoke_v1.json` | 冒烟评测聚合摘要（`npc.eval` 输出）：全 16 关 × 3 种子，胜率 + 三星达成率 + 逐目标达成率 |
| `npc_pve_smoke_v1.stages.csv` | 逐关一行的紧凑表（胜率 / 三星率 / 平均星数 / 平均回合） |
| `npc_pve_smoke_v1.md` | B2/B3 交付说明（本文件） |
| `pve_v1_gate.json` / `pve_v1_gate.md` | **PvE-v1 门禁（B5）**：decklab 池 C 组牌 × 16 关 × 24 种子（384 局）+ 组牌/打牌消融 + 门禁判定 |

复现（冒烟）：

```bash
python -m npc.eval --stages all --deck superconduct_aggro \
  --policy expert:superconduct_aggro --seeds 1,2,3 --workers 6 \
  --tag npc_pve_smoke --run-id npc_pve_smoke_v1
```

复现（门禁，可断点续跑）：

```bash
python -m scripts.run_npc_gate --stages all --seed-count 24 --workers 8 --tag pve_v1_gate
```

原始逐局记录写入 `data/npc/<run-id>.jsonl` / `data/npc/<tag>__<arm>.jsonl`（+ `.meta.json` 指纹护栏）；
重复同命令即断点续跑。
