# howto — NPC PvE 调试系统

> 数据权威：`configs/npc/README.md`；代码入口：`npc/README.md`；
> 报告：`reports/npc/pve_v1_gate.md`。定位：调试/训练辅助，独立 PvE 门禁。

## 范围

- 蒙德/璃月全量：**19 类敌人 = 17 个新增 `.gts` + 2 个上游复用**；16 关。
- `NpcPolicy`（greedy + rotation）；`NpcSpec`/三星判定；decklab 自动组牌。

## 门禁 PvE-v1

- decklab 组牌 × 16 关 × 多种子；三星达成率 ≥60%。
- 当前 PASS 74.7%；但支援区/多轮挑战与游戏内机制不一致，报告需带限定。

## 命令

```bash
python -m scripts.run_npc_gate --seeds 3 --workers 8
python -m pytest tests/test_npc_*.py -q
```

## 已知差异

- 支援区/多轮挑战的替代描述方式仍是 TODO。
- NPC 阵容人数不为 3 也原样保留（出战+预备）。
- 敌人数据改动走 `engine-overlay/`，见 `docs/howto/engine-overlay.md`。
