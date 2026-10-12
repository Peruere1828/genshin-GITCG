# engine-overlay/overlay — 我们的引擎改动（入库）

本目录是**引擎仓根目录的镜像**：`overlay/` 下的相对路径会被
`scripts/apply-overlay.sh` 原样复制进 `build/engine/`（引擎源码树根）。
纪律（PLAN §6.5）：`refs/genius-invokation` 只读；我们的改动只放这里。

## B0 现状

B0（构建链 go/no-go）**overlay 为空**——本阶段只证明工具链能重建出与 wheel
0.21.0 ABI 一致、可被 conda `gitcg` dlopen、且现有测试全绿的 `libgitcg.so`。
我们的敌人 `.gts` 数据在 B1 阶段才开始注入。

## B1 起怎么放敌人数据（路径对照）

引擎的数据源在 `packages/data/src/`：

| 我们要加的 | 放到 overlay 的路径 |
|---|---|
| NPC 敌人角色 `.gts`（技能/被动/HP/充能） | `packages/data/src/characters/<element>/<enemy>.gts` |
| 注册新角色/卡牌 | `packages/data/src/index.ts`（或 `begin.ts`/`end.ts` 的导出聚合处） |
| 若需新增数据版本 | `packages/data/src/` 内的版本常量 + `packages/pybinding/src/gitcg/create_param.py` 的 `DATA_VERSION` 同步 |

参考现有敌人实现（如 `packages/data/src/characters/anemo/consecrated_flying_serpent.gts`）
的 `.gts` 语法；数据权威来源见 `configs/npc/npc_enemies_timaeus.json` 与
`reports/explore/npc_pve_doc.md`。

## 快速循环

改 overlay 后**无需重编 V8**，只需重跑 TS 构建 + 重链：

```sh
bash engine-overlay/scripts/apply-overlay.sh
V8_OFFLINE=1 SKIP_PNPM=0 bash engine-overlay/scripts/build-in-docker.sh
```

（`SKIP_MATERIALIZE=1` 复用 `build/engine`；`SKIP_PNPM=1` 连 TS 构建也跳过，
仅重链 C++。）
