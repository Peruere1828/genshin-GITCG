# howto — 引擎 overlay 与构建

> 权威细节：`engine-overlay/README.md`、`engine-overlay/overlay/README.md`。

## 约束

- `refs/` 只读；不修改、不 apply 补丁。
- 我们的敌人 `.gts` / standalone 改动放 `engine-overlay/`（入库）；构建工作区放 `build/`（gitignore）。
- 引擎 pin `v0.21.0`；V8 11.6.189.4 纯 CMake；Docker 内一次性首建。
- `libgitcg.so` 只建一台（本机/WSL 通用）；改敌人数据走增量重链（分钟级）。

## 流程

1. 修改 `engine-overlay/overlay/` 下的数据/前端。
2. Docker 内构建/重链；产物复制到 `build/engine/`。
3. 用 `custom-data-loader` 免构建验证敌人数据，定稿后再重链。
4. 冒烟：跑一局 + 同种子重放一致 + `pytest -m "not slow"`。

## 敌人/关卡

- 数据权威：`configs/npc/README.md`（19 类 = 17 个新增 `.gts` + 2 个上游复用；16 关）。
- 实现：`engine-overlay/overlay/README.md`。
- PvE 入口：`npc/README.md`、`docs/howto/npc-pve.md`。
