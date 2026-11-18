# howto — NPC PvE 调试系统

> 数据权威：`configs/npc/README.md`；代码入口：`npc/README.md`；
> 报告：`reports/npc/pve_v1_gate.md`。定位：调试/训练辅助，独立 PvE 门禁。

## 范围

- 蒙德/璃月全量：**19 类敌人 = 17 个新增 `.gts` + 2 个 NPC 变体**；16 关。
- Fatui 火债处理人 / 镜女 使用独立 NPC 变体 `9301`（9 HP）/ `9202`（10 HP）；
  官方可玩卡 `2301`（11）/ `2202`（10）保持引擎原版，不再被 PvE 血量污染。
- `NpcPolicy`（greedy + rotation）；`NpcSpec`/三星判定；decklab 自动组牌。

## 门禁 PvE-v1

- decklab 组牌 × 16 关 × 多种子；三星达成率 ≥60%。
- 当前 PASS 74.7% 是**无关卡规则**口径；`npc/eval.py` 现已默认发送关卡规则，
  旧基线需重跑后再引用（spec 指纹已包含 `special_rules.json` 的 sha1）。

## 关卡特殊规则（冒险挑战）

16 关的「初始支援 + 开局效果」由 `configs/npc/special_rules.json` 权威定义
（10 关有初始支援、7 关有额外效果、3 关无规则；每条带 HoYoWiki 出处与逐字文本），
经 `npc/stage_rules.py` 生成 payload，由 `envs.match.run_match(..., stage_rules=<json>)`
写入引擎 `CreateParam` 字符串属性 `ATTR_CREATEPARAM_STAGE_RULES = 2002`：

```json
{"player": 0,
 "supports": [{"card_id": 322007, "removable": true}],
 "effects": [{"kind": "initial_energy", "value": 2, "target": "friendly_all"}]}
```

- 初始支援走正常支援牌打出路径（`on staged` 生效、计入 4 张上限）；
  `removable=false` 不出现在替换列表、不会被弃置。
- 效果只有两种引擎契约：`initial_energy`（开局充能）与
  `element_application`（行动阶段起点附元素）；`weapon_free_equip` /
  `artifact_free_equip` 由注入的支援牌自身卡面实现，不单独发送。
- 不传该属性时行为与补丁前逐字节一致（同种子初始 state 相同），旧对局不受影响。
- `npc/eval.py` 默认开启，`--no-stage-rules` 关闭。
- 引擎补丁与溯源：`engine-overlay/patches/stage-rules.patch`；
  overlay 全量文件在 `engine-overlay/overlay/packages/{core,cbinding}/`；
  构建 sha256 见 `configs/engine.lock` 的 `[overlay]`。

验证：

```bash
python -m npc.stage_rules --verify
python -m pytest tests/test_stage_rules.py tests/test_stage_rules_engine.py -q
```

## 命令

```bash
python -m scripts.run_npc_gate --seeds 3 --workers 8
python -m pytest tests/test_npc_*.py -q
```

## 已知差异

- 初始支援的 `removable` 为推断值（`removable_verified: false`，未做游戏内溢出实测）。
- 「多轮挑战」仍不在模拟范围（单局制）；15 回合上限是引擎默认
  `ATTR_STATE_CONFIG_MAX_ROUNDS_COUNT`（可设，`run_match` 未设置）。
- NPC 阵容人数不为 3 也原样保留（出战+预备）。
- 敌人数据改动走 `engine-overlay/`，见 `docs/howto/engine-overlay.md`。
