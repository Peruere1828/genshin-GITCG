# 价值头真值测试（2026-11-21，0 新采集）

> 这是 D30 留下、`docs/failures/008` 与 E4（TD(λ)/GAE）都点名要过的前置闸门：
> **价值头到底有没有信号**。用现有 store 的 held-out 游戏直接量，不新采一局。

## 方法

- 数据：`data/store/r6_dvalin_sp2000.pt` 的 **val 划分**（不同种子产生，共 9,992 条；取前 6,000 条），
  标签 = 该局结果（我方视角 0/1），基率 **0.672**。
- 指标：Brier（越低越好）、相对常数预测器的 skill、**分位数校准 ECE**（按预测值排序切十档，比较
  平均预测 vs 实际胜率）。

## 结果

| checkpoint | Brier | 常数基线 Brier | skill | ECE（十分位） | 平均预测 |
|---|---:|---:|---:|---:|---:|
| `tg3_bc_e20_s0`（BC 臂） | **0.1565** | 0.2205 | **+29.0%** | **0.0235** | 0.673（标签 0.672） |
| `r7_gb50_lr2e5_e2_s0`（PG 臂） | 0.1588 | 0.2205 | +28.0% | 0.0603 | 0.732（标签 0.672） |

BC 臂的分档校准（预测 → 实际）：

```
0.14->0.12  0.38->0.42  0.50->0.57  0.59->0.57  0.66->0.64
0.75->0.75  0.84->0.83  0.90->0.89  0.96->0.94  1.00->0.97
```

## 结论

1. **价值头有真实信号，且 BC 阶段校准良好**（skill +29%，ECE 2.4%）：D30 的"价值能不能用"这一问
   **答案为能**（在公开观测 + 本 pool 上），TD(λ)/GAE 的前置条件满足。
2. **PG 训练会让价值头变乐观**（mean_pred 0.732 vs 实际 0.672，ECE 升到 6.0%）：作为 baseline 会
   系统性低估优势（`adv = G − V` 偏负）。因此 E4 必须：
   - 价值头与策略**分开拟合**（冻结 critic / 单独 value 阶段 / 价值用 BC 数据重拟合），或
   - 至少在 PG 后做一次 value-only 校准（brier/ECE 门禁 <5%）；
   - 这条同时解释了 D31 里 `r7` 的 Δ 只有 +0.031：优势被乐观 baseline 压缩了约 6pp。
3. 对特权 critic 的含义：形式不是"1022 维直方图"，但**基础价值信号确实存在**；
   若重开 PTIE，应先在这套校准口径上证明"结构化特权特征把 ECE 从 6% 降回去"，而不是看 val MSE。

## 复现

```bash
python - <<'PY'
import torch, dataclasses
from train.model import load_cvpn_model
import scripts.train_bc_scale as T
st = torch.load('data/store/r6_dvalin_sp2000.pt', map_location='cpu', weights_only=False)
val = st['val']; f = {x.name for x in dataclasses.fields(val)}
sub = dataclasses.replace(val, **{k: getattr(val,k)[:6000] for k in f
                                  if isinstance(getattr(val,k), list) and len(getattr(val,k))>6000})
labels = torch.tensor([float(v) for v in sub.value]); n=len(labels)
m = load_cvpn_model('data/checkpoints/tg3_bc_e20_s0.pt').eval()
p=[]
with torch.no_grad():
    for s in range(0,n,512):
        i=list(range(s,min(s+512,n)))
        tok,tm,opt,om,_,_ = T._collate(sub,i,'cpu'); p.append(m(tok,tm,opt,om)[1])
p=torch.cat(p).clamp(0,1)
print('Brier', float(((p-labels)**2).mean()), 'base', float(labels.mean()))
PY
```
