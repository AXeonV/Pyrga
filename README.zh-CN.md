# Pyrga

轻量级 AlphaZero 风格管线：面向自定义 4×4 叠放夺塔博弈的自博生成、PUCT MCTS（含 Dirichlet 根噪声）、策略-价值残差网络、监督更新、迭代 Arena 筛选。

## Rules

- 棋盘：4×4（16 格）。
- 每方棋子：5×square，5×circle，5×arrow（放置时选择方向：上/右/下/左）。
- 受上一手影响的放置约束：
  - 上一手为 square：下一手须落在其四个正交相邻格之一。
  - 上一手为 arrow(方向 d)：下一手须在该格沿 d 方向射线（含起点后的全部格，直到边界）任一格。
  - 上一手为 circle：下一手（若仍合法）须继续落在同一格。
  - 回退：若按上述约束无合法格，则可在任意“空格”（当前无任何棋子）放置；若再无空格则游戏结束。
- 格约束：每格最多容纳 3 枚棋子；同一格内每种类型至多 1（类型唯一）。
- 塔与计分：格子达到 3 枚即成“塔”，归属为该格己方放置数多的一方（2–1 或 3–0）。终局比分=己方拥有塔数量，胜负值 z ∈ {+1,0,-1}（从当前玩家视角）。
- 动作编码（共 96）：
  - 0–15：在格 i 放置 square
  - 16–31：在格 i 放置 circle
  - 32–95：在格 i 放置 arrow，方向 = (a−32) % 4（0上 1右 2下 3左）

## Engine

核心训练三元组：(s, π, z)。

1. 观测 (Observation)：通道堆叠（己/敌各类型占位、箭头方向 one-hot、剩余棋子计数、轮到方）。
2. 网络 (PolicyValueNet)：轻量残差 CNN → 输出 96 维策略 logits + 标量价值 v ∈ [-1,1]。
3. 搜索 (MCTS)：PUCT 选择 Q+U；根节点注入 Dirichlet 噪声；扩展时屏蔽非法并重新归一；沿父链回传时交替翻转价值号。
4. 自博 (Self-Play)：每步运行若干模拟，访问计数归一为 π；前若干步按温度采样，后期直接 argmax；对局结束得到 z。
5. 训练 (train.py / learn.py)：最小化 L = CE(策略, π_target) + MSE(v, z)；使用 AdamW + 可选 AMP；迭代版本包含重放缓冲与 Arena 胜率阈值门控。
6. 稳定性：严格合法动作掩码；温度退火；重放缓冲缓解分布漂移。

技术要点：
- PUCT：U ∝ P[a] * sqrt(N_total) / (1 + N[a])，平衡探索/利用。
- Dirichlet：根先验扰动防止早期策略塌缩。
- 价值符号翻转：适配零和、轮换玩家视角。
- AMP：降低显存占用，若关闭或在 CPU 上自动回退。

## Files

```
src/
  game.py        # 规则与状态转换
  mcts.py        # PUCT 搜索
  model.py       # 策略-价值网络
  self_play.py   # 生成 (s, π, z) 数据
  train.py       # NPZ 数据监督训练
  learn.py       # 自博→训练→Arena→门控 循环
  config.py      # 常量 / 默认超参
tests/           # 规则 & 冒烟测试
data/            # 数据与快照
ckpt/            # 模型权重
```

### Quick
生成自博数据：
```bash
python -m src.self_play --games 50 --mcts-sims 200 --out data/sp.npz --temperature 1.0 --temp-moves 8
```
监督训练（单批）：
```bash
python -m src.train --data data/sp.npz --epochs 5 --batch-size 256 --lr 1e-3 --seed 123
```
迭代自提升：
```bash
python -m src.learn --iters 5 --games-per-iter 80 --mcts-sims-sp 200 \
  --eval-games 40 --accept-rate 0.55 --temperature 1.0 --temp-moves 8
```
禁用 AMP：`--no-amp`；使用 CPU：`--device cpu`。

### Data Format
NPZ 字段：
- `s`: float32 (N, C, 4, 4)
- `p`: float32 (N, 96)
- `z`: float32 (N,)

## Changelog

相较初始基线的主要实现改动：
1. 统一 AMP：单一 `autocast` + `GradScaler` 模式，提供 `--no-amp` 关闭开关。
2. 可复现性：新增 `--seed`（同步 Python / NumPy / Torch / CUDA 设定，尽力确定性）。
3. CLI 扩展：学习率、权重衰减、梯度裁剪、温度调度(`--temperature`/`--temp-moves`)、Dirichlet 关闭(`--no-dirichlet`)。
4. 自博增强：早期温度采样；可选择禁止根 Dirichlet；更严格合法掩码。
5. 重放缓冲 + Gating：`learn.py` 引入 Arena 胜率阈值模型替换机制。
6. 代码瘦身：移除临时工具模块；合并训练逻辑；精简 `train_once`。
7. 损失结构明确：CE + MSE，便于后续加辅助头。
8. 数据格式固定：`(s,p,z)` NPZ 规范，留扩展入口。
9. 文档重构：规则 / 引擎 / 结构 / 未来路线清晰化。
10. 稳定措施：梯度缩放、随机种子、非法动作显式屏蔽、温度退火。

## Future

潜在扩展（复杂度递增）：
1. 8 重对称数据增强（旋转/翻转）。
2. 重放采样策略：随机/优先级 (PER)。
3. 策略正则：与旧策略 KL 或温度退火组合。
4. 辅助头：塔归属 / 剩余步数预测。
5. 搜索效率：根重用 / 批量推理 / 部分树持久化。
6. 分布式：多进程/多机自博 + 参数服务器。
7. 高级搜索调参：动态 c_puct / Progressive Widening。
8. 评估体系：Elo 追踪、长期稳定性、对称一致性。
9. 网络扩展：更深残差、SE / Attention、混合多头结构。
10. 可靠性：NaN/Inf 监控与梯度爆炸熔断。

欢迎 PR 与实验衍生。

---
简洁 · 可复现 · 可扩展。玩得开心。🧠
