# Pyrga

Lightweight AlphaZero-style pipeline for a custom 4×4 stacking / tower control game: self-play generation, PUCT MCTS (with Dirichlet noise), policy–value network, supervised updates, iterative gating.

## Rules

- Board: 4×4 (16 cells).
- Pieces per player: 5 squares, 5 circles, 5 arrows (arrow direction chosen on placement: up/right/down/left).
- Placement constraints (depend on previous move):
  - Previous was a square at (r,c): next move must go to one of its 4 orthogonal neighbours.
  - Previous was an arrow at (r,c) with direction d: next move must lie anywhere along the ray from (r,c) in direction d (inclusive) until the edge.
  - Previous was a circle at (r,c): next move must (if still legal) be on the same cell.
  - Fallback: if no legal cell under these constraints, you may place on any empty cell (with zero pieces). If none exist the game ends.
- Cell constraints: each cell holds at most 3 pieces; piece types are unique within a cell (≤1 square, ≤1 circle, ≤1 arrow).
- Tower & scoring: when a cell reaches 3 pieces it becomes a tower. Ownership: player with strictly more pieces there (2–1 or 3–0). Final score = number of owned towers. Outcome z ∈ {+1, 0, -1} from the current player’s perspective.
- Action encoding (96 total):
  - 0–15: place square on cell i
  - 16–31: place circle on cell i
  - 32–95: place arrow on cell i; direction = (a−32) % 4 (0 up, 1 right, 2 down, 3 left)

## Engine

Core learning triple: (s, π, z).

1. Observation: stacked planes (own/opponent occupancy per type, arrow direction one-hot, remaining piece counts, side-to-move).
2. Network (PolicyValueNet): light residual CNN → 96 policy logits + scalar value v ∈ [−1,1].
3. MCTS: PUCT selection Q+U; root Dirichlet noise for exploration; illegal actions masked then renormalised; value signs flipped up the path.
4. Self-Play: run N simulations per move, convert visit counts to π; use temperature sampling for early moves then argmax; game end produces z.
5. Training (train.py / learn.py): minimise L = CE(policy_logits, π_target) + MSE(v, z); AdamW + optional AMP; iterative mode adds replay buffer + arena gating (accept if new model win rate ≥ threshold).
6. Stability: strict legality masking, temperature cooling, replay buffer balancing distribution shift.

Technical notes:
- PUCT: U ∝ P[a] * sqrt(N_total) / (1 + N[a]) balancing exploration–exploitation.
- Dirichlet: root prior perturbation avoids early policy collapse.
- Value sign flip: zero-sum perspective alignment.
- AMP: reduces memory + latency; falls back cleanly on CPU or when disabled.

## Files

```
src/
  game.py        # Rules & state transitions
  mcts.py        # PUCT search
  model.py       # Policy-value network
  self_play.py   # Self-play generation (s, π, z)
  train.py       # Supervised training on NPZ data
  learn.py       # Iterative loop: self-play → train → arena → gating
  config.py      # Constants / default hyperparams
tests/           # Rule & smoke tests
data/            # Generated data & snapshots
ckpt/            # Model weights
```

### Quick
Generate self-play data:
```bash
python -m src.self_play --games 50 --mcts-sims 200 --out data/sp.npz --temperature 1.0 --temp-moves 8
```
Supervised training (single dataset):
```bash
python -m src.train --data data/sp.npz --epochs 5 --batch-size 256 --lr 1e-3 --seed 123
```
Iterative self-improvement:
```bash
python -m src.learn --iters 5 --games-per-iter 80 --mcts-sims-sp 200 \
  --eval-games 40 --accept-rate 0.55 --temperature 1.0 --temp-moves 8
```
Disable AMP: add `--no-amp`. Use CPU: `--device cpu`.

### Data Format
NPZ file fields:
- `s`: float32 (N, C, 4, 4)
- `p`: float32 (N, 96)
- `z`: float32 (N,)

## Changelog

High-impact implemented changes (relative to initial baseline):
1. Unified AMP handling: single `autocast` + `GradScaler` pattern across `train.py` and `learn.py` with `--no-amp` switch.
2. Added reproducibility controls: `--seed` (covers Python, NumPy, Torch, CUDA determinism best-effort).
3. Extended CLI hyperparameters: learning rate, weight decay, clip norm, temperature scheduling (`--temperature`, `--temp-moves`), Dirichlet toggle (`--no-dirichlet`).
4. Self-play enhancements: early-move temperature sampling; optional root Dirichlet suppression; legal mask strictness clarified.
5. Replay buffer + gating: iterative loop (`learn.py`) with arena evaluation and accept-rate threshold.
6. Code simplification: removed interim utility module; consolidated training logic; streamlined `train_once`.
7. Policy/Value loss structure: explicit CE + MSE with clear logging points; ready for auxiliary heads.
8. Dataset specification: consistent NPZ schema (`s,p,z`) documented; easy future augmentation hook.
9. Documentation overhaul: concise English README + formalised rules, engine, file map, future roadmap.
10. Safety & stability: gradient scaling, deterministic seeds, explicit illegal action masking, temperature decay.

## Future

Potential extensions (roughly ascending sophistication):
1. 8-fold symmetry augmentation (rotations / reflections).
2. Replay sampling strategies: stochastic or prioritized (PER).
3. Policy regularisation: KL to previous policy or temperature ramps.
4. Auxiliary heads: tower ownership or remaining-move prediction.
5. Search efficiency: root reuse / batched GPU inference / partial tree persistence.
6. Distributed self-play: multi-process or multi-node with parameter server.
7. Advanced search tuning: dynamic c_puct, progressive widening.
8. Evaluation suite: Elo tracking, long-horizon stability, symmetry consistency checks.
9. Network scaling: deeper residual stacks, Squeeze-Excitation / attention, mixed-head designs.
10. Reliability: NaN/Inf watchdog & gradient explosion fuse.

PRs / experiments are welcome.

---
Concise, reproducible, extensible. Have fun. 🧠
