# Pyrga

## 4x4 Stacking Placement Game — AlphaZero/KataGo Style (PyTorch)

This repository implements a compact AlphaZero/KataGo-style pipeline for a custom 4×4 stacking placement game. It focuses on the essential ingredients: a policy–value network, PUCT Monte Carlo Tree Search (MCTS) with root Dirichlet noise, self-play data generation, and a simple training loop.

## Game definition

- Board: 4×4 grid.
- Pieces (per player): 5 squares, 5 circles, 5 arrows. Arrow is a single piece type whose direction (up/right/down/left) is chosen at placement.
- Previous-move constraints:
  - If the previous move placed a square at (r,c), the next move must be placed in one of its 4 orthogonal neighbors (subject to legality).
  - If the previous move placed an arrow at (r,c) pointing direction d, the next move must be placed on any cell along the ray from (r,c) in direction d, up to the board edge (subject to legality).
  - If the previous move placed a circle at (r,c), the next move must be placed on the same cell (subject to legality).
  - Fallback: if no legal move exists under these constraints, the player may place on any completely empty cell. If no empty cells remain, the game ends.
- Cell capacity and uniqueness: each cell holds at most 3 pieces total, and piece types within a cell must be unique (at most one square, one circle, one arrow).
- Tower and scoring: a cell with 3 pieces becomes a “tower”. The tower is owned by the side who placed strictly more pieces in that cell (2–1 or 3–0). Final score is the number of owned towers.

## Action encoding (96 actions)

- 0..15: place a square on cell i (i ∈ [0,15])
- 16..31: place a circle on cell i
- 32..95: place an arrow on cell i with direction d = (a−32) % 4; 0: up, 1: right, 2: down, 3: left

Legal moves are strictly masked by the previous-move constraints, cell capacity/type uniqueness, and remaining pieces. If no legal actions exist, the fallback to “any empty cell” is applied.

## AlphaZero/KataGo mapping in this codebase

- Policy–value network (ResNet): `src/model.py`
  - A lightweight residual CNN consumes stacked planes (piece presence, arrow direction one-hot, remaining counts, side-to-move) and outputs policy logits over the 96 actions and a scalar value in [−1,1].

- PUCT MCTS with root Dirichlet noise: `src/mcts.py`
  - Selection: argmax over Q + U, where U = c_puct × P[a] × sqrt(∑N) / (1 + N[a]).
  - Expansion: network provides policy priors P (masked to legal moves and normalized) and the leaf value v.
  - Root exploration: inject Dirichlet noise at the root (α and ε configurable) to improve early-game diversity.
  - Backup: alternate signs of v up the path and update W/N/Q per edge.

- Self-play data generation: `src/self_play.py`
  - For each state, run MCTS and collect the normalized visit counts as π (soft target for the policy), sample an action (temperature can be tuned), and continue until terminal.
  - Upon termination, compute z ∈ {−1,0,1} based on tower majority and assign it from each state’s player-to-move perspective.

- Training loop: `src/train.py`
  - Loss = policy cross-entropy on soft targets π + value MSE against z.
  - Mixed precision (AMP) enabled on CUDA by default. Optimizer: AdamW.
  - The trained network is the “expert” prior for subsequent self-play iterations.

## Architectural components

- `src/game.py`: complete rules, legal move generation (constraints + fallback), action encoding/decoding, terminal and scoring logic, observation planes.
- `src/model.py`: compact residual policy–value network.
- `src/mcts.py`: PUCT search with root Dirichlet noise.
- `src/self_play.py`: self-play rollouts to produce (s, π, z) triplets.
- `src/train.py`: supervised updates on π and z.
- `src/learn.py`: iterative orchestrator (self-play → train → arena evaluate → model gating), moving the system towards a full AlphaZero loop.
- `src/config.py`: board/action constants and default hyperparameters.

## Notes on design and efficiency

- Strict legal move masking keeps the policy focused and reduces wasted search.
- The 4×4 size enables fast experimentation; increase MCTS simulations or network width/depth for stronger play.
- CUDA is the default device for both self-play and training; switch to CPU by passing `--device cpu` if needed.

## Possible extensions (towards KataGo-style engineering)

- Self-play/Training orchestration loop with model gating (arena evaluation).
- Symmetry augmentation (8 board symmetries) to improve sample efficiency.
- Progressive widening or prior-based expansion to limit branching.
- Auxiliary heads (e.g., tower ownership prediction) for denser learning signals.
- Replay buffer with deduplication (e.g., Zobrist hashing) and temperature scheduling.
