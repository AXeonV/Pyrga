from __future__ import annotations
import argparse
import numpy as np
import torch
from tqdm import trange
from .game import GameState
from .model import PolicyValueNet
from .mcts import MCTS
from .config import MCTS_SIMS, TEMPERATURE
import os, random


def _set_seed(seed: int | None):
	if seed is None: return
	random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)

def play_game(
	net,
	device: str = "cuda",
	mcts_sims: int = MCTS_SIMS,
	temperature: float = TEMPERATURE,
	temp_moves: int = 8,
	add_root_noise: bool = True,
):
	mcts = MCTS(net, device=device, add_root_noise=add_root_noise)
	states: list[np.ndarray] = []
	policies: list[np.ndarray] = []
	players: list[int] = []
	st = GameState()
	move_index = 0
	while not st.game_over():
		pi, _ = mcts.run(st, sims=mcts_sims)
		if temperature > 0 and move_index < temp_moves:
			probs = pi ** (1.0 / max(1e-6, temperature))
			probs /= probs.sum() + 1e-8
			a = int(np.random.choice(len(probs), p=probs))
		else:
			a = int(pi.argmax())
		states.append(st.obs())
		policies.append(pi)
		players.append(st.player)
		st.apply(a)
		move_index += 1
	z = st.result()
	returns = np.array([z if p == 0 else -z for p in players], dtype=np.float32)
	return np.array(states, dtype=np.float32), np.array(policies, dtype=np.float32), returns


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--games", type=int, default=10)
	ap.add_argument("--mcts-sims", type=int, default=MCTS_SIMS)
	ap.add_argument("--device", type=str, default="cuda")
	ap.add_argument("--out", type=str, default="data/sp.npz")
	ap.add_argument("--temperature", type=float, default=TEMPERATURE, help="初期温度(>0采样)；与 --temp-moves 联合使用")
	ap.add_argument("--temp-moves", type=int, default=8, help="前多少步使用温度采样")
	ap.add_argument("--seed", type=int, default=None)
	ap.add_argument("--no-dirichlet", action="store_true", help="关闭根节点Dirichlet噪声")
	args = ap.parse_args()

	if args.device.startswith("cuda") and not torch.cuda.is_available():
		raise RuntimeError("CUDA device requested but not available. Please install a CUDA-enabled PyTorch build or set --device cpu.")

	_set_seed(args.seed)
	net = PolicyValueNet().to(args.device); net.eval()

	all_s, all_p, all_z = [], [], []
	for _ in trange(args.games, desc="self-play"):
		s, p, z = play_game(net, device=args.device, mcts_sims=args.mcts_sims,
			temperature=args.temperature, temp_moves=args.temp_moves, add_root_noise=not args.no_dirichlet)
		all_s.append(s); all_p.append(p); all_z.append(z)
	S = np.concatenate(all_s, axis=0)
	P = np.concatenate(all_p, axis=0)
	Z = np.concatenate(all_z, axis=0)
	np.savez_compressed(args.out, s=S, p=P, z=Z)
	print(f"Saved to {args.out}: states={len(S)}")

if __name__ == "__main__":
	main()
