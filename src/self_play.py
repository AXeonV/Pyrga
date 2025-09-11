from __future__ import annotations
import argparse
import numpy as np
import torch
from tqdm import trange
from .game import GameState
from .model import PolicyValueNet
from .mcts import MCTS
from .config import MCTS_SIMS, TEMPERATURE


def play_game(net, device="cpu", mcts_sims=MCTS_SIMS):
	mcts = MCTS(net, device=device, add_root_noise=True)
	states = []
	policies = []
	players = []
	st = GameState()
	while not st.game_over():
		pi, root = mcts.run(st, sims=mcts_sims)
		# temperature sampling
		if TEMPERATURE > 0:
			probs = pi ** (1.0 / TEMPERATURE)
			probs /= probs.sum() + 1e-8
			a = int(np.random.choice(len(probs), p=probs))
		else:
			a = int(pi.argmax())
		states.append(st.obs())
		policies.append(pi)
		players.append(st.player)
		st.apply(a)
	z = st.result()
	# assign outcomes from each state's player perspective
	returns = np.array([z if p == 0 else -z for p in players], dtype=np.float32)
	return np.array(states, dtype=np.float32), np.array(policies, dtype=np.float32), returns


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--games", type=int, default=10)
	ap.add_argument("--mcts-sims", type=int, default=MCTS_SIMS)
	ap.add_argument("--device", type=str, default="cuda")
	ap.add_argument("--out", type=str, default="data/sp.npz")
	args = ap.parse_args()

	if args.device.startswith("cuda") and not torch.cuda.is_available():
		raise RuntimeError("CUDA device requested but not available. Please install a CUDA-enabled PyTorch build or set --device cpu.")

	net = PolicyValueNet().to(args.device)
	net.eval()

	all_s, all_p, all_z = [], [], []
	for _ in trange(args.games, desc="self-play"):
		s, p, z = play_game(net, device=args.device, mcts_sims=args.mcts_sims)
		all_s.append(s); all_p.append(p); all_z.append(z)
	S = np.concatenate(all_s, axis=0)
	P = np.concatenate(all_p, axis=0)
	Z = np.concatenate(all_z, axis=0)
	np.savez_compressed(args.out, s=S, p=P, z=Z)
	print(f"Saved to {args.out}: states={len(S)}")

if __name__ == "__main__":
	main()
