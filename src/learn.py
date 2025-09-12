from __future__ import annotations
import argparse
import os
import numpy as np
import torch
from tqdm import trange

from src.model import PolicyValueNet
from src.self_play import play_game
from src.mcts import MCTS
from src.game import GameState
from src.config import MCTS_SIMS, NUM_ACTIONS

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def train_once(net: PolicyValueNet, S: np.ndarray, P: np.ndarray, Z: np.ndarray, device: str,
			   epochs: int = 5, batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-4):
	ds = TensorDataset(torch.from_numpy(S), torch.from_numpy(P), torch.from_numpy(Z).float())
	dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
	net = net.to(device)
	net.train()
	opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
	scaler = torch.cuda.amp.GradScaler(device_type='cuda', enabled=device.startswith('cuda'))
	ce = nn.CrossEntropyLoss()
	mse = nn.MSELoss()
	for _ in range(epochs):
		for s, p, z in dl:
			s = s.to(device); p = p.to(device); z = z.to(device)
			with torch.cuda.amp.autocast(device_type='cuda', enabled=device.startswith('cuda')):
				logits, v = net(s)
				log_probs = torch.log_softmax(logits, dim=-1)
				policy_loss = -(p * log_probs).sum(dim=-1).mean()
				value_loss = mse(v, z)
				loss = policy_loss + value_loss
			opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
	net.eval()
	return net


@torch.no_grad()
def arena_win_rate(net_new: PolicyValueNet, net_old: PolicyValueNet, games: int, sims: int, device: str) -> float:
	# deterministic evaluation: no root noise, temperature -> choose argmax
	wins = 0
	for g in trange(games, desc="arena"):
		st = GameState()
		players = [net_new, net_old] if (g % 2 == 0) else [net_old, net_new]
		add_root_noise = False
		while not st.game_over():
			net = players[st.player]
			mcts = MCTS(net, device=device, add_root_noise=add_root_noise)
			pi, _ = mcts.run(st, sims=sims)
			a = int(pi.argmax())
			st.apply(a)
		z = st.result()  # +1 if player 0 wins
		# If net_new was player 0 in this game (g%2==0), its score is z; else -z
		score = z if (g % 2 == 0) else -z
		if score > 0:
			wins += 1
		elif score == 0:
			wins += 0.5
	return wins / games


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--device", type=str, default="cuda")
	ap.add_argument("--iters", type=int, default=5)
	ap.add_argument("--games-per-iter", type=int, default=100)
	ap.add_argument("--mcts-sims-sp", type=int, default=MCTS_SIMS)
	ap.add_argument("--mcts-sims-eval", type=int, default=400)
	ap.add_argument("--epochs", type=int, default=5)
	ap.add_argument("--batch-size", type=int, default=256)
	ap.add_argument("--eval-games", type=int, default=40)
	ap.add_argument("--accept-rate", type=float, default=0.55)
	ap.add_argument("--save-dir", type=str, default="ckpt")
	ap.add_argument("--data-dir", type=str, default="data")
	args = ap.parse_args()

	if args.device.startswith("cuda") and not torch.cuda.is_available():
		raise RuntimeError("CUDA requested but not available. Install CUDA PyTorch or set --device cpu.")

	os.makedirs(args.save_dir, exist_ok=True)
	os.makedirs(args.data_dir, exist_ok=True)

	# Initialize best network (random at start)
	best = PolicyValueNet().to(args.device)
	best.eval()
	torch.save(best.state_dict(), os.path.join(args.save_dir, "best.pt"))

	for it in range(1, args.iters + 1):
		# 1) Self-play with current best to generate dataset
		all_s, all_p, all_z = [], [], []
		for _ in trange(args.games_per_iter, desc=f"self-play it={it}"):
			s, p, z = play_game(best, device=args.device, mcts_sims=args.mcts_sims_sp)
			all_s.append(s); all_p.append(p); all_z.append(z)
		S = np.concatenate(all_s, axis=0)
		P = np.concatenate(all_p, axis=0)
		Z = np.concatenate(all_z, axis=0)
		np.savez_compressed(os.path.join(args.data_dir, f"sp_it{it}.npz"), s=S, p=P, z=Z)

		# 2) Train a new network starting from best weights
		new_net = PolicyValueNet().to(args.device)
		new_net.load_state_dict(best.state_dict())
		new_net = train_once(new_net, S, P, Z, device=args.device, epochs=args.epochs, batch_size=args.batch_size)

		# 3) Evaluate new vs best (arena) and gate
		wr = arena_win_rate(new_net, best, games=args.eval_games, sims=args.mcts_sims_eval, device=args.device)
		print(f"Iteration {it}: win-rate(new vs best) = {wr:.3f}")
		if wr >= args.accept_rate:
			best = new_net
			best.eval()
			torch.save(best.state_dict(), os.path.join(args.save_dir, "best.pt"))
			torch.save(best.state_dict(), os.path.join(args.save_dir, f"best_it{it}.pt"))
			print("Accepted new model.")
		else:
			print("Rejected new model; keep previous best.")


if __name__ == "__main__":
	main()
