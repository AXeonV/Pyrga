from __future__ import annotations
import argparse, os, random, time
import numpy as np
import torch
from collections import deque
from tqdm import trange

from src.model import PolicyValueNet
from src.self_play import play_game
from src.mcts import MCTS
from src.game import GameState
from src.config import MCTS_SIMS, TEMPERATURE

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def _set_seed(seed: int | None):
	if seed is None:
		return
	random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)

def train_once(net: PolicyValueNet, S: np.ndarray, P: np.ndarray, Z: np.ndarray, device: str,
			   epochs: int = 5, batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-4,
			   clip_norm: float = 1.0, amp: bool = True):
	"""最小监督训练 (policy + value) 若干 epoch，与 train.py AMP 写法统一。"""
	ds = TensorDataset(torch.from_numpy(S), torch.from_numpy(P), torch.from_numpy(Z).float())
	dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
	net = net.to(device); net.train()
	opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
	use_amp = amp and device.startswith('cuda') and torch.cuda.is_available()
	if use_amp:
		from torch.cuda.amp import GradScaler, autocast  # type: ignore
		scaler = GradScaler()
	else:
		from contextlib import nullcontext
		autocast = nullcontext  # type: ignore
		class _DummyScaler:
			def scale(self, x): return x
			def step(self, opt): opt.step()
			def update(self): pass
			def unscale_(self, opt): pass
		scaler = _DummyScaler()  # type: ignore
	mse = nn.MSELoss()
	for _ in range(epochs):
		for s,p,z in dl:
			s=s.to(device); p=p.to(device); z=z.to(device)
			with autocast():
				logits,v = net(s)
				log_probs = torch.log_softmax(logits, dim=-1)
				policy_loss = -(p*log_probs).sum(dim=-1).mean()
				value_loss = mse(v,z)
				loss = policy_loss + value_loss
			opt.zero_grad(set_to_none=True)
			scaler.scale(loss).backward()
			if clip_norm and clip_norm>0:
				scaler.unscale_(opt)
				torch.nn.utils.clip_grad_norm_(net.parameters(), clip_norm)
			scaler.step(opt); scaler.update()
	net.eval(); return net


@torch.no_grad()
def arena_win_rate(net_new: PolicyValueNet, net_old: PolicyValueNet, games: int, sims: int, device: str) -> float:
	"""评估新旧模型胜率(和棋0.5)，无噪声，直接 argmax。"""
	wins = 0.0
	for g in trange(games, desc="arena"):
		st = GameState()
		order = (net_new, net_old) if g % 2 == 0 else (net_old, net_new)
		while not st.game_over():
			cur = order[st.player]
			pi,_ = MCTS(cur, device=device, add_root_noise=False).run(st, sims=sims)
			st.apply(int(pi.argmax()))
		z = st.result()
		score = z if g % 2 == 0 else -z
		wins += 1.0 if score>0 else (0.5 if score==0 else 0.0)
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
	ap.add_argument("--replay-size", type=int, default=50000, help="重放缓冲最大样本数")
	ap.add_argument("--seed", type=int, default=None)
	ap.add_argument("--temperature", type=float, default=TEMPERATURE)
	ap.add_argument("--temp-moves", type=int, default=8)
	ap.add_argument("--no-dirichlet", action="store_true")
	ap.add_argument("--clip-norm", type=float, default=1.0)
	ap.add_argument("--lr", type=float, default=1e-3)
	ap.add_argument("--weight-decay", type=float, default=1e-4)
	ap.add_argument("--no-amp", action="store_true", help="禁用 AMP")
	args = ap.parse_args()

	if args.device.startswith("cuda") and not torch.cuda.is_available():
		raise RuntimeError("CUDA requested but not available. Install CUDA PyTorch or set --device cpu.")

	os.makedirs(args.save_dir, exist_ok=True)
	os.makedirs(args.data_dir, exist_ok=True)

	_set_seed(args.seed)
	best = PolicyValueNet().to(args.device); best.eval()
	torch.save(best.state_dict(), os.path.join(args.save_dir, "best.pt"))
	replay_s, replay_p, replay_z = deque(), deque(), deque(); cap = args.replay_size

	for it in range(1, args.iters + 1):
		steps = 0
		for _ in trange(args.games_per_iter, desc=f"self-play it={it}"):
			s, p, z = play_game(
				best,
				device=args.device,
				mcts_sims=args.mcts_sims_sp,
				temperature=args.temperature,
				temp_moves=args.temp_moves,
				add_root_noise=not args.no_dirichlet,
			)
			for a_s, a_p, a_z in zip(s, p, z):
				replay_s.append(a_s); replay_p.append(a_p); replay_z.append(a_z)
				steps += 1
				if len(replay_s) > cap:
					replay_s.popleft(); replay_p.popleft(); replay_z.popleft()
		print(f"[iter {it}] generated {steps} positions; replay size={len(replay_s)}")
		S = np.stack(replay_s, axis=0); P = np.stack(replay_p, axis=0); Z = np.stack(replay_z, axis=0)
		np.savez_compressed(os.path.join(args.data_dir, f"replay_snapshot_it{it}.npz"), s=S, p=P, z=Z)

		new_net = PolicyValueNet().to(args.device); new_net.load_state_dict(best.state_dict())
		new_net = train_once(new_net, S, P, Z, device=args.device, epochs=args.epochs, batch_size=args.batch_size,
						 lr=args.lr, weight_decay=args.weight_decay, clip_norm=args.clip_norm, amp=not args.no_amp)
		wr = arena_win_rate(new_net, best, games=args.eval_games, sims=args.mcts_sims_eval, device=args.device)
		print(f"Iteration {it}: win-rate(new vs best) = {wr:.3f}")
		if wr >= args.accept_rate:
			best = new_net; best.eval()
			torch.save(best.state_dict(), os.path.join(args.save_dir, "best.pt"))
			torch.save(best.state_dict(), os.path.join(args.save_dir, f"best_it{it}.pt"))
			print("Accepted new model.")
		else:
			print("Rejected new model; keep previous best.")


if __name__ == "__main__":
	main()
