from __future__ import annotations
import argparse, os, torch
from tqdm import trange
from typing import Optional
from .model import PolicyValueNet
from .mcts import MCTS
from .game import GameState

from .utils import print_board
from datetime import datetime

@torch.no_grad()
def run_arena(new_net: PolicyValueNet, old_net: PolicyValueNet, games: int, sims: int, device: str, debug: bool=False) -> float:
	wins = 0
	for g in trange(games, desc="arena"):
		if debug:
			cur_time = datetime.now().strftime('%Y%m%d.%H%M%S')
			print_board(None, cur_time, 0)
			actions = []
		st = GameState()
		# Alternate first mover: even -> new_net as player0; odd -> old_net as player0
		new_is_p0 = (g % 2 == 0)
		order = (new_net, old_net) if new_is_p0 else (old_net, new_net)
		steps = 0
		while not st.game_over():
			steps += 1
			cur = order[st.player]
			pi, _ = MCTS(cur, device=device, add_root_noise=False).run(st, sims=sims)
			a = int(pi.argmax())
			st.apply(a)
			if debug:
				cur_step = GameState.decode_action(a)
				actions.append([cur_step, st.player])
				print_board(actions, cur_time, steps)
		z = st.result()  # +1 if absolute player0 wins, -1 otherwise
		new_win = (z == 1) if new_is_p0 else (z == -1)
		wins += 1 if new_win else 0
		if debug:
			print(f"[GAME {g:03d}] new_is_p0={new_is_p0} result={z:+d} -> new_win={new_win} steps={steps}")
	return wins / games

def load_model(path: str, device: str) -> PolicyValueNet:
	net = PolicyValueNet().to(device)
	state = torch.load(path, map_location=device)
	net.load_state_dict(state)
	net.eval()
	return net

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--candidate', type=str, required=True, help='Path to candidate model weights (.pt)')
	ap.add_argument('--best', type=str, required=True, help='Path to current best model weights (.pt)')
	ap.add_argument('--out-best', type=str, default=None, help='Output path to save new best (if accepted); default overwrite --best')
	ap.add_argument('--accept-rate', type=float, default=0.55, help='Win rate threshold to accept candidate')
	ap.add_argument('--eval-games', type=int, default=50, help='Number of evaluation games')
	ap.add_argument('--mcts-sims', type=int, default=400, help='MCTS simulations per move during evaluation')
	ap.add_argument('--device', type=str, default='cuda')
	ap.add_argument('--debug', action='store_true', help='Print per-game debug info for win mapping')
	args = ap.parse_args()

	if args.device.startswith('cuda') and not torch.cuda.is_available():
		raise RuntimeError('CUDA requested but not available. Use --device cpu.')

	if not os.path.isfile(args.candidate):
		raise FileNotFoundError(f'Candidate model not found: {args.candidate}')
	if not os.path.isfile(args.best):
		raise FileNotFoundError(f'Best model not found: {args.best}')

	device = args.device
	new_net = load_model(args.candidate, device)
	old_net = load_model(args.best, device)

	wr = run_arena(new_net, old_net, games=args.eval_games, sims=args.mcts_sims, device=device, debug=getattr(args,'debug',False))
	print(f'Win-rate (candidate vs best): {wr:.3f}')

	if wr >= args.accept_rate:
		out_path = args.out_best or args.best
		torch.save(new_net.state_dict(), out_path)
		print(f'Accepted new model. Saved to {out_path}')
	else:
		print('Rejected candidate model.')

if __name__ == '__main__':
	main()
