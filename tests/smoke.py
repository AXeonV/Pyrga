import torch
from src.game import GameState
from src.model import PolicyValueNet
from src.mcts import MCTS
from src.utils import print_board
from datetime import datetime

def main():
	st = GameState()
	net = PolicyValueNet().to("cuda")
	mcts = MCTS(net, device="cuda", add_root_noise=False)
	pi, _ = mcts.run(st, sims=10)
	cur_time = datetime.now().strftime('%Y%m%d.%H%M%S')
	print_board(None, cur_time, 0)
	print("Policy sum:", pi.sum(), "Nonzero:", (pi>0).sum())
	# play a short game deterministically
	steps = 0
	actions = []
	while not st.game_over() and steps < 30:
		steps += 1
		pi, _ = mcts.run(st, sims=20)
		a = int(pi.argmax())
		st.apply(a)
		cur_step = GameState.decode_action(a)
		actions.append([cur_step, st.player])
		print("Step:", steps, "Player:", st.player, "Action:", cur_step)
		print_board(actions, cur_time, steps)
	print("Game over! Result:", st.result(), "Moves:", steps)

if __name__ == "__main__":
	main()
