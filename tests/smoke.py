import torch
from src.game import GameState
from src.model import PolicyValueNet
from src.mcts import MCTS


def main():
	st = GameState()
	net = PolicyValueNet().to("cuda")
	mcts = MCTS(net, device="cuda", add_root_noise=False)
	pi, root = mcts.run(st, sims=10)
	print("Policy sum:", pi.sum(), "Nonzero:", (pi>0).sum())
	# play a short game deterministically
	steps = 0
	while not st.game_over() and steps < 20:
		pi, _ = mcts.run(st, sims=20)
		a = int(pi.argmax())
		print("Step", steps, "Player", st.player, "Action", GameState.decode_action(a))
		st.apply(a)
		steps += 1
	print("Game over:", st.game_over(), "Result:", st.result(), "Moves:", steps)

if __name__ == "__main__":
	main()
