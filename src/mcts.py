from __future__ import annotations
import math
import numpy as np
import torch
from typing import Dict, Tuple
from .config import C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPS, NUM_ACTIONS

class Node:
	__slots__ = ("P","N","W","Q","children","is_expanded","legal")
	def __init__(self):
		self.P = None  # prior policy logits/probs for legal actions
		self.N = np.zeros(NUM_ACTIONS, dtype=np.int32)
		self.W = np.zeros(NUM_ACTIONS, dtype=np.float32)
		self.Q = np.zeros(NUM_ACTIONS, dtype=np.float32)
		self.children: Dict[int, Node] = {}
		self.is_expanded = False
		self.legal = None

class MCTS:
	def __init__(self, net, device="cpu", add_root_noise=True):
		self.net = net
		self.device = device
		self.add_root_noise = add_root_noise

	@torch.no_grad()
	def run(self, state, sims: int):
		root = Node()
		self._expand(root, state, add_noise=self.add_root_noise)
		for _ in range(sims):
			self._simulate(root, state)
		visits = root.N.astype(np.float32)
		pi = visits / (visits.sum() + 1e-8)
		return pi, root

	def _simulate(self, node: Node, state):
		path = []
		cur = node
		st = state.clone()
		# selection
		while cur.is_expanded:
			a = self._select_action(cur)
			path.append((cur, a))
			if a not in cur.children:
				st.apply(a)
				child = Node()
				cur.children[a] = child
				cur = child
				break
			else:
				st.apply(a)
				cur = cur.children[a]
		# expand
		v = self._expand(cur, st, add_noise=False)
		# backup
		self._backup(path, v, st.player)

	def _select_action(self, node: Node) -> int:
		# PUCT: a = argmax Q + U
		assert node.P is not None and node.legal is not None, "Node must be expanded before selecting action"
		legal = node.legal
		N_total = node.N.sum() + 1
		# U = c_puct * P * sqrt(N_total)/(1+N[a])
		U = np.zeros(NUM_ACTIONS, dtype=np.float32)
		denom = 1.0 + node.N
		U[legal] = (C_PUCT * node.P[legal] * math.sqrt(N_total) / denom[legal]).astype(np.float32)
		score = node.Q + U
		score[~legal] = -1e9
		a = int(score.argmax())
		return a

	@torch.no_grad()
	def _expand(self, node: Node, state, add_noise: bool):
		if node.is_expanded:
			# if already expanded, return leaf value using stored policy/v?
			# Use fresh net eval for value
			obs = torch.from_numpy(state.obs()[None, ...]).to(self.device)
			p_logits, v = self.net(obs)
			return float(v.item())
		legal_mask = state.legal_actions_mask()
		node.legal = legal_mask
		obs = torch.from_numpy(state.obs()[None, ...]).to(self.device)
		p_logits, v = self.net(obs)
		p = torch.softmax(p_logits, dim=-1).cpu().numpy()[0]
		p = p * legal_mask.astype(np.float32)
		s = p.sum()
		if s > 0:
			p = p / s
		else:
			# no legal -> terminal; value from state, adjusted to the current player to move at this state
			# state.result() returns +1 if player 0 wins, -1 if player 1 wins, from absolute perspective.
			# If it's player 0 to move, leaf value is result; if it's player 1 to move, flip sign.
			node.is_expanded = True
			res = state.result()
			v_leaf = float(res if state.player == 0 else -res)
			return v_leaf
		if add_noise:
			noise = np.random.dirichlet([DIRICHLET_ALPHA] * legal_mask.sum())
			p[legal_mask] = (1 - DIRICHLET_EPS) * p[legal_mask] + DIRICHLET_EPS * noise
		node.P = p
		node.is_expanded = True
		return float(v.item())

	def _backup(self, path, leaf_value: float, to_play_after_leaf: int):
		# leaf_value is value for current player at leaf state;
		# convert along path alternating signs (since players alternate)
		v = leaf_value
		# to_play_after_leaf is player to move at leaf; if first backup step corresponds to the opponent move, we need sign
		# We can simply alternate sign along the path from leaf backwards
		for node, a in reversed(path):
			n = node.N[a] + 1
			node.N[a] = n
			node.W[a] += v
			node.Q[a] = node.W[a] / n
			v = -v
