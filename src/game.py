from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from .config import BOARD_SIZE, NUM_CELLS, NUM_ACTIONS, DIRS, MAX_MOVES
from .utils import rc_to_i, i_to_rc, in_bounds, empty_mask, cell_full, cell_has_type, clone_board

# piece types
SQUARE, CIRCLE, ARROW = 0, 1, 2

@dataclass
class Move:
	cell: int
	ptype: int  # 0 square, 1 circle, 2 arrow
	dir: int = 0  # only for arrow 0..3

@dataclass
class LastMove:
	cell: Optional[int]
	ptype: Optional[int]
	dir: int = 0

class GameState:
	def __init__(self):
		# board_types[r,c,3] booleans of presence per type
		self.board_types = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.bool_)
		# who placed pieces: for majority in tower; store counts per cell per player
		# owner_counts[r,c,2] -> how many pieces in cell placed by player 0 or 1
		self.owner_counts = np.zeros((BOARD_SIZE, BOARD_SIZE, 2), dtype=np.int8)
		# arrow dir per cell: -1 if none, else 0..3 (only meaningful if ARROW present)
		self.arrow_dir = -np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
		# remaining pieces per player [player, type] with counts; each player has 5 of each type
		self.remaining = np.full((2, 3), 5, dtype=np.int8)
		self.player = 0
		self.last = LastMove(cell=None, ptype=None, dir=0)
		self.move_count = 0
		self.is_1st_square = False

	def clone(self) -> 'GameState':
		st = GameState()
		st.board_types = self.board_types.copy()
		st.owner_counts = self.owner_counts.copy()
		st.arrow_dir = self.arrow_dir.copy()
		st.remaining = self.remaining.copy()
		st.player = self.player
		st.last = LastMove(self.last.cell, self.last.ptype, self.last.dir)
		st.move_count = self.move_count
		st.is_1st_square = self.is_1st_square
		return st

	# action encoding: 0..15 square, 16..31 circle, 32..95 arrow (dir = (a-32)%4)
	@staticmethod
	def decode_action(a: int) -> Move:
		# squares occupy [0, NUM_CELLS-1]
		if a < NUM_CELLS:
			return Move(cell=a, ptype=SQUARE, dir=0)
		# circles occupy [NUM_CELLS, 2*NUM_CELLS-1]
		elif a < 2 * NUM_CELLS:
			return Move(cell=a - NUM_CELLS, ptype=CIRCLE, dir=0)
		else:
			# arrows occupy [2*NUM_CELLS, 2*NUM_CELLS + 4*NUM_CELLS - 1]
			idx = a - 2 * NUM_CELLS
			cell = idx // 4
			d = idx % 4
			return Move(cell=cell, ptype=ARROW, dir=d)

	@staticmethod
	def encode_action(cell: int, ptype: int, d: int=0) -> int:
		if ptype == SQUARE:
			return cell
		if ptype == CIRCLE:
			return NUM_CELLS + cell
		return 2 * NUM_CELLS + cell * 4 + d

	def legal_actions_mask(self) -> np.ndarray:
		mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
		# gather candidate cells according to previous move
		cand = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.bool_)
		if self.last.ptype is None:
			cand[:, :] = True
		else:
			# mypy/pylance: ensure non-None before passing to i_to_rc
			assert self.last.cell is not None, "last.cell must be set when last.ptype is not None"
			r, c = i_to_rc(self.last.cell)
			if self.last.ptype == SQUARE:
				for dr, dc in DIRS:
					rr, cc = r+dr, c+dc
					if in_bounds(rr, cc):
						cand[rr, cc] = True
			elif self.last.ptype == ARROW:
				dr, dc = DIRS[self.last.dir]
				rr, cc = r+dr, c+dc
				while in_bounds(rr, cc):
					cand[rr, cc] = True
					rr += dr; cc += dc
			else:  # CIRCLE
				cand[r, c] = True
			# if no candidate has any legal placement, we may fallback to any empty cell
		# build legality
		any_legal = False
		# iterate all cells and piece types
		for i in range(NUM_CELLS):
			r, c = i_to_rc(i)
			types_here = self.board_types[r, c]
			cap_ok = (types_here.sum() < 3)
			empty_here = (types_here.sum() == 0)
			# for fallback we need "completely empty"
			allow_fallback_cell = empty_here
			# for arrow, direction-specific
			# SQUARE
			if self.remaining[self.player, SQUARE] > 0 and cap_ok and not types_here[SQUARE]:
				if self.last.ptype is None or cand[r, c]:
					if not (self.is_1st_square and self.player == 0 and self.move_count == 2):
       			# if player 0's first move is square, he cannot place square on step 3
						mask[self.encode_action(i, SQUARE)] = True; any_legal = True
			# CIRCLE
			if self.remaining[self.player, CIRCLE] > 0 and cap_ok and not types_here[CIRCLE]:
				if self.last.ptype is None or cand[r, c]:
					mask[self.encode_action(i, CIRCLE)] = True; any_legal = True
			# ARROW
			if self.remaining[self.player, ARROW] > 0 and cap_ok and not types_here[ARROW]:
				# constrain by last move cells; dir does not affect cell candidacy except it's encoded per action
				if self.last.ptype is None or cand[r, c]:
					base = 32 + i*4
					for d in range(4):
						# also need to check arrow direction does not point to the wall
						dr, dc = DIRS[d]
						rr, cc = r+dr, c+dc
						if in_bounds(rr, cc):
							mask[base + d] = True; any_legal = True
		# Fallback: if under constraints no legal move, allow any empty cell with any type available
		if not any_legal:
			# if no empty cells too -> game over (no action will be legal)
			empties = np.where((self.board_types.sum(axis=2) == 0).flatten())[0]
			for i in empties:
				r, c = i_to_rc(i)
				if self.remaining[self.player, SQUARE] > 0:
					mask[self.encode_action(i, SQUARE)] = True
				if self.remaining[self.player, CIRCLE] > 0:
					mask[self.encode_action(i, CIRCLE)] = True
				if self.remaining[self.player, ARROW] > 0:
					base = 32 + i*4
					for d in range(4):
						# also need to check arrow direction does not point to the wall
						dr, dc = DIRS[d]
						rr, cc = r+dr, c+dc
						if in_bounds(rr, cc):
							mask[base + d] = True
		return mask

	def apply(self, a: int):
		# Enforce action legality according to current state
		mask = self.legal_actions_mask()
		assert 0 <= a < NUM_ACTIONS and mask[a], "illegal action for current state"
		mv = self.decode_action(a)
		r, c = i_to_rc(mv.cell)
		assert not cell_full(self.board_types, r, c), "cell full"
		assert not self.board_types[r, c, mv.ptype], "type already in cell"
		assert self.remaining[self.player, mv.ptype] > 0, "no piece remaining"
		# place
		self.board_types[r, c, mv.ptype] = True
		self.owner_counts[r, c, self.player] += 1
		if mv.ptype == SQUARE and self.player == 0 and self.move_count == 0:
			self.is_1st_square = True
		if mv.ptype == ARROW:
			self.arrow_dir[r, c] = mv.dir
		self.remaining[self.player, mv.ptype] -= 1
		# update last
		self.last = LastMove(cell=mv.cell, ptype=mv.ptype, dir=mv.dir)
		self.player ^= 1
		self.move_count += 1

	def game_over(self) -> bool:
		if self.move_count >= MAX_MOVES:
			return True
		# Game ends only when fallback also impossible (no empty cells) and constraints give nothing
		mask = self.legal_actions_mask()
		if mask.any():
			return False
		# fallback would allow empty cells if any; so if mask is all-false, there are no empty cells either
		return True

	def result(self) -> int:
		# return +1 if player 0 wins, -1 if player 1 wins
		# count towers (cells with 3 pieces) majority by owner_counts
		p0, p1 = 0, 0
		for r in range(BOARD_SIZE):
			for c in range(BOARD_SIZE):
				if self.board_types[r, c].sum() == 3:
					if self.owner_counts[r, c, 0] > self.owner_counts[r, c, 1]:
						p0 += 1
					elif self.owner_counts[r, c, 1] > self.owner_counts[r, c, 0]:
						p1 += 1
		if p0 > p1:
			return 1
		else: # p0 == p1, the second player wins ties
			return -1

	def obs(self) -> np.ndarray:
		# observation planes [C,H,W]
		# planes: for current player perspective
		# 3 type presence planes (regardless of who placed), plus remaining counts per type (broadcast), plus side-to-move
		H = BOARD_SIZE; W = BOARD_SIZE
		planes = []
		# presence per type
		for t in range(3):
			planes.append(self.board_types[:, :, t].astype(np.float32))
		# arrow direction one-hot per cell (4 planes)
		arrow_dir_oh = np.zeros((H, W, 4), dtype=np.float32)
		for r in range(H):
			for c in range(W):
				d = self.arrow_dir[r, c]
				if d >= 0:
					arrow_dir_oh[r, c, d] = 1.0
		for k in range(4):
			planes.append(arrow_dir_oh[:, :, k])
		# remaining counts for current player (normalized by 5)
		rem = self.remaining[self.player].astype(np.float32) / 5.0
		for t in range(3):
			planes.append(np.full((H, W), rem[t], dtype=np.float32))
		# side to move
		planes.append(np.full((H, W), 1.0 if self.player == 0 else 0.0, dtype=np.float32))
		return np.stack(planes, axis=0)

