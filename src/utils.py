from __future__ import annotations
import numpy as np
from typing import List, Tuple
from .config import BOARD_SIZE, NUM_CELLS


def rc_to_i(r: int, c: int) -> int:
	return r * BOARD_SIZE + c


def i_to_rc(i: int) -> Tuple[int, int]:
	return divmod(i, BOARD_SIZE)


def in_bounds(r: int, c: int) -> bool:
	return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def empty_mask(board_types: np.ndarray) -> np.ndarray:
	# board_types shape: (BOARD_SIZE, BOARD_SIZE, 3) boolean: [square, circle, arrow] presence
	# empty if all three are False
	return (board_types.sum(axis=2) == 0)


def cell_full(board_types: np.ndarray, r: int, c: int) -> bool:
	return board_types[r, c].sum() >= 3


def cell_has_type(board_types: np.ndarray, r: int, c: int, t: int) -> bool:
	return bool(board_types[r, c, t])


def clone_board(board_types: np.ndarray, owner_counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	return board_types.copy(), owner_counts.copy()
