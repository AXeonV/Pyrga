from __future__ import annotations
import cv2, os
import numpy as np
from typing import Tuple
from .config import BOARD_SIZE, DIRS

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

def print_board(actions, timestamp, frame, silence=True):
	# 4*4 board by cv2
	cell_size = 100
	board_size = BOARD_SIZE * cell_size
	img = np.ones((board_size + 4, board_size + 4, 3), dtype=np.uint8) * 255
	player_color = [(255, 0, 0), (0, 0, 255)]

	for i in range(BOARD_SIZE + 1):
		x = i * cell_size
		cv2.line(img, (x, 0), (x, board_size), (0, 0, 0), 5)
		cv2.line(img, (0, x), (board_size, x), (0, 0, 0), 5)

	if actions is not None:
		for action in actions:
			cur_step, player = action
			r, c = i_to_rc(cur_step.cell)
			cx = c * cell_size + cell_size // 2
			cy = r * cell_size + cell_size // 2
			if cur_step.ptype == 0:   # SQUARE
				x1, y1 = c * cell_size + 10, r * cell_size + 10
				x2, y2 = (c + 1) * cell_size - 10, (r + 1) * cell_size - 10
				cv2.rectangle(img, (x1, y1), (x2, y2), player_color[player], 5)
			elif cur_step.ptype == 1: # CIRCLE
				radius = int(cell_size // 2 - 12)
				cv2.circle(img, (cx, cy), radius, player_color[player], 5)
			elif cur_step.ptype == 2: # ARROW
				dy, dx = DIRS[cur_step.dir]
				# print(dx, dy)
				x1 = int(cx - dx * (cell_size // 2 - 14))
				y1 = int(cy - dy * (cell_size // 2 - 14))
				x2 = int(cx + dx * (cell_size // 2 - 14))
				y2 = int(cy + dy * (cell_size // 2 - 14))
				cv2.arrowedLine(img, (x1, y1), (x2, y2), player_color[player], 5, tipLength=0.4)
 
	if silence:
		save_dir = os.path.join('logs/', timestamp)
		os.makedirs(save_dir, exist_ok=True)
		save_path = os.path.join(save_dir, 'board' + str(frame) + '.png')
		cv2.imwrite(save_path, img)
	else:
		cv2.imshow("Pyrga Board", img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()