from __future__ import annotations
"""简易人机对战 (含预览):
交互:
	1. 鼠标点击格子 -> 高亮
	2. 按 s/c/a 选择 方形/圆形/箭头 (箭头仅通过重复按 a 轮换方向 0->1->2->3)
	3. 预览: 选中格显示候选棋子/箭头
	4. 回车 / 空格 确认落子; q/ESC 退出
"""
import argparse, os, cv2, torch
from datetime import datetime
from src.game import GameState, SQUARE, CIRCLE, ARROW
from src.config import DIRS
from src.model import PolicyValueNet
from src.mcts import MCTS
from src.utils import print_board

def encode_if_legal(st: GameState, cell: int, ptype: int, direction: int):
	mask = st.legal_actions_mask()
	a = GameState.encode_action(cell, ARROW, direction) if ptype == ARROW else GameState.encode_action(cell, ptype)
	return a if mask[a] else None

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--model', type=str, default=None)
	ap.add_argument('--mcts-sims', type=int, default=400)
	ap.add_argument('--device', type=str, default='cuda')
	args = ap.parse_args()

	if args.device.startswith('cuda') and not torch.cuda.is_available():
		args.device = 'cpu'

	net = PolicyValueNet().to(args.device)
	if args.model:
		state = torch.load(args.model, map_location=args.device)
		net.load_state_dict(state)
	net.eval()
	mcts_ai = MCTS(net, device=args.device, add_root_noise=False)

	st = GameState(); actions = []
	timestamp = datetime.now().strftime('%Y%m%d.%H%M%S')
	step = 0
	win = 'Pyrga'
	cv2.namedWindow(win)
	selected_type = None; selected_dir = 0; current_cell = None

	clicked = {}
	def _cb(event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			if 0 <= x < 404 and 0 <= y < 404:
				c = x // 100; r = y // 100
				if c <= 3 and r <= 3:
					clicked['cell'] = int(r*4 + c)
	cv2.setMouseCallback(win, _cb)

	while True:
		if st.game_over():
			res = st.result()
			print('Result:', 'You win!' if res == 1 else 'You lose!')
			print_board(actions, timestamp, step)
			img = cv2.imread(os.path.join('logs', timestamp, f'board{step}.png'))
			if img is not None:
				cv2.imshow(win, img); cv2.waitKey(10)
			break

		print_board(actions, timestamp, step)
		img = cv2.imread(os.path.join('logs', timestamp, f'board{step}.png'))
		if img is not None:
			if current_cell is not None:
				cell_size = 100
				col = current_cell % 4; row = current_cell // 4
				x0, y0 = col * cell_size, row * cell_size
				# 高亮边框
				cv2.rectangle(img, (x0+2, y0+2), (x0+cell_size-2, y0+cell_size-2), (0, 255, 255), 2)
				# 预览棋子
				preview_color = (0, 255, 0)
				cx, cy = x0 + cell_size//2, y0 + cell_size//2
				if selected_type is not None:
					if selected_type == SQUARE:
						cv2.rectangle(img, (x0+10, y0+10), (x0+cell_size-10, y0+cell_size-10), preview_color, 3)
					elif selected_type == CIRCLE:
						radius = cell_size // 2 - 12
						cv2.circle(img, (cx, cy), radius, preview_color, 3)
					elif selected_type == ARROW:
						dy, dx = DIRS[selected_dir]
						L = cell_size // 2 - 14
						x1 = int(cx - dx * L); y1 = int(cy - dy * L)
						x2 = int(cx + dx * L); y2 = int(cy + dy * L)
						cv2.arrowedLine(img, (x1, y1), (x2, y2), preview_color, 4, tipLength=0.4)
			cv2.imshow(win, img)

		if st.player == 0:
			k = cv2.waitKey(10) & 0xFF
			if k in (ord('q'), 27): break
			if k in (ord('s'), ord('S')): selected_type = SQUARE
			elif k in (ord('c'), ord('C')): selected_type = CIRCLE
			elif k in (ord('a'), ord('A')):
				if selected_type == ARROW: selected_dir = (selected_dir + 1) % 4
				else: selected_type = ARROW
			elif k in (13, 32):
				if current_cell is not None and selected_type is not None:
					a = encode_if_legal(st, current_cell, selected_type, selected_dir)
					if a is not None:
						st.apply(a)
						mv = GameState.decode_action(a)
						actions.append([mv, st.player]); step += 1; current_cell = None
						# 立即刷新：先展示玩家落子，再进入下一轮 AI 计算
						print_board(actions, timestamp, step)
						img2 = cv2.imread(os.path.join('logs', timestamp, f'board{step}.png'))
						if img2 is not None:
							cv2.imshow(win, img2)
							cv2.waitKey(10)
			if 'cell' in clicked:
				current_cell = clicked.pop('cell')
		else:
			pi, _ = mcts_ai.run(st, sims=args.mcts_sims)
			a = int(pi.argmax())
			st.apply(a)
			mv = GameState.decode_action(a)
			actions.append([mv, st.player]); step += 1
			cv2.waitKey(10)

	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
