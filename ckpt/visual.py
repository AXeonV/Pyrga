from __future__ import annotations
import argparse, re, os
import matplotlib.pyplot as plt
from typing import List

# epoch <num>: loss=<float> policy=<float> value=<float> time=<float>s
PATTERN = re.compile(r"^epoch\s+(\d+):\s+loss=([-+eE0-9\.]+)\s+policy=([-+eE0-9\.]+)\s+value=([-+eE0-9\.]+)\s+time=([-+eE0-9\.]+)s$")

def moving_avg(xs: List[float], k: int) -> List[float]:
	if k <= 1:
		return xs
	out, acc = [], 0.0
	for i, v in enumerate(xs):
		acc += v
		if i >= k:
			acc -= xs[i - k]
		out.append(acc / min(i + 1, k))
	return out

def parse_log(path: str):
	epochs, loss, policy, value = [], [], [], []
	with open(path, 'r', encoding='utf-8') as f:
		for line in f:
			m = PATTERN.match(line.strip())
			if not m:
				continue
			ep, l, p, v, _t = m.groups()
			epochs.append(int(ep)); loss.append(float(l)); policy.append(float(p)); value.append(float(v))
	return epochs, loss, policy, value

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--log', type=str, required=True, help='训练日志文件（包含 epoch ... 行）')
	ap.add_argument('--out', type=str, default=None, help='可选：保存图片路径 (例如 ckpt/loss.png)')
	ap.add_argument('--smooth', type=int, default=1, help='移动平均窗口（>1 开启平滑）')
	ap.add_argument('--title', type=str, default='Training Loss Curve')
	args = ap.parse_args()

	if not os.path.isfile(args.log):
		raise FileNotFoundError(args.log)
	epochs, loss, policy, value = parse_log(args.log)
	if not epochs:
		raise RuntimeError('Log lines must match: epoch <n>: loss=... policy=... value=... time=...s')

	loss_s   = moving_avg(loss, args.smooth)
	policy_s = moving_avg(policy, args.smooth)
	value_s  = moving_avg(value, args.smooth)

	plt.figure(figsize=(8,5))
	plt.plot(epochs, loss,   label='Total Loss', color='#1f77b4', alpha=0.35)
	plt.plot(epochs, policy, label='Policy Loss', color='#ff7f0e', alpha=0.35)
	plt.plot(epochs, value,  label='Value Loss', color='#2ca02c', alpha=0.35)
	if args.smooth > 1:
		plt.plot(epochs, loss_s,   label=f'Total Loss (MA{args.smooth})', color='#1f77b4')
		plt.plot(epochs, policy_s, label=f'Policy Loss (MA{args.smooth})', color='#ff7f0e')
		plt.plot(epochs, value_s,  label=f'Value Loss (MA{args.smooth})', color='#2ca02c')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title(args.title)
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	if args.out:
		out_dir = os.path.dirname(args.out)
		if out_dir:
			os.makedirs(out_dir, exist_ok=True)
		plt.savefig(args.out, dpi=160)
		print(f'Saved figure to {args.out}')
	else:
		plt.show()

if __name__ == '__main__':
	main()