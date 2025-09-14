from __future__ import annotations
import argparse
import os
import random
import time
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .model import PolicyValueNet


def set_seed(seed: Optional[int]):
	if seed is None:
		return
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)


class AverageMeter:
	def __init__(self):
		self.sum = 0.0
		self.n = 0

	def update(self, v, k=1):
		self.sum += float(v) * k
		self.n += k

	@property
	def avg(self):
		return self.sum / max(1, self.n)


def build_dataset(S, P, Z):
	return TensorDataset(
		torch.from_numpy(S),
		torch.from_numpy(P),
		torch.from_numpy(Z).float(),
	)


def train_model(
	net: torch.nn.Module,
	dataset: TensorDataset,
	device: str,
	epochs: int,
	batch_size: int,
	lr: float,
	weight_decay: float,
	clip_norm: float,
	amp: bool = True,
	seed: Optional[int] = None,
	logger: Optional[Callable[[str], None]] = None,
	return_best: bool = True,
):
	set_seed(seed)
	dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
	net = net.to(device)
	net.train()
	opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

	use_amp = amp and device.startswith("cuda") and torch.cuda.is_available()
	if use_amp:
		from torch.amp import GradScaler, autocast  # type: ignore
		scaler = GradScaler('cuda')
		autocast_ctx = lambda: autocast('cuda')
	else:
		from contextlib import nullcontext
		autocast_ctx = nullcontext  # type: ignore

		class _DummyScaler:  # minimal shim
			def scale(self, x):
				return x
			def step(self, opt):
				opt.step()
			def update(self):
				pass
			def unscale_(self, opt):
				pass

		scaler = _DummyScaler()  # type: ignore

	mse = nn.MSELoss()
	best_state = None; best_loss = float('inf')
	for ep in range(1, epochs + 1):
		m_loss = AverageMeter(); m_p = AverageMeter(); m_v = AverageMeter(); t0 = time.time()
		for s, p, z in dl:
			s = s.to(device); p = p.to(device); z = z.to(device)
			with autocast_ctx():
				logits, v = net(s)
				log_probs = torch.log_softmax(logits, dim=-1)
				policy_loss = -(p * log_probs).sum(dim=-1).mean()
				value_loss = mse(v, z)
				loss = policy_loss + value_loss
			opt.zero_grad(set_to_none=True)
			scaler.scale(loss).backward()
			if clip_norm and clip_norm > 0:
				scaler.unscale_(opt)
				torch.nn.utils.clip_grad_norm_(net.parameters(), clip_norm)
			scaler.step(opt); scaler.update()
			bs = s.size(0)
			m_loss.update(loss.item(), bs); m_p.update(policy_loss.item(), bs); m_v.update(value_loss.item(), bs)
		dt = time.time()-t0
		msg = f"epoch {ep}: loss={m_loss.avg:.4f} policy={m_p.avg:.4f} value={m_v.avg:.4f} time={dt:.1f}s"
		if logger is not None:
			logger(msg)
		else:
			print(msg)
		if return_best and m_loss.avg < best_loss:
			best_loss = m_loss.avg
			best_state = {k: v.cpu() for k, v in net.state_dict().items()}
	net.eval()
	if return_best and best_state is not None:
		return net, best_state, best_loss
	return net, None, None


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data", type=str, default="data/sp.npz")
	ap.add_argument("--epochs", type=int, default=100)
	ap.add_argument("--batch-size", type=int, default=256)
	ap.add_argument("--lr", type=float, default=1e-3)
	ap.add_argument("--weight-decay", type=float, default=1e-4)
	ap.add_argument("--device", type=str, default="cuda")
	ap.add_argument("--save", type=str, default="ckpt/model.pt")
	ap.add_argument("--clip-norm", type=float, default=1.0)
	ap.add_argument("--seed", type=int, default=42, help="随机种子 (默认42)")
	ap.add_argument("--no-amp", action="store_true", help="禁用自动混合精度")
	ap.add_argument("--model", type=str, default=None, help="可选：预训练/上一轮 best 权重；若提供将在其基础上继续训练")
	ap.add_argument("--log", type=str, default=None, help="可选：将每个 epoch 指标追加写入该日志文件")
	args = ap.parse_args()

	if args.device.startswith("cuda") and not torch.cuda.is_available():
		raise RuntimeError(
			"CUDA device requested but not available. Install CUDA PyTorch or set --device cpu."
		)

	set_seed(args.seed)
	data = np.load(args.data)
	ds = build_dataset(data["s"], data["p"], data["z"])
	net = PolicyValueNet()
	if args.model is not None:
		if not os.path.isfile(args.model):
			raise FileNotFoundError(f"--model file not found: {args.model}")
		state = torch.load(args.model, map_location=args.device)
		net.load_state_dict(state)
		print(f"Loaded pretrained weights from {args.model}")

	log_f = None
	if args.log is not None:
		os.makedirs(os.path.dirname(args.log), exist_ok=True)
		mode = "a" if os.path.exists(args.log) else "w"
		log_f = open(args.log, mode, encoding="utf-8")
		if mode == "w":
			log_f.write("epoch,loss,policy,value,time\n")
	def logger(msg: str):
		print(msg)
		if log_f:
			log_f.write(msg+"\n"); log_f.flush()

	net, best_state, best_loss = train_model(
		net=net,
		dataset=ds,
		device=args.device,
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		weight_decay=args.weight_decay,
		clip_norm=args.clip_norm,
		amp=not args.no_amp,
		seed=args.seed,
		logger=logger,
		return_best=True,
	)
	os.makedirs(os.path.dirname(args.save), exist_ok=True)
	torch.save(net.state_dict(), args.save)
	logger(f"saved final {args.save}")
	if best_state is not None:
		best_path = args.save.replace('.pt', '_best.pt')
		torch.save(best_state, best_path)
		logger(f"saved best(by loss {best_loss:.4f}) -> {best_path}")
	if log_f:
		log_f.close()


if __name__ == "__main__":
	main()
