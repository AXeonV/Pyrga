from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .model import PolicyValueNet
from .config import NUM_ACTIONS


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data", type=str, default="data/sp.npz")
	ap.add_argument("--epochs", type=int, default=5)
	ap.add_argument("--batch-size", type=int, default=256)
	ap.add_argument("--lr", type=float, default=1e-3)
	ap.add_argument("--weight-decay", type=float, default=1e-4)
	ap.add_argument("--device", type=str, default="cuda")
	ap.add_argument("--save", type=str, default="ckpt/model.pt")
	ap.add_argument("--clip-norm", type=float, default=1.0)
	args = ap.parse_args()

	if args.device.startswith("cuda") and not torch.cuda.is_available():
		raise RuntimeError("CUDA device requested but not available. Please install a CUDA-enabled PyTorch build or set --device cpu.")

	data = np.load(args.data)
	S = torch.from_numpy(data['s'])  # [N,C,H,W]
	P = torch.from_numpy(data['p'])  # [N,A]
	Z = torch.from_numpy(data['z']).float()  # [N]

	ds = TensorDataset(S, P, Z)
	dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

	net = PolicyValueNet().to(args.device)
	opt = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scaler = torch.cuda.amp.GradScaler(device_type='cuda', enabled=args.device.startswith('cuda'))

	ce = nn.CrossEntropyLoss()
	mse = nn.MSELoss()

	net.train()
	for epoch in range(args.epochs):
		total_loss = 0.0
		total_p = 0.0
		total_v = 0.0
		total_n = 0
		for s, p, z in dl:
			s = s.to(args.device)
			p = p.to(args.device)
			z = z.to(args.device)
			with torch.cuda.amp.autocast(device_type='cuda', enabled=args.device.startswith('cuda')):
				logits, v = net(s)
				# policy loss: cross-entropy with target probabilities -> use soft targets via KL (or CE on probs)
				log_probs = torch.log_softmax(logits, dim=-1)
				policy_loss = -(p * log_probs).sum(dim=-1).mean()
				value_loss = mse(v, z)
				loss = policy_loss + value_loss * 1.0
			opt.zero_grad()
			scaler.scale(loss).backward()
			if args.clip_norm is not None and args.clip_norm > 0:
				scaler.unscale_(opt)
				torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_norm)
			scaler.step(opt)
			scaler.update()
			bs = s.size(0)
			total_loss += loss.item() * bs
			total_p += policy_loss.item() * bs
			total_v += value_loss.item() * bs
			total_n += bs
		print(f"epoch {epoch+1}: loss={total_loss/total_n:.4f} policy={total_p/total_n:.4f} value={total_v/total_n:.4f}")

	torch.save(net.state_dict(), args.save)
	print(f"saved {args.save}")

if __name__ == "__main__":
	main()
