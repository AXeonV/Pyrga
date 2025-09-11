from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import BOARD_SIZE, NUM_ACTIONS, POLICY_CHANNELS, RES_BLOCKS

class ResidualBlock(nn.Module):
	def __init__(self, ch: int):
		super().__init__()
		self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(ch)
		self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(ch)

	def forward(self, x):
		h = F.relu(self.bn1(self.conv1(x)))
		h = self.bn2(self.conv2(h))
		return F.relu(h + x)

class PolicyValueNet(nn.Module):
	def __init__(self, in_planes: int = 3+4+3+1, ch: int = POLICY_CHANNELS, blocks: int = RES_BLOCKS):
		super().__init__()
		self.stem = nn.Sequential(
			nn.Conv2d(in_planes, ch, 3, padding=1, bias=False),
			nn.BatchNorm2d(ch),
			nn.ReLU(inplace=True),
		)
		self.res = nn.Sequential(*[ResidualBlock(ch) for _ in range(blocks)])
		# policy head: conv to planes then linear to NUM_ACTIONS
		self.policy_head = nn.Sequential(
			nn.Conv2d(ch, 2, 1),
			nn.BatchNorm2d(2),
			nn.ReLU(inplace=True),
		)
		self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, NUM_ACTIONS)
		# value head
		self.value_head = nn.Sequential(
			nn.Conv2d(ch, 1, 1),
			nn.BatchNorm2d(1),
			nn.ReLU(inplace=True),
		)
		self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, ch)
		self.value_fc2 = nn.Linear(ch, 1)

	def forward(self, x):
		# x: [B,C,H,W]
		h = self.stem(x)
		h = self.res(h)
		# policy
		p = self.policy_head(h)
		p = p.view(p.size(0), -1)
		p = self.policy_fc(p)
		# value
		v = self.value_head(h)
		v = v.view(v.size(0), -1)
		v = F.relu(self.value_fc1(v))
		v = torch.tanh(self.value_fc2(v)).squeeze(-1)
		return p, v
