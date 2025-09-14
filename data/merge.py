import numpy as np

# 加载所有数据
files = ['data/sp01.npz', 'data/sp02.npz', 'data/sp03.npz']
S, P, Z = [], [], []
for f in files:
    data = np.load(f)
    S.append(data['s'])
    P.append(data['p'])
    Z.append(data['z'])

# 合并
S = np.concatenate(S, axis=0)
P = np.concatenate(P, axis=0)
Z = np.concatenate(Z, axis=0)

# 保存为新 npz
np.savez('data/sp01-03.npz', s=S, p=P, z=Z)