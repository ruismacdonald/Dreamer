import os
import numpy as np

cnt_grid = 150
bb, aa = np.meshgrid(
    np.linspace(-0.2, 0.2, cnt_grid), np.linspace(-0.2, 0.2, cnt_grid)
)
pos_pairs = np.vstack((bb.flatten(), aa.flatten())).T

T2_center = np.array([-0.1, -0.1])
T1_center = np.array([0.13, 0.1])
radius = 0.07

data = []
for pos in pos_pairs:
    x, y = pos[0], pos[1]
    if np.linalg.norm(pos - T2_center) < radius:
        reward = 2.0
    elif np.linalg.norm(pos - T1_center) < radius:
        reward = 1.0
    else:
        reward = 0.0
    data.append([x, y, reward])

out_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rewards/reacherloca_optimal/heat_data_optimal.npy")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "wb") as f:
    np.save(f, np.array(data))