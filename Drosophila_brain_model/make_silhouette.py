import json
import numpy as np
import pandas as pd

ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
pos = ann.drop_duplicates('root_id')[['pos_x', 'pos_y']].dropna()
X_MIN, X_MAX, Y_MIN, Y_MAX = 20000, 228000, 12000, 112000
GRID_W, GRID_H = 100, 62

xs, ys = pos['pos_x'].to_numpy(), pos['pos_y'].to_numpy()
gx = np.clip(((xs - X_MIN) / (X_MAX - X_MIN) * GRID_W).astype(int), 0, GRID_W - 1)
gy = np.clip(((ys - Y_MIN) / (Y_MAX - Y_MIN) * GRID_H).astype(int), 0, GRID_H - 1)
grid = np.zeros((GRID_H, GRID_W), dtype=np.int32)
np.add.at(grid, (gy, gx), 1)

out = {'grid': grid.tolist(), 'grid_w': GRID_W, 'grid_h': GRID_H,
       'x_min': X_MIN, 'x_max': X_MAX, 'y_min': Y_MIN, 'y_max': Y_MAX}
with open('../dashboard_silhouette.json', 'w') as f:
    json.dump(out, f)
import os
print('silhouette size (KB):', os.path.getsize('../dashboard_silhouette.json') / 1024)
