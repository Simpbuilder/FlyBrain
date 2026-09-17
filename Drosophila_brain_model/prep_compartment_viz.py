import json
import numpy as np
import pandas as pd

X_MIN, X_MAX = 20000, 228000
Y_MIN, Y_MAX = 12000, 112000
GRID_W, GRID_H = 90, 55
N_TBINS = 30
T_MAX = 300.0


def process(path, label):
    with open(path) as f:
        d = json.load(f)
    i = np.array(d['i']); t = np.array(d['t_ms']); cat = np.array(d['cat'])
    x = np.array(d['x']); y = np.array(d['y'])

    bg_mask = ~np.isin(cat, ['odor_A_kc', 'ppl_mbon', 'pam_mbon'])
    gx = np.clip(((x[bg_mask] - X_MIN) / (X_MAX - X_MIN) * GRID_W).astype(int), 0, GRID_W - 1)
    gy = np.clip(((y[bg_mask] - Y_MIN) / (Y_MAX - Y_MIN) * GRID_H).astype(int), 0, GRID_H - 1)
    gt = np.clip((t[bg_mask] / T_MAX * N_TBINS).astype(int), 0, N_TBINS - 1)
    grid = np.zeros((N_TBINS, GRID_H, GRID_W), dtype=np.int32)
    np.add.at(grid, (gt, gy, gx), 1)

    hi_mask = np.isin(cat, ['odor_A_kc', 'ppl_mbon', 'pam_mbon'])
    hi_i, hi_t, hi_cat, hi_x, hi_y = i[hi_mask], t[hi_mask], cat[hi_mask], x[hi_mask], y[hi_mask]
    neurons = {}
    for nid, nx, ny, ncat in zip(hi_i, hi_x, hi_y, hi_cat):
        nid = int(nid)
        if nid not in neurons:
            neurons[nid] = {'x': float(nx), 'y': float(ny), 'cat': str(ncat)}
    spikes = sorted(zip(hi_i.tolist(), np.round(hi_t, 1).tolist()), key=lambda p: p[1])
    spikes = [[int(a), round(float(b), 1)] for a, b in spikes]

    print(f'{label}: {len(neurons)} highlighted neurons ({sum(1 for v in neurons.values() if v["cat"]=="ppl_mbon")} PPL-MBON, {sum(1 for v in neurons.values() if v["cat"]=="pam_mbon")} PAM-MBON), {len(spikes)} highlighted spikes')
    return {'grid': grid.tolist(), 'grid_w': GRID_W, 'grid_h': GRID_H, 'n_tbins': N_TBINS, 't_max': T_MAX,
            'x_min': X_MIN, 'x_max': X_MAX, 'y_min': Y_MIN, 'y_max': Y_MAX,
            'neurons': neurons, 'spikes': spikes}


before = process('../compartment_before.json', 'before')
after = process('../compartment_after.json', 'after')

df_comp = pd.read_csv('Completeness_783.csv', index_col=0)
ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
ann_pos = ann.drop_duplicates('root_id').set_index('root_id')[['pos_x', 'pos_y']]
pos_df = ann_pos.reindex(df_comp.index).dropna()
sx = pos_df['pos_x'].to_numpy(); sy = pos_df['pos_y'].to_numpy()
SIL_W, SIL_H = 160, 100
sgx = np.clip(((sx - X_MIN) / (X_MAX - X_MIN) * SIL_W).astype(int), 0, SIL_W - 1)
sgy = np.clip(((sy - Y_MIN) / (Y_MAX - Y_MIN) * SIL_H).astype(int), 0, SIL_H - 1)
sil_grid = np.zeros((SIL_H, SIL_W), dtype=np.int32)
np.add.at(sil_grid, (sgy, sgx), 1)

with open('../compartment_viz.json', 'w') as f:
    json.dump({'before': before, 'after': after, 'silhouette': sil_grid.tolist(), 'sil_w': SIL_W, 'sil_h': SIL_H}, f)
import os
print('output size (KB):', os.path.getsize('../compartment_viz.json') / 1024)
