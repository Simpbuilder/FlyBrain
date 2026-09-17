import json
import numpy as np
import pandas as pd


def process(path):
    with open(path) as f:
        d = json.load(f)
    i = np.array(d['i']); t = np.array(d['t_ms']); cat = np.array(d['cat'])
    x = np.array(d['x']); y = np.array(d['y']); z = np.array(d['z'])

    hi_mask = np.isin(cat, ['odor_A_kc', 'ppl_mbon', 'pam_mbon'])
    hi_i, hi_t, hi_cat = i[hi_mask], t[hi_mask], cat[hi_mask]
    hi_x, hi_y, hi_z = x[hi_mask], y[hi_mask], z[hi_mask]
    neurons = {}
    for nid, nx, ny, nz, ncat in zip(hi_i, hi_x, hi_y, hi_z, hi_cat):
        nid = int(nid)
        if nid not in neurons:
            neurons[nid] = {'x': float(nx), 'y': float(ny), 'z': float(nz), 'cat': str(ncat)}
    spikes = sorted(zip(hi_i.tolist(), np.round(hi_t, 1).tolist()), key=lambda p: p[1])
    spikes = [[int(a), round(float(b), 1)] for a, b in spikes]
    print(f'{path}: {len(neurons)} highlighted neurons, {len(spikes)} spikes')
    return {'neurons': neurons, 'spikes': spikes}


before = process('../compartment_before_3d.json')
after = process('../compartment_after_3d.json')

# a subsampled real 3D point cloud for the silhouette (every neuron would be
# too many points for smooth WebGL interaction - 15k is plenty for shape)
ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
pos = ann.drop_duplicates('root_id')[['pos_x', 'pos_y', 'pos_z']].dropna()
rng = np.random.default_rng(0)
sample = pos.sample(n=min(15000, len(pos)), random_state=0)
silhouette = sample.round(0).astype(int).values.tolist()

xs, ys, zs = pos['pos_x'].to_numpy(), pos['pos_y'].to_numpy(), pos['pos_z'].to_numpy()
bounds = {'x_min': float(xs.min()), 'x_max': float(xs.max()),
          'y_min': float(ys.min()), 'y_max': float(ys.max()),
          'z_min': float(zs.min()), 'z_max': float(zs.max())}

with open('../viz_3d.json', 'w') as f:
    json.dump({'before': before, 'after': after, 'silhouette': silhouette, 'bounds': bounds}, f)
import os
print('viz_3d.json size (KB):', os.path.getsize('../viz_3d.json') / 1024)
