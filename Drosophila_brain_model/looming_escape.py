import time
import numpy as np
import pandas as pd
import json

from model import create_model, default_params as params
from brian2 import Network, NeuronGroup, Synapses, ms, Hz, mV, second, defaultclock, seed as brian_seed

# Brian2's own RNG (drives the Poisson-threshold stim mechanism) is
# separate from numpy's rng - without seeding it, identical scripts can
# land in different stochastic outcomes run to run. Fixed for reproducibility.
brian_seed(7)

config = {'path_comp': './Completeness_783.csv', 'path_con': './Connectivity_783.parquet'}

df_comp = pd.read_csv(config['path_comp'], index_col=0)
ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
flyid2i = {j: i for i, j in enumerate(df_comp.index)}

lc4_ids = set(ann.loc[ann['cell_type'] == 'LC4', 'root_id'])
lplc2_ids = set(ann.loc[ann['cell_type'] == 'LPLC2', 'root_id'])
gf_ids = set(ann.loc[ann['cell_type'] == 'DNp01', 'root_id'])

lc4_idx = sorted(flyid2i[x] for x in lc4_ids if x in flyid2i)
lplc2_idx = sorted(flyid2i[x] for x in lplc2_ids if x in flyid2i)
gf_idx = sorted(flyid2i[x] for x in gf_ids if x in flyid2i)
loom_idx = lc4_idx + lplc2_idx
lc4_arr = np.array(lc4_idx)
lplc2_arr = np.array(lplc2_idx)
print(f'>>> LC4: {len(lc4_idx)}, LPLC2: {len(lplc2_idx)}, Giant Fiber (DNp01): {len(gf_idx)} -> {gf_idx}')

ann_pos = ann.drop_duplicates('root_id').set_index('root_id')[['pos_x', 'pos_y']]
comp_ids = df_comp.index.to_numpy()
pos_df = ann_pos.reindex(comp_ids)
pos_x = pos_df['pos_x'].to_numpy()
pos_y = pos_df['pos_y'].to_numpy()

t0 = time.time()
neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
print(f'>>> network built in {time.time() - t0:.1f}s')

stim = NeuronGroup(len(loom_idx), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(loom_idx)), j=np.array(loom_idx))

net = Network(neu, syn, spk_mon, stim, drive)

t_run = 300 * ms
r_stim = 150 * Hz
gf_arr = np.array(gf_idx)


def category_for(idx_arr):
    cat = np.full(idx_arr.shape, 'other', dtype=object)
    cat[np.isin(idx_arr, lc4_arr)] = 'lc4'
    cat[np.isin(idx_arr, lplc2_arr)] = 'lplc2'
    cat[np.isin(idx_arr, gf_arr)] = 'gf'
    return cat


def run_trial(capture=False):
    neu.v = params['v_0']
    neu.g = 0 * mV
    stim.rate = 0 * Hz
    stim.rate[np.arange(len(loom_idx))] = r_stim
    t_start = defaultclock.t / second
    net.run(t_run)
    mask = np.asarray(spk_mon.t / second) >= t_start
    trial_i = np.asarray(spk_mon.i)[mask]
    trial_t = (np.asarray(spk_mon.t / second)[mask] - t_start) * 1000
    gf_spikes = trial_i[np.isin(trial_i, gf_arr)]
    gf_times = trial_t[np.isin(trial_i, gf_arr)]

    cap = None
    if capture:
        cats = category_for(trial_i)
        has_pos = ~np.isnan(pos_x[trial_i])
        cap = {
            'i': trial_i[has_pos].tolist(),
            't_ms': np.round(trial_t[has_pos], 2).tolist(),
            'cat': cats[has_pos].tolist(),
            'x': pos_x[trial_i][has_pos].tolist(),
            'y': pos_y[trial_i][has_pos].tolist(),
        }
    return len(trial_i), gf_spikes, gf_times, cap


# warm-up (same transient as the mushroom body experiment)
for _ in range(3):
    run_trial()
print('>>> warm-up complete')

n_reps = 5
capture = None
for r in range(n_reps):
    total, gf_spikes, gf_times, cap = run_trial(capture=(r == n_reps - 1))
    if cap is not None:
        capture = cap
    gf0 = gf_times[gf_spikes == gf_idx[0]] if len(gf_idx) > 0 else []
    gf1 = gf_times[gf_spikes == gf_idx[1]] if len(gf_idx) > 1 else []
    print(f'trial {r+1}: {total} total spikes brain-wide, Giant Fiber L={len(gf0)} spikes, R={len(gf1)} spikes'
          f'  (first GF spike at {round(float(gf_times.min()),1) if len(gf_times) else "never"} ms)')

with open('../looming_capture.json', 'w') as f:
    json.dump(capture, f)
print(f'>>> wrote looming_capture.json ({len(capture["i"])} spikes)')
