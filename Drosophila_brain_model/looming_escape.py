import time
import numpy as np
import pandas as pd
import json

from model import create_model, default_params as params
from brian2 import Network, NeuronGroup, Synapses, ms, Hz, mV, second, defaultclock

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
print(f'>>> LC4: {len(lc4_idx)}, LPLC2: {len(lplc2_idx)}, Giant Fiber (DNp01): {len(gf_idx)} -> {gf_idx}')

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


def run_trial():
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
    return len(trial_i), gf_spikes, gf_times


# warm-up (same transient as the mushroom body experiment)
for _ in range(3):
    run_trial()
print('>>> warm-up complete')

n_reps = 5
for r in range(n_reps):
    total, gf_spikes, gf_times = run_trial()
    gf0 = gf_times[gf_spikes == gf_idx[0]] if len(gf_idx) > 0 else []
    gf1 = gf_times[gf_spikes == gf_idx[1]] if len(gf_idx) > 1 else []
    print(f'trial {r+1}: {total} total spikes brain-wide, Giant Fiber L={len(gf0)} spikes, R={len(gf1)} spikes'
          f'  (first GF spike at {round(float(gf_times.min()),1) if len(gf_times) else "never"} ms)')
