import time
import numpy as np
import pandas as pd

from model import create_model, default_params as params
from brian2 import Network, NeuronGroup, Synapses, ms, Hz, mV, second, defaultclock, seed as brian_seed

brian_seed(7)

config = {'path_comp': './Completeness_783.csv', 'path_con': './Connectivity_783.parquet'}
df_comp = pd.read_csv(config['path_comp'], index_col=0)
ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
flyid2i = {j: i for i, j in enumerate(df_comp.index)}

hsvs_ids = set(ann.loc[ann['cell_type'].astype(str).str.match('^HS|^VS', na=False), 'root_id'])
hsvs_idx = sorted(flyid2i[x] for x in hsvs_ids if x in flyid2i)
dng46_ids = set(ann.loc[ann['cell_type'] == 'DNg46', 'root_id'])
dng46_idx = sorted(flyid2i[x] for x in dng46_ids if x in flyid2i)
print(f'>>> HS/VS (wide-field motion neurons): {len(hsvs_idx)}, DNg46 (steering descending neuron): {len(dng46_idx)}')

neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
stim = NeuronGroup(len(hsvs_idx), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(hsvs_idx)), j=np.array(hsvs_idx))
net = Network(neu, syn, spk_mon, stim, drive)
t_run = 300 * ms
dng46_arr = np.array(dng46_idx)


def run_trial():
    neu.v = params['v_0']
    neu.g = 0 * mV
    stim.rate = 0 * Hz
    stim.rate[np.arange(len(hsvs_idx))] = 150 * Hz
    t_start = defaultclock.t / second
    net.run(t_run)
    mask = np.asarray(spk_mon.t / second) >= t_start
    trial_i = np.asarray(spk_mon.i)[mask]
    trial_t = (np.asarray(spk_mon.t / second)[mask] - t_start) * 1000
    dn_spikes = trial_i[np.isin(trial_i, dng46_arr)]
    dn_times = trial_t[np.isin(trial_i, dng46_arr)]
    return len(dn_spikes), (float(dn_times.min()) if len(dn_times) else None)


for _ in range(3):
    run_trial()
print('>>> warm-up complete')

n_reps = 15
spikes_list, first_ts = [], []
for r in range(n_reps):
    n, first_t = run_trial()
    spikes_list.append(n)
    if first_t is not None:
        first_ts.append(first_t)
    print(f'trial {r+1:2d}: DNg46 spikes = {n}, first spike at {first_t} ms' if first_t else f'trial {r+1}: DNg46 silent')

import numpy as np
print()
print(f'=== summary across {n_reps} trials ===')
print(f'DNg46 fired in {sum(1 for s in spikes_list if s > 0)}/{n_reps} trials')
print(f'spikes/trial: {np.mean(spikes_list):.1f} +/- {np.std(spikes_list):.1f}')
print(f'first-spike latency: {np.mean(first_ts):.1f} +/- {np.std(first_ts):.1f} ms (range {min(first_ts):.1f}-{max(first_ts):.1f})')
