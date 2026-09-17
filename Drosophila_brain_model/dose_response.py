import numpy as np
import pandas as pd

from model import create_model, default_params as params
from brian2 import Network, NeuronGroup, Synapses, ms, Hz, mV, second, defaultclock, seed as brian_seed

brian_seed(7)

config = {'path_comp': './Completeness_783.csv', 'path_con': './Connectivity_783.parquet'}
df_comp = pd.read_csv(config['path_comp'], index_col=0)
ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
flyid2i = {j: i for i, j in enumerate(df_comp.index)}

loom_ids = set(ann.loc[ann['cell_type'].isin(['LC4', 'LPLC2']), 'root_id'])
gf_ids = set(ann.loc[ann['cell_type'] == 'DNp01', 'root_id'])
rng = np.random.default_rng(0)
loom_idx = list(rng.permutation(sorted(flyid2i[x] for x in loom_ids if x in flyid2i)))
gf_arr = np.array(sorted(flyid2i[x] for x in gf_ids if x in flyid2i))

neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
stim = NeuronGroup(len(loom_idx), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(loom_idx)), j=np.array(loom_idx))
net = Network(neu, syn, spk_mon, stim, drive)
t_run = 300 * ms


def run_trial(n_active, n_reps=4):
    gf_n = []
    for _ in range(n_reps):
        neu.v = params['v_0']
        neu.g = 0 * mV
        stim.rate = 0 * Hz
        stim.rate[np.arange(n_active)] = 150 * Hz
        t_start = defaultclock.t / second
        net.run(t_run)
        mask = np.asarray(spk_mon.t / second) >= t_start
        trial_i = np.asarray(spk_mon.i)[mask]
        gf_n.append(np.isin(trial_i, gf_arr).sum())
    return np.mean(gf_n), np.std(gf_n)


for _ in range(3):
    run_trial(len(loom_idx), n_reps=1)
print('>>> warm-up complete')

fractions = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
for frac in fractions:
    n_active = max(1, int(round(frac * len(loom_idx))))
    mean_gf, std_gf = run_trial(n_active)
    print(f'{100*frac:5.0f}% of looming detectors ({n_active:3d}/{len(loom_idx)}): GF = {mean_gf:.1f} +/- {std_gf:.1f} spikes')
