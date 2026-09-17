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
kc_ids = set(ann.loc[ann['cell_class'] == 'Kenyon_Cell', 'root_id'])
mbon_ids = set(ann.loc[ann['cell_type'].astype(str).str.startswith('MBON', na=False), 'root_id'])
kc_idx = sorted(flyid2i[x] for x in kc_ids if x in flyid2i)
mbon_idx = sorted(flyid2i[x] for x in mbon_ids if x in flyid2i)
mbon_arr = np.array(mbon_idx)

rng = np.random.default_rng(0)
shuffled = rng.permutation(kc_idx)
max_n = 250
superset = shuffled[:max_n].tolist()  # nested prefixes = nested odor ensembles

t0 = time.time()
neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
print(f'>>> network built in {time.time()-t0:.1f}s')

stim = NeuronGroup(len(superset), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(superset)), j=np.array(superset))

net = Network(neu, syn, spk_mon, stim, drive)

t_run = 300 * ms
r_stim = 150 * Hz
n_reps = 8
sizes = [50, 75, 100, 125, 150, 200, 250]
high_thresh = 40  # Hz - midpoint between the two observed clusters (~5Hz vs ~80Hz)

results = []
for n in sizes:
    positions = np.arange(0, n)
    rates = []
    for r in range(n_reps):
        neu.v = params['v_0']
        neu.g = 0 * mV
        stim.rate = 0 * Hz
        stim.rate[positions] = r_stim
        t_start = defaultclock.t / second
        net.run(t_run)
        mask = np.asarray(spk_mon.t / second) >= t_start
        trial_i = np.asarray(spk_mon.i)[mask]
        mbon_spikes = np.isin(trial_i, mbon_arr).sum()
        rate = mbon_spikes / len(mbon_idx) / (t_run / (1000 * ms))
        rates.append(rate)
    p_high = float(np.mean(np.array(rates) > high_thresh))
    print(f'n={n:3d} KCs: rates={[round(x,1) for x in rates]}  P(high state)={p_high:.2f}')
    results.append({'n_kc': n, 'rates': rates, 'p_high': p_high, 'mean': float(np.mean(rates)), 'std': float(np.std(rates))})

with open('../ensemble_sweep_results.json', 'w') as f:
    json.dump(results, f)
print('>>> wrote ensemble_sweep_results.json')
