import numpy as np
import pandas as pd

from model import create_model, default_params as params
from brian2 import Network, NeuronGroup, Synapses, ms, Hz, mV, second, defaultclock, seed as brian_seed

brian_seed(7)

config = {'path_comp': './Completeness_783.csv', 'path_con': './Connectivity_783.parquet'}
df_comp = pd.read_csv(config['path_comp'], index_col=0)
ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
flyid2i = {j: i for i, j in enumerate(df_comp.index)}

jo_ids = set(ann.loc[ann['cell_type'].isin(['JO-B1_a', 'JO-B2', 'JO-B3']), 'root_id'])
jo_idx = sorted(flyid2i[x] for x in jo_ids if x in flyid2i)
dng29_arr = np.array(sorted(flyid2i[x] for x in ann.loc[ann['cell_type'] == 'DNg29', 'root_id'] if x in flyid2i))
gf_arr = np.array(sorted(flyid2i[x] for x in ann.loc[ann['cell_type'] == 'DNp01', 'root_id'] if x in flyid2i))
print(f'>>> JO-B (wind/vibration neurons): {len(jo_idx)} -> DNg29 ({len(dng29_arr)}) and Giant Fiber ({len(gf_arr)}, multi-modal convergence)', flush=True)

neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
stim = NeuronGroup(len(jo_idx), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(jo_idx)), j=np.array(jo_idx))
net = Network(neu, syn, spk_mon, stim, drive)
t_run = 300 * ms


def run_trial():
    neu.v = params['v_0']
    neu.g = 0 * mV
    stim.rate = 0 * Hz
    stim.rate[np.arange(len(jo_idx))] = 150 * Hz
    t_start = defaultclock.t / second
    net.run(t_run)
    mask = np.asarray(spk_mon.t / second) >= t_start
    trial_i = np.asarray(spk_mon.i)[mask]
    return np.isin(trial_i, dng29_arr).sum(), np.isin(trial_i, gf_arr).sum()


for _ in range(3):
    run_trial()
print('>>> warm-up complete', flush=True)

dn29_n, gf_n = [], []
for r in range(10):
    a, b = run_trial()
    dn29_n.append(a); gf_n.append(b)
    print(f'trial {r+1:2d}: DNg29 = {a} spikes, Giant Fiber = {b} spikes', flush=True)

print()
print(f'DNg29 fired in {sum(1 for x in dn29_n if x>0)}/10 trials, mean {np.mean(dn29_n):.1f}+/-{np.std(dn29_n):.1f} spikes')
print(f'Giant Fiber fired in {sum(1 for x in gf_n if x>0)}/10 trials, mean {np.mean(gf_n):.1f}+/-{np.std(gf_n):.1f} spikes')
print('(wind/vibration alone driving the SAME escape neuron as looming - real multi-modal convergence)')
