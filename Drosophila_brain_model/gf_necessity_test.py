import numpy as np
import pandas as pd

from model import create_model, default_params as params
from brian2 import Network, NeuronGroup, Synapses, ms, Hz, mV, second, defaultclock, seed as brian_seed

brian_seed(7)

config = {'path_comp': './Completeness_783.csv', 'path_con': './Connectivity_783.parquet'}
df_comp = pd.read_csv(config['path_comp'], index_col=0)
ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
flyid2i = {j: i for i, j in enumerate(df_comp.index)}

lc4_ids = set(ann.loc[ann['cell_type'] == 'LC4', 'root_id'])
lplc2_ids = set(ann.loc[ann['cell_type'] == 'LPLC2', 'root_id'])
gf_ids = set(ann.loc[ann['cell_type'] == 'DNp01', 'root_id'])
loom_idx = sorted(flyid2i[x] for x in (lc4_ids | lplc2_ids) if x in flyid2i)
gf_idx = sorted(flyid2i[x] for x in gf_ids if x in flyid2i)

neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
syn_i = np.asarray(syn.i[:])
gf_arr = np.array(gf_idx)
gf_out_mask = np.isin(syn_i, gf_arr)  # all synapses FROM the Giant Fiber
gf_out_idx = np.where(gf_out_mask)[0]
downstream = np.unique(np.asarray(syn.j[:])[gf_out_idx])
print(f'>>> {len(gf_out_idx)} synapses out of the Giant Fiber, reaching {len(downstream)} distinct downstream neurons')

stim = NeuronGroup(len(loom_idx), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(loom_idx)), j=np.array(loom_idx))
net = Network(neu, syn, spk_mon, stim, drive)
t_run = 300 * ms
gf_w_saved = np.array(syn.w[gf_out_idx] / mV)


def run_trial(silence_gf, n_reps=5):
    rates = []
    for _ in range(n_reps):
        neu.v = params['v_0']
        neu.g = 0 * mV
        stim.rate = 0 * Hz
        stim.rate[np.arange(len(loom_idx))] = 150 * Hz
        syn.w[gf_out_idx] = (np.zeros_like(gf_w_saved) if silence_gf else gf_w_saved) * mV
        t_start = defaultclock.t / second
        net.run(t_run)
        mask = np.asarray(spk_mon.t / second) >= t_start
        trial_i = np.asarray(spk_mon.i)[mask]
        n_downstream_spikes = np.isin(trial_i, downstream).sum()
        rates.append(n_downstream_spikes / len(downstream) / (t_run / (1000 * ms)))
    return rates


for _ in range(3):
    run_trial(False, n_reps=1)
print('>>> warm-up complete')

intact = run_trial(False)
silenced = run_trial(True)
print(f'GF downstream population rate, GF intact:   {np.mean(intact):.2f} +/- {np.std(intact):.2f} Hz')
print(f'GF downstream population rate, GF silenced: {np.mean(silenced):.2f} +/- {np.std(silenced):.2f} Hz')
pct = 100 * (np.mean(silenced) - np.mean(intact)) / np.mean(intact)
print(f'change: {pct:+.1f}% (looming detectors driven identically both times, only GF silenced)')
