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

neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
syn_i = np.asarray(syn.i[:])
dn_arr = np.array(dng46_idx)
dn_out_idx = np.where(np.isin(syn_i, dn_arr))[0]
downstream = np.unique(np.asarray(syn.j[:])[dn_out_idx])
print(f'>>> {len(dn_out_idx)} synapses out of DNg46, reaching {len(downstream)} downstream neurons')

stim = NeuronGroup(len(hsvs_idx), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(hsvs_idx)), j=np.array(hsvs_idx))
net = Network(neu, syn, spk_mon, stim, drive)
t_run = 300 * ms
dn_w_saved = np.array(syn.w[dn_out_idx] / mV)


def run_trial(silence, n_reps=5):
    rates = []
    for _ in range(n_reps):
        neu.v = params['v_0']
        neu.g = 0 * mV
        stim.rate = 0 * Hz
        stim.rate[np.arange(len(hsvs_idx))] = 150 * Hz
        syn.w[dn_out_idx] = (np.zeros_like(dn_w_saved) if silence else dn_w_saved) * mV
        t_start = defaultclock.t / second
        net.run(t_run)
        mask = np.asarray(spk_mon.t / second) >= t_start
        trial_i = np.asarray(spk_mon.i)[mask]
        rates.append(np.isin(trial_i, downstream).sum() / len(downstream) / (t_run / (1000 * ms)))
    return rates


for _ in range(3):
    run_trial(False, n_reps=1)
print('>>> warm-up complete')

intact = run_trial(False)
silenced = run_trial(True)
print(f'DNg46 downstream, intact:   {np.mean(intact):.2f} +/- {np.std(intact):.2f} Hz')
print(f'DNg46 downstream, silenced: {np.mean(silenced):.2f} +/- {np.std(silenced):.2f} Hz')
print(f'change: {100*(np.mean(silenced)-np.mean(intact))/np.mean(intact):+.1f}%')
