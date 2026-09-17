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
hsvs_ids = set(ann.loc[ann['cell_type'].astype(str).str.match('^HS|^VS', na=False), 'root_id'])
gf_ids = set(ann.loc[ann['cell_type'] == 'DNp01', 'root_id'])
dng46_ids = set(ann.loc[ann['cell_type'] == 'DNg46', 'root_id'])

loom_idx = sorted(flyid2i[x] for x in loom_ids if x in flyid2i)
hsvs_idx = sorted(flyid2i[x] for x in hsvs_ids if x in flyid2i)
gf_arr = np.array(sorted(flyid2i[x] for x in gf_ids if x in flyid2i))
dng46_arr = np.array(sorted(flyid2i[x] for x in dng46_ids if x in flyid2i))
print(f'>>> looming: {len(loom_idx)} neurons -> GF; optomotor: {len(hsvs_idx)} neurons -> DNg46')

neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
all_stim = loom_idx + hsvs_idx
stim = NeuronGroup(len(all_stim), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(all_stim)), j=np.array(all_stim))
loom_stim = np.arange(0, len(loom_idx))
hsvs_stim = np.arange(len(loom_idx), len(all_stim))
net = Network(neu, syn, spk_mon, stim, drive)
t_run = 300 * ms


def run_trial(loom_on, hsvs_on, n_reps=5):
    gf_n, dn_n = [], []
    for _ in range(n_reps):
        neu.v = params['v_0']
        neu.g = 0 * mV
        stim.rate = 0 * Hz
        if loom_on:
            stim.rate[loom_stim] = 150 * Hz
        if hsvs_on:
            stim.rate[hsvs_stim] = 150 * Hz
        t_start = defaultclock.t / second
        net.run(t_run)
        mask = np.asarray(spk_mon.t / second) >= t_start
        trial_i = np.asarray(spk_mon.i)[mask]
        gf_n.append(np.isin(trial_i, gf_arr).sum())
        dn_n.append(np.isin(trial_i, dng46_arr).sum())
    return np.mean(gf_n), np.mean(dn_n)


for _ in range(3):
    run_trial(True, False, n_reps=1)
print('>>> warm-up complete')

gf_alone, dn_alone_with_loom = run_trial(True, False)
gf_with_hsvs, dn_alone = run_trial(False, True)
gf_both, dn_both = run_trial(True, True)

print(f'looming alone:        GF={gf_alone:.1f} spikes, DNg46={dn_alone_with_loom:.1f} spikes')
print(f'optomotor alone:      GF={gf_with_hsvs:.1f} spikes, DNg46={dn_alone:.1f} spikes')
print(f'both simultaneously:  GF={gf_both:.1f} spikes, DNg46={dn_both:.1f} spikes')
print()
print(f'GF: {gf_alone:.1f} (alone) -> {gf_both:.1f} (both)  ({100*(gf_both-gf_alone)/gf_alone:+.1f}%)')
print(f'DNg46: {dn_alone:.1f} (alone) -> {dn_both:.1f} (both)  ({100*(dn_both-dn_alone)/dn_alone:+.1f}%)')
