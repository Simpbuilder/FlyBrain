import numpy as np
import pandas as pd
import json

from model import create_model, default_params as params
from brian2 import Network, NeuronGroup, Synapses, ms, Hz, mV, second, defaultclock, seed as brian_seed

brian_seed(7)

config = {'path_comp': './Completeness_783.csv', 'path_con': './Connectivity_783.parquet'}
df_comp = pd.read_csv(config['path_comp'], index_col=0)
ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
flyid2i = {j: i for i, j in enumerate(df_comp.index)}
i2flyid = {i: j for j, i in flyid2i.items()}

odor_A_glom = ['DA1_lPN', 'DA2_lPN', 'DL3_lPN', 'DL2d_adPN']
odor_B_glom = ['VM5d_adPN', 'VA1v_adPN', 'DL2v_adPN', 'VC3_adPN']
pnA_ids = set(ann.loc[ann['cell_type'].isin(odor_A_glom), 'root_id'])
pnB_ids = set(ann.loc[ann['cell_type'].isin(odor_B_glom), 'root_id'])
pnA_idx = sorted(flyid2i[x] for x in pnA_ids if x in flyid2i)
pnB_idx = sorted(flyid2i[x] for x in pnB_ids if x in flyid2i)
kc_ids = set(ann.loc[ann['cell_class'] == 'Kenyon_Cell', 'root_id'])
kc_idx = np.array(sorted(flyid2i[x] for x in kc_ids if x in flyid2i))
print(f'>>> odor A: {len(pnA_idx)} real PNs, odor B: {len(pnB_idx)} real PNs')

neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
stim = NeuronGroup(len(pnA_idx) + len(pnB_idx), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
all_pn = pnA_idx + pnB_idx
drive.connect(i=np.arange(len(all_pn)), j=np.array(all_pn))
A_stim = np.arange(0, len(pnA_idx))
B_stim = np.arange(len(pnA_idx), len(all_pn))
net = Network(neu, syn, spk_mon, stim, drive)

sniff_ms = 12  # short enough to stay in the sparse-coding regime (see sparsity_sweep.py)


def sniff(positions, n_reps=5):
    kc_counts = {}
    for _ in range(n_reps):
        neu.v = params['v_0']
        neu.g = 0 * mV
        stim.rate = 0 * Hz
        stim.rate[positions] = 150 * Hz
        t_start = defaultclock.t / second
        net.run(sniff_ms * ms)
        mask = np.asarray(spk_mon.t / second) >= t_start
        trial_i = np.asarray(spk_mon.i)[mask]
        for k in np.unique(trial_i[np.isin(trial_i, kc_idx)]):
            kc_counts[k] = kc_counts.get(k, 0) + 1
    return kc_counts


for _ in range(3):
    sniff(B_stim, n_reps=1)
print('>>> warm-up complete')

counts_A = sniff(A_stim, n_reps=8)
counts_B = sniff(B_stim, n_reps=8)
# reliable responders: KCs that fired in at least half the sniff reps
reliable_A = sorted(int(k) for k, c in counts_A.items() if c >= 4)
reliable_B = sorted(int(k) for k, c in counts_B.items() if c >= 4)
overlap = set(reliable_A) & set(reliable_B)
print(f'odor A: {len(counts_A)} KCs fired at least once, {len(reliable_A)} reliably (>=4/8 sniffs)')
print(f'odor B: {len(counts_B)} KCs fired at least once, {len(reliable_B)} reliably (>=4/8 sniffs)')
print(f'overlap between reliable A and reliable B: {len(overlap)} ({100*len(overlap)/max(1,min(len(reliable_A),len(reliable_B))):.1f}% of the smaller set)')

with open('../sniff_identified_kcs.json', 'w') as f:
    json.dump({'odor_A_kc': reliable_A, 'odor_B_kc': reliable_B}, f)
print('>>> wrote sniff_identified_kcs.json')
