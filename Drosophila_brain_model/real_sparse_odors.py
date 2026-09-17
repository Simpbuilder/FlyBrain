import json
import numpy as np
import pandas as pd

from model import create_model, default_params as params
from brian2 import Network, NeuronGroup, Synapses, ms, Hz, mV, second, defaultclock, seed as brian_seed

brian_seed(7)

config = {'path_comp': './Completeness_783.csv', 'path_con': './Connectivity_783.parquet'}
df_comp = pd.read_csv(config['path_comp'], index_col=0)
ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
flyid2i = {j: i for i, j in enumerate(df_comp.index)}

odor_A_glom = ['DA1_lPN', 'DA2_lPN', 'DL3_lPN', 'DL2d_adPN']
odor_B_glom = ['VM5d_adPN', 'VA1v_adPN', 'DL2v_adPN', 'VC3_adPN']
pnA_ids = set(ann.loc[ann['cell_type'].isin(odor_A_glom), 'root_id'])
pnB_ids = set(ann.loc[ann['cell_type'].isin(odor_B_glom), 'root_id'])
pnA_idx = sorted(flyid2i[x] for x in pnA_ids if x in flyid2i)
pnB_idx = sorted(flyid2i[x] for x in pnB_ids if x in flyid2i)
kc_ids = set(ann.loc[ann['cell_class'] == 'Kenyon_Cell', 'root_id'])
kc_idx = np.array(sorted(flyid2i[x] for x in kc_ids if x in flyid2i))
print(f'>>> odor A: {len(pnA_idx)} PNs, odor B: {len(pnB_idx)} PNs', flush=True)

neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
all_pn = pnA_idx + pnB_idx
stim = NeuronGroup(len(all_pn), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(all_pn)), j=np.array(all_pn))
A_stim = np.arange(0, len(pnA_idx))
B_stim = np.arange(len(pnA_idx), len(all_pn))
net = Network(neu, syn, spk_mon, stim, drive)

sniff_ms = 10  # the sparse-coding sweet spot found by the grid search


def sniff(positions):
    neu.v = params['v_0']
    neu.g = 0 * mV
    stim.rate = 0 * Hz
    stim.rate[positions] = 150 * Hz
    t_start = defaultclock.t / second
    net.run(sniff_ms * ms)
    mask = np.asarray(spk_mon.t / second) >= t_start
    trial_i = np.asarray(spk_mon.i)[mask]
    return set(np.unique(trial_i[np.isin(trial_i, kc_idx)]).tolist())


for _ in range(3):
    sniff(A_stim)
print('>>> warm-up complete', flush=True)

n_sniffs = 20  # RELIABILITY across independent 10ms sniffs, not union - a
                # neuron's identity code is what it does *consistently*, not
                # everything it ever did once (union grows unboundedly and
                # merges in noise as more sniffs are added; verified this
                # empirically - union overlap was 68.8%, much worse than any
                # single-sniff sample in the grid search)
from collections import Counter
counts_A, counts_B = Counter(), Counter()
for _ in range(n_sniffs):
    for k in sniff(A_stim):
        counts_A[k] += 1
for _ in range(n_sniffs):
    for k in sniff(B_stim):
        counts_B[k] += 1

for reliability in [0.3, 0.5, 0.7]:
    thresh = reliability * n_sniffs
    rel_A = {k for k, c in counts_A.items() if c >= thresh}
    rel_B = {k for k, c in counts_B.items() if c >= thresh}
    ov = rel_A & rel_B
    smaller = max(1, min(len(rel_A), len(rel_B)))
    print(f'reliability>={reliability:.0%}: A={len(rel_A)} KCs, B={len(rel_B)} KCs, overlap={len(ov)} ({100*len(ov)/smaller:.1f}% of smaller)', flush=True)

# use the 50% threshold as the working definition
thresh = 0.5 * n_sniffs
rel_A = {k for k, c in counts_A.items() if c >= thresh}
rel_B = {k for k, c in counts_B.items() if c >= thresh}
overlap = rel_A & rel_B
excl_A = sorted(int(k) for k in (rel_A - overlap))
excl_B = sorted(int(k) for k in (rel_B - overlap))
print(f'FINAL (50% reliability, exclusive): odor-A={len(excl_A)} KCs, odor-B={len(excl_B)} KCs', flush=True)

with open('../real_sparse_odors.json', 'w') as f:
    json.dump({
        'odor_A_reliable': sorted(int(k) for k in rel_A), 'odor_B_reliable': sorted(int(k) for k in rel_B),
        'odor_A_exclusive': excl_A, 'odor_B_exclusive': excl_B,
    }, f)
print('>>> wrote real_sparse_odors.json', flush=True)
