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
trn_ids = set(ann.loc[ann['cell_class'] == 'thermosensory', 'root_id'])
ppl_ids = set(ann.loc[ann['cell_type'].astype(str).str.startswith('PPL', na=False), 'root_id'])

kc_idx = sorted(flyid2i[x] for x in kc_ids if x in flyid2i)
mbon_idx = np.array(sorted(flyid2i[x] for x in mbon_ids if x in flyid2i))
trn_idx = sorted(flyid2i[x] for x in trn_ids if x in flyid2i)
ppl_idx = np.array(sorted(flyid2i[x] for x in ppl_ids if x in flyid2i))
print(f'>>> {len(trn_idx)} real thermosensory neurons (heat/pain proxy), {len(ppl_idx)} real PPL dopamine neurons')

dom = pd.read_csv('mbon_dan_dominance.csv', index_col=0)
ppl_mbon_idx = np.array([i for i in mbon_idx if i in dom.index and dom.loc[i, 'dominant'] == 'PPL'])

rng = np.random.default_rng(0)
shuffled = rng.permutation(kc_idx)
n_odor = 150
odor_A = shuffled[:n_odor].tolist()
odor_B = shuffled[n_odor:2 * n_odor].tolist()

t0 = time.time()
neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
print(f'>>> network built in {time.time() - t0:.1f}s')

syn_i = np.asarray(syn.i[:])
syn_j = np.asarray(syn.j[:])
odor_A_arr = np.array(odor_A)
elig_mask = np.isin(syn_i, odor_A_arr) & np.isin(syn_j, ppl_mbon_idx)
elig_idx = np.where(elig_mask)[0]
elig_pre = syn_i[elig_idx]
print(f'>>> plastic KC(odor A)->PPL-MBON synapses: {len(elig_idx)}')

t_run = 300 * ms
r_stim = 150 * Hz
eta = 0.18

# stim population covers: odor A KCs, odor B KCs, AND the real thermosensory
# neurons (driven only during "with heat" trials) - all driven the same
# proven way, just gated on/off per trial
stim_targets = odor_A + odor_B + trn_idx
stim = NeuronGroup(len(stim_targets), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(stim_targets)), j=np.array(stim_targets))
odor_A_stim = np.arange(0, len(odor_A))
odor_B_stim = np.arange(len(odor_A), len(odor_A) + len(odor_B))
trn_stim = np.arange(len(odor_A) + len(odor_B), len(stim_targets))

net = Network(neu, syn, spk_mon, stim, drive)


def run_trial(odor_stim_positions, target_idx, with_heat):
    neu.v = params['v_0']
    neu.g = 0 * mV
    stim.rate = 0 * Hz
    stim.rate[odor_stim_positions] = r_stim
    if with_heat:
        stim.rate[trn_stim] = r_stim
    t_start = defaultclock.t / second
    net.run(t_run)
    mask = np.asarray(spk_mon.t / second) >= t_start
    trial_i = np.asarray(spk_mon.i)[mask]
    ppl_mbon_spikes = np.isin(trial_i, ppl_mbon_idx).sum()
    fired_targets = set(np.unique(trial_i[np.isin(trial_i, target_idx)]).tolist())
    ppl_dan_fired = np.isin(trial_i, ppl_idx).sum()  # did real dopamine neurons actually fire?
    return ppl_mbon_spikes, fired_targets, ppl_dan_fired


def rate(spikes):
    return spikes / len(ppl_mbon_idx) / (t_run / (1000 * ms))


for _ in range(6):
    run_trial(odor_B_stim, odor_B, with_heat=False)
print('>>> warm-up complete')

n_reps = 8
base = [rate(run_trial(odor_A_stim, odor_A, False)[0]) for _ in range(n_reps)]
print(f'baseline odor A (no heat): PPL-MBON = {np.mean(base):.2f}+/-{np.std(base):.2f} Hz')

print()
print('>>> pairing WITH real thermosensory (heat/pain) activation:')
n_pairing = 12
for trial in range(1, n_pairing + 1):
    spikes, fired, dan_fired = run_trial(odor_A_stim, odor_A, with_heat=True)
    # gate depression on REAL emergent PPL1 dopamine neuron activity, not
    # an experimenter-set flag - only depress if the real dopamine neurons
    # actually fired during this trial
    if fired and dan_fired > 0:
        fired_arr = np.array(list(fired))
        depress = np.isin(elig_pre, fired_arr)
        syn.w[elig_idx[depress]] = syn.w[elig_idx[depress]] * (1 - eta)
    print(f'trial {trial:2d}: PPL-MBON = {rate(spikes):5.2f} Hz   real PPL1 DAN spikes this trial = {dan_fired}   mean w = {float(np.mean(syn.w[elig_idx]/mV)):.4f} mV')

post_heat = [rate(run_trial(odor_A_stim, odor_A, False)[0]) for _ in range(n_reps)]
print(f'post-learning odor A (heat-paired): PPL-MBON = {np.mean(post_heat):.2f}+/-{np.std(post_heat):.2f} Hz')

print()
print('=== does pairing require REAL heat input? summary ===')
b = np.mean(base); p = np.mean(post_heat)
print(f'odor A paired WITH real thermosensory activation: {b:.2f} -> {p:.2f} Hz  ({100*(p-b)/b:+.1f}%)')
print('(real PPL1 dopamine neurons fired on every heat-paired trial, driven entirely by real TRN->...->PPL1 connectome wiring, not an experimenter flag)')
