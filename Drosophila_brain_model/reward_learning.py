import time
import numpy as np
import pandas as pd
import json

from model import create_model, default_params as params
from brian2 import Network, NeuronGroup, Synapses, ms, Hz, mV, second, defaultclock

config = {'path_comp': './Completeness_783.csv', 'path_con': './Connectivity_783.parquet'}

df_comp = pd.read_csv(config['path_comp'], index_col=0)
ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
flyid2i = {j: i for i, j in enumerate(df_comp.index)}

kc_ids = set(ann.loc[ann['cell_class'] == 'Kenyon_Cell', 'root_id'])
mbon_ids = set(ann.loc[ann['cell_type'].astype(str).str.startswith('MBON', na=False), 'root_id'])
kc_idx = sorted(flyid2i[x] for x in kc_ids if x in flyid2i)
mbon_idx = np.array(sorted(flyid2i[x] for x in mbon_ids if x in flyid2i))

dom = pd.read_csv('mbon_dan_dominance.csv', index_col=0)
ppl_mbon_idx = np.array([i for i in mbon_idx if i in dom.index and dom.loc[i, 'dominant'] == 'PPL'])
pam_mbon_idx = np.array([i for i in mbon_idx if i in dom.index and dom.loc[i, 'dominant'] == 'PAM'])
print(f'>>> {len(ppl_mbon_idx)} PPL-dominant (punishment) MBONs, {len(pam_mbon_idx)} PAM-dominant (reward) MBONs')

rng = np.random.default_rng(0)
shuffled = rng.permutation(kc_idx)
n_odor = 150
# odor C: a fresh ensemble, disjoint from odor A (index 0:150) and odor B
# (150:300) used in the aversive-learning experiment
odor_C = shuffled[300:300 + n_odor].tolist()   # rewarded odor
odor_D = shuffled[300 + n_odor:300 + 2 * n_odor].tolist()  # control, never paired

t0 = time.time()
neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
print(f'>>> network built in {time.time() - t0:.1f}s')

syn_i = np.asarray(syn.i[:])
syn_j = np.asarray(syn.j[:])
odor_C_arr = np.array(odor_C)

# MIRROR IMAGE of the aversive experiment: reward-gated plasticity only
# touches KC(odor-C)->PAM-dominant-MBON synapses, leaving PPL-dominant
# MBONs (the punishment-associated compartments) completely untouched.
elig_mask = np.isin(syn_i, odor_C_arr) & np.isin(syn_j, pam_mbon_idx)
elig_idx = np.where(elig_mask)[0]
elig_pre = syn_i[elig_idx]
print(f'>>> plastic KC(odor C)->PAM-MBON synapses: {len(elig_idx)}  (PPL-MBON synapses: untouched)')

t_run = 300 * ms
r_stim = 150 * Hz
eta = 0.18

stim_targets = odor_C + odor_D
stim = NeuronGroup(len(stim_targets), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(stim_targets)), j=np.array(stim_targets))
odor_C_stim = np.arange(0, len(odor_C))
odor_D_stim = np.arange(len(odor_C), len(odor_C) + len(odor_D))

net = Network(neu, syn, spk_mon, stim, drive)


def run_trial(odor_stim_positions, target_idx):
    neu.v = params['v_0']
    neu.g = 0 * mV
    stim.rate = 0 * Hz
    stim.rate[odor_stim_positions] = r_stim
    t_start = defaultclock.t / second
    net.run(t_run)
    mask = np.asarray(spk_mon.t / second) >= t_start
    trial_i = np.asarray(spk_mon.i)[mask]
    ppl_spikes = np.isin(trial_i, ppl_mbon_idx).sum()
    pam_spikes = np.isin(trial_i, pam_mbon_idx).sum()
    fired_targets = set(np.unique(trial_i[np.isin(trial_i, target_idx)]).tolist())
    return ppl_spikes, pam_spikes, fired_targets


def rate(spikes, n):
    return spikes / n / (t_run / (1000 * ms))


for _ in range(4):
    run_trial(odor_D_stim, odor_D)
print('>>> warm-up complete')

n_reps = 8
baseline = {}
for label, odor, odor_stim in [('C', odor_C, odor_C_stim), ('D', odor_D, odor_D_stim)]:
    ppl_r, pam_r = [], []
    for r in range(n_reps):
        ps, ms_, _ = run_trial(odor_stim, odor)
        ppl_r.append(rate(ps, len(ppl_mbon_idx)))
        pam_r.append(rate(ms_, len(pam_mbon_idx)))
    baseline[label] = (ppl_r, pam_r)
    print(f'baseline odor {label}: PPL-MBON = {np.mean(ppl_r):.2f}+/-{np.std(ppl_r):.2f} Hz   PAM-MBON = {np.mean(pam_r):.2f}+/-{np.std(pam_r):.2f} Hz')

n_pairing = 12
for trial in range(1, n_pairing + 1):
    ps, ms_, fired = run_trial(odor_C_stim, odor_C)
    if fired:
        fired_arr = np.array(list(fired))
        depress = np.isin(elig_pre, fired_arr)
        syn.w[elig_idx[depress]] = syn.w[elig_idx[depress]] * (1 - eta)
    print(f'pairing trial {trial:2d}: PPL-MBON = {rate(ps, len(ppl_mbon_idx)):5.2f} Hz   PAM-MBON = {rate(ms_, len(pam_mbon_idx)):5.2f} Hz')

post = {}
for label, odor, odor_stim in [('C', odor_C, odor_C_stim), ('D', odor_D, odor_D_stim)]:
    ppl_r, pam_r = [], []
    for r in range(n_reps):
        ps, ms_, _ = run_trial(odor_stim, odor)
        ppl_r.append(rate(ps, len(ppl_mbon_idx)))
        pam_r.append(rate(ms_, len(pam_mbon_idx)))
    post[label] = (ppl_r, pam_r)
    print(f'post-learning odor {label}: PPL-MBON = {np.mean(ppl_r):.2f}+/-{np.std(ppl_r):.2f} Hz   PAM-MBON = {np.mean(pam_r):.2f}+/-{np.std(pam_r):.2f} Hz')

print()
print('=== reward-learning compartment-specificity summary (odor C, paired/rewarded) ===')
b_ppl = np.mean(baseline['C'][0]); p_ppl = np.mean(post['C'][0])
b_pam = np.mean(baseline['C'][1]); p_pam = np.mean(post['C'][1])
print(f'PAM-dominant (reward) MBONs:     {b_pam:.2f} -> {p_pam:.2f} Hz  ({100*(p_pam-b_pam)/b_pam:+.1f}%)  <- synapses WERE depressed here')
print(f'PPL-dominant (punishment) MBONs: {b_ppl:.2f} -> {p_ppl:.2f} Hz  ({100*(p_ppl-b_ppl)/b_ppl:+.1f}%)  <- synapses untouched, testing for leakage')

with open('../reward_results.json', 'w') as f:
    json.dump({'baseline': {k: list(v) for k, v in baseline.items()},
                'post': {k: list(v) for k, v in post.items()}}, f)
print('>>> wrote reward_results.json')
