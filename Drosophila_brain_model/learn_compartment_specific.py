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
mbon_idx = np.array(sorted(flyid2i[x] for x in mbon_ids if x in flyid2i))

# classify MBONs by which DAN cluster dominates their dopaminergic input -
# PPL1-dominant compartments are punishment/aversive-learning associated,
# PAM-dominant compartments are reward/appetitive-learning associated
# (Aso et al. 2014). We derive this from the connectome itself rather than
# assuming a literature table, since the actual wiring is right here.
dom = pd.read_csv('mbon_dan_dominance.csv', index_col=0)
ppl_mbon_idx = np.array([i for i in mbon_idx if i in dom.index and dom.loc[i, 'dominant'] == 'PPL'])
pam_mbon_idx = np.array([i for i in mbon_idx if i in dom.index and dom.loc[i, 'dominant'] == 'PAM'])
print(f'>>> MBONs: {len(mbon_idx)} total, {len(ppl_mbon_idx)} PPL-dominant (punishment), {len(pam_mbon_idx)} PAM-dominant (reward)')

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

# KEY DIFFERENCE from the earlier (non-compartment-specific) experiment:
# only synapses landing on PPL-dominant (punishment-associated) MBONs are
# allowed to plasticize. synapses onto PAM-dominant MBONs are left
# completely untouched, exactly as real compartmentalized dopamine release
# would only modulate the compartment it innervates.
elig_mask = np.isin(syn_i, odor_A_arr) & np.isin(syn_j, ppl_mbon_idx)
elig_idx = np.where(elig_mask)[0]
elig_pre = syn_i[elig_idx]
print(f'>>> plastic KC(odor A)->PPL-MBON synapses: {len(elig_idx)}  (PAM-MBON synapses: untouched)')

t_run = 300 * ms
r_stim = 150 * Hz
eta = 0.18

stim_targets = odor_A + odor_B
stim = NeuronGroup(len(stim_targets), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(stim_targets)), j=np.array(stim_targets))
odor_A_stim = np.arange(0, len(odor_A))
odor_B_stim = np.arange(len(odor_A), len(odor_A) + len(odor_B))

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


# warm-up
for _ in range(4):
    run_trial(odor_B_stim, odor_B)
print('>>> warm-up complete')

n_reps = 8
baseline = {}
for label, odor, odor_stim in [('A', odor_A, odor_A_stim), ('B', odor_B, odor_B_stim)]:
    ppl_r, pam_r = [], []
    for r in range(n_reps):
        ps, ms_, _ = run_trial(odor_stim, odor)
        ppl_r.append(rate(ps, len(ppl_mbon_idx)))
        pam_r.append(rate(ms_, len(pam_mbon_idx)))
    baseline[label] = (ppl_r, pam_r)
    print(f'baseline odor {label}: PPL-MBON = {np.mean(ppl_r):.2f}+/-{np.std(ppl_r):.2f} Hz   PAM-MBON = {np.mean(pam_r):.2f}+/-{np.std(pam_r):.2f} Hz')

n_pairing = 12
for trial in range(1, n_pairing + 1):
    ps, ms_, fired = run_trial(odor_A_stim, odor_A)
    if fired:
        fired_arr = np.array(list(fired))
        depress = np.isin(elig_pre, fired_arr)
        syn.w[elig_idx[depress]] = syn.w[elig_idx[depress]] * (1 - eta)
    print(f'pairing trial {trial:2d}: PPL-MBON = {rate(ps, len(ppl_mbon_idx)):5.2f} Hz   PAM-MBON = {rate(ms_, len(pam_mbon_idx)):5.2f} Hz')

post = {}
for label, odor, odor_stim in [('A', odor_A, odor_A_stim), ('B', odor_B, odor_B_stim)]:
    ppl_r, pam_r = [], []
    for r in range(n_reps):
        ps, ms_, _ = run_trial(odor_stim, odor)
        ppl_r.append(rate(ps, len(ppl_mbon_idx)))
        pam_r.append(rate(ms_, len(pam_mbon_idx)))
    post[label] = (ppl_r, pam_r)
    print(f'post-learning odor {label}: PPL-MBON = {np.mean(ppl_r):.2f}+/-{np.std(ppl_r):.2f} Hz   PAM-MBON = {np.mean(pam_r):.2f}+/-{np.std(pam_r):.2f} Hz')

print()
print('=== compartment-specificity summary (odor A, paired/punished) ===')
b_ppl = np.mean(baseline['A'][0]); p_ppl = np.mean(post['A'][0])
b_pam = np.mean(baseline['A'][1]); p_pam = np.mean(post['A'][1])
print(f'PPL-dominant (punishment) MBONs: {b_ppl:.2f} -> {p_ppl:.2f} Hz  ({100*(p_ppl-b_ppl)/b_ppl:+.1f}%)  <- synapses WERE depressed here')
print(f'PAM-dominant (reward) MBONs:     {b_pam:.2f} -> {p_pam:.2f} Hz  ({100*(p_pam-b_pam)/b_pam:+.1f}%)  <- synapses untouched, testing for indirect/network-wide leakage')

with open('../compartment_results.json', 'w') as f:
    json.dump({'baseline': {k: list(v) for k, v in baseline.items()},
                'post': {k: list(v) for k, v in post.items()}}, f)
print('>>> wrote compartment_results.json')
