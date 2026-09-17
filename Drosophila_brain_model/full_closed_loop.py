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

# real, disjoint, named-glomerulus odors - no hand-picked Kenyon cells
odor_A_glom = ['DA1_lPN', 'DA2_lPN', 'DL3_lPN', 'DL2d_adPN']
odor_B_glom = ['VM5d_adPN', 'VA1v_adPN', 'DL2v_adPN', 'VC3_adPN']
pnA_ids = set(ann.loc[ann['cell_type'].isin(odor_A_glom), 'root_id'])
pnB_ids = set(ann.loc[ann['cell_type'].isin(odor_B_glom), 'root_id'])
pnA_idx = sorted(flyid2i[x] for x in pnA_ids if x in flyid2i)
pnB_idx = sorted(flyid2i[x] for x in pnB_ids if x in flyid2i)

trn_ids = set(ann.loc[ann['cell_class'] == 'thermosensory', 'root_id'])
trn_idx = sorted(flyid2i[x] for x in trn_ids if x in flyid2i)

kc_ids = set(ann.loc[ann['cell_class'] == 'Kenyon_Cell', 'root_id'])
mbon_ids = set(ann.loc[ann['cell_type'].astype(str).str.startswith('MBON', na=False), 'root_id'])
kc_idx = np.array(sorted(flyid2i[x] for x in kc_ids if x in flyid2i))
mbon_idx = np.array(sorted(flyid2i[x] for x in mbon_ids if x in flyid2i))

dom = pd.read_csv('mbon_dan_dominance.csv', index_col=0)
ppl_mbon_idx = np.array([i for i in mbon_idx if i in dom.index and dom.loc[i, 'dominant'] == 'PPL'])
pam_mbon_idx = np.array([i for i in mbon_idx if i in dom.index and dom.loc[i, 'dominant'] == 'PAM'])

ppl_ids = set(ann.loc[ann['cell_type'].astype(str).str.startswith('PPL', na=False), 'root_id'])
ppl_dan_idx = np.array(sorted(flyid2i[x] for x in ppl_ids if x in flyid2i))

print(f'>>> odor A: {len(pnA_idx)} real PNs across {len(odor_A_glom)} glomeruli')
print(f'>>> odor B: {len(pnB_idx)} real PNs across {len(odor_B_glom)} glomeruli')
print(f'>>> {len(trn_idx)} real thermosensory (heat) neurons, {len(ppl_dan_idx)} real PPL1 dopamine neurons')
print(f'>>> {len(ppl_mbon_idx)} PPL-dominant MBONs, {len(pam_mbon_idx)} PAM-dominant MBONs')

t0 = time.time()
neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
print(f'>>> network built in {time.time() - t0:.1f}s')

syn_i = np.asarray(syn.i[:])
syn_j = np.asarray(syn.j[:])

# eligible pool: ALL real KC->PPL-dominant-MBON synapses (compartment
# restriction). odor-specificity comes entirely from which KCs the real
# PN drive actually recruits each trial, not from a pre-picked KC list.
elig_mask = np.isin(syn_i, kc_idx) & np.isin(syn_j, ppl_mbon_idx)
elig_idx = np.where(elig_mask)[0]
elig_pre = syn_i[elig_idx]
print(f'>>> plastic KC->PPL-MBON synapse pool (compartment-restricted, odor-unrestricted): {len(elig_idx)}')

t_run = 300 * ms
r_stim = 150 * Hz
eta = 0.10  # smaller step since the eligible pool + fired-KC count is much larger here

stim_targets = pnA_idx + pnB_idx + trn_idx
stim = NeuronGroup(len(stim_targets), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(stim_targets)), j=np.array(stim_targets))
A_stim = np.arange(0, len(pnA_idx))
B_stim = np.arange(len(pnA_idx), len(pnA_idx) + len(pnB_idx))
trn_stim = np.arange(len(pnA_idx) + len(pnB_idx), len(stim_targets))

net = Network(neu, syn, spk_mon, stim, drive)


# a brief external pulse doesn't work on its own: this is a fully
# recurrent whole-brain network, so once the initial cascade starts it
# keeps propagating on its own for the rest of the 300ms trial even
# after the external drive stops (tried it - KC recruitment was
# unchanged, ~64%, because the settling window let the same recurrent
# cascade develop just a few ms later). the actual fix: define odor
# identity by which KCs respond EARLY (matching real sparse coincidence
# detection - real odor identity is read out in the first ~20ms of a
# response, not from everything that eventually fires over 300ms),
# while the stimulus and trial can still run their full realistic
# duration for the MBON readout to develop properly.
early_cutoff_ms = 20


def run_trial(odor_positions, with_heat):
    neu.v = params['v_0']
    neu.g = 0 * mV
    stim.rate = 0 * Hz
    stim.rate[odor_positions] = r_stim
    if with_heat:
        stim.rate[trn_stim] = r_stim
    t_start = defaultclock.t / second
    net.run(t_run)
    mask = np.asarray(spk_mon.t / second) >= t_start
    trial_i = np.asarray(spk_mon.i)[mask]
    trial_t = (np.asarray(spk_mon.t / second)[mask] - t_start) * 1000
    ppl_mbon_spikes = np.isin(trial_i, ppl_mbon_idx).sum()
    pam_mbon_spikes = np.isin(trial_i, pam_mbon_idx).sum()
    early_mask = np.isin(trial_i, kc_idx) & (trial_t <= early_cutoff_ms)
    fired_kc = np.unique(trial_i[early_mask])
    dan_fired = np.isin(trial_i, ppl_dan_idx).sum()
    return ppl_mbon_spikes, pam_mbon_spikes, fired_kc, dan_fired


def rate(spikes, n):
    return spikes / n / (t_run / (1000 * ms))


for _ in range(6):
    run_trial(B_stim, with_heat=False)
print('>>> warm-up complete')

n_reps = 8
baseline = {}
for label, pos in [('A', A_stim), ('B', B_stim)]:
    ppl_r, pam_r, kcs = [], [], []
    for r in range(n_reps):
        ps, ms_, fkc, _ = run_trial(pos, with_heat=False)
        ppl_r.append(rate(ps, len(ppl_mbon_idx)))
        pam_r.append(rate(ms_, len(pam_mbon_idx)))
        kcs.append(len(fkc))
    baseline[label] = (ppl_r, pam_r)
    print(f'baseline odor {label} (no heat): PPL-MBON = {np.mean(ppl_r):.2f}+/-{np.std(ppl_r):.2f} Hz   PAM-MBON = {np.mean(pam_r):.2f}+/-{np.std(pam_r):.2f} Hz   KCs recruited ~{int(np.mean(kcs))}')

n_pairing = 12
for trial in range(1, n_pairing + 1):
    ps, ms_, fkc, dan_fired = run_trial(A_stim, with_heat=True)
    if len(fkc) and dan_fired > 0:
        depress = np.isin(elig_pre, fkc)
        syn.w[elig_idx[depress]] = syn.w[elig_idx[depress]] * (1 - eta)
    print(f'pairing {trial:2d}: PPL-MBON={rate(ps, len(ppl_mbon_idx)):5.2f} Hz  PAM-MBON={rate(ms_, len(pam_mbon_idx)):5.2f} Hz  '
          f'KCs recruited={len(fkc)}  real PPL1 DAN spikes={dan_fired}  mean w={float(np.mean(syn.w[elig_idx]/mV)):.4f} mV')

post = {}
for label, pos in [('A', A_stim), ('B', B_stim)]:
    ppl_r, pam_r = [], []
    for r in range(n_reps):
        ps, ms_, fkc, _ = run_trial(pos, with_heat=False)
        ppl_r.append(rate(ps, len(ppl_mbon_idx)))
        pam_r.append(rate(ms_, len(pam_mbon_idx)))
    post[label] = (ppl_r, pam_r)
    print(f'post odor {label}: PPL-MBON = {np.mean(ppl_r):.2f}+/-{np.std(ppl_r):.2f} Hz   PAM-MBON = {np.mean(pam_r):.2f}+/-{np.std(pam_r):.2f} Hz')

print()
print('=== full closed loop summary (real PN odor + real TRN punishment + compartment-specific depression) ===')
for label in ['A', 'B']:
    b_ppl, p_ppl = np.mean(baseline[label][0]), np.mean(post[label][0])
    b_pam, p_pam = np.mean(baseline[label][1]), np.mean(post[label][1])
    tag = '(real odor, paired with real heat)' if label == 'A' else '(real odor, control, never paired)'
    print(f'odor {label} {tag}:')
    print(f'  PPL-dominant MBONs: {b_ppl:.2f} -> {p_ppl:.2f} Hz ({100*(p_ppl-b_ppl)/b_ppl:+.1f}%)')
    print(f'  PAM-dominant MBONs: {b_pam:.2f} -> {p_pam:.2f} Hz ({100*(p_pam-b_pam)/b_pam:+.1f}%)')

with open('../full_closed_loop_results.json', 'w') as f:
    json.dump({'baseline': {k: list(v) for k, v in baseline.items()}, 'post': {k: list(v) for k, v in post.items()}}, f)
print('>>> wrote full_closed_loop_results.json')
