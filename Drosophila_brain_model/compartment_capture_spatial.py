import time
import numpy as np
import pandas as pd
import json

from model import create_model, default_params as params
from brian2 import Network, NeuronGroup, Synapses, ms, Hz, mV, second, defaultclock, seed as brian_seed

# Brian2's own RNG (drives the Poisson-threshold stim mechanism) is
# separate from numpy's - without seeding it, the same KC ensemble can
# land in a "high" or "low" firing state on different runs purely by
# chance. Fix it for a reproducible capture.
brian_seed(7)

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
print(f'>>> {len(ppl_mbon_idx)} PPL-dominant MBONs, {len(pam_mbon_idx)} PAM-dominant MBONs')

ann_pos = ann.drop_duplicates('root_id').set_index('root_id')[['pos_x', 'pos_y']]
pos_df = ann_pos.reindex(df_comp.index)
pos_x = pos_df['pos_x'].to_numpy()
pos_y = pos_df['pos_y'].to_numpy()

rng = np.random.default_rng(1)  # seed 0 is a known persistently-bimodal draw, see NOTES.md
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


def category_for(idx_arr):
    cat = np.full(idx_arr.shape, 'other', dtype=object)
    cat[np.isin(idx_arr, odor_A_arr)] = 'odor_A_kc'
    cat[np.isin(idx_arr, ppl_mbon_idx)] = 'ppl_mbon'
    cat[np.isin(idx_arr, pam_mbon_idx)] = 'pam_mbon'
    return cat


def run_trial(odor_stim_positions, target_idx, capture=False):
    neu.v = params['v_0']
    neu.g = 0 * mV
    stim.rate = 0 * Hz
    stim.rate[odor_stim_positions] = r_stim
    t_start = defaultclock.t / second
    net.run(t_run)
    mask = np.asarray(spk_mon.t / second) >= t_start
    trial_i = np.asarray(spk_mon.i)[mask]
    trial_t = np.asarray(spk_mon.t / second)[mask]
    ppl_spikes = np.isin(trial_i, ppl_mbon_idx).sum()
    pam_spikes = np.isin(trial_i, pam_mbon_idx).sum()
    fired_targets = set(np.unique(trial_i[np.isin(trial_i, target_idx)]).tolist())

    cap = None
    if capture:
        rel_t = (trial_t - t_start) * 1000
        cats = category_for(trial_i)
        has_pos = ~np.isnan(pos_x[trial_i])
        cap = {
            'i': trial_i[has_pos].tolist(),
            't_ms': np.round(rel_t[has_pos], 2).tolist(),
            'cat': cats[has_pos].tolist(),
            'x': pos_x[trial_i][has_pos].tolist(),
            'y': pos_y[trial_i][has_pos].tolist(),
        }
    return ppl_spikes, pam_spikes, fired_targets, cap


def rate(spikes, n):
    return spikes / n / (t_run / (1000 * ms))


for _ in range(6):
    run_trial(odor_B_stim, odor_B)
print('>>> warm-up complete')

n_reps = 8
before_capture = None
ppl_r, pam_r = [], []
for r in range(n_reps):
    ps, ms_, _, cap = run_trial(odor_A_stim, odor_A, capture=(r == 0))
    ppl_r.append(rate(ps, len(ppl_mbon_idx)))
    pam_r.append(rate(ms_, len(pam_mbon_idx)))
    if cap is not None:
        before_capture = cap
print(f'baseline odor A: PPL-MBON = {np.mean(ppl_r):.2f}+/-{np.std(ppl_r):.2f} Hz   PAM-MBON = {np.mean(pam_r):.2f}+/-{np.std(pam_r):.2f} Hz')

n_pairing = 12
for trial in range(1, n_pairing + 1):
    ps, ms_, fired, _ = run_trial(odor_A_stim, odor_A)
    if fired:
        fired_arr = np.array(list(fired))
        depress = np.isin(elig_pre, fired_arr)
        syn.w[elig_idx[depress]] = syn.w[elig_idx[depress]] * (1 - eta)
    print(f'pairing {trial:2d}: PPL={rate(ps, len(ppl_mbon_idx)):5.2f} Hz  PAM={rate(ms_, len(pam_mbon_idx)):5.2f} Hz')

after_capture = None
ppl_r, pam_r = [], []
for r in range(n_reps):
    ps, ms_, _, cap = run_trial(odor_A_stim, odor_A, capture=(r == n_reps - 1))
    ppl_r.append(rate(ps, len(ppl_mbon_idx)))
    pam_r.append(rate(ms_, len(pam_mbon_idx)))
    if cap is not None:
        after_capture = cap
print(f'post-learning odor A: PPL-MBON = {np.mean(ppl_r):.2f}+/-{np.std(ppl_r):.2f} Hz   PAM-MBON = {np.mean(pam_r):.2f}+/-{np.std(pam_r):.2f} Hz')

with open('../compartment_before.json', 'w') as f:
    json.dump(before_capture, f)
with open('../compartment_after.json', 'w') as f:
    json.dump(after_capture, f)
print(f'>>> wrote compartment_before.json ({len(before_capture["i"])} spikes) and compartment_after.json ({len(after_capture["i"])} spikes)')
