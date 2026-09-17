import time
import numpy as np
import pandas as pd
import json

from model import create_model, default_params as params
from brian2 import Network, NeuronGroup, Synapses, ms, Hz, mV, second, defaultclock

config = {
    'path_comp': './Completeness_783.csv',
    'path_con':  './Connectivity_783.parquet',
}

# ---------------------------------------------------------------
# identify Kenyon cells / MBONs, and build a brian-index -> position lookup
# ---------------------------------------------------------------
df_comp = pd.read_csv(config['path_comp'], index_col=0)
ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
flyid2i = {j: i for i, j in enumerate(df_comp.index)}

kc_ids = set(ann.loc[ann['cell_class'] == 'Kenyon_Cell', 'root_id'])
mbon_ids = set(ann.loc[ann['cell_type'].astype(str).str.startswith('MBON', na=False), 'root_id'])
kc_idx = sorted(flyid2i[x] for x in kc_ids if x in flyid2i)
mbon_idx = sorted(flyid2i[x] for x in mbon_ids if x in flyid2i)

# position lookup, indexed by brian id (one row per traced neuron)
ann_pos = ann.drop_duplicates('root_id').set_index('root_id')[['pos_x', 'pos_y', 'pos_z']]
comp_ids = df_comp.index.to_numpy()
pos_df = ann_pos.reindex(comp_ids)  # brian-id order, NaN where position unknown
pos_x = pos_df['pos_x'].to_numpy()
pos_y = pos_df['pos_y'].to_numpy()
pos_z = pos_df['pos_z'].to_numpy()
print(f'>>> positions available for {(~np.isnan(pos_x)).sum()} / {len(pos_x)} neurons')

rng = np.random.default_rng(0)
shuffled = rng.permutation(kc_idx)
n_odor = 150
odor_A = shuffled[:n_odor].tolist()
odor_B = shuffled[n_odor:2 * n_odor].tolist()

print(f'>>> Kenyon cells available: {len(kc_idx)}, MBONs: {len(mbon_idx)}')
print(f'>>> odor A ensemble: {len(odor_A)} KCs, odor B ensemble: {len(odor_B)} KCs')

t0 = time.time()
neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
print(f'>>> network built in {time.time() - t0:.1f}s  ({len(neu)} neurons, {len(syn)} synapses)')

syn_i = np.asarray(syn.i[:])
syn_j = np.asarray(syn.j[:])
mbon_arr = np.array(mbon_idx)
odor_A_arr = np.array(odor_A)
odor_B_arr = np.array(odor_B)

elig_mask = np.isin(syn_i, odor_A_arr) & np.isin(syn_j, mbon_arr)
elig_idx = np.where(elig_mask)[0]
elig_pre = syn_i[elig_idx]
print(f'>>> plastic KC(odor A)->MBON synapses: {len(elig_idx)}')

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
    cat[np.isin(idx_arr, odor_B_arr)] = 'odor_B_kc'
    cat[np.isin(idx_arr, mbon_arr)] = 'mbon'
    return cat


def run_trial(odor_stim_positions, target_idx, capture=False):
    neu.v = params['v_0']
    neu.g = 0 * mV
    stim.rate = 0 * Hz
    stim.rate[odor_stim_positions] = r_stim

    t_start = defaultclock.t / second
    net.run(t_run)

    mask_new = np.asarray(spk_mon.t / second) >= t_start
    trial_i = np.asarray(spk_mon.i)[mask_new]
    trial_t = np.asarray(spk_mon.t / second)[mask_new]
    mbon_spikes = np.isin(trial_i, mbon_arr).sum()
    fired_targets = set(np.unique(trial_i[np.isin(trial_i, target_idx)]).tolist())

    capture_data = None
    if capture:
        rel_t = (trial_t - t_start) * 1000  # ms since trial start
        cats = category_for(trial_i)
        has_pos = ~np.isnan(pos_x[trial_i])
        capture_data = {
            'i': trial_i[has_pos].tolist(),
            't_ms': np.round(rel_t[has_pos], 2).tolist(),
            'cat': cats[has_pos].tolist(),
            'x': pos_x[trial_i][has_pos].tolist(),
            'y': pos_y[trial_i][has_pos].tolist(),
            'z': pos_z[trial_i][has_pos].tolist(),
        }
    return mbon_spikes, fired_targets, capture_data


def mbon_rate(spikes):
    return spikes / len(mbon_idx) / (t_run / (1000 * ms))


# ---------------------------------------------------------------
# warm-up: a freshly-built network shows a transient before settling into
# a consistent operating regime (its first couple of trials respond
# weaker regardless of stimulus) - burn a few trials before collecting
# any real data, same as discarding habituation trials in a real
# electrophysiology experiment
# ---------------------------------------------------------------
for _ in range(4):
    run_trial(odor_B_stim, odor_B)
print('>>> warm-up complete')

# ---------------------------------------------------------------
# baseline (capture trial 1 of odor A as the "before learning" snapshot)
# ---------------------------------------------------------------
n_reps = 5
before_capture = None
for label, odor, odor_stim in [('A', odor_A, odor_A_stim), ('B', odor_B, odor_B_stim)]:
    reps = []
    for r in range(n_reps):
        spikes, _, cap = run_trial(odor_stim, odor, capture=(label == 'A' and r == 0))
        reps.append(mbon_rate(spikes))
        if cap is not None:
            before_capture = cap
    rate = float(np.mean(reps))
    print(f'baseline odor {label}: MBON pop rate = {rate:.2f} +/- {np.std(reps):.2f} Hz')

# ---------------------------------------------------------------
# pairing
# ---------------------------------------------------------------
n_pairing = 12
for trial in range(1, n_pairing + 1):
    spikes, fired, _ = run_trial(odor_A_stim, odor_A)
    rate = mbon_rate(spikes)
    if fired:
        fired_arr = np.array(list(fired))
        depress = np.isin(elig_pre, fired_arr)
        syn.w[elig_idx[depress]] = syn.w[elig_idx[depress]] * (1 - eta)
    mean_w = float(np.mean(syn.w[elig_idx] / mV))
    print(f'pairing trial {trial:2d}: MBON pop rate = {rate:5.2f} Hz   mean plastic weight = {mean_w:.4f} mV')

# ---------------------------------------------------------------
# post-learning (capture last rep of odor A as the "after learning" snapshot)
# ---------------------------------------------------------------
after_capture = None
for label, odor, odor_stim in [('A', odor_A, odor_A_stim), ('B', odor_B, odor_B_stim)]:
    reps = []
    for r in range(n_reps):
        spikes, _, cap = run_trial(odor_stim, odor, capture=(label == 'A' and r == n_reps - 1))
        reps.append(mbon_rate(spikes))
        if cap is not None:
            after_capture = cap
    rate = float(np.mean(reps))
    print(f'post-learning odor {label}: MBON pop rate = {rate:.2f} +/- {np.std(reps):.2f} Hz')

with open('../spatial_before.json', 'w') as f:
    json.dump(before_capture, f)
with open('../spatial_after.json', 'w') as f:
    json.dump(after_capture, f)
print(f'>>> wrote spatial_before.json ({len(before_capture["i"])} spikes) and spatial_after.json ({len(after_capture["i"])} spikes)')
