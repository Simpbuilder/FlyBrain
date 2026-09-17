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

config = {
    'path_comp': './Completeness_783.csv',
    'path_con':  './Connectivity_783.parquet',
}

# ---------------------------------------------------------------
# identify Kenyon cells / MBONs from public FlyWire annotations
# ---------------------------------------------------------------
df_comp = pd.read_csv(config['path_comp'], index_col=0)
ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
flyid2i = {j: i for i, j in enumerate(df_comp.index)}

kc_ids = set(ann.loc[ann['cell_class'] == 'Kenyon_Cell', 'root_id'])
mbon_ids = set(ann.loc[ann['cell_type'].astype(str).str.startswith('MBON', na=False), 'root_id'])

kc_idx = sorted(flyid2i[x] for x in kc_ids if x in flyid2i)
mbon_idx = sorted(flyid2i[x] for x in mbon_ids if x in flyid2i)

rng = np.random.default_rng(0)
shuffled = rng.permutation(kc_idx)
n_odor = 150
odor_A = shuffled[:n_odor].tolist()          # "punished" odor
odor_B = shuffled[n_odor:2 * n_odor].tolist()  # control odor, never paired

print(f'>>> Kenyon cells available: {len(kc_idx)}, MBONs: {len(mbon_idx)}')
print(f'>>> odor A ensemble: {len(odor_A)} KCs, odor B ensemble: {len(odor_B)} KCs')

# ---------------------------------------------------------------
# build the model ONCE - weights persist across trials, this is memory
# ---------------------------------------------------------------
t0 = time.time()
neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
print(f'>>> network built in {time.time() - t0:.1f}s  ({len(neu)} neurons, {len(syn)} synapses)')

syn_i = np.asarray(syn.i[:])
syn_j = np.asarray(syn.j[:])

mbon_arr = np.array(mbon_idx)
odor_A_arr = np.array(odor_A)

# candidate KC(odor A)->MBON synapses: the only synapses allowed to plasticize
elig_mask = np.isin(syn_i, odor_A_arr) & np.isin(syn_j, mbon_arr)
elig_idx = np.where(elig_mask)[0]
elig_pre = syn_i[elig_idx]
w_init = np.asarray(syn.w[elig_idx])
print(f'>>> plastic KC(odor A)->MBON synapses: {len(elig_idx)}')

t_run = 300 * ms
r_stim = 150 * Hz
eta = 0.18  # multiplicative depression per pairing trial for active synapses

results = {'trial': [], 'phase': [], 'odor': [], 'mbon_rate_hz': [], 'mean_w_mV': []}

# persistent stimulus population, one unit per odor-ensemble KC, one-to-one
# wired in. its per-unit rate is just toggled between trials instead of
# adding/removing PoissonInput objects on an already-run network (Brian2
# silently drops spikes from PoissonInputs added after a network has run)
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
    """Drive the neurons at `target_idx` for one trial via the persistent
    stim population, return (mbon spike count, set of target neurons that fired)."""
    neu.v = params['v_0']
    neu.g = 0 * mV
    stim.rate = 0 * Hz
    stim.rate[odor_stim_positions] = r_stim

    t_start = defaultclock.t / second  # plain float snapshot - defaultclock.t is a live view
    net.run(t_run)

    # spikes recorded during this run only
    mask_new = np.asarray(spk_mon.t / second) >= t_start
    trial_i = np.asarray(spk_mon.i)[mask_new]
    mbon_spikes = np.isin(trial_i, mbon_arr).sum()
    fired_targets = set(np.unique(trial_i[np.isin(trial_i, target_idx)]).tolist())
    return mbon_spikes, fired_targets


def mbon_rate(mbon_spikes):
    return mbon_spikes / len(mbon_idx) / (t_run / (1000 * ms))


# ---------------------------------------------------------------
# baseline probes (before any learning) - averaged over repeats since a
# single trial is noisy (the mushroom body is a coincidence detector,
# sensitive to the exact random realization of the Poisson drive)
# ---------------------------------------------------------------
n_reps = 8
probe_rates = {}
for label, odor, odor_stim in [('A', odor_A, odor_A_stim), ('B', odor_B, odor_B_stim)]:
    reps = []
    for r in range(n_reps):
        spikes, _ = run_trial(odor_stim, odor)
        reps.append(mbon_rate(spikes))
    probe_rates[('baseline', label)] = reps
    rate = float(np.mean(reps))
    results['trial'].append(0); results['phase'].append('baseline'); results['odor'].append(label)
    results['mbon_rate_hz'].append(rate); results['mean_w_mV'].append(float(np.mean(syn.w[elig_idx] / mV)))
    results.setdefault('reps', {})[f'baseline_{label}'] = reps
    print(f'baseline odor {label}: MBON pop rate = {rate:.2f} +/- {np.std(reps):.2f} Hz  (n={n_reps}: {[round(x,1) for x in reps]})')

# ---------------------------------------------------------------
# pairing phase: odor A + (programmatic) punishment signal
# ---------------------------------------------------------------
n_pairing = 12
for trial in range(1, n_pairing + 1):
    spikes, fired = run_trial(odor_A_stim, odor_A)
    rate = mbon_rate(spikes)

    # dopamine-gated depression: any plastic synapse whose presynaptic KC
    # fired during this (punished) trial gets weakened
    if fired:
        fired_arr = np.array(list(fired))
        depress = np.isin(elig_pre, fired_arr)
        syn.w[elig_idx[depress]] = syn.w[elig_idx[depress]] * (1 - eta)

    mean_w = float(np.mean(syn.w[elig_idx] / mV))
    results['trial'].append(trial); results['phase'].append('pairing'); results['odor'].append('A')
    results['mbon_rate_hz'].append(rate); results['mean_w_mV'].append(mean_w)
    print(f'pairing trial {trial:2d}: MBON pop rate = {rate:5.2f} Hz   mean plastic weight = {mean_w:.4f} mV   ({len(fired)}/{n_odor} odor-A KCs fired)')

# ---------------------------------------------------------------
# post-learning probes, same averaging as baseline for a fair comparison
# ---------------------------------------------------------------
for label, odor, odor_stim in [('A', odor_A, odor_A_stim), ('B', odor_B, odor_B_stim)]:
    reps = []
    for r in range(n_reps):
        spikes, _ = run_trial(odor_stim, odor)
        reps.append(mbon_rate(spikes))
    probe_rates[('post', label)] = reps
    rate = float(np.mean(reps))
    results['trial'].append(n_pairing + 1); results['phase'].append('post'); results['odor'].append(label)
    results['mbon_rate_hz'].append(rate); results['mean_w_mV'].append(float(np.mean(syn.w[elig_idx] / mV)))
    results.setdefault('reps', {})[f'post_{label}'] = reps
    print(f'post-learning odor {label}: MBON pop rate = {rate:.2f} +/- {np.std(reps):.2f} Hz  (n={n_reps}: {[round(x,1) for x in reps]})')

print()
print('=== summary ===')
for label in ['A', 'B']:
    b = np.mean(probe_rates[('baseline', label)])
    p = np.mean(probe_rates[('post', label)])
    pct = 100 * (p - b) / b if b > 0 else float('nan')
    tag = '(paired/punished)' if label == 'A' else '(control, never paired)'
    print(f'odor {label} {tag}: baseline {b:.2f} Hz -> post {p:.2f} Hz  ({pct:+.1f}%)')

with open('../learning_results.json', 'w') as f:
    json.dump(results, f)
print('>>> wrote learning_results.json')
