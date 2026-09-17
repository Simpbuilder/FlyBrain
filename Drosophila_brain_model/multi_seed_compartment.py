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
kc_idx_all = sorted(flyid2i[x] for x in kc_ids if x in flyid2i)
mbon_idx = np.array(sorted(flyid2i[x] for x in mbon_ids if x in flyid2i))

dom = pd.read_csv('mbon_dan_dominance.csv', index_col=0)
ppl_mbon_idx = np.array([i for i in mbon_idx if i in dom.index and dom.loc[i, 'dominant'] == 'PPL'])
pam_mbon_idx = np.array([i for i in mbon_idx if i in dom.index and dom.loc[i, 'dominant'] == 'PAM'])
print(f'>>> {len(ppl_mbon_idx)} PPL-dominant MBONs, {len(pam_mbon_idx)} PAM-dominant MBONs')

n_odor = 150
t_run = 300 * ms
r_stim = 150 * Hz
eta = 0.18
n_reps = 8
n_pairing = 12
n_seeds = 5
n_warmup = 6

results = []

for seed in range(n_seeds):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(kc_idx_all)
    odor_A = shuffled[:n_odor].tolist()
    odor_B = shuffled[n_odor:2 * n_odor].tolist()

    neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
    syn_i = np.asarray(syn.i[:])
    syn_j = np.asarray(syn.j[:])
    odor_A_arr = np.array(odor_A)
    elig_mask = np.isin(syn_i, odor_A_arr) & np.isin(syn_j, ppl_mbon_idx)
    elig_idx = np.where(elig_mask)[0]
    elig_pre = syn_i[elig_idx]

    stim_targets = odor_A + odor_B
    stim = NeuronGroup(len(stim_targets), 'rate : Hz', threshold='rand() < rate*dt', name=f'stim{seed}')
    stim.rate = 0 * Hz
    w_stim = params['w_syn'] * params['f_poi']
    drive = Synapses(stim, neu, on_pre='v_post += w_stim', name=f'stim_syn{seed}')
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

    for _ in range(n_warmup):
        run_trial(odor_B_stim, odor_B)

    base_ppl, base_pam = [], []
    for r in range(n_reps):
        ps, ms_, _ = run_trial(odor_A_stim, odor_A)
        base_ppl.append(rate(ps, len(ppl_mbon_idx)))
        base_pam.append(rate(ms_, len(pam_mbon_idx)))

    for trial in range(n_pairing):
        ps, ms_, fired = run_trial(odor_A_stim, odor_A)
        if fired:
            fired_arr = np.array(list(fired))
            depress = np.isin(elig_pre, fired_arr)
            syn.w[elig_idx[depress]] = syn.w[elig_idx[depress]] * (1 - eta)

    post_ppl, post_pam = [], []
    for r in range(n_reps):
        ps, ms_, _ = run_trial(odor_A_stim, odor_A)
        post_ppl.append(rate(ps, len(ppl_mbon_idx)))
        post_pam.append(rate(ms_, len(pam_mbon_idx)))

    b_ppl, p_ppl = np.mean(base_ppl), np.mean(post_ppl)
    b_pam, p_pam = np.mean(base_pam), np.mean(post_pam)
    ppl_pct = 100 * (p_ppl - b_ppl) / b_ppl
    pam_pct = 100 * (p_pam - b_pam) / b_pam
    dt = time.time() - t0
    noisy = (np.std(base_ppl) > 15) or (np.std(base_pam) > 15)
    flag = '  <-- HIGH BASELINE VARIANCE, still bimodal, treat with caution' if noisy else ''
    print(f'seed {seed} ({dt:.0f}s): PPL {b_ppl:.1f}->{p_ppl:.1f} Hz ({ppl_pct:+.1f}%)   PAM {b_pam:.1f}->{p_pam:.1f} Hz ({pam_pct:+.1f}%)   n_eligible_syn={len(elig_idx)}{flag}')
    print(f'          baseline reps -- PPL: {[round(x,1) for x in base_ppl]}   PAM: {[round(x,1) for x in base_pam]}')
    results.append({'seed': seed, 'ppl_baseline': b_ppl, 'ppl_post': p_ppl, 'ppl_pct': ppl_pct,
                     'pam_baseline': b_pam, 'pam_post': p_pam, 'pam_pct': pam_pct, 'n_eligible': int(len(elig_idx)),
                     'noisy_baseline': bool(noisy)})

print()
clean = [r for r in results if not r['noisy_baseline']]
ppl_pcts = [r['ppl_pct'] for r in clean]
pam_pcts = [r['pam_pct'] for r in clean]
print(f'=== across {n_seeds} independent random odor ensembles ({len(clean)} with a stable baseline) ===')
print(f'PPL-dominant (punishment) MBON change: {np.mean(ppl_pcts):+.1f}% +/- {np.std(ppl_pcts):.1f}%   (clean seeds: {[round(x,1) for x in ppl_pcts]})')
print(f'PAM-dominant (reward) MBON change:     {np.mean(pam_pcts):+.1f}% +/- {np.std(pam_pcts):.1f}%   (clean seeds: {[round(x,1) for x in pam_pcts]})')
if len(clean) < n_seeds:
    print(f'({n_seeds - len(clean)} seed(s) excluded for high baseline variance - see per-seed output above)')

with open('../multi_seed_results.json', 'w') as f:
    json.dump(results, f)
print('>>> wrote multi_seed_results.json')
