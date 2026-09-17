import csv
import os
import time
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
mbon_ids = set(ann.loc[ann['cell_type'].astype(str).str.startswith('MBON', na=False), 'root_id'])
mbon_idx = np.array(sorted(flyid2i[x] for x in mbon_ids if x in flyid2i))
print(f'>>> odor A: {len(pnA_idx)} PNs, odor B: {len(pnB_idx)} PNs, {len(kc_idx)} total KCs', flush=True)

neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
all_pn = pnA_idx + pnB_idx
stim = NeuronGroup(len(all_pn), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_syn_base = params['w_syn']
drive = Synapses(stim, neu, 'w_d : volt', on_pre='v_post += w_d', name='stim_syn')
drive.connect(i=np.arange(len(all_pn)), j=np.array(all_pn))
A_stim = np.arange(0, len(pnA_idx))
B_stim = np.arange(len(pnA_idx), len(all_pn))
net = Network(neu, syn, spk_mon, stim, drive)


def run_trial(positions, factor, dur_ms):
    drive.w_d = w_syn_base * factor
    neu.v = params['v_0']
    neu.g = 0 * mV
    stim.rate = 0 * Hz
    stim.rate[positions] = 150 * Hz
    t_start = defaultclock.t / second
    net.run(dur_ms * ms)
    mask = np.asarray(spk_mon.t / second) >= t_start
    trial_i = np.asarray(spk_mon.i)[mask]
    kc_fired = set(np.unique(trial_i[np.isin(trial_i, kc_idx)]).tolist())
    mbon_spikes = np.isin(trial_i, mbon_idx).sum()
    mbon_rate = mbon_spikes / len(mbon_idx) / (dur_ms / 1000)
    return kc_fired, mbon_rate


for _ in range(3):
    run_trial(A_stim, 250, 300)
print('>>> warm-up complete', flush=True)

out_path = '../sparsity_grid_results.csv'
fields = ['factor', 'dur_ms', 'rep', 'n_kc_A', 'n_kc_B', 'overlap', 'overlap_pct_of_smaller',
          'mbon_rate_A', 'mbon_rate_B', 'wall_s']
write_header = not os.path.exists(out_path)
f = open(out_path, 'a', newline='')
writer = csv.DictWriter(f, fieldnames=fields)
if write_header:
    writer.writeheader()
    f.flush()

factors = [2, 10, 50, 100, 250]
durations = [10, 15, 20, 30, 50, 100, 200, 300]
n_reps = 2

total = len(factors) * len(durations) * n_reps
done = 0
t_start_all = time.time()

for factor in factors:
    for dur_ms in durations:
        for rep in range(n_reps):
            t0 = time.time()
            kc_A, mbon_A = run_trial(A_stim, factor, dur_ms)
            kc_B, mbon_B = run_trial(B_stim, factor, dur_ms)
            overlap = kc_A & kc_B
            smaller = max(1, min(len(kc_A), len(kc_B)))
            row = {
                'factor': factor, 'dur_ms': dur_ms, 'rep': rep,
                'n_kc_A': len(kc_A), 'n_kc_B': len(kc_B),
                'overlap': len(overlap), 'overlap_pct_of_smaller': round(100 * len(overlap) / smaller, 1),
                'mbon_rate_A': round(mbon_A, 2), 'mbon_rate_B': round(mbon_B, 2),
                'wall_s': round(time.time() - t0, 1),
            }
            writer.writerow(row)
            f.flush()
            done += 1
            elapsed = time.time() - t_start_all
            print(f'[{done}/{total}] factor={factor} dur={dur_ms}ms rep={rep}: '
                  f'KC_A={len(kc_A)} KC_B={len(kc_B)} overlap={row["overlap_pct_of_smaller"]}% '
                  f'MBON_A={mbon_A:.1f} MBON_B={mbon_B:.1f} ({elapsed:.0f}s elapsed)', flush=True)

f.close()
print('>>> grid search complete, wrote sparsity_grid_results.csv', flush=True)
