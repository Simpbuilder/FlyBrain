"""
Bounded batch run of the fly obstacle-dodge game (NOT endless - runs a
fixed number of lives then stops), recording the full tick-by-tick
trajectory of every life so a real 2D game viewer can replay them,
including comparing an early (untrained) life against a later
(trained) one to actually SEE improvement.

Same real circuits as endless_fly_game.py: real looming detectors ->
Giant Fiber (danger), real left/right HS/VS -> DNg46 (steering,
repurposed from their natural yaw-rotation role to an up/down cue -
a simplification, noted honestly), real thermosensory pathway fires
on collision (the aversive/failure signal). One trainable, explicitly
non-connectome readout maps those three real signals to a move
decision, updated via simple stochastic hill-climbing.
"""
import json
import time
import numpy as np
import pandas as pd

from model import create_model, default_params as params
from brian2 import Network, NeuronGroup, Synapses, ms, Hz, mV, second, defaultclock, seed as brian_seed

brian_seed(7)
rng = np.random.default_rng(0)

config = {'path_comp': './Completeness_783.csv', 'path_con': './Connectivity_783.parquet'}
df_comp = pd.read_csv(config['path_comp'], index_col=0)
ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
flyid2i = {j: i for i, j in enumerate(df_comp.index)}

loom_idx = sorted(flyid2i[x] for x in ann.loc[ann['cell_type'].isin(['LC4', 'LPLC2']), 'root_id'] if x in flyid2i)
gf_idx = sorted(flyid2i[x] for x in ann.loc[ann['cell_type'] == 'DNp01', 'root_id'] if x in flyid2i)
hsvs = ann[ann['cell_type'].astype(str).str.match('^HS|^VS', na=False)]
hsvs_left = sorted(flyid2i[x] for x in hsvs.loc[hsvs['side'] == 'left', 'root_id'] if x in flyid2i)
hsvs_right = sorted(flyid2i[x] for x in hsvs.loc[hsvs['side'] == 'right', 'root_id'] if x in flyid2i)
dng46 = ann[ann['cell_type'] == 'DNg46']
dng46_left = np.array(sorted(flyid2i[x] for x in dng46.loc[dng46['side'] == 'left', 'root_id'] if x in flyid2i))
dng46_right = np.array(sorted(flyid2i[x] for x in dng46.loc[dng46['side'] == 'right', 'root_id'] if x in flyid2i))
trn_idx = sorted(flyid2i[x] for x in ann.loc[ann['cell_class'] == 'thermosensory', 'root_id'] if x in flyid2i)
print(f'>>> loom={len(loom_idx)} gf={len(gf_idx)} hsvs_L={len(hsvs_left)} hsvs_R={len(hsvs_right)} trn={len(trn_idx)}', flush=True)

neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
all_stim = loom_idx + hsvs_left + hsvs_right + trn_idx
n_loom, n_l, n_r, n_trn = len(loom_idx), len(hsvs_left), len(hsvs_right), len(trn_idx)
LOOM = slice(0, n_loom)
HL = slice(n_loom, n_loom + n_l)
HR = slice(n_loom + n_l, n_loom + n_l + n_r)
TRN = slice(n_loom + n_l + n_r, n_loom + n_l + n_r + n_trn)

stim = NeuronGroup(len(all_stim), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(all_stim)), j=np.array(all_stim))
net = Network(neu, syn, spk_mon, stim, drive)

TICK_MS = 30 * ms
N_LANES = 5
gf_arr = np.array(gf_idx)


def sense_and_act(fly_lane, obs_lane, obs_dist):
    neu.v = params['v_0']
    neu.g = 0 * mV
    stim.rate = 0 * Hz
    danger = max(0.0, 1.0 - obs_dist / 4.0)
    if danger > 0:
        n_active = max(1, int(round(danger * n_loom)))
        stim.rate[np.arange(n_active)] = 150 * Hz
        if obs_lane < fly_lane:
            stim.rate[HL] = 150 * Hz
        elif obs_lane > fly_lane:
            stim.rate[HR] = 150 * Hz
    t_start = defaultclock.t / second
    net.run(TICK_MS)
    mask = np.asarray(spk_mon.t / second) >= t_start
    trial_i = np.asarray(spk_mon.i)[mask]
    return (np.isin(trial_i, dng46_left).sum(), np.isin(trial_i, dng46_right).sum(),
            np.isin(trial_i, gf_arr).sum())


def pain():
    neu.v = params['v_0']
    neu.g = 0 * mV
    stim.rate = 0 * Hz
    stim.rate[TRN] = 150 * Hz
    net.run(TICK_MS)


for _ in range(3):
    sense_and_act(2, 2, 4)
print('>>> warm-up complete', flush=True)

w = np.array([1.0, -1.0, 0.3])
bias = 0.0
survival_avg = 5.0
best_survival = 0
lives = []

N_LIVES = 80
t0 = time.time()
for episode in range(1, N_LIVES + 1):
    fly_lane = N_LANES // 2
    obs_lane = int(rng.integers(0, N_LANES))
    obs_dist = 5
    ticks = 0
    perturb = rng.normal(0, 0.15, size=3)
    w_try = w + perturb
    trajectory = []

    while True:
        ticks += 1
        left_r, right_r, gf_r = sense_and_act(fly_lane, obs_lane, obs_dist)
        signal = w_try[0] * left_r + w_try[1] * right_r + w_try[2] * gf_r + bias
        if signal > 1.0 and fly_lane > 0:
            fly_lane -= 1
        elif signal < -1.0 and fly_lane < N_LANES - 1:
            fly_lane += 1

        trajectory.append([fly_lane, obs_lane, obs_dist])
        obs_dist -= 1
        if obs_dist <= 0:
            if fly_lane == obs_lane:
                pain()
                break
            obs_lane = int(rng.integers(0, N_LANES))
            obs_dist = 5
        if ticks > 400:  # safety cap so a batch run can't hang forever
            break

    if ticks > survival_avg:
        w = w_try
    survival_avg = 0.95 * survival_avg + 0.05 * ticks
    best_survival = max(best_survival, ticks)
    lives.append({'episode': episode, 'ticks': ticks, 'trajectory': trajectory, 'weights': w_try.round(2).tolist()})
    print(f'life {episode:3d}: survived {ticks:3d} ticks (avg {survival_avg:.1f}, best {best_survival})', flush=True)

print(f'>>> batch complete in {time.time()-t0:.0f}s', flush=True)
with open('../fly_game_batch.json', 'w') as f:
    json.dump({'n_lanes': N_LANES, 'tick_ms': 30, 'lives': lives, 'final_weights': w.tolist()}, f)
print('>>> wrote fly_game_batch.json', flush=True)
