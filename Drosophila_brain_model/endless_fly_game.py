"""
Endless fly survival game, driven by the real connectome.

World: 5 discrete vertical lanes (0=top .. 4=bottom). A single obstacle
approaches at a random lane; the fly must be in a different lane when it
arrives. Every game tick:
  - obstacle proximity drives the REAL looming detectors (LC4/LPLC2) that
    already reliably fire the real Giant Fiber (escape urgency signal)
  - obstacle direction (above/below the fly) drives the REAL left- or
    right-side HS/VS optomotor neurons (a real anatomical asymmetry,
    repurposed here as an up/down cue rather than their natural yaw-
    rotation role - a simplification, noted honestly)
  - a small TRAINABLE readout (NOT part of the real connectome - the one
    deliberately artificial piece) turns [left DNg46 rate, right DNg46
    rate, GF rate] into move up / down / stay
  - on collision, the real thermosensory ("pain") pathway fires, which is
    also the real signal that drives aversive learning elsewhere in this
    project - failure has a real physiological correlate, not just a
    score decrement
  - the readout's weights get a small stochastic hill-climbing nudge
    after each life, based on whether it beat its own running-average
    survival time (simple, cheap, genuinely improves with experience -
    not a full RL framework, but real trial-and-error learning)

Runs forever (or until stopped). Periodically writes status to
../game_state.json for dashboard pushes.
"""
import json
import time
import numpy as np
import pandas as pd

from model import create_model, default_params as params
from brian2 import Network, NeuronGroup, Synapses, ms, Hz, mV, second, defaultclock, seed as brian_seed

brian_seed(7)
rng = np.random.default_rng()

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

ann_pos = ann.drop_duplicates('root_id').set_index('root_id')[['pos_x', 'pos_y']]
pos_df = ann_pos.reindex(df_comp.index)
pos_x = pos_df['pos_x'].to_numpy()
pos_y = pos_df['pos_y'].to_numpy()

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
    """One game tick: drive real sensors based on world state, run the
    brain briefly, read real motor-relevant neurons, return their rates."""
    neu.v = params['v_0']
    neu.g = 0 * mV
    stim.rate = 0 * Hz
    danger = max(0.0, 1.0 - obs_dist / 4.0)  # 0 far away .. 1 imminent
    if danger > 0:
        n_active = max(1, int(round(danger * n_loom)))
        stim.rate[np.arange(n_active)] = 150 * Hz
        if obs_lane < fly_lane:
            stim.rate[HL] = 150 * Hz  # threat above -> left HS/VS
        elif obs_lane > fly_lane:
            stim.rate[HR] = 150 * Hz  # threat below -> right HS/VS
    t_start = defaultclock.t / second
    net.run(TICK_MS)
    mask = np.asarray(spk_mon.t / second) >= t_start
    trial_i = np.asarray(spk_mon.i)[mask]
    left_r = np.isin(trial_i, dng46_left).sum()
    right_r = np.isin(trial_i, dng46_right).sum()
    gf_r = np.isin(trial_i, gf_arr).sum()
    return left_r, right_r, gf_r


def pain():
    """Fire the real thermosensory pathway - the aversive/failure signal."""
    neu.v = params['v_0']
    neu.g = 0 * mV
    stim.rate = 0 * Hz
    stim.rate[TRN] = 150 * Hz
    net.run(TICK_MS)


for _ in range(3):
    sense_and_act(2, 2, 4)
print('>>> warm-up complete', flush=True)

# the one trainable, non-connectome piece
w = np.array([1.0, -1.0, 0.3])  # [left_rate, right_rate, gf_rate] -> move signal
bias = 0.0
survival_avg = 5.0
episode = 0
best_survival = 0
last_export = 0

print('>>> starting endless run', flush=True)
t_loop_start = time.time()

while True:
    episode += 1
    fly_lane = N_LANES // 2
    obs_lane = rng.integers(0, N_LANES)
    obs_dist = 5
    ticks = 0
    perturb = rng.normal(0, 0.15, size=3)
    w_try = w + perturb

    while True:
        ticks += 1
        left_r, right_r, gf_r = sense_and_act(fly_lane, obs_lane, obs_dist)
        signal = w_try[0] * left_r + w_try[1] * right_r + w_try[2] * gf_r + bias
        if signal > 1.0 and fly_lane > 0:
            fly_lane -= 1
        elif signal < -1.0 and fly_lane < N_LANES - 1:
            fly_lane += 1

        obs_dist -= 1
        if obs_dist <= 0:
            if fly_lane == obs_lane:
                pain()
                break
            obs_lane = rng.integers(0, N_LANES)
            obs_dist = 5

    # simple stochastic hill-climbing: keep the perturbation if this life
    # beat the running average, otherwise drift back
    if ticks > survival_avg:
        w = w_try
    survival_avg = 0.95 * survival_avg + 0.05 * ticks
    best_survival = max(best_survival, ticks)

    if ticks % 1 == 0:
        print(f'life {episode:4d}: survived {ticks:3d} ticks (avg {survival_avg:.1f}, best {best_survival}) '
              f'w={np.round(w,2).tolist()}', flush=True)

    if time.time() - last_export > 20:
        last_export = time.time()
        with open('../game_state.json', 'w') as f:
            json.dump({'episode': episode, 'last_survival': ticks, 'avg_survival': round(survival_avg, 2),
                       'best_survival': int(best_survival), 'weights': w.tolist(),
                       'elapsed_s': round(time.time() - t_loop_start, 1)}, f)
