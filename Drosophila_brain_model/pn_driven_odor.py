import time
import numpy as np
import pandas as pd

from model import create_model, default_params as params
from brian2 import Network, NeuronGroup, Synapses, ms, Hz, mV, second, defaultclock

config = {'path_comp': './Completeness_783.csv', 'path_con': './Connectivity_783.parquet'}

df_comp = pd.read_csv(config['path_comp'], index_col=0)
ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
flyid2i = {j: i for i, j in enumerate(df_comp.index)}

# a real multi-glomerulus "odor" - 8 real projection-neuron types together,
# simulating combinatorial activation the way a real odorant would (a
# single glomerulus alone is subthreshold for most Kenyon cells; real
# odor identity is encoded combinatorially across ~a dozen glomeruli)
glomeruli = ['DA1_lPN', 'DA2_lPN', 'DL3_lPN', 'DL2d_adPN', 'VM5d_adPN', 'VA1v_adPN', 'DL2v_adPN', 'VC3_adPN']
pn_ids = set(ann.loc[ann['cell_type'].isin(glomeruli), 'root_id'])
pn_idx = sorted(flyid2i[x] for x in pn_ids if x in flyid2i)
print(f'>>> real odor: {len(glomeruli)} glomeruli, {len(pn_idx)} real projection neurons')

kc_ids = set(ann.loc[ann['cell_class'] == 'Kenyon_Cell', 'root_id'])
mbon_ids = set(ann.loc[ann['cell_type'].astype(str).str.startswith('MBON', na=False), 'root_id'])
kc_idx = np.array(sorted(flyid2i[x] for x in kc_ids if x in flyid2i))
mbon_idx = np.array(sorted(flyid2i[x] for x in mbon_ids if x in flyid2i))

t0 = time.time()
neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
print(f'>>> network built in {time.time() - t0:.1f}s')

stim = NeuronGroup(len(pn_idx), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(pn_idx)), j=np.array(pn_idx))

net = Network(neu, syn, spk_mon, stim, drive)
t_run = 300 * ms
r_stim = 150 * Hz


def run_trial():
    neu.v = params['v_0']
    neu.g = 0 * mV
    stim.rate = 0 * Hz
    stim.rate[np.arange(len(pn_idx))] = r_stim
    t_start = defaultclock.t / second
    net.run(t_run)
    mask = np.asarray(spk_mon.t / second) >= t_start
    trial_i = np.asarray(spk_mon.i)[mask]
    pn_fired = np.isin(trial_i, np.array(pn_idx)).sum()
    kc_fired = np.unique(trial_i[np.isin(trial_i, kc_idx)])
    mbon_spikes = np.isin(trial_i, mbon_idx).sum()
    return pn_fired, kc_fired, mbon_spikes


for _ in range(3):
    run_trial()
print('>>> warm-up complete')

n_reps = 6
for r in range(n_reps):
    pn_fired, kc_fired, mbon_spikes = run_trial()
    mbon_rate = mbon_spikes / len(mbon_idx) / (t_run / (1000 * ms))
    print(f'trial {r+1}: {pn_fired} PN spikes, {len(kc_fired)} distinct KCs recruited (of {len(kc_idx)}), MBON pop rate = {mbon_rate:.2f} Hz')
