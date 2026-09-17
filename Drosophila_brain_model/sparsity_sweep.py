import time
import numpy as np
import pandas as pd

from model import create_model, default_params as params
from brian2 import Network, NeuronGroup, Synapses, ms, Hz, mV, second, defaultclock

config = {'path_comp': './Completeness_783.csv', 'path_con': './Connectivity_783.parquet'}
df_comp = pd.read_csv(config['path_comp'], index_col=0)
ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
flyid2i = {j: i for i, j in enumerate(df_comp.index)}

glomeruli = ['DA1_lPN', 'DA2_lPN', 'DL3_lPN', 'DL2d_adPN']
pn_ids = set(ann.loc[ann['cell_type'].isin(glomeruli), 'root_id'])
pn_idx = sorted(flyid2i[x] for x in pn_ids if x in flyid2i)
kc_ids = set(ann.loc[ann['cell_class'] == 'Kenyon_Cell', 'root_id'])
kc_idx = np.array(sorted(flyid2i[x] for x in kc_ids if x in flyid2i))
mbon_ids = set(ann.loc[ann['cell_type'].astype(str).str.startswith('MBON', na=False), 'root_id'])
mbon_idx = np.array(sorted(flyid2i[x] for x in mbon_ids if x in flyid2i))
print(f'>>> {len(pn_idx)} real PNs across {len(glomeruli)} glomeruli, {len(kc_idx)} total KCs')

neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
stim = NeuronGroup(len(pn_idx), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']  # placeholder, overwritten per test below
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(pn_idx)), j=np.array(pn_idx))
net = Network(neu, syn, spk_mon, stim, drive)


def run_trial(rate_hz, dur_ms, factor):
    global w_stim
    w_stim = params['w_syn'] * factor
    neu.v = params['v_0']
    neu.g = 0 * mV
    stim.rate = 0 * Hz
    stim.rate[np.arange(len(pn_idx))] = rate_hz * Hz
    t_start = defaultclock.t / second
    net.run(dur_ms * ms)
    mask = np.asarray(spk_mon.t / second) >= t_start
    trial_i = np.asarray(spk_mon.i)[mask]
    pn_fired = np.unique(trial_i[np.isin(trial_i, np.array(pn_idx))])
    kc_fired = np.unique(trial_i[np.isin(trial_i, kc_idx)])
    mbon_spikes = np.isin(trial_i, mbon_idx).sum()
    mbon_rate = mbon_spikes / len(mbon_idx) / (dur_ms / 1000)
    return len(pn_fired), len(kc_fired), mbon_rate


for _ in range(3):
    run_trial(150, 300, 250)
print('>>> warm-up complete\n')

print('--- varying external drive weight factor (w_syn * factor), 300ms trial, 150Hz ---')
for factor in [250, 100, 50, 20, 10, 5, 2]:
    n_pn, n_kc, mbon_rate = run_trial(150, 300, factor)
    pct = 100 * n_kc / len(kc_idx)
    print(f'factor={factor:4d} (w={0.275*factor:.2f}mV): {n_pn}/{len(pn_idx)} PNs fired, {n_kc} KCs recruited ({pct:.1f}%), MBON rate={mbon_rate:.2f} Hz')
