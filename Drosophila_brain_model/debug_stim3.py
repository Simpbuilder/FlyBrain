import time
import numpy as np
import pandas as pd

from model import create_model, default_params as params
from brian2 import Network, NeuronGroup, Synapses, SpikeMonitor, ms, Hz, mV, defaultclock

config = {'path_comp': './Completeness_783.csv', 'path_con': './Connectivity_783.parquet'}

df_comp = pd.read_csv(config['path_comp'], index_col=0)
ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
flyid2i = {j: i for i, j in enumerate(df_comp.index)}
kc_ids = set(ann.loc[ann['cell_class'] == 'Kenyon_Cell', 'root_id'])
kc_idx = sorted(flyid2i[x] for x in kc_ids if x in flyid2i)

rng = np.random.default_rng(0)
shuffled = rng.permutation(kc_idx)
odor_A = shuffled[:100].tolist()

t0 = time.time()
neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
print(f'built in {time.time()-t0:.1f}s')

stim_targets = odor_A
stim = NeuronGroup(len(stim_targets), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
print('w_stim =', w_stim)
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(stim_targets)), j=np.array(stim_targets))
print('drive synapse count:', len(drive))

stim_spk = SpikeMonitor(stim)
net = Network(neu, syn, spk_mon, stim, drive, stim_spk)

neu.v = params['v_0']
neu.g = 0 * mV
stim.rate = 0 * Hz
stim.rate[np.arange(len(stim_targets))] = 150 * Hz
print('stim.rate set, sample:', stim.rate[:5])

t0 = time.time()
net.run(300 * ms)
print(f'ran in {time.time()-t0:.1f}s')

print('stim total spikes:', stim_spk.num_spikes, 'stim.count sample:', np.asarray(stim_spk.count[:])[:10])
print('neu total spikes:', spk_mon.num_spikes)
target_arr = np.array(odor_A)
neu_i = np.asarray(spk_mon.i)
print('of the 100 target KCs, spikes recorded for:', np.isin(neu_i, target_arr).sum())
print('unique target KCs that spiked:', len(np.unique(neu_i[np.isin(neu_i, target_arr)])))

print()
print('=== SECOND trial, same setup, testing repeated net.run() ===')
neu.v = params['v_0']
neu.g = 0 * mV
stim.rate = 0 * Hz
stim.rate[np.arange(len(stim_targets))] = 150 * Hz
t_start2 = defaultclock.t
t0 = time.time()
net.run(300 * ms)
print(f'ran in {time.time()-t0:.1f}s, t_start2={t_start2}')

mask2 = spk_mon.t >= t_start2
neu_i2 = np.asarray(spk_mon.i)[mask2]
print('trial2 total neu spikes:', len(neu_i2))
print('trial2: of the 100 target KCs, spikes recorded for:', np.isin(neu_i2, target_arr).sum())
print('trial2: unique target KCs that spiked:', len(np.unique(neu_i2[np.isin(neu_i2, target_arr)])))
stim_i2_mask = stim_spk.t >= t_start2
print('trial2 stim spikes:', np.asarray(stim_spk.i)[stim_i2_mask].shape[0])
