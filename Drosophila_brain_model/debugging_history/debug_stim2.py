import numpy as np
import pandas as pd
from model import create_model, default_params as params
from brian2 import NeuronGroup, Synapses, Network, SpikeMonitor, ms, mV, Hz, defaultclock

# tiny synthetic connectome: 5 neurons, a couple of weak synapses
ids = [100, 101, 102, 103, 104]
pd.DataFrame({'Completed': [True]*5}, index=pd.Index(ids, name='')).to_csv('tiny_comp.csv')
df_con = pd.DataFrame({
    'Presynaptic_ID': [100], 'Postsynaptic_ID': [101],
    'Presynaptic_Index': [0], 'Postsynaptic_Index': [1],
    'Connectivity': [1], 'Excitatory': [1], 'Excitatory x Connectivity': [1],
})
df_con.to_parquet('tiny_con.parquet')

neu, syn, spk_mon = create_model('tiny_comp.csv', 'tiny_con.parquet', params)
print('built tiny model:', len(neu), 'neurons', len(syn), 'synapses')

stim_targets = [0, 2]  # drive neuron indices 0 and 2 directly
stim = NeuronGroup(len(stim_targets), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(stim_targets)), j=np.array(stim_targets))

net = Network(neu, syn, spk_mon, stim, drive)

print('--- trial 1: rate=0 ---')
neu.v = params['v_0']; neu.g = 0*mV
stim.rate = 0*Hz
net.run(50*ms)
print('neu spikes so far:', spk_mon.count[:])

print('--- trial 2: rate=150Hz via fancy-index assignment (matches production code) ---')
neu.v = params['v_0']; neu.g = 0*mV
stim.rate = 0*Hz
positions = np.arange(0, len(stim_targets))
stim.rate[positions] = 150*Hz
net.run(100*ms)
print('neu spikes (cumulative):', spk_mon.count[:])
print('stim.rate[:] =', stim.rate[:])
