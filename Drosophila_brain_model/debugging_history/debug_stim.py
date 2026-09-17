import numpy as np
from brian2 import NeuronGroup, Synapses, Network, SpikeMonitor, StateMonitor, ms, mV, Hz, defaultclock
from textwrap import dedent

v_0, v_rst, v_th, t_mbr, tau, t_rfc, t_dly = -52*mV, -52*mV, -45*mV, 20*ms, 5*ms, 2.2*ms, 1.8*ms
eqs = dedent('''
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
''')
neu = NeuronGroup(3, eqs, method='linear', threshold='v > v_th', reset='v = v_rst; g = 0*mV',
                   refractory='rfc', name='neu', namespace={'v_0': v_0, 'v_th': v_th, 't_mbr': t_mbr, 'tau': tau})
neu.v = v_0
neu.g = 0
neu.rfc = t_rfc

stim = NeuronGroup(3, 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0*Hz

w_stim = 68.75*mV
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='drive', namespace={'w_stim': w_stim})
drive.connect(i=[0, 1, 2], j=[0, 1, 2])

spk = SpikeMonitor(neu)
stim_spk = SpikeMonitor(stim)
vmon = StateMonitor(neu, 'v', record=True)

net = Network(neu, stim, drive, spk, stim_spk, vmon)

print('--- trial 1: rate=0 (should be silent) ---')
stim.rate = 0*Hz
net.run(50*ms)
print('stim spikes:', stim_spk.count[:], 'neu spikes:', spk.count[:])

print('--- trial 2: rate=150Hz on all 3 ---')
stim.rate = 150*Hz
net.run(100*ms)
print('stim spikes (cumulative):', stim_spk.count[:], 'neu spikes (cumulative):', spk.count[:])
print('v sample around stim onset (neuron 0):', vmon.v[0][490:520]/mV)

print('--- trial 3: rate=150Hz AGAIN (repeat, same value, testing re-activation) ---')
neu.v = v_0
neu.g = 0*mV
stim.rate = 0*Hz
stim.rate = 150*Hz
t3 = defaultclock.t
net.run(100*ms)
m3 = spk.t >= t3
sm3 = stim_spk.t >= t3
print('trial3: stim spikes =', sm3.sum(), 'neu spikes =', m3.sum())
