import numpy as np
from brian2 import NeuronGroup, Synapses, Network, SpikeMonitor, PoissonGroup, ms, mV, Hz, defaultclock
from textwrap import dedent

v_0, v_th, t_mbr, tau, t_rfc = -52*mV, -45*mV, 20*ms, 5*ms, 2.2*ms
eqs = dedent('''
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
''')
neu = NeuronGroup(3, eqs, method='linear', threshold='v > v_th', reset='v = v_0; g = 0*mV',
                   refractory='rfc', name='neu', namespace={'v_0': v_0, 'v_th': v_th, 't_mbr': t_mbr, 'tau': tau})
neu.v = v_0
neu.g = 0
neu.rfc = t_rfc

stim = PoissonGroup(3, rates=0*Hz, name='stim')

w_stim = 68.75*mV
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='drive', namespace={'w_stim': w_stim})
drive.connect(i=[0, 1, 2], j=[0, 1, 2])

spk = SpikeMonitor(neu)
stim_spk = SpikeMonitor(stim)
net = Network(neu, stim, drive, spk, stim_spk)

for trial in range(1, 4):
    neu.v = v_0
    neu.g = 0*mV
    stim.rates = 0*Hz
    stim.rates[[0, 1, 2]] = 150*Hz
    t_start = defaultclock.t
    net.run(100*ms)
    mask = spk.t >= t_start
    smask = stim_spk.t >= t_start
    print(f'trial {trial}: stim spikes = {smask.sum()}, neu spikes = {mask.sum()}')
