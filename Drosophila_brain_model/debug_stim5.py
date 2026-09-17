import numpy as np
from brian2 import NeuronGroup, Synapses, Network, SpikeMonitor, SpikeGeneratorGroup, ms, mV, Hz, second, defaultclock
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

rng = np.random.default_rng(0)
rate = 150.0  # Hz
dur = 0.1     # s per trial

# trial pattern: silent, active, active again
active = [False, True, True]
idx_list, t_list = [], []
for k, on in enumerate(active):
    t0 = k * dur
    if on:
        for n in range(3):
            t = t0
            while True:
                t += rng.exponential(1.0 / rate)
                if t >= t0 + dur:
                    break
                idx_list.append(n)
                t_list.append(t)

t_arr = np.round(np.array(t_list) / 1e-4) * 1e-4  # snap to 100us grid, dt of default clock
idx_arr = np.array(idx_list)
pairs = sorted(set(zip(idx_arr.tolist(), t_arr.tolist())))
idx_arr = np.array([p[0] for p in pairs])
t_arr = np.array([p[1] for p in pairs])
gen = SpikeGeneratorGroup(3, idx_arr, t_arr * second, name='gen')
w_stim = 68.75 * mV
drive = Synapses(gen, neu, on_pre='v_post += w_stim', name='drive', namespace={'w_stim': w_stim})
drive.connect(i=[0, 1, 2], j=[0, 1, 2])

spk = SpikeMonitor(neu)
gen_spk = SpikeMonitor(gen)
net = Network(neu, gen, drive, spk, gen_spk)

print('total scheduled spikes:', len(idx_arr), 'sample times:', t_arr[:5])
for trial in range(3):
    neu.v = v_0
    neu.g = 0 * mV
    t_start = defaultclock.t
    net.run(dur * second)
    m = spk.t >= t_start
    gm = gen_spk.t >= t_start
    print(f'trial {trial+1}: t_start={t_start}, gen_spk.t[-5:]={np.asarray(gen_spk.t[:])[-5:]}, gm.sum()={gm.sum()}')
    print(f'  gen spikes = {gm.sum()}, neu spikes = {m.sum()}, gen.count total = {np.asarray(gen_spk.count[:])}')
