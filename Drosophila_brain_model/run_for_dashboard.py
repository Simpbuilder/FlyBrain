import numpy as np
import pandas as pd

from model import create_model, default_params as params
from brian2 import Network, NeuronGroup, Synapses, ms, Hz, mV, second, defaultclock, seed as brian_seed
from dashboard_export import export_for_dashboard

brian_seed(7)

config = {'path_comp': './Completeness_783.csv', 'path_con': './Connectivity_783.parquet'}
df_comp = pd.read_csv(config['path_comp'], index_col=0)
ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
flyid2i = {j: i for i, j in enumerate(df_comp.index)}

loom_ids = set(ann.loc[ann['cell_type'].isin(['LC4', 'LPLC2']), 'root_id'])
gf_ids = set(ann.loc[ann['cell_type'] == 'DNp01', 'root_id'])
loom_idx = sorted(flyid2i[x] for x in loom_ids if x in flyid2i)
gf_idx = sorted(flyid2i[x] for x in gf_ids if x in flyid2i)

ann_pos = ann.drop_duplicates('root_id').set_index('root_id')[['pos_x', 'pos_y']]
pos_df = ann_pos.reindex(df_comp.index)
pos_x = pos_df['pos_x'].to_numpy()
pos_y = pos_df['pos_y'].to_numpy()

neu, syn, spk_mon = create_model(config['path_comp'], config['path_con'], params)
stim = NeuronGroup(len(loom_idx), 'rate : Hz', threshold='rand() < rate*dt', name='stim')
stim.rate = 0 * Hz
w_stim = params['w_syn'] * params['f_poi']
drive = Synapses(stim, neu, on_pre='v_post += w_stim', name='stim_syn')
drive.connect(i=np.arange(len(loom_idx)), j=np.array(loom_idx))
net = Network(neu, syn, spk_mon, stim, drive)
t_run = 300 * ms

for _ in range(3):
    neu.v = params['v_0']; neu.g = 0 * mV
    stim.rate = 0 * Hz; stim.rate[np.arange(len(loom_idx))] = 150 * Hz
    net.run(t_run)
print('>>> warm-up complete', flush=True)

neu.v = params['v_0']; neu.g = 0 * mV
stim.rate = 0 * Hz
stim.rate[np.arange(len(loom_idx))] = 150 * Hz
t_start = defaultclock.t / second
net.run(t_run)

export_for_dashboard(
    name='Looming escape: LC4+LPLC2 -> Giant Fiber',
    spk_mon=spk_mon, t_start=t_start, duration_ms=300,
    pos_x=pos_x, pos_y=pos_y,
    highlight={'loom': loom_idx, 'gf': gf_idx},
)
