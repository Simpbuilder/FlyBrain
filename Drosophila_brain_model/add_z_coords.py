import json
import numpy as np
import pandas as pd

ann = pd.read_csv('annotations_783.tsv', sep='\t', low_memory=False)
z_lookup = ann.drop_duplicates('root_id').set_index('root_id')['pos_z']

df_comp = pd.read_csv('Completeness_783.csv', index_col=0)
flyid_by_idx = df_comp.index.to_numpy()  # brian index -> root_id, in order


def enrich(path_in, path_out):
    with open(path_in) as f:
        d = json.load(f)
    idx = np.array(d['i'])
    root_ids = flyid_by_idx[idx]
    z = z_lookup.reindex(root_ids).to_numpy()
    z = np.nan_to_num(z, nan=float(np.nanmedian(z)))
    d['z'] = z.round(1).tolist()
    with open(path_out, 'w') as f:
        json.dump(d, f)
    print(f'{path_out}: {len(idx)} spikes enriched with real z coordinates')


enrich('../compartment_before.json', '../compartment_before_3d.json')
enrich('../compartment_after.json', '../compartment_after_3d.json')
