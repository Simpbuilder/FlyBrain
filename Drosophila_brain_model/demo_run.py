import time
from model import run_exp
from model import default_params as params
import utils as utl
from brian2 import Hz, ms

config = {
    'path_res'  : './results/demo',
    'path_comp' : './2023_03_23_completeness_630_final.csv',
    'path_con'  : './2023_03_23_connectivity_630_final.parquet',
    'n_proc'    : -1,
}

neu_sugar = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
id_mn9 = 720575940660219265

# quick smoke run: fewer trials, shorter duration
params['n_run'] = 2
params['t_run'] = 300 * ms

t0 = time.time()
run_exp(exp_name='demo_sugar', neu_exc=neu_sugar, params=params, force_overwrite=True, **config)
print(f'>>> total wall time: {time.time() - t0:.1f}s')

df_spike = utl.load_exps(['./results/demo/demo_sugar.parquet'])
print('total spikes:', len(df_spike))
print('unique active neurons:', df_spike['flywire_id'].nunique())

df_rate, df_std = utl.get_rate(df_spike, t_run=params['t_run'], n_run=params['n_run'])
print('\nMN9 rate:')
print(df_rate.loc[id_mn9] if id_mn9 in df_rate.index else 'MN9 did not fire')

print('\ntop 10 active neurons:')
print(df_rate.sort_values('demo_sugar', ascending=False).head(10))
