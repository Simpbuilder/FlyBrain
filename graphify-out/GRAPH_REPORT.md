# Graph Report - Sigma  (2026-09-17)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 38 nodes · 67 edges · 5 communities
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- brian2
- model.py
- run_trial
- learn_mushroom_body.py
- utils.py

## God Nodes (most connected - your core abstractions)
1. `run_trial()` - 7 edges
2. `create_model()` - 6 edges
3. `run_exp()` - 5 edges
4. `construct_dataframe()` - 3 edges
5. `get_spk_trn()` - 3 edges
6. `poi()` - 3 edges
7. `silence()` - 3 edges
8. `run_trial()` - 2 edges
9. `get_rate()` - 2 edges
10. `load_exps()` - 2 edges

## Surprising Connections (you probably didn't know these)
- `run_exp()` --indirect_call--> `run_trial()`  [INFERRED]
  Drosophila_brain_model/model.py → Drosophila_brain_model/model.py  _Bridges community 1 → community 2_
- `run_trial()` --calls--> `create_model()`  [EXTRACTED]
  Drosophila_brain_model/model.py → Drosophila_brain_model/model.py  _Bridges community 2 → community 3_

## Import Cycles
- None detected.

## Communities (5 total, 0 thin omitted)

### Community 0 - "brian2"
Cohesion: 0.42
Nodes (4): brian2, numpy, pandas, textwrap

### Community 1 - "model.py"
Cohesion: 0.31
Nodes (7): construct_dataframe(), Take spike time dict and collects spikes in pandas dataframe Parameters…, Run default network experiment Neurons in `neu_exc` are Poisson external inputs…, run_exp(), joblib, pathlib, time

### Community 2 - "run_trial"
Cohesion: 0.25
Nodes (8): get_spk_trn(), poi(), Silence neuron by setting weights of all synapses from it to 0 Parameters…, Extracts spike times from 'spk_mon' The spike times recorded in the…, Run single trial of coactivation/silencng experiment During the coactivation…, Create PoissonInput for neurons. For each neuron in 'names' a PoissonInput is…, run_trial(), silence()

### Community 3 - "learn_mushroom_body.py"
Cohesion: 0.29
Nodes (5): Drive the neurons at `target_idx` for one trial via the persistent stim…, run_trial(), create_model(), Create default network model. Convert the "completeness materialization" and…, json

### Community 4 - "utils.py"
Cohesion: 0.40
Nodes (4): get_rate(), load_exps(), Calculate rate and standard deviation for all experiments in df Parameters…, Load simulation results from disk Parameters ---------- l_pkl : list List of…

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_trial()` connect `run_trial` to `model.py`, `learn_mushroom_body.py`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Why does `run_exp()` connect `model.py` to `run_trial`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Why does `create_model()` connect `learn_mushroom_body.py` to `brian2`, `model.py`, `run_trial`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._