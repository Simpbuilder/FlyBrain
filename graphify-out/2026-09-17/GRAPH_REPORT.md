# Graph Report - Sigma  (2026-09-17)

## Corpus Check
- 70 files · ~6,539,477 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 22 file(s) not represented in the graph (top: .parquet 10, .csv 5, (none) 3)

## Summary
- 162 nodes · 343 edges · 13 communities (8 shown, 5 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `43538dba`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- brian2
- model.py
- What's in here
- numpy
- utils.py
- Session notes: extended autonomous run
- time
- create_model
- Installation
- endless_fly_game.py
- compartment_capture_spatial.py
- learn_and_capture_spatial.py
- looming_escape.py

## God Nodes (most connected - your core abstractions)
1. `create_model()` - 31 edges
2. `Session notes: extended autonomous run` - 14 edges
3. `What's in here` - 8 edges
4. `run_trial()` - 7 edges
5. `export_for_dashboard()` - 6 edges
6. `Installation` - 6 edges
7. `Results summary` - 6 edges
8. `run_exp()` - 5 edges
9. `silence()` - 4 edges
10. `Milestone 5: closing the two biggest honest gaps` - 4 edges

## Surprising Connections (you probably didn't know these)
- `Milestone 11: live dashboard (db-backed, extensible for future sessions)` --references--> `export_for_dashboard()`  [INFERRED]
  Drosophila_brain_model/NOTES.md → Drosophila_brain_model/dashboard_export.py
- `Debugging history` --references--> `create_model()`  [INFERRED]
  Drosophila_brain_model/debugging_history/README.md → Drosophila_brain_model/model.py
- `Milestone 8: necessity and integration tests` --references--> `silence()`  [INFERRED]
  Drosophila_brain_model/NOTES.md → Drosophila_brain_model/model.py

## Import Cycles
- None detected.

## Communities (13 total, 5 thin omitted)

### Community 1 - "model.py"
Cohesion: 0.11
Nodes (15): construct_dataframe(), get_spk_trn(), poi(), Silence neuron by setting weights of all synapses from it to 0 Parameters…, Extracts spike times from 'spk_mon' The spike times recorded in the…, Take spike time dict and collects spikes in pandas dataframe Parameters…, Run single trial of coactivation/silencng experiment During the coactivation…, Run default network experiment Neurons in `neu_exc` are Poisson external inputs… (+7 more)

### Community 2 - "What's in here"
Cohesion: 0.11
Nodes (16): 1. Sensory-to-motor replication, 2. Mushroom body learning, 3. Looming-escape reflex, 4. Optomotor steering reflex, 5. Multi-modal convergence, 6. Dose-response and necessity tests, 6. Real 3D visualization, FlyBrain (+8 more)

### Community 3 - "numpy"
Cohesion: 0.12
Nodes (8): collections, csv, Drive the neurons at `target_idx` for one trial via the persistent stim…, run_trial(), json, numpy, os, pandas

### Community 4 - "utils.py"
Cohesion: 0.40
Nodes (4): get_rate(), load_exps(), Calculate rate and standard deviation for all experiments in df Parameters…, Load simulation results from disk Parameters ---------- l_pkl : list List of…

### Community 5 - "Session notes: extended autonomous run"
Cohesion: 0.12
Nodes (16): Future ideas (not started), Known limitations / honest caveats, Milestone 10: chasing the sparse-coding fix further - corrected conclusion, Milestone 11: live dashboard (db-backed, extensible for future sessions), Milestone 1: clean learning statistics, Milestone 2: spatial visualization, Milestone 3: looming-escape reflex (second independent real circuit), Milestone 4: literature comparison (+8 more)

### Community 7 - "create_model"
Cohesion: 0.24
Nodes (6): export_for_dashboard(), Standardized export for pushing simulation results to the live dashboard. Any…, name: short string describing the experiment, shown in the dashboard spk_mon:…, Debugging history, create_model(), Create default network model. Convert the "completeness materialization" and…

### Community 8 - "Installation"
Cohesion: 0.20
Nodes (9): Brian 2 performance, dependencies, Installation, Model for the _Drosophila_ brain, Paper, Quick Start, Usage, Version 783 (+1 more)

### Community 9 - "endless_fly_game.py"
Cohesion: 0.33
Nodes (5): pain(), Endless fly survival game, driven by the real connectome. World: 5 discrete…, Fire the real thermosensory pathway - the aversive/failure signal., One game tick: drive real sensors based on world state, run the brain briefly,…, sense_and_act()

## Knowledge Gaps
- **34 isolated node(s):** `Future ideas (not started)`, `Milestone 1: clean learning statistics`, `Milestone 2: spatial visualization`, `Milestone 3: looming-escape reflex (second independent real circuit)`, `Milestone 4: literature comparison` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 83 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Session notes: extended autonomous run` connect `Session notes: extended autonomous run` to `model.py`, `What's in here`?**
  _High betweenness centrality (0.330) - this node is a cross-community bridge._
- **Why does `Milestone 8: necessity and integration tests` connect `model.py` to `Session notes: extended autonomous run`?**
  _High betweenness centrality (0.229) - this node is a cross-community bridge._
- **What connects `Future ideas (not started)`, `Milestone 1: clean learning statistics`, `Milestone 2: spatial visualization` to the rest of the system?**
  _34 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `model.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11067193675889328 - nodes in this community are weakly interconnected._
- **Should `What's in here` be split into smaller, more focused modules?**
  _Cohesion score 0.1111111111111111 - nodes in this community are weakly interconnected._
- **Should `numpy` be split into smaller, more focused modules?**
  _Cohesion score 0.12258064516129032 - nodes in this community are weakly interconnected._
- **Should `Session notes: extended autonomous run` be split into smaller, more focused modules?**
  _Cohesion score 0.125 - nodes in this community are weakly interconnected._