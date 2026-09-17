# Session notes: extended autonomous run

## Milestone 1: clean learning statistics
- Swept KC odor-ensemble size (50-250) against MBON response reliability.
  Below ~100 KCs the mushroom body almost never crosses its response
  threshold; 125 is a near-exact coin flip (P=0.75 across 8 reps); 150+
  is consistently reliable. See `ensemble_sweep.py` / `ensemble_sweep_results.json`.
- Found and fixed a network "warm-up" transient: a freshly-built model's
  first few trials respond weaker regardless of stimulus. Fixed by
  discarding a handful of habituation trials before measuring, same as
  a real electrophysiology protocol would.
- Final clean result (`learn_and_capture_spatial.py`, 150 KCs/odor, 8
  reps, 12 pairing trials, eta=0.18 multiplicative depression):
  - odor A (paired with punishment signal): 86.24 -> 80.75 Hz (-6.4%)
  - odor B (control, never paired): 86.76 -> 85.27 Hz (-1.7%, within noise)
  - mean weight at the 1,789 plastic KC(A)->MBON synapses: 0.87mV -> 0.098mV (-89%)
  - the two odors are statistically indistinguishable at baseline and
    clearly diverge after learning, with tight error bars (+/-0.25-0.6Hz)

## Milestone 2: spatial visualization
- `brain_activity_map.html` (published as Artifact "Mushroom body
  memory") - real neuron 3D coordinates from FlyWire annotations,
  animated before/after comparison. Background activity is spatially
  binned (30 time bins x 90x55 grid) for a density "glow"; the 150
  odor-A KCs and 96 MBONs are tracked individually with exact spike
  timing for a proper pulse animation.
- Data pipeline: `learn_and_capture_spatial.py` -> spatial_before/after.json
  -> `prep_spatial_viz.py` -> spatial_viz.json (compact, ~1.2MB)

## Milestone 3: looming-escape reflex (second independent real circuit)
- Identified real LC4 (104 neurons) and LPLC2 (210 neurons) looming
  detectors, and the real Giant Fiber (DNp01, 2 neurons, one per
  hemisphere) in the FlyWire annotations.
- Confirmed a real direct anatomical pathway: 293 direct synapses from
  LC4/LPLC2 onto the Giant Fiber, plus 504 second-hop reinforcing
  neurons - matches published escape-circuit anatomy (Ache et al. 2019).
- `looming_escape.py`: driving LC4+LPLC2 reliably fires the Giant Fiber
  every trial (46-54 spikes/trial on each side, 5/5 reps), with first-spike
  latency of 1.7-4.0ms after stimulus onset.

## Milestone 4: literature comparison
- Qualitative check verified: the GF escape pathway is well documented
  as one of the fastest visual-to-motor circuits in any animal, with
  response latencies described in the literature as "a few
  milliseconds." Our simulated 1.7-4.0ms first-spike latency is
  consistent with that characterization. Could NOT pin an exact
  published millisecond figure for direct numeric comparison - the
  specific numbers live in paywalled figures, not extractable text.
  Should not be overclaimed as a precise numeric match, only a
  qualitative one.
- MBON valence check (approach- vs avoidance-promoting MBON types) was
  NOT completed as a validated finding. Real compartment-specific
  dopamine biology (PAM=reward vs PPL1=punishment, targeting different
  MB compartments) is not yet modeled - our current rule depresses
  KC->MBON synapses uniformly regardless of which MBON/compartment
  they target. Listed which real MBON types (MBON09, MBON11, MBON12,
  MBON22...) receive the most odor-A KC input for future reference,
  but a real approach/avoidance validation needs compartment-specific
  PAM/PPL1 modeling first - a legitimate next step, not done here.

## Known limitations / honest caveats
- Single hemisphere / simplified odor representation (directly driving
  KC ensembles rather than real PN->KC input from the antennal lobe).
- Dopamine signal is an experimenter-controlled flag per trial, not
  actual spiking DAN neurons.
- No compartment-specificity in the plasticity rule.
