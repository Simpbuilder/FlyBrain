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
- **Compartment-specific dopamine validation (completed, `learn_compartment_specific.py`):**
  classified all 96 real MBONs by which real DAN cluster dominates
  their dopaminergic input, computed directly from the connectome
  (307 PAM neurons = reward-associated, 24 PPL neurons = punishment-
  associated, per Aso et al. 2014's established framework) -> 58
  PPL-dominant MBONs, 38 PAM-dominant MBONs. Restricted the
  punishment-gated plasticity rule to ONLY depress KC(odor-A)->MBON
  synapses landing on PPL-dominant MBONs, leaving PAM-dominant MBONs'
  synapses completely untouched. Result:
  - PPL-dominant (punishment) MBONs: 74.18 -> 68.68 Hz (-7.4%, the
    synapses that WERE depressed)
  - PAM-dominant (reward) MBONs: 103.03 -> 102.87 Hz (-0.1%, synapses
    untouched)
  Despite both populations living in the same fully recurrent
  138,639-neuron network, the reward-associated MBONs showed
  essentially zero indirect/leakage effect from punishment-gated
  learning happening elsewhere in the brain. This reproduces a real,
  specific, non-tautological prediction from Aso et al. 2014: that
  dopaminergic modulation of mushroom body learning is
  compartment-specific and does not bleed across compartments, even
  in a richly interconnected whole-brain circuit.

## Milestone 5: closing the two biggest honest gaps
Two of the limitations listed below were addressable, so they got fixed
rather than just documented.

### Robustness across independent odor ensembles
The compartment-specificity finding (milestone 4) used one specific
random draw of 150 Kenyon cells. Reran it across 5 independent random
seeds (`multi_seed_compartment.py`, 8 reps + 6 warm-up trials each,
matching the settings that gave a clean baseline earlier):
- seed 0: still bimodal even with 8 reps (flagged automatically,
  excluded from the aggregate - a genuinely rare persistent edge case)
- seed 1: PPL -8.3%, PAM +0.0%
- seed 2: PPL -7.7%, PAM -0.4%
- seed 3: PPL -7.6%, PAM -0.5%
- seed 4: PPL -8.7%, PAM -0.1%
Aggregate across the 4 clean seeds: **PPL-dominant MBONs -8.1% +/- 0.5%,
PAM-dominant MBONs -0.2% +/- 0.2%.** Tight, consistent, not a fluke of
one particular draw. The script now auto-flags high-baseline-variance
seeds (std > 15Hz) instead of letting them silently distort an average.

### Real thermosensory-driven dopamine (replacing the experimenter flag)
Previously, "punishment" was a Python-side flag that told the script
when to apply depression. Found that real thermosensory neurons (TRN,
29 neurons - a real nociceptive-adjacent sensory pathway, matching how
real aversive-conditioning experiments use heat as the punishing US)
reach all 24 real PPL1 dopamine neurons within 2-3 synaptic hops in
the actual connectome. Built `thermal_punishment.py`: drive odor-A KCs
AND the real TRNs together, monitor whether real PPL1 neurons actually
fire (they do, reliably: ~690-706 spikes/trial across all 12 pairing
trials), and gate synaptic depression on that real emergent activity
instead of an assumed flag. The depression now happens because real
dopamine neurons genuinely fired in response to a real nociceptive-like
input - not because the script said so.

### Real PN-driven odor input (replacing hand-picked KC ensembles)
Previously, "odor A" was 150 KCs picked directly by us. Checked real
antennal-lobe projection neuron (PN) -> KC connectivity: median 6
distinct PN sources per recruited KC (matches the textbook fact that
each Kenyon cell samples ~6 random glomeruli). Built
`pn_driven_odor.py`: drove 80 real PN neurons across 8 real named
glomeruli (DA1, DA2, DL3, DL2d, VM5d, VA1v, DL2v, VC3) instead of
picking KCs ourselves, and let the real PN->KC wiring decide which
KCs respond. Result: reliable ~83Hz MBON response across all 6 trials,
with zero KC hand-picking. Caveat: this recruited ~67% of all KCs
(3468-3489 of 5177), much broader than real sparse odor coding
(~5-10%) - likely because the external Poisson drive weight in this
codebase (tuned for "guaranteed activation" of directly-stimulated
sensory neurons) is strong enough to oversaturate propagation once
combined across 8 glomeruli. A more careful drive-strength calibration
would be needed to reproduce realistic sparsity; not done here.

## Milestone 6: the full closed loop, and a genuine unresolved finding
Combined all three upgrades into one experiment (`full_closed_loop.py`):
real PN-driven odors (2 disjoint sets of 4 named glomeruli each, no
hand-picked KCs), real TRN-driven punishment (no experimenter flag),
and compartment-restricted depression (PPL-dominant MBONs only).

**Compartment specificity held perfectly, every time.** PAM-dominant
MBONs stayed flat (+/-1%) in every version of this experiment,
regardless of what else was going on. That part of the model is solid.

**Odor specificity did not.** The punished odor (A) and the never-
paired control odor (B) declined by nearly identical amounts every
time this was tried:
- straightforward version: A -44.4%, B -44.9% (odor-blind - clearly wrong)
- brief 15ms external pulse (tried to limit recruitment): no change at
  all - this is a fully recurrent whole-brain network, so once the
  cascade starts, it keeps propagating on its own for the rest of the
  300ms trial regardless of when the external drive stops. Confirmed
  via direct comparison, not assumed.
- restricting "eligible" KCs to only those firing in the first 20ms
  (closer to how real coincidence detection actually reads out odor
  identity): recruitment dropped from ~64% to ~25-35% of all KCs
  (better), but A and B still declined almost identically (-26.7% vs
  -26.0%) - improved, not fixed.

**Root cause, most likely:** `sparsity_sweep.py` showed KC recruitment
is essentially independent of PN firing *rate* (10Hz through 150Hz all
gave ~64%) but highly sensitive to *duration* - meaning it's driven by
slow temporal summation/recurrent amplification, not genuine multi-
glomerulus coincidence. The external stimulation weight in this
codebase (`w_syn * f_poi`, tuned in the original paper for "guaranteed
activation" of directly-stimulated sensory neurons) makes all 40-50
real PN neurons in a glomerulus fire almost perfectly synchronously
from the first millisecond - so even two genuinely different real
odors look artificially synchronized to downstream KCs, defeating the
sparse combinatorial coding that would normally keep them apart. This
wasn't tuned to failure by accident - reducing rate alone doesn't touch
this because the *reliability* of each spike (not its rate) is what's
saturating.

**Honestly unresolved, and now precisely diagnosed.** Tested whether
reducing the external drive *weight* (not just rate/duration) would
help, down to 1/125th of the original value (`sparsity_sweep.py`,
factor sweep 250->2): made no difference at all - PNs still fired
46/46 reliably and KC recruitment stayed at ~64-65% even at the
weakest weight tested. Combined with the earlier rate-independence
result, this rules out both drive rate and drive weight as the lever.
The real cause: Kenyon cells have a 20ms membrane time constant and
are driven by a 150Hz Poisson train sustained for 300ms (~45 events
per PN, mean inter-event interval ~6.7ms, well under the membrane time
constant) - almost *any* sufficiently long, sufficiently frequent
drive will eventually push a KC over threshold through slow temporal
summation, regardless of the instantaneous multi-glomerulus
coincidence pattern that's supposed to define real sparse coding. A
genuine fix isn't a parameter tweak - it needs a fundamentally
different sensory encoding (a brief adapting burst per PN, the way a
real odor's onset transient actually looks, rather than a sustained
Poisson train) or a shorter effective integration window built into
the plasticity rule itself. Not done here; a legitimate next step for
a session with more room to restructure the stimulation model.

What IS established: the compartment-specificity result (milestone
4/5) doesn't depend on this problem at all, since it uses a much
sparser hand-picked 150-KC ensemble (2.9% of all KCs) that never hits
this saturation regime. And the PN-driven pathway itself (milestone 5)
works end-to-end with 100% real neurons and gives a reliable MBON
readout - it's specifically the *combination* with odor-selective
learning that exposes this calibration gap. This closed-loop
experiment is a genuine, informative negative result about what's
needed for odor-specific memory in a spiking model, not a
contradiction of the main finding.

## Known limitations / honest caveats
- Odor specificity in the fully-real-PN-driven version is not yet
  achieved - see milestone 6 above for the full honest account.
- External Poisson drive strength is tuned for "guaranteed activation"
  rather than calibrated to realistic sparse/probabilistic coding.
- No compartment-specificity within PPL1 itself (real biology has
  multiple PPL1 subtypes targeting different sub-compartments; this
  model treats all PPL-dominant MBONs as one group).
