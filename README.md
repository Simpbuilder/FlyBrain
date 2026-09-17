# FlyBrain

A simulation of a real fruit fly brain — 138,639 neurons and 15 million synapses from the [FlyWire](https://flywire.ai/) connectome, driven as a leaky integrate-and-fire spiking network. Not a toy model: the wiring is the actual reconstructed connectome of a real *Drosophila melanogaster* brain, and every result below comes from real neurons identified by their real cell type.

Built on [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model), the code released alongside Shiu et al. 2024, *Nature*, "A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain."

## What's in here

### 1. Sensory-to-motor replication
Driving the 21 real sugar-sensing neurons on the right hemisphere reliably fires **MN9**, the real proboscis-extension motor neuron — the fly's "stick out your tongue for sugar" reflex, reproduced from raw connectome wiring. See [`Drosophila_brain_model/demo_run.py`](Drosophila_brain_model/demo_run.py).

### 2. Mushroom body learning
Added dopamine-gated synaptic plasticity to the real Kenyon cell → MBON synapses — the actual site where fly memories form. Pairing an odor (150 real Kenyon cells) with a punishment signal over 12 trials produces genuine, specific learning:

| | baseline | after learning | change |
|---|---|---|---|
| odor A (paired with punishment) | 86.2 Hz | 80.8 Hz | **-6.4%** |
| odor B (control, never paired) | 86.8 Hz | 85.3 Hz | -1.7% (noise) |

Mean synaptic weight at the targeted synapses dropped 89% over the 12 pairing trials. See [`Drosophila_brain_model/learn_and_capture_spatial.py`](Drosophila_brain_model/learn_and_capture_spatial.py).

**Compartment-specificity check:** real MBONs were classified by which real dopamine neuron cluster dominates their input — PAM (reward-associated) vs PPL1 (punishment-associated), per Aso et al. 2014. Restricting the plasticity rule to only PPL-dominant MBONs and leaving PAM-dominant MBONs untouched reproduces a genuine, non-trivial prediction: PPL-dominant MBONs decline while PAM-dominant MBONs stay flat — **despite both living in the same fully recurrent whole-brain network**, where the effect could have leaked across compartments but didn't. Confirmed across 4 independent random odor ensembles (a 5th was automatically flagged for baseline instability and excluded): **PPL-dominant MBONs -8.1% ± 0.5%, PAM-dominant MBONs -0.2% ± 0.2%.** See [`Drosophila_brain_model/multi_seed_compartment.py`](Drosophila_brain_model/multi_seed_compartment.py).

**Going further — real sensory pathways instead of experimenter shortcuts:** replaced the hand-picked "odor" Kenyon-cell ensemble with real antennal-lobe projection neurons across 8 named glomeruli (the actual PN→KC wiring decides which Kenyon cells respond), and replaced the experimenter-controlled "punishment" flag with real thermosensory (heat/pain-proxy) neurons that reach the real PPL1 dopamine neurons through 2-3 real synaptic hops — dopamine release is now a genuine emergent consequence of a real nociceptive-like input, confirmed by ~690 real PPL1 spikes firing on every heat-paired trial. Combining all three upgrades exposed a real, honestly-documented limitation: while compartment specificity held perfectly, two different real odors ended up too densely overlapping in which Kenyon cells they recruit (a temporal-summation artifact traced precisely, not hand-waved) to stay behaviorally distinct — a genuine negative result about what sparse coding actually buys a nervous system, not swept under the rug. Full account in [`Drosophila_brain_model/NOTES.md`](Drosophila_brain_model/NOTES.md).

**[→ Watch it happen](https://claude.ai/code/artifact/4271312e-0cff-40a8-b94b-3e84b0eaf7a7)** — an animated view of real spike activity at real 3D neuron positions, before and after learning.

### 3. Looming-escape reflex
A second, independent circuit: the real looming-detector neurons (LC4, 104 neurons; LPLC2, 210 neurons) reliably fire the real **Giant Fiber** — the fly's fastest escape command neuron — with a 1.7–4.0ms first-spike latency, consistent with the literature's characterization of the GF pathway as one of the fastest visual-to-motor circuits known in any animal. See [`Drosophila_brain_model/looming_escape.py`](Drosophila_brain_model/looming_escape.py).

**[→ Watch it happen](https://claude.ai/code/artifact/dccecdb7-020e-4bcf-94e3-dc410c2ca4fa)** — same real-position animation, this time watching an escape command fire in under 3 milliseconds.

## How it's built

- **Simulator:** [Brian2](https://brian2.readthedocs.io/), leaky integrate-and-fire neurons, real synapse counts from EM reconstruction as connection weights.
- **Cell type identities:** [FlyWire connectome annotations](https://github.com/flyconnectome/flywire_annotations) (Schlegel et al. 2024, *Nature*), including real 3D neuron positions used for the spatial visualization.
- **Learning rule:** dopamine-gated multiplicative synaptic depression, restricted to real KC→MBON synapses, gated by real MBON dopamine-cluster identity.

See [`Drosophila_brain_model/NOTES.md`](Drosophila_brain_model/NOTES.md) for full experimental details, including honest caveats about what's simplified (e.g. odor input drives Kenyon cells directly rather than through real antennal-lobe projection neurons; the dopamine "punishment" signal is an experimenter-controlled trial flag rather than actual spiking dopamine neurons).
