# Results summary

All numbers from real connectome simulations (FlyWire, 138,639 neurons). Full narrative and caveats in [`Drosophila_brain_model/NOTES.md`](Drosophila_brain_model/NOTES.md).

## Circuit replications (innate reflexes)

| Circuit | Input | Output | Result |
|---|---|---|---|
| Sugar → proboscis | 21 sugar-sensing neurons | MN9 (proboscis motor neuron) | reliably fires, 63 Hz |
| Looming → escape | LC4+LPLC2 (314 neurons) | Giant Fiber (DNp01) | 5/5 trials, 1.7-4.0ms latency |
| Optomotor steering | HS/VS (38 neurons) | DNg46 (steering neuron) | 15/15 trials, 45.9±3.7 spikes, 5.7±2.7ms latency |

## Mushroom body learning (dopamine-gated plasticity)

| Experiment | Result |
|---|---|
| Odor A (punished) vs B (control), 150-KC ensembles | A: 86.2→80.8 Hz (-6.4%); B: 86.8→85.3 Hz (-1.7%, noise) |
| Compartment specificity (PPL vs PAM MBONs), 5 clean seeds | PPL: -7.96% (touched); PAM: -0.16% (untouched) |
| Real thermosensory-driven punishment (no experimenter flag) | 74.19→68.79 Hz (-7.3%), real PPL1 DAN firing confirmed every trial |
| Reward learning (symmetric, PAM-gated) | PAM: -4.6% (touched); PPL: +1.7% (untouched) |
| Full closed loop (real PN odor + real heat + compartment-specific) | compartment specificity held; odor specificity did not (see NOTES.md milestone 6) |

## Visualizations published

- [Mushroom body memory](https://claude.ai/code/artifact/4271312e-0cff-40a8-b94b-3e84b0eaf7a7) — before/after spatial view of learning
- [Giant Fiber escape](https://claude.ai/code/artifact/dccecdb7-020e-4bcf-94e3-dc410c2ca4fa) — sub-3ms escape response
- [Compartment specificity](https://claude.ai/code/artifact/41377e0e-d918-438d-a909-009f0dbebc45) — PPL vs PAM spatial comparison

## Infrastructure fixes made along the way

- Diagnosed and fixed a Brian2 gotcha: `defaultclock.t` returns a live view, not a snapshot, silently breaking all "spikes during this trial" windowing logic.
- Diagnosed and fixed Brian2's unseeded internal RNG (separate from numpy's), which caused identical scripts to land in different stochastic outcomes run to run.
- Installed MSVC Build Tools, moving simulations off the slow numpy codegen fallback onto compiled Cython/C++.
