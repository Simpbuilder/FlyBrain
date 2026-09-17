# Debugging history

One-off diagnostic scripts kept for transparency, not part of the main results.

`debug_stim.py` through `debug_stim5.py` chase down the same bug in progressively
larger test cases: a persistent "zero spikes" symptom when reusing a Brian2
`Network` across repeated trials. The real cause (found in `debug_stim.py`'s
third trial) was that `defaultclock.t` returns a live-updating view, not a
snapshot — so `t_start = defaultclock.t` captured before a trial's `net.run()`
silently reflected the *post-run* time by the time it was compared against,
making every "spikes during this trial" filter come back empty. Fixed
everywhere by snapshotting as a plain float (`defaultclock.t / second`)
instead. See the main `NOTES.md` for the full story.

`tiny_comp.csv` / `tiny_con.parquet` are a synthetic 5-neuron connectome used
in `debug_stim2.py` to reproduce the bug against the real `create_model()`
function without needing the full 138k-neuron network.
