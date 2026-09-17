"""
Standardized export for pushing simulation results to the live dashboard.

Any experiment script calls export_for_dashboard(...) after a run to produce
one JSON file in the exact shape the dashboard page expects. Claude then
reads that file and pushes it to the db capability's brain_live/current doc
via the Artifact tool's write_db action - no custom formatting needed per
experiment, now or in future sessions.

Usage (inside an experiment script, after a trial's spk_mon has data):
    from dashboard_export import export_for_dashboard
    export_for_dashboard(
        name='Wind-sensing: JO-B -> DNg29 / Giant Fiber',
        spk_mon=spk_mon, t_start=t_start, duration_ms=300,
        pos_x=pos_x, pos_y=pos_y,                    # brian-index -> real coords
        highlight={'jo_b': jo_idx, 'dng29': list(dng29_arr), 'gf': list(gf_arr)},
    )
"""
import json
import numpy as np
from brian2 import second


def export_for_dashboard(name, spk_mon, t_start, duration_ms, pos_x, pos_y, highlight,
                          out_path='../dashboard_live.json'):
    """
    name: short string describing the experiment, shown in the dashboard
    spk_mon: brian2 SpikeMonitor with data from the run (spk_mon.i, spk_mon.t)
    t_start: float seconds - defaultclock.t/second captured BEFORE this trial's
             net.run() (as a plain float snapshot, not the live view - see NOTES.md)
    duration_ms: trial duration in ms
    pos_x, pos_y: arrays indexed by brian id giving real x/y coordinates
    highlight: dict of category_name -> list of brian ids to track individually.
               category names become the 'cat' field the page colors by.
    """
    trial_i = np.asarray(spk_mon.i)
    trial_t = np.asarray(spk_mon.t / second)  # plain float seconds
    mask = trial_t >= t_start
    trial_i = trial_i[mask]
    trial_t = (trial_t[mask] - t_start) * 1000  # ms since trial start

    cat_of = {}
    for cat, ids in highlight.items():
        for nid in ids:
            cat_of[int(nid)] = cat

    highlight_ids = set(cat_of.keys())
    keep = np.isin(trial_i, list(highlight_ids))
    hi_i, hi_t = trial_i[keep], trial_t[keep]

    neurons = {}
    for nid in np.unique(hi_i):
        nid = int(nid)
        if np.isnan(pos_x[nid]) or np.isnan(pos_y[nid]):
            continue
        neurons[str(nid)] = {'x': round(float(pos_x[nid]), 1), 'y': round(float(pos_y[nid]), 1), 'cat': cat_of[nid]}

    spikes = sorted(
        ([int(i), round(float(t), 1)] for i, t in zip(hi_i, hi_t) if str(int(i)) in neurons),
        key=lambda p: p[1]
    )

    out = {
        'name': name,
        'duration_ms': duration_ms,
        'neurons': neurons,
        'spikes': spikes,
        'categories': sorted(highlight.keys()),
    }
    with open(out_path, 'w') as f:
        json.dump(out, f)
    print(f'>>> exported {len(neurons)} neurons, {len(spikes)} spikes to {out_path} for dashboard push', flush=True)
    return out
